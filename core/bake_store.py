"""Bakes as Blender data, read with no Python at all (FUTURE.md 1.3).

**The store.** One hidden mesh per source, `.ACE_bake_<source uid>`, kept with a fake user:
one point per frame of the baked range and one attribute per signal output (named after its
output key, which is the signal's identity, ARCHITECTURE.md §1). Its custom properties hold the
range and the source file's identity (path, size, modification time, content hash). It is saved
with the .blend, so a bake survives reloads. An output whose key is in the store is baked; an
edit that changes a signal gives it a new key, so only the signals an edit changes lose their
curve.

**The layout.** After every compile and every bake, the curves go where the frontends read:

- Geometry Nodes: the proxy mesh (and ACE_Values) gets one point per frame, the baked
  attributes hold their curve, and ``ACE_FRAME0`` the frame of point 0. The node groups sample
  point ``floor(frame) - ACE_frame0``, clamped; outputs not baked are written live to every
  point, so they read the same at any index.
- Shaders: the scene properties of baked outputs are animated by F-curves converted to sample
  points, in the action ``ACE_Bake`` on its own NLA track of the scene, so the scene's own action
  stays untouched. View Layer Attribute nodes read them as before.

A source switched to live (``source["ace_bake_use"] = False``) keeps its store but is laid out as
not baked. A store whose file changed (and still exists) is stale and is removed. A store whose
file is missing is kept: the bake is all there is (a render farm without the audio).
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import bpy  # type: ignore
import numpy as np

from .. import utils
from .proxy.attributes import FRAME0_ATTR

STORE_PREFIX = ".ACE_bake_"
ACTION_NAME = "ACE_Bake"
NLA_TRACK_NAME = "ACE Bake"
SLOT_NAME = "ACE"
_COMPONENTS = {"FLOAT": 1, "FLOAT_VECTOR": 3, "FLOAT_COLOR": 4}
_FOREACH_KEY = {"FLOAT": "value", "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color"}


def store_name(src_uid: str) -> str:
    return f"{STORE_PREFIX}{src_uid}"[:63]


def _attr_name(attr_key: str) -> str:
    return str(attr_key).replace(":", "__")


def _attr_key(attr_name: str) -> str:
    node_id, sep, socket = str(attr_name).rpartition("__")
    return f"{node_id}:{socket}" if sep else str(attr_name)


def file_identity(filepath: str, with_hash: bool = True) -> Dict[str, Any]:
    """{"path", "size", "mtime", "hash"} of a file; empty when it does not exist."""
    path = str(filepath or "")
    try:
        stat = os.stat(path)
    except OSError:
        return {}
    identity: Dict[str, Any] = {"path": path, "size": int(stat.st_size), "mtime": float(stat.st_mtime), "hash": ""}
    if with_hash:
        digest = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        identity["hash"] = digest.hexdigest()
    return identity


@dataclass
class BakeView:
    """A source's store, read into arrays: what is baked, over which frames."""

    src_uid: str
    frame_start: int
    frame_end: int
    version: int
    identity: Dict[str, Any]
    series: Dict[str, np.ndarray] = field(default_factory=dict)  # output key -> (n,) or (n, k)
    types: Dict[str, str] = field(default_factory=dict)
    origin: str = "user"  # "user" (the Bake button) or "take" (a recorded live session)

    def index(self, frame: int) -> int:
        return int(min(max(int(frame) - self.frame_start, 0), self.frame_end - self.frame_start))

    def values_at(self, frame: int, keys: Iterable[str]) -> Dict[str, Any]:
        i = self.index(frame)
        out: Dict[str, Any] = {}
        for key in keys:
            arr = self.series.get(key)
            if arr is None:
                continue
            out[key] = float(arr[i]) if arr.ndim == 1 else tuple(float(x) for x in arr[i])
        return out

    def covers(self, frame_start: int, frame_end: int) -> bool:
        return self.frame_start <= int(frame_start) and self.frame_end >= int(frame_end)


class BakeStore:
    """Read and write the per-source store meshes (main thread)."""

    def __init__(self):
        self._views: Dict[str, BakeView] = {}
        self._hash_cache: Dict[Tuple[str, int, float], str] = {}
        self._scene_signatures: Dict[str, tuple] = {}

    def clear_runtime(self) -> None:
        """Forget what was read (a file load): the stores themselves are Blender data."""
        self._views.clear()
        self._scene_signatures.clear()

    @staticmethod
    def _mesh(src_uid: str):
        return bpy.data.meshes.get(store_name(src_uid))

    def read(self, src_uid: str) -> Optional[BakeView]:
        mesh = self._mesh(src_uid)
        if mesh is None:
            self._views.pop(str(src_uid), None)
            return None
        version = int(mesh.get("ace_bake_version", 0) or 0)
        cached = self._views.get(str(src_uid))
        if cached is not None and cached.version == version:
            return cached
        frame_start = int(mesh.get("ace_frame_start", 0) or 0)
        n = len(mesh.vertices)
        view = BakeView(
            src_uid=str(src_uid),
            frame_start=frame_start,
            frame_end=frame_start + max(n - 1, 0),
            version=version,
            identity={k: mesh.get(f"ace_file_{k}") for k in ("path", "size", "mtime", "hash")},
            origin=str(mesh.get("ace_bake_origin", "user") or "user"),
        )
        for attr in mesh.attributes:
            if attr.domain != "POINT" or attr.data_type not in _COMPONENTS or attr.name == "position":
                continue
            comps = _COMPONENTS[attr.data_type]
            buf = np.empty(n * comps, dtype=np.float32)
            attr.data.foreach_get(_FOREACH_KEY[attr.data_type], buf)
            key = _attr_key(attr.name)
            view.series[key] = buf if comps == 1 else buf.reshape(n, comps)
            view.types[key] = attr.data_type
        self._views[str(src_uid)] = view
        return view

    def write(self, src_uid: str, frame_start: int, frame_end: int, series: Dict[str, Any],
              types: Dict[str, str], identity: Dict[str, Any], origin: str = "user") -> BakeView:
        """Store curves over [frame_start, frame_end]; another range or file starts the store again.

        *origin* is "user" (the Bake button) or "take" (a recorded live session).
        """
        utils.assert_main_thread("BakeStore.write")
        n = max(int(frame_end) - int(frame_start) + 1, 1)
        mesh = self._mesh(src_uid)
        same = (
            mesh is not None
            and int(mesh.get("ace_frame_start", -1)) == int(frame_start)
            and len(mesh.vertices) == n
            and str(mesh.get("ace_file_hash", "") or "") == str(identity.get("hash", "") or "")
        )
        if mesh is not None and not same:
            bpy.data.meshes.remove(mesh)
            mesh = None
        if mesh is None:
            mesh = bpy.data.meshes.new(store_name(src_uid))
            mesh.use_fake_user = True
            mesh.vertices.add(n)
        for key, values in series.items():
            attr_type = str(types.get(key, "FLOAT") or "FLOAT")
            comps = _COMPONENTS.get(attr_type, 1)
            arr = np.asarray(values, dtype=np.float32).reshape(-1, comps) if comps > 1 else np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.shape[0] != n:
                continue
            name = _attr_name(key)
            attr = mesh.attributes.get(name)
            if attr is not None and attr.data_type != attr_type:
                mesh.attributes.remove(attr)
                attr = None
            if attr is None:
                attr = mesh.attributes.new(name=name, type=attr_type, domain="POINT")
            attr.data.foreach_set(_FOREACH_KEY[attr_type], np.ascontiguousarray(arr, dtype=np.float32).ravel())
        mesh["ace_source_uid"] = str(src_uid)
        mesh["ace_bake_origin"] = str(origin)
        mesh["ace_frame_start"] = int(frame_start)
        for k in ("path", "size", "mtime", "hash"):
            value = identity.get(k)
            if value is not None:
                mesh[f"ace_file_{k}"] = value
        mesh["ace_bake_version"] = int(mesh.get("ace_bake_version", 0) or 0) + 1
        return self.read(src_uid)

    def clear(self, src_uid: str) -> None:
        mesh = self._mesh(src_uid)
        if mesh is not None:
            bpy.data.meshes.remove(mesh)
        self._views.pop(str(src_uid), None)

    def is_stale(self, view: BakeView, filepath: str) -> bool:
        """True when the source's file exists and is not the one baked."""
        now = file_identity(filepath, with_hash=False)
        if not now:
            return False  # nothing to compare with: keep the bake
        baked = view.identity
        if int(baked.get("size") or -1) != now["size"]:
            return True
        if baked.get("mtime") is not None and abs(float(baked["mtime"]) - now["mtime"]) < 1e-6:
            return False
        key = (now["path"], now["size"], now["mtime"])
        digest = self._hash_cache.get(key)
        if digest is None:
            digest = file_identity(filepath)["hash"]
            self._hash_cache[key] = digest
        return digest != str(baked.get("hash") or "")


# ---------------------------------------------------------------------------
# Layout: from the stores to what the frontends read.
# ---------------------------------------------------------------------------

def _resize_points(mesh, n: int) -> bool:
    """Give *mesh* exactly *n* points, keeping its point attributes (values are rewritten anyway)."""
    count = len(mesh.vertices)
    if count == n:
        return False
    if count < n:
        mesh.vertices.add(n - count)
        return True
    layout = [(a.name, a.data_type) for a in mesh.attributes if a.domain == "POINT" and a.name != "position" and not a.name.startswith(".")]
    mesh.clear_geometry()
    mesh.vertices.add(n)
    for name, data_type in layout:
        if mesh.attributes.get(name) is None:
            mesh.attributes.new(name=name, type=data_type, domain="POINT")
    return True


def _padded(series: np.ndarray, offset: int, n: int) -> np.ndarray:
    """*series* placed at *offset* in *n* frames, its first and last value held outside it."""
    arr = np.asarray(series, dtype=np.float32)
    idx = np.clip(np.arange(n) - int(offset), 0, arr.shape[0] - 1)
    return arr[idx]


def _layout_mesh(obj, entries) -> bool:
    """Write the baked curves of one proxy (or ACE_Values); True when its point count changed."""
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "attributes"):
        return False
    if entries:
        first = min(fs for _n, _t, s, fs in entries)
        last = max(fs + s.shape[0] - 1 for _n, _t, s, fs in entries)
        n = last - first + 1
    else:
        first, n = 0, 1
    resized = _resize_points(mesh, n)
    frame0 = mesh.attributes.get(FRAME0_ATTR)
    if entries:
        if frame0 is None or frame0.data_type != "INT":
            if frame0 is not None:
                mesh.attributes.remove(frame0)
            frame0 = mesh.attributes.new(name=FRAME0_ATTR, type="INT", domain="POINT")
        frame0.data.foreach_set("value", np.full(n, first, dtype=np.int32))
        for name, attr_type, series, fs in entries:
            attr = mesh.attributes.get(name)
            if attr is not None and attr.data_type != attr_type:
                mesh.attributes.remove(attr)
                attr = None
            if attr is None:
                attr = mesh.attributes.new(name=name, type=attr_type, domain="POINT")
            attr.data.foreach_set(_FOREACH_KEY[attr_type], np.ascontiguousarray(_padded(series, fs - first, n)).ravel())
    elif frame0 is not None:
        mesh.attributes.remove(frame0)
    return resized


def _clear_scene_curves(scene) -> None:
    adt = getattr(scene, "animation_data", None)
    if adt is not None:
        track = adt.nla_tracks.get(NLA_TRACK_NAME)
        if track is not None:
            adt.nla_tracks.remove(track)
    action = bpy.data.actions.get(ACTION_NAME)
    if action is not None and action.users == 0:
        bpy.data.actions.remove(action)


def _layout_scene(scene, entries) -> None:
    """Animate the baked scene properties with sample-point F-curves on the ACE NLA track."""
    _clear_scene_curves(scene)
    if not entries:
        return
    action = bpy.data.actions.new(ACTION_NAME)
    slot = action.slots.new(scene.id_type, SLOT_NAME)
    strip = action.layers.new(SLOT_NAME).strips.new(type="KEYFRAME")
    channelbag = strip.channelbag(slot, ensure=True)
    first = min(fs for _n, s, fs in entries)
    for prop_name, series, fs in entries:
        arr = np.asarray(series, dtype=np.float32)
        columns = [arr] if arr.ndim == 1 else [arr[:, c] for c in range(arr.shape[1])]
        frames = np.arange(fs, fs + arr.shape[0], dtype=np.float32)
        for index, column in enumerate(columns):
            fcurve = channelbag.fcurves.new(data_path=f'["{prop_name}"]', index=index)
            co = np.empty(frames.size * 2, dtype=np.float32)
            co[0::2] = frames
            co[1::2] = column
            fcurve.keyframe_points.add(frames.size)
            fcurve.keyframe_points.foreach_set("co", co)
            fcurve.update()
            # Sample points: one value per frame, much lighter than keyframes. The end is exclusive.
            fcurve.convert_to_samples(int(fs), int(fs + arr.shape[0]))
    adt = scene.animation_data_create()
    track = adt.nla_tracks.new()
    track.name = NLA_TRACK_NAME
    nla_strip = track.strips.new(ACTION_NAME, int(first), action)
    nla_strip.action_slot = slot
    nla_strip.blend_type = "REPLACE"
    nla_strip.extrapolation = "HOLD"


def uses_bake(source, view: Optional[BakeView]) -> bool:
    """True when *source* reads its store: a file source unless switched to live, a live input
    only when told to use its take (it plays live otherwise)."""
    if view is None:
        return False
    source_type = str(getattr(source, "source_type", "") or "").upper()
    getter = getattr(source, "get", None)
    if source_type == "FILE":
        return bool(getter("ace_bake_use", True)) if callable(getter) else True
    if source_type == "LIVE":
        return view.origin == "take" and (bool(getter("ace_bake_use", False)) if callable(getter) else False)
    return False


def apply_layout(engine, store: BakeStore) -> None:
    """Lay the stores of the compiled sources out onto their proxies and scene properties.

    Sets ``graph.baked_attr_keys``: the outputs the frame runner leaves to Blender data.
    """
    utils.assert_main_thread("bake_store.apply_layout")
    writer = getattr(engine, "writer", None)
    per_object: Dict[str, list] = {}
    per_scene: Dict[str, list] = {}
    scene_signature: Dict[str, list] = {}
    objects: set = set()
    for uid, graph in engine.graphs.items():
        graph.baked_attr_keys = set()
        proxy_name = str(getattr(graph, "proxy_name", "") or "")
        if proxy_name:
            objects.add(proxy_name)
        source = getattr(graph, "source_context", None)
        view = store.read(str(uid))
        if view is None or not uses_bake(source, view):
            continue
        for key in graph.output_attr_keys:
            series = view.series.get(key)
            if series is None:
                continue
            graph.baked_attr_keys.add(key)
            for binding in (getattr(graph, "sink_bindings_by_attr", {}) or {}).get(key, []) or []:
                target = str(getattr(binding, "target_object_name", "") or "")
                name = str(getattr(binding, "target_data_name", "") or "")
                if not target or not name:
                    continue
                if binding.sink_kind == "PROXY_ATTRIBUTE":
                    per_object.setdefault(target, []).append((name[:63], view.types.get(key, "FLOAT"), series, view.frame_start))
                elif binding.sink_kind == "SCENE_PROPERTY":
                    per_scene.setdefault(target, []).append((name, series, view.frame_start))
                    scene_signature.setdefault(target, []).append((name, view.src_uid, view.version, key))
    if writer is not None and getattr(writer, "values_object_name", ""):
        objects.add(str(writer.values_object_name))
    objects |= set(per_object)
    for obj_name in sorted(objects):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        with utils.logged_errors("bake_store.apply_layout#mesh"):
            resized = _layout_mesh(obj, per_object.get(obj_name, []))
            if resized and writer is not None:
                # Cached attribute handles and last values belong to the old geometry.
                writer.clear_runtime_caches()
    for scene in bpy.data.scenes:
        signature = tuple(sorted(scene_signature.get(scene.name, [])))
        if store._scene_signatures.get(scene.name) == signature:
            continue
        with utils.logged_errors("bake_store.apply_layout#scene"):
            _layout_scene(scene, per_scene.get(scene.name, []))
            store._scene_signatures[scene.name] = signature
