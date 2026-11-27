"""Proxy mesh attribute sinks: schema and per-frame writes."""

import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any
from ..id_management import ACEIdManager
from ... import utils


# A source's tempo, on its proxy mesh for the GN Source node's BPM and Beats per Bar outputs.
SOURCE_TEMPO_ATTRS = {"ACE_bpm": "FLOAT", "ACE_beats_per_bar": "INT"}


def write_source_tempo(proxy, bpm: float, beats_per_bar: int) -> bool:
    """Write a source's tempo on its proxy mesh; True when a value changed."""
    utils.assert_main_thread("write_source_tempo")
    mesh = getattr(proxy, "data", None)
    attributes = getattr(mesh, "attributes", None)
    if attributes is None or not len(getattr(mesh, "vertices", ())):
        return False
    changed = False
    for name, value in (("ACE_bpm", float(bpm)), ("ACE_beats_per_bar", int(beats_per_bar))):
        attr = attributes.get(name)
        if attr is not None and (attr.data_type != SOURCE_TEMPO_ATTRS[name] or attr.domain != "POINT"):
            attributes.remove(attr)
            attr = None
        if attr is None:
            attr = attributes.new(name=name, type=SOURCE_TEMPO_ATTRS[name], domain="POINT")
        elif all(item.value == value for item in attr.data):
            continue
        for item in attr.data:
            item.value = value
        changed = True
    return changed


class AttributeSinks:
    def _cache_output_defs(self):
        """Precompute attribute slugs and Blender data types for all writeable outputs."""
        if not self.abstract_tree:
            return

        for node in self.abstract_tree.nodes.values():
            is_stream = node.category == 'stream'
            is_rolling_avg = node.type == 'util_rolling_avg' and node.category == 'utilities'
            is_stream_accumulator = node.type == 'stream_accumulator' and node.category == 'utilities'
            if not (is_stream or is_rolling_avg or is_stream_accumulator):
                continue

            for sock in node.outputs:
                # Audio passthroughs have no value to write.
                if getattr(sock, "data_type", "") in ("NodeSocketBundle", "BUNDLE", "BUNDLE_SOCKET"):
                    continue
                # Key is unique per node instance and socket index
                key = f"{node.id}:{sock.socket_idx}"
                node_slug = ACEIdManager._clean_string(node.id if node.id else node.type)  # type: ignore
                socket_slug = ACEIdManager._clean_string(sock.name or sock.id or f"out{sock.socket_idx}")  # type: ignore
                slug = f"{node_slug}_{socket_slug or sock.socket_idx}"
                attr_type = self._resolve_attr_type(sock.data_type)
                params = node.parameters if isinstance(getattr(node, "parameters", {}), dict) else {}
                attr_id = params.get(f"attr_id_out_{sock.socket_idx}")
                attr_custom = bool(params.get(f"attr_id_custom_out_{sock.socket_idx}", False))
                self.output_defs[key] = {
                    "slug": slug,
                    "type": attr_type,
                    "node_id": node.id,
                    "socket_idx": sock.socket_idx,
                    "output_name": sock.name,
                    "attr_id": attr_id,
                    "attr_custom": attr_custom,
                    # Every use's sink of this output (a signal tree node stands for several uses).
                    "sinks": list((getattr(node, "instance_sinks", {}) or {}).get(sock.socket_idx, [])),
                    "frontend_uid": str(getattr(node, "frontend_uid", "") or ""),
                    "frontend_kind": str(getattr(node, "frontend_kind", "") or ""),
                }

    @staticmethod
    def _resolve_attr_type(sock_type: str) -> str:
        attr_type = 'FLOAT'
        if sock_type in ('VECTOR', 'NodeSocketVector'):
            attr_type = 'FLOAT_VECTOR'
        elif sock_type in ('RGBA', 'NodeSocketColor'):
            attr_type = 'FLOAT_COLOR'
        return attr_type

    @staticmethod
    def _is_prunable(attr_name: str, required) -> bool:
        """An attribute of a proxy mesh that no output writes any more (proxies are ACE's own meshes)."""
        return not (
            attr_name in required
            or attr_name in SOURCE_TEMPO_ATTRS
            or attr_name == "position"
            or attr_name.startswith(".")
        )

    def ensure_attributes_exist(self, proxy: bpy.types.Object, required: Dict[str, str], prune_existing: bool = True):
        """Create or repair the point attributes *required* names ({name: type}) on the proxy mesh.

        With *prune_existing*, attributes no output writes any more go.
        """
        utils.assert_main_thread("ensure_attributes_exist")
        mesh = getattr(proxy, "data", None)
        if not mesh:
            return
        attrs_changed = False
        for name, attr_type in required.items():
            attr = mesh.attributes.get(name)
            if attr and attr.data_type == attr_type:
                continue
            if attr:
                mesh.attributes.remove(attr)
            mesh.attributes.new(name=name, type=attr_type, domain='POINT')
            attrs_changed = True

        if prune_existing:
            for attr in list(mesh.attributes):
                if self._is_prunable(attr.name, required):
                    mesh.attributes.remove(attr)
                    attrs_changed = True

        if attrs_changed:
            # Attribute layout changed: cached RNA handles and last written values
            # may no longer map to valid sockets, so flush per-mesh caches.
            self._attr_cache.pop(mesh.name, None)
            self._last_values.pop(mesh.name, None)
            self._write_buffer_cache.pop(mesh.name, None)

    def _get_write_buffer(self, mesh_name: str, attr_name: str, length: int) -> np.ndarray:
        mesh_buffers = self._write_buffer_cache.setdefault(mesh_name, {})
        key = f"{attr_name}:{int(length)}"
        arr = mesh_buffers.get(key)
        if arr is None or int(arr.shape[0]) != int(length):
            arr = np.empty(int(length), dtype=np.float32)
            mesh_buffers[key] = arr
        return arr

    @staticmethod
    def _zero_value_for_type(attr_type: str):
        if attr_type == "FLOAT_VECTOR":
            return [0.0, 0.0, 0.0]
        if attr_type == "FLOAT_COLOR":
            return [0.0, 0.0, 0.0, 1.0]
        return 0.0

    @classmethod
    def _values_equal(cls, prev, curr) -> bool:
        """True when a write would store the same thing: the same float32 values, as they render.

        Not "close": a change skipped as small would leave a frame's value depending on the
        frames written before it, and a bake would no longer equal playback.
        """
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            return bool(np.float32(prev) == np.float32(curr))
        if isinstance(prev, np.ndarray):
            prev = prev.tolist()
        if isinstance(curr, np.ndarray):
            curr = curr.tolist()
        if isinstance(prev, (list, tuple)) and isinstance(curr, (list, tuple)):
            try:
                return bool(np.array_equal(np.asarray(prev, dtype=np.float32), np.asarray(curr, dtype=np.float32)))
            except Exception as exc:
                utils.note_swallowed("attributes.AttributeSinks._values_equal", exc)
                return False
        return prev == curr

    def update_proxy_object(self, proxy: bpy.types.Object, values: Dict[str, Any], pulses: Set[str] = frozenset()):
        """Write values ({attribute name: value}) into the proxy mesh's attributes.

        Names in *pulses* are written every frame, even when unchanged (pulse outputs reset).
        """
        utils.assert_main_thread("update_proxy_object")
        mesh = getattr(proxy, "data", None)
        if not mesh:
            return

        for attr_name, value in values.items():
            if not attr_name:
                continue
            attr = None
            cache = self._attr_cache.setdefault(mesh.name, {})
            if attr_name in cache:
                attr = cache[attr_name]
            else:
                attr = mesh.attributes.get(attr_name)
                if attr:
                    cache[attr_name] = attr
            if not attr or not attr.data:
                continue
            last_vals = self._last_values.setdefault(mesh.name, {})
            dtype = attr.data_type
            _is_transient = attr_name in pulses
            if dtype == 'FLOAT':
                try:
                    val = float(value)
                except Exception as exc:
                    utils.note_swallowed("attributes.AttributeSinks.update_proxy_object", exc)
                    continue
                prev = last_vals.get(attr_name)
                if not _is_transient and isinstance(prev, float) and np.float32(prev) == np.float32(val):
                    continue
                last_vals[attr_name] = val
            elif dtype in ('FLOAT_VECTOR', 'FLOAT_COLOR'):
                arr = None
                if hasattr(value, "tolist"):
                    arr = np.asarray(value.tolist(), dtype=np.float32)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value, dtype=np.float32)
                if arr is None:
                    continue
                target_len = 4 if dtype == 'FLOAT_COLOR' else 3
                flat = np.zeros(target_len, dtype=np.float32)
                flat[: min(target_len, arr.size)] = arr.ravel()[:target_len]
                prev = last_vals.get(attr_name)
                if isinstance(prev, np.ndarray) and prev.shape == flat.shape and np.array_equal(prev, flat):
                    continue
                last_vals[attr_name] = flat
            else:
                continue

            dtype = attr.data_type
            count = len(attr.data)
            try:
                if dtype == 'FLOAT':
                    if count == 1:
                        attr.data[0].value = float(val)
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count)
                        buf.fill(float(val))
                        attr.data.foreach_set("value", buf)
                elif dtype == 'FLOAT_VECTOR':
                    vec = last_vals.get(attr_name)
                    if vec is None:
                        continue
                    if count == 1:
                        attr.data[0].vector = tuple(float(x) for x in vec.tolist())
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count * 3)
                        np.copyto(buf.reshape(count, 3), vec)
                        attr.data.foreach_set("vector", buf)
                elif dtype == 'FLOAT_COLOR':
                    col = last_vals.get(attr_name)
                    if col is None:
                        continue
                    if count == 1:
                        attr.data[0].color = tuple(float(x) for x in col.tolist())
                    else:
                        buf = self._get_write_buffer(mesh.name, attr_name, count * 4)
                        np.copyto(buf.reshape(count, 4), col)
                        attr.data.foreach_set("color", buf)
            except Exception as exc:
                utils.note_swallowed("attributes.AttributeSinks.update_proxy_object#2", exc)
                # Fallback to zeroing on failure
                with utils.logged_errors("attributes.AttributeSinks.update_proxy_object#3"):
                    if dtype == 'FLOAT':
                        if count == 1:
                            attr.data[0].value = 0.0
                        else:
                            buf = self._get_write_buffer(mesh.name, attr_name, count)
                            buf.fill(0.0)
                            attr.data.foreach_set("value", buf)
                    elif dtype == 'FLOAT_VECTOR':
                        if count == 1:
                            attr.data[0].vector = (0.0, 0.0, 0.0)
                        else:
                            buf = self._get_write_buffer(mesh.name, attr_name, count * 3)
                            buf.fill(0.0)
                            attr.data.foreach_set("vector", buf)
                    elif dtype == 'FLOAT_COLOR':
                        if count == 1:
                            attr.data[0].color = (0.0, 0.0, 0.0, 1.0)
                        else:
                            buf = self._get_write_buffer(mesh.name, attr_name, count * 4)
                            buf.fill(0.0)
                            buf.reshape(count, 4)[:, 3] = 1.0
                            attr.data.foreach_set("color", buf)
