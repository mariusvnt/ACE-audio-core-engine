"""Bake jobs: a file source's outputs over the frame range, into Blender data (core/bake_store.py)."""

import os
import queue
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

import bpy
import numpy as np

from ... import utils
from ..bake_store import BakeStore, apply_layout, file_identity
from ..id_management import ACEIdManager
from ..tasks.graph import ACEAudioGraph
from .common import close_task


class BakeManager:
    """Runs bakes on fresh copies of the source graphs and keeps the stores laid out."""

    def __init__(self, engine):
        self.engine = engine
        self.store = BakeStore()
        self._bake_jobs: Dict[str, Dict[str, Any]] = {}
        self._bake_result_queue: "queue.Queue[tuple]" = queue.Queue()
        # Bake nodes and simulation zones downstream of ACE nodes (find_downstream_bakes).
        self.downstream_bakes: List[Dict[str, Any]] = []

    # -- reading ------------------------------------------------------------

    def view(self, graph: ACEAudioGraph):
        """The store of *graph*'s source when its outputs are laid out as baked, else None."""
        if not getattr(graph, "baked_attr_keys", None):
            return None
        return self.store.read(str(getattr(graph, "source_uid", "") or ""))

    def covers(self, graph: ACEAudioGraph, frame_start: int, frame_end: int) -> bool:
        """True when every output of *graph* is baked over [frame_start, frame_end]."""
        view = self.view(graph)
        return (
            view is not None
            and set(graph.output_attr_keys) <= set(getattr(graph, "baked_attr_keys", ()) or ())
            and view.covers(frame_start, frame_end)
        )

    def _get_baked_frame_results(self, graph: ACEAudioGraph, frame_index: int) -> Optional[Dict[str, Any]]:
        """The baked outputs' values at a frame (they are in Blender data already), or None."""
        view = self.view(graph)
        if view is None:
            return None
        return view.values_at(frame_index, graph.baked_attr_keys)

    # -- compile hooks ------------------------------------------------------

    # Edits must have been quiet this long before a held source bakes what it is missing.
    DOWNSTREAM_DEBOUNCE_S = 1.0

    def find_downstream_bakes(self, scene) -> List[Dict[str, Any]]:
        """The Geometry Nodes bakes (Bake nodes, simulation zones) of modifiers whose trees read ACE.

        A native bake records what the proxies hold at each frame it steps, so the ACE sources
        such a modifier reads are kept baked over the bake's range (FUTURE.md 1.4).
        """
        tree = getattr(self.engine, "instance_tree_ref", None)
        if tree is None or scene is None:
            return []
        sources_by_owner: Dict[str, set] = {}
        for node in tree.nodes.values():
            frontend = tree.frontends.get(node.frontend_uid)
            for owner in tuple(getattr(frontend, "owner_keys", ()) or ()):
                sources_by_owner.setdefault(str(owner), set()).update(str(s) for s in (node.root_sources or ()) if str(s))
        found: List[Dict[str, Any]] = []
        for obj in scene.objects:
            for mod in getattr(obj, "modifiers", []) or []:
                if getattr(mod, "type", "") != "NODES" or not len(getattr(mod, "bakes", ()) or ()):
                    continue
                sources = sources_by_owner.get(f"{obj.name}:{mod.name}")
                if not sources:
                    continue
                for bake in mod.bakes:
                    node = getattr(bake, "node", None)
                    if node is None or getattr(node, "mute", False):
                        continue
                    if bake.use_custom_simulation_frame_range:
                        frame_range = (int(bake.frame_start), int(bake.frame_end))
                    elif scene.use_custom_simulation_range:
                        frame_range = (int(scene.simulation_frame_start), int(scene.simulation_frame_end))
                    else:
                        frame_range = (int(scene.frame_start), int(scene.frame_end))
                    found.append({
                        "label": f"{obj.name} › {mod.name} › {node.name}",
                        "object": obj.name,
                        "modifier": mod.name,
                        "bake_id": int(bake.bake_id),
                        "sources": sorted(sources),
                        "frame_start": min(frame_range),
                        "frame_end": max(frame_range),
                    })
        return found

    def held_ranges(self) -> Dict[str, tuple]:
        """{source uid: (first, last frame)} every downstream bake needs baked."""
        held: Dict[str, tuple] = {}
        for bake in self.downstream_bakes:
            for uid in bake["sources"]:
                first, last = held.get(uid, (bake["frame_start"], bake["frame_end"]))
                held[uid] = (min(first, bake["frame_start"]), max(last, bake["frame_end"]))
        return held

    def maintain_downstream(self, scene, force: bool = False) -> Optional[str]:
        """Keep the sources downstream bakes need baked; release what none needs (the 0.25 s poll).

        Only the outputs missing from a source's store are baked (an edit changes only some
        signals), one job at a time, once edits have been quiet for DOWNSTREAM_DEBOUNCE_S; the
        proxies read live values meanwhile. Returns the uid of a source it started baking.
        """
        if self._bake_jobs or scene is None:
            return None
        frontends = self.engine.frontends
        if not force:
            quiet_since = float(getattr(self.engine.compiler, "_last_compile_time", 0.0) or 0.0)
            if getattr(frontends, "_dirty_frontend_uids", None) or getattr(frontends, "_pending_scene_rebuild", False):
                return None
            if time.time() - quiet_since < self.DOWNSTREAM_DEBOUNCE_S:
                return None
        held = self.held_ranges()
        sources = {str(s.stable_id): s for s in getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []}
        for uid, (first, last) in sorted(held.items()):
            source = sources.get(uid)
            graph = self.engine.graphs.get(uid)
            if source is None or graph is None or getattr(source, "source_type", "") != "FILE":
                continue
            if not bool(source.get("ace_bake_use", True)):
                continue
            missing = self.missing_keys(graph, first, last)
            if missing and self.bake_source_to_cache(
                source, scene=scene, keys=missing, frame_range=(first, last), compile=False, origin="downstream"
            ):
                return uid
        released = False
        for uid in sources:
            view = self.store.read(uid)
            if view is not None and view.origin == "downstream" and uid not in held:
                self.store.clear(uid)
                released = True
        if released:
            self.layout()
        return None

    def after_compile(self, sources, scene=None) -> None:
        """Drop the stores of files that changed, then lay every store out (main thread)."""
        for source in sources or []:
            uid = str(getattr(source, "stable_id", "") or "")
            if not uid or getattr(source, "source_type", "") != "FILE":
                continue
            view = self.store.read(uid)
            if view is not None and self.store.is_stale(view, str(getattr(source, "filepath", "") or "")):
                utils.log(f"Bake of '{getattr(source, 'name', uid)}' dropped: its file changed.", "WARNING")
                self.store.clear(uid)
                with utils.logged_errors("bake.BakeManager.after_compile"):
                    source["ace_bake_status"] = "stale"
                    source["ace_bake_error"] = "The file changed since the bake."
        with utils.logged_errors("bake.BakeManager.after_compile#downstream"):
            self.downstream_bakes = self.find_downstream_bakes(scene or getattr(bpy.context, "scene", None))
        self.layout()

    def layout(self) -> None:
        with utils.logged_errors("bake.BakeManager.layout"):
            apply_layout(self.engine, self.store)

    def clear_source(self, src_uid: str) -> None:
        self.store.clear(src_uid)
        self.layout()

    def _clear_bake_cache(self):
        """A file load or an engine reset: what was read goes, the stores stay (they are Blender data)."""
        self.store.clear_runtime()

    # -- jobs ----------------------------------------------------------------

    @staticmethod
    def _find_source(scene_name: str, src_uid: str):
        """The source with *src_uid* in the scene, looked up again: a job holds no Blender references."""
        scene = bpy.data.scenes.get(scene_name) if scene_name else None
        if scene is None:
            scene = getattr(getattr(bpy, "context", None), "scene", None)
        ace = getattr(scene, "audiocore", None)
        for src in getattr(ace, "audio_sources", []) or []:
            if str(getattr(src, "stable_id", "") or "") == str(src_uid):
                return src
        return None

    def _set_source_status(self, job: Dict[str, Any], **values) -> None:
        source = self._find_source(str(job.get("scene_name", "") or ""), str(job.get("source_uid", "") or ""))
        if source is None:
            return
        with utils.logged_errors("bake.BakeManager._set_source_status"):
            for key, value in values.items():
                source[f"ace_bake_{key}"] = value

    def cancel_jobs(self) -> None:
        """Stop every running bake and drop its results (file load, add-on shutdown)."""
        for job in self._bake_jobs.values():
            cancel = job.get("cancel")
            if cancel is not None:
                cancel.set()
            for task in (job.get("tasks") or {}).values():
                close_task(task)
        self._bake_jobs.clear()
        try:
            while True:
                self._bake_result_queue.get_nowait()
        except queue.Empty:
            pass

    def _finalise(self, src_uid: str, payload: Dict[str, Any]) -> None:
        """Store a finished bake and lay it out (main thread)."""
        job = self._bake_jobs.pop(str(src_uid), None)
        if not isinstance(job, dict):
            return
        for task in (job.get("tasks") or {}).values():
            close_task(task)
        frame_start = int(job.get("frame_start", 0) or 0)
        frame_end = int(job.get("frame_end", frame_start) or frame_start)
        series: Dict[str, np.ndarray] = {}
        for key, values in dict(payload.get("values") or {}).items():
            if values:
                series[key] = np.asarray(values, dtype=np.float32)
        # Output keys are signal identities: a curve stays right for its key whatever was edited
        # meanwhile, so nothing is discarded here.
        self.store.write(
            str(src_uid), frame_start, frame_end, series, dict(job.get("types") or {}),
            dict(payload.get("identity") or {}), origin=str(job.get("origin", "user") or "user"),
        )
        self.layout()
        self._set_source_status(
            job,
            status="done",
            error="",
            progress=1.0,
            frame_start=frame_start,
            frame_end=frame_end,
            samples=max(frame_end - frame_start + 1, 0),
        )

    def flush_bake_results(self) -> None:
        """Drain completed async bake events and write results on the main thread."""
        q = getattr(self, "_bake_result_queue", None)
        if q is None:
            return
        try:
            while True:
                event = q.get_nowait()
                if not isinstance(event, tuple) or len(event) < 3:
                    continue
                kind = str(event[0] or "")
                src_uid = str(event[1] or "")
                payload = event[2]
                job = self._bake_jobs.get(src_uid)
                if not src_uid or not isinstance(job, dict):
                    continue
                if kind == "progress":
                    self._set_source_status(job, progress=float(max(0.0, min(float(payload), 1.0))))
                    continue
                if kind == "error":
                    self._bake_jobs.pop(src_uid, None)
                    for task in (job.get("tasks") or {}).values():
                        close_task(task)
                    self._set_source_status(job, status="error", error=str(payload or "Bake failed."))
                    continue
                if kind == "done" and isinstance(payload, dict):
                    self._finalise(src_uid, payload)
        except queue.Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("bake.BakeManager.flush_bake_results#3", exc)

    def missing_keys(self, graph: ACEAudioGraph, frame_start: int, frame_end: int) -> List[str]:
        """The outputs of *graph* with no curve in its store over the range."""
        view = self.store.read(str(getattr(graph, "source_uid", "") or ""))
        if view is None or view.frame_start != int(frame_start) or view.frame_end != int(frame_end):
            return list(graph.output_attr_keys)
        return [key for key in graph.output_attr_keys if key not in view.series]

    def bake_source_to_cache(self, source, tree=None, scene=None, keys: Optional[Iterable[str]] = None,
                             frame_range: Optional[tuple] = None, compile: bool = True, origin: str = "user"):
        """Bake a file source's outputs over the scene's frame range (or *frame_range*), on a thread.

        With *keys*, only those outputs (the others keep their curves); without, every output.
        """
        if not source or getattr(source, "source_type", "") != "FILE":
            return False
        if scene is None:
            scene = getattr(getattr(bpy, "context", None), "scene", None)
        if scene is None:
            return False
        filepath = getattr(source, "filepath", "") or ""
        if not filepath or not os.path.isfile(filepath):
            with utils.logged_errors("bake.BakeManager.bake_source_to_cache"):
                source["ace_bake_status"] = "error"
                source["ace_bake_error"] = "Missing source file."
            return False

        if compile:
            target_tree = tree if tree and self.engine.frontends._tree_has_ace_nodes(tree) else None
            try:
                self.engine.compiler.compile_scene_frontends(scene, preferred_tree=target_tree)
            except Exception as e:
                utils.log_throttled("bake_compile", f"Bake compile failed: {e}", "ERROR", min_interval=0.0)
                return False

        ace = getattr(scene, "audiocore", None)
        existing_ids = [getattr(s, "stable_id", "") for s in getattr(ace, "audio_sources", []) if getattr(s, "stable_id", "")]
        src_uid = ACEIdManager.ensure_source_uid(source, existing_ids=existing_ids)
        if str(src_uid) in self._bake_jobs:
            return True
        graph = self.engine.graphs.get(src_uid)
        if graph is None:
            src_proxy = getattr(source, "proxy_object_name", "")
            graph = next((g for g in self.engine.graphs.values() if getattr(g, "proxy_name", "") == src_proxy), None)
        if not self.engine.writer:
            return False
        if graph is None:
            with utils.logged_errors("bake.BakeManager.bake_source_to_cache#2"):
                source["ace_bake_status"] = "error"
                source["ace_bake_error"] = "Source graph unavailable."
            return False
        output_attr_keys = [k for k in (keys if keys is not None else graph.output_attr_keys) if k in graph.output_attr_keys]
        if not output_attr_keys:
            with utils.logged_errors("bake.BakeManager.bake_source_to_cache#3"):
                source["ace_bake_status"] = "error" if keys is None else "done"
                source["ace_bake_error"] = "No stream/spec outputs to bake." if keys is None else ""
            return keys is not None

        frame_start, frame_end = frame_range if frame_range else (int(getattr(scene, "frame_start", 1)), int(getattr(scene, "frame_end", 1)))
        frame_start, frame_end = int(frame_start), int(frame_end)
        if frame_end < frame_start:
            frame_start, frame_end = frame_end, frame_start
        total_frames = max(frame_end - frame_start + 1, 1)

        with utils.logged_errors("bake.BakeManager.bake_source_to_cache#4"):
            source["ace_bake_status"] = "baking"
            source["ace_bake_error"] = ""
            source["ace_bake_progress"] = 0.0

        fresh = self.engine.compiler.fresh_graph_copy(graph)
        output_defs = getattr(self.engine.writer, "output_defs", {}) or {}
        cancel = threading.Event()
        self._bake_jobs[str(src_uid)] = {
            "source_uid": str(src_uid),
            "scene_name": str(getattr(scene, "name", "") or ""),
            "filepath": filepath,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "types": {k: str((output_defs.get(k) or {}).get("type", "FLOAT") or "FLOAT") for k in output_attr_keys},
            "keys": list(output_attr_keys),
            "origin": str(origin),
            "tasks": fresh.tasks,
            "cancel": cancel,
        }
        frame_runner = self.engine.frame
        result_queue = self._bake_result_queue

        def _bake_worker():
            baked_values: Dict[str, List[Any]] = {k: [] for k in output_attr_keys}
            try:
                identity = file_identity(filepath)
                for i, frame_idx in enumerate(range(frame_start, frame_end + 1)):
                    if cancel.is_set():
                        return
                    frame_results = frame_runner._collect_graph_results_for_frame(fresh, frame_idx, attr_keys=output_attr_keys)
                    for attr_key in output_attr_keys:
                        baked_values[attr_key].append(
                            frame_results.get(attr_key, frame_runner._zero_value_for_attr(attr_key))
                        )
                    if (i % 32) == 0 or frame_idx == frame_end:
                        result_queue.put_nowait(("progress", str(src_uid), float(i + 1) / float(total_frames)))
            except Exception as exc:
                utils.note_swallowed("bake.BakeManager.bake_source_to_cache._bake_worker", exc)
                result_queue.put_nowait(("error", str(src_uid), str(exc)))
                return
            if not cancel.is_set():
                result_queue.put_nowait(("done", str(src_uid), {"values": baked_values, "identity": identity}))

        t = threading.Thread(target=_bake_worker, name=f"ACE_BakeWorker_{src_uid}", daemon=True)
        t.start()
        return True
