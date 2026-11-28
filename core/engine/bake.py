"""Bake jobs: a file source's outputs over the frame range, into Blender data (core/bake_store.py)."""

import os
import queue
import threading
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
        # How many controller Simulation Zones the last bake baked after its sources.
        self.controllers_baked: int = 0

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
            dict(payload.get("identity") or {}),
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
        if not self._bake_jobs:
            # The controllers read the values just baked: their Simulation Zones bake after them.
            scene = bpy.data.scenes.get(str(job.get("scene_name", "") or "")) or getattr(bpy.context, "scene", None)
            if scene is not None:
                with utils.logged_errors("bake.BakeManager._finalise#controllers"):
                    from ...nodes.geometry.controllers import bake_controllers

                    self.controllers_baked = bake_controllers(scene)

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

    def bake_source_to_cache(self, source, tree=None, scene=None, keys: Optional[Iterable[str]] = None):
        """Bake a file source's outputs over the scene's frame range, on a thread.

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

        frame_start = int(getattr(scene, "frame_start", 1))
        frame_end = int(getattr(scene, "frame_end", frame_start))
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
