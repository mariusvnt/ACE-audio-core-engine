"""Offline bake of a file source's outputs into an in-memory cache."""

import bpy
import hashlib
import json
import numpy as np
import os
import queue
import tempfile
import threading
from typing import Any, Dict, List, Optional
from ... import utils
from ..tasks.graph import ACEAudioGraph
from ..id_management import ACEIdManager
from .common import close_task


class BakeManager:
    """Offline bake of a file source's outputs into an in-memory cache."""

    def __init__(self, engine):
        self.engine = engine
        self._bake_cache: Dict[str, Dict[str, Any]] = {}
        self._bake_jobs: Dict[str, Dict[str, Any]] = {}
        self._bake_result_queue: "queue.Queue[tuple]" = queue.Queue()

    @staticmethod
    def _safe_file_mtime(filepath: str) -> Optional[float]:
        if not filepath:
            return None
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return None

    @staticmethod
    def _cleanup_bake_entry(entry: Dict[str, Any] | None):
        if not isinstance(entry, dict):
            return
        npz_obj = entry.pop("_npz_cache", None)
        if npz_obj is not None:
            with utils.logged_errors("bake.BakeManager._cleanup_bake_entry"):
                close_fn = getattr(npz_obj, "close", None)
                if callable(close_fn):
                    close_fn()
        npz_path = str(entry.get("npz_path", "") or "")
        if npz_path:
            with utils.logged_errors("bake.BakeManager._cleanup_bake_entry#2"):
                if os.path.isfile(npz_path):
                    os.remove(npz_path)

    def _prune_bake_cache(self, sources, signatures):
        """Drop the bakes of sources gone, and of sources whose graph computes something else now.

        *signatures* maps a source uid to its graph's signal signature: an edit that leaves a
        source's signals as they were (a sink renamed, another source's branch) keeps its bake.
        """
        valid_ids: set[str] = set()
        existing_ids = [getattr(s, "stable_id", "") for s in sources if getattr(s, "stable_id", "")]
        for src in sources or []:
            sid = getattr(src, "stable_id", "") or ACEIdManager.ensure_source_uid(src, existing_ids=existing_ids)
            if sid:
                valid_ids.add(sid)
                if sid not in existing_ids:
                    existing_ids.append(sid)

        for sid in list(self._bake_cache.keys()):
            entry = self._bake_cache.get(sid, {})
            if sid not in valid_ids or entry.get("signal_signature") != signatures.get(sid):
                self._cleanup_bake_entry(entry if isinstance(entry, dict) else None)
                self._bake_cache.pop(sid, None)

    def covers(self, graph: ACEAudioGraph, frame_start: int, frame_end: int) -> bool:
        """True when *graph*'s source has a valid bake over the frames [frame_start, frame_end]."""
        entry = self._bake_cache.get(str(getattr(graph, "source_uid", "") or ""))
        if not isinstance(entry, dict):
            return False
        if int(entry.get("frame_start", 0)) > int(frame_start) or int(entry.get("frame_end", -1)) < int(frame_end):
            return False
        return self._get_baked_frame_results(graph, int(frame_start)) is not None

    def _get_baked_frame_results(self, graph: ACEAudioGraph, frame_index: int) -> Optional[Dict[str, Any]]:
        source_uid = getattr(graph, "source_uid", "") or getattr(getattr(graph, "source_context", None), "stable_id", "")
        if not source_uid:
            return None
        entry = self._bake_cache.get(source_uid)
        if not entry:
            return None
        if entry.get("signal_signature") != str(getattr(graph, "signal_signature", "") or ""):
            return None
        if int(entry.get("block_size", -1)) != int(getattr(graph, "block_size", 0)):
            return None

        source_ctx = getattr(graph, "source_context", None)
        if not source_ctx or getattr(source_ctx, "source_type", "") != "FILE":
            return None
        filepath = getattr(source_ctx, "filepath", "") or ""
        if not filepath or filepath != entry.get("filepath", ""):
            return None
        if self._safe_file_mtime(filepath) != entry.get("mtime"):
            return None

        frame_start = int(entry.get("frame_start", 0))
        frame_end = int(entry.get("frame_end", -1))
        if frame_index < frame_start or frame_index > frame_end:
            return None

        frame_offset = frame_index - frame_start
        values = entry.get("values", {})
        npz_cache = entry.get("_npz_cache", None)
        npz_path = str(entry.get("npz_path", "") or "")
        npz_keys = entry.get("npz_keys", {}) if isinstance(entry.get("npz_keys", {}), dict) else {}
        if npz_cache is None and npz_path:
            if not os.path.isfile(npz_path):
                return None
            try:
                npz_cache = np.load(npz_path, allow_pickle=False)
                entry["_npz_cache"] = npz_cache
            except Exception as exc:
                utils.note_swallowed("bake.BakeManager._get_baked_frame_results", exc)
                return None
        results: Dict[str, Any] = {}
        for attr_key in graph.output_attr_keys:
            series = values.get(attr_key) if isinstance(values, dict) else None
            if series is None and npz_cache is not None:
                npz_key = npz_keys.get(attr_key)
                if isinstance(npz_key, str) and npz_key in npz_cache:
                    series = npz_cache[npz_key]
            if isinstance(series, np.ndarray):
                if frame_offset >= int(series.shape[0]):
                    continue
                sample = series[frame_offset]
                if series.ndim <= 1:
                    results[attr_key] = float(sample)
                else:
                    arr = np.asarray(sample, dtype=np.float32).ravel()
                    results[attr_key] = tuple(float(x) for x in arr.tolist())
                continue
            if not isinstance(series, list) or frame_offset >= len(series):
                continue
            results[attr_key] = series[frame_offset]
        return results if results else None

    def _clear_bake_cache(self):
        for source_id in list(self._bake_cache.keys()):
            entry = self._bake_cache.get(source_id)
            self._cleanup_bake_entry(entry if isinstance(entry, dict) else None)
            self._bake_cache.pop(source_id, None)

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

    def _finalise_bake_arrays(self, src_uid: str, baked_values: Dict[str, List[Any]]) -> None:
        """Commit a finished bake on the main thread, if the source still computes what was baked."""
        job = self._bake_jobs.pop(str(src_uid), None)
        if not isinstance(job, dict):
            return
        for task in (job.get("tasks") or {}).values():
            close_task(task)
        signature = str(job.get("signal_signature", "") or "")
        filepath = str(job.get("filepath", "") or "")
        frame_start = int(job.get("frame_start", 0) or 0)
        frame_end = int(job.get("frame_end", frame_start) or frame_start)
        graph = self.engine.graphs.get(str(src_uid))
        if graph is not None and str(getattr(graph, "signal_signature", "") or "") != signature:
            # The source's signals changed while it baked: the curves are of a graph that no longer exists.
            utils.log(f"Bake of '{src_uid}' discarded: its signals changed while it ran.", "WARNING")
            self._set_source_status(job, status="stale", error="The nodes changed during the bake.", progress=0.0)
            return

        baked_arrays: Dict[str, np.ndarray] = {}
        for attr_key, series in dict(baked_values or {}).items():
            if not series:
                baked_arrays[attr_key] = np.zeros(0, dtype=np.float32)
                continue
            first = series[0]
            if isinstance(first, (tuple, list, np.ndarray)):
                arr = np.asarray(series, dtype=np.float32)
            else:
                arr = np.asarray(series, dtype=np.float32).reshape(-1)
            baked_arrays[attr_key] = arr.astype(np.float32, copy=False)

        total_bytes = int(sum(int(arr.nbytes) for arr in baked_arrays.values()))
        npz_path = ""
        npz_keys: Dict[str, str] = {}
        values_store: Dict[str, Any] = dict(baked_arrays)
        # Spill very large bake payloads to compressed disk cache to cap memory usage.
        if total_bytes >= (16 * 1024 * 1024) and baked_arrays:
            try:
                cache_dir = os.path.join(tempfile.gettempdir(), "ace_bake_cache")
                os.makedirs(cache_dir, exist_ok=True)
                sig_seed = json.dumps(
                    {
                        "source_uid": str(src_uid or ""),
                        "frame_start": int(frame_start),
                        "frame_end": int(frame_end),
                        "signal_signature": signature,
                    },
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
                sig = int.from_bytes(hashlib.blake2b(sig_seed, digest_size=4).digest(), byteorder="big")
                npz_path = os.path.join(cache_dir, f"ace_bake_{sig:08x}.npz")
                payload: Dict[str, np.ndarray] = {}
                for idx, (attr_key, arr) in enumerate(baked_arrays.items()):
                    npz_key = f"a{idx}"
                    payload[npz_key] = arr
                    npz_keys[attr_key] = npz_key
                np.savez_compressed(npz_path, **payload)
                values_store = {}
            except Exception as e:
                npz_path = ""
                npz_keys = {}
                values_store = dict(baked_arrays)
                utils.log_throttled("bake_spill", f"Could not spill bake cache to disk: {e}", "WARNING", min_interval=0.0)

        self._cleanup_bake_entry(self._bake_cache.get(str(src_uid)))
        self._bake_cache[str(src_uid)] = {
            "signal_signature": signature,
            "block_size": int(job.get("block_size", 0) or 0),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "values": values_store,
            "npz_path": npz_path,
            "npz_keys": npz_keys,
            "bake_bytes": total_bytes,
            "filepath": filepath,
            "mtime": self._safe_file_mtime(filepath),
        }
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
                    self._finalise_bake_arrays(src_uid, payload)
        except queue.Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("bake.BakeManager.flush_bake_results#3", exc)

    def bake_source_to_cache(self, source, tree=None, scene=None):
        """Precompute stream/spec outputs for a file source and cache them in memory."""
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
        if not graph.output_attr_keys:
            with utils.logged_errors("bake.BakeManager.bake_source_to_cache#3"):
                source["ace_bake_status"] = "error"
                source["ace_bake_error"] = "No stream/spec outputs to bake."
            return False

        frame_start = int(getattr(scene, "frame_start", 1))
        frame_end = int(getattr(scene, "frame_end", frame_start))
        if frame_end < frame_start:
            frame_start, frame_end = frame_end, frame_start
        total_frames = max(frame_end - frame_start + 1, 1)
        output_attr_keys = list(graph.output_attr_keys)

        with utils.logged_errors("bake.BakeManager.bake_source_to_cache#4"):
            source["ace_bake_status"] = "baking"
            source["ace_bake_error"] = ""
            source["ace_bake_progress"] = 0.0

        fresh = self.engine.compiler.fresh_graph_copy(graph)
        cancel = threading.Event()
        self._bake_jobs[str(src_uid)] = {
            "source_uid": str(src_uid),
            "scene_name": str(getattr(scene, "name", "") or ""),
            "signal_signature": str(getattr(graph, "signal_signature", "") or ""),
            "block_size": int(getattr(graph, "block_size", 0) or 0),
            "filepath": filepath,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "tasks": fresh.tasks,
            "cancel": cancel,
        }
        frame_runner = self.engine.frame
        result_queue = self._bake_result_queue

        def _bake_worker():
            baked_values: Dict[str, List[Any]] = {k: [] for k in output_attr_keys}
            try:
                for i, frame_idx in enumerate(range(frame_start, frame_end + 1)):
                    if cancel.is_set():
                        return
                    frame_results = frame_runner._collect_graph_results_for_frame(fresh, frame_idx)
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
                result_queue.put_nowait(("done", str(src_uid), baked_values))

        t = threading.Thread(target=_bake_worker, name=f"ACE_BakeWorker_{src_uid}", daemon=True)
        t.start()
        return True
