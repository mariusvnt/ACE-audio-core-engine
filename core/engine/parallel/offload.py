"""Worker-process DSP offload (Stage B): job submission, polling and result reuse."""

import time
from collections import deque
from typing import Any, Dict, List, Optional

import bpy
import numpy as np

from .... import utils
from ...abstract_tree import ACEAbstractTree
from ...tasks.chunk import SAMPLE_RATE
from ...tasks.graph import ACEAudioGraph
from ...tasks.source import SourceTask
from ...worker.job import evaluate_parallel_job
from ..common import BrokenProcessPool, engine_settings
from . import ir
from .pool import WorkerPool
from .slots import PayloadSlots, source_payload

_NO_FRAME = -10_000_000


class ParallelOffload:
    MAX_INFLIGHT_PER_SOURCE = 2
    STALE_JOB_MAX_MS = 66
    STALE_FRAME_GRACE = 2
    STALE_REUSE_MS = 50
    CANCEL_STALE_FRAME_DELTA = 4
    PAUSE_DRAIN_FRAMES = 3
    KEPT_RESULT_FRAMES = 4

    shard_attr_keys = staticmethod(ir.shard_attr_keys)
    is_offload_candidate = staticmethod(ir.is_offload_candidate)

    def __init__(self, engine):
        self.engine = engine
        self.pool = WorkerPool()
        self.slots = PayloadSlots()
        self.generation_id = 0
        self.job_seq = 0
        self.inflight: Dict[str, deque] = {}
        self.future_meta: Dict[Any, Dict[str, Any]] = {}
        # Untracked jobs still running in a worker, with the payload slots they read.
        self.retired: Dict[Any, List[str]] = {}
        self.results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.last_valid: Dict[str, Dict[str, Any]] = {}
        self.last_valid_frame: Dict[str, int] = {}
        self.failures: Dict[str, int] = {}
        self.disabled_sources: set = set()
        self.ir_required: set = set()
        # Worker pids that hold the current graph IR, per graph.
        self.ir_acks: Dict[str, set] = {}
        self.last_submit_frame = _NO_FRAME
        self.result_accepts = 0
        self.stale_job_drops = 0
        self.stale_result_drops = 0
        self.queue_pressure_drops = 0
        self.job_exceptions = 0

    @classmethod
    def mode(cls, scene_or_engine_settings=None) -> str:
        settings = engine_settings(scene_or_engine_settings)
        value = str(getattr(settings, "ace_parallel_mode", "processes_dsp") or "processes_dsp").strip().lower()
        return value if value in {"processes_dsp", "threads_io", "off"} else "processes_dsp"

    def inflight_limit(self) -> int:
        return max(int(self.MAX_INFLIGHT_PER_SOURCE), 1)

    def has_jobs(self) -> bool:
        return bool(self.future_meta)

    # Pool lifecycle

    def ensure_pool(self, scene) -> bool:
        """True when the pool runs and its workers are warm."""
        if self.mode(scene) != "processes_dsp":
            self.shutdown_pool(wait=False)
            return False
        if self.pool.failed:
            return False
        if self.pool.executor is None:
            if not self.pool.start():
                return False
            self.mark_ir_required_for_all_graphs()
        try:
            return self.pool.warm(self._warm_demand())
        except (BrokenProcessPool, RuntimeError, OSError) as exc:
            self.restart_pool(scene, f"warm-up failed: {exc}")
            return False

    def _warm_demand(self) -> int:
        shards = 0
        for graph in self.engine.graphs.values():
            if getattr(graph, "parallel_ir", {}) and getattr(graph, "parallel_offload_attr_keys", []):
                shards += max(len(getattr(graph, "parallel_output_shards", []) or []), 1)
        return shards * self.inflight_limit()

    def restart_pool(self, scene, reason: str, source_key: str = "") -> bool:
        if not self.pool.restart_budget_left():
            if source_key:
                self.disabled_sources.add(source_key)
            return False
        delay = self.pool.take_restart()
        if delay is None:
            return False
        utils.log_throttled(
            "parallel_restart",
            f"Restarting Processes-DSP pool ({reason}), "
            f"attempt {self.pool.restart_count}/{self.pool.RESTART_LIMIT}, backoff {delay:.1f}s.",
            "WARNING",
            min_interval=0.0,
        )
        self.shutdown_pool(wait=False)
        self.pool.failed = False
        ok = self.ensure_pool(scene)
        if not ok and source_key and self.pool.executor is None:
            self.disabled_sources.add(source_key)
        return ok

    def shutdown_pool(self, wait: bool = False):
        self.pool.shutdown(wait, pending=list(self.future_meta))
        self.forget_jobs()
        self.retired.clear()
        self.slots.destroy_all()

    def reset_generation(self):
        """Start a new job generation on the running pool; workers drop older graphs themselves."""
        self.generation_id += 1
        self.job_seq = 0
        self.pool.failed = False
        self.forget_jobs()
        self.slots.destroy_idle()

    def forget_jobs(self):
        """Stop tracking every job; jobs still running keep their payload slots until done."""
        for future in list(self.future_meta):
            self.drop_future(future)
        self.inflight.clear()
        self.future_meta.clear()
        self.results.clear()
        self.failures.clear()
        self.disabled_sources.clear()
        self.ir_required.clear()
        self.ir_acks.clear()
        self.slots.frame_slots.clear()
        self.last_valid.clear()
        self.last_valid_frame.clear()

    # Graph IR

    def prepare_graph_ir(self, graph: ACEAudioGraph, abstract_tree: Optional[ACEAbstractTree]):
        writer = self.engine.writer
        tree = abstract_tree if writer is not None else None
        output_defs = writer.output_defs if writer is not None else {}
        self.ir_required.discard(ir.graph_uid(graph))
        if ir.build_graph_ir(graph, tree, output_defs, self.engine._current_scene_fps):
            self.require_ir(graph.parallel_graph_uid)

    def refresh_specs_params(self):
        """Keep tempo and fps in each graph's IR current without a recompile."""
        fps = float(getattr(self.engine, "_current_scene_fps", 24.0) or 24.0)
        for graph in self.engine.graphs.values():
            if ir.refresh_specs_params(graph, fps):
                graph_key = str(getattr(graph, "parallel_graph_uid", "") or "")
                if graph_key:
                    self.require_ir(graph_key)

    def require_ir(self, graph_key: str):
        self.ir_required.add(graph_key)
        self.ir_acks[graph_key] = set()

    def ack_ir(self, graph_key: str, pid: int):
        acks = self.ir_acks.setdefault(graph_key, set())
        if pid > 0:
            acks.add(pid)
        if len(acks) >= self.pool.process_count():
            self.ir_required.discard(graph_key)

    def mark_ir_required_for_all_graphs(self):
        self.ir_required.clear()
        for graph in self.engine.graphs.values():
            graph_key = str(getattr(graph, "parallel_graph_uid", "") or ir.graph_uid(graph))
            if graph_key and getattr(graph, "parallel_ir", {}) and getattr(graph, "parallel_offload_attr_keys", []):
                self.require_ir(graph_key)

    def source_task_requires_local_frontend(self, task: Optional[SourceTask]) -> bool:
        """False when workers compute every output of the task's graph."""
        if not isinstance(task, SourceTask):
            return True
        node_id = str(getattr(task, "node_id", "") or "")
        graph = next((g for g in self.engine.graphs.values() if g.tasks.get(node_id) is task), None)
        if graph is None:
            return True
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        offloaded = (
            self.mode(scene) == "processes_dsp"
            and bool(getattr(graph, "parallel_ir", {}))
            and bool(getattr(graph, "parallel_offload_attr_keys", []))
        )
        return not offloaded or bool(getattr(graph, "parallel_local_attr_keys", []))

    # Jobs

    def submit_graph_job(
        self,
        scene,
        graph: ACEAudioGraph,
        graph_key: str,
        queue_key: str,
        owner_graph_key: str,
        frame_index: int,
        source_chunks: Dict[str, Any],
        output_keys: Optional[List[str]] = None,
        include_graph_ir: Optional[bool] = None,
    ) -> bool:
        if owner_graph_key in self.disabled_sources or not getattr(graph, "parallel_ir", {}):
            return False
        output_keys = [str(v) for v in list(output_keys or []) if str(v)] or list(
            getattr(graph, "parallel_offload_attr_keys", []) or []
        )
        if not output_keys or not self.ensure_pool(scene) or self.pool.executor is None:
            return False
        queue_key = str(queue_key or graph_key)
        pending = self._make_room(queue_key, frame_index)
        if pending is None:
            return False
        chunks = {
            str(src_id): chunk
            for src_id, chunk in dict(source_chunks or {}).items()
            if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate")
        }
        if not chunks:
            return False
        if include_graph_ir is None:
            include_graph_ir = graph_key in self.ir_required
        job, slot_keys = self._job(graph, graph_key, queue_key, owner_graph_key, frame_index, chunks, output_keys, include_graph_ir)
        try:
            future = self.pool.submit(evaluate_parallel_job, job)
        except (BrokenProcessPool, RuntimeError, OSError) as exc:
            self.slots.release_keys(slot_keys)
            self._on_submit_failure(scene, owner_graph_key, exc)
            return False
        pending.append(future)
        self.future_meta[future] = {
            "graph_key": graph_key,
            "queue_key": queue_key,
            "owner_graph_key": owner_graph_key,
            "frame_id": int(frame_index),
            "generation_id": self.generation_id,
            "undo_generation": int(getattr(self.engine, "undo_generation", 0) or 0),
            "submit_ts": time.time(),
            "payload_slot_keys": list(slot_keys),
        }
        self.last_submit_frame = int(frame_index)
        return True

    def _make_room(self, queue_key: str, frame_index: int) -> Optional[deque]:
        """The queue's in-flight jobs once there is room for one more, or None."""
        limit = self.inflight_limit()
        for future in list(self.inflight.get(queue_key, ())):
            if future.done():
                self.drop_future(future)
        pending = self.inflight.get(queue_key)
        if pending is None or pending.maxlen != limit:
            pending = deque(list(pending or []), maxlen=limit)
            self.inflight[queue_key] = pending
        if len(pending) < limit:
            return pending
        oldest = pending[0]
        oldest_frame = utils.as_int(dict(self.future_meta.get(oldest, {}) or {}).get("frame_id"), frame_index)
        if int(frame_index) - oldest_frame > self.CANCEL_STALE_FRAME_DELTA and oldest.cancel():
            self.drop_future(oldest)
            return pending
        self.queue_pressure_drops += 1
        return None

    def _job(self, graph, graph_key, queue_key, owner_graph_key, frame_index, chunks, output_keys, include_graph_ir):
        block_size = int(getattr(graph, "block_size", 0) or 0)
        if block_size <= 0:
            block_size = max(int(np.asarray(getattr(c, "mono", np.zeros(1))).shape[-1] or 1) for c in chunks.values())
        block_size = max(block_size, 1)
        sample_rate = max(int(getattr(c, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE) for c in chunks.values())
        payloads = {
            src_id: source_payload(chunk, block_size, int(getattr(chunk, "sample_rate", sample_rate) or sample_rate))
            for src_id, chunk in chunks.items()
        }
        self.job_seq += 1
        packed, slot_keys = self.slots.pack(payloads, graph_key=graph_key, frame_index=frame_index)
        job = {
            "generation_id": self.generation_id,
            "graph_uid": str(getattr(graph, "parallel_graph_uid", "") or owner_graph_key),
            "frame_id": int(frame_index),
            "block_size": block_size,
            "sample_rate": max(sample_rate, 1),
            "graph_ir_digest": str(getattr(graph, "parallel_ir_digest", "") or ""),
            "output_keys": list(output_keys),
            "job_key": f"g{self.generation_id}:q{queue_key}:f{int(frame_index)}:j{self.job_seq}",
        }
        job.update(packed)
        if include_graph_ir:
            job["graph_ir"] = dict(getattr(graph, "parallel_ir", {}) or {})
        return job, slot_keys

    def _on_submit_failure(self, scene, owner_graph_key: str, exc: BaseException):
        fail_count = self.failures.get(owner_graph_key, 0) + 1
        self.failures[owner_graph_key] = fail_count
        if fail_count <= 1 and self.pool.restart_budget_left():
            self.restart_pool(scene, f"submit failure: {exc}", source_key=owner_graph_key)
        else:
            self.disabled_sources.add(owner_graph_key)
        utils.log_throttled(
            f"parallel_submit:{owner_graph_key}",
            f"Parallel submit failed for '{owner_graph_key}', reusing last-valid offload values: {exc}",
            "WARNING",
            min_interval=0.0,
        )

    def drop_future(self, future):
        meta = self.future_meta.pop(future, None)
        if not meta:
            return
        queue_key = str(meta.get("queue_key", "") or meta.get("graph_key", "") or "")
        pending = self.inflight.get(queue_key)
        if pending:
            try:
                pending.remove(future)
            except ValueError:
                pass
            if not pending:
                self.inflight.pop(queue_key, None)
        slot_keys = [str(v) for v in meta.get("payload_slot_keys", []) or []]
        if future.done() or future.cancel():
            self.slots.release_keys(slot_keys)
        elif slot_keys:
            self.retired[future] = slot_keys

    def reap_retired(self):
        for future, slot_keys in list(self.retired.items()):
            if future.done():
                del self.retired[future]
                self.slots.release_keys(slot_keys)

    def poll(self, scene, current_frame: int):
        """Collect finished jobs, drop over-budget ones and restart the pool after a crash."""
        self.reap_retired()
        if not self.future_meta:
            return
        now = time.time()
        undo_generation = int(getattr(self.engine, "undo_generation", 0) or 0)
        restart = None
        for future in list(self.future_meta):
            meta = dict(self.future_meta.get(future, {}) or {})
            if int(meta.get("undo_generation", undo_generation) or 0) != undo_generation:
                self.drop_future(future)
                continue
            if not future.done():
                if now - float(meta.get("submit_ts", now) or now) > self.STALE_JOB_MAX_MS / 1000.0:
                    self.stale_job_drops += 1
                    self.drop_future(future)
                continue
            self.drop_future(future)
            restart = self._collect(future, meta, current_frame)
            if restart:
                break
        if restart:
            self.restart_pool(scene, restart[1], source_key=restart[0])
        if int(current_frame) - self.last_submit_frame > self.PAUSE_DRAIN_FRAMES:
            for future in [f for f in self.future_meta if f.done()]:
                self.drop_future(future)
        self._trim_results(int(current_frame) - self.KEPT_RESULT_FRAMES)

    def _collect(self, future, meta, current_frame):
        """Store one finished job's values; (source, reason) when the pool must restart."""
        graph_key = str(meta.get("graph_key", "") or "")
        try:
            payload = future.result(timeout=0.0)
        except Exception as exc:  # noqa: BLE001 boundary: any worker-side error comes back here
            return self._on_job_failure(str(meta.get("owner_graph_key", "") or graph_key) or "unknown", exc)
        if not isinstance(payload, dict) or utils.as_int(payload.get("generation_id"), -1) != self.generation_id:
            return None
        res_key = str(payload.get("graph_uid", "") or graph_key)
        worker = utils.as_int(payload.get("pid"), 0)
        if payload.get("needs_graph_ir"):
            if res_key:
                self.ir_required.add(res_key)
                self.ir_acks.get(res_key, set()).discard(worker)
            return None
        elapsed = float(payload.get("elapsed_s", 0.0) or 0.0)
        if elapsed > 0.0:
            self.engine.diagnostics._perf_add("dsp", elapsed)
        frame_id = utils.as_int(payload.get("frame_id"), -1)
        if frame_id < int(current_frame) - self.STALE_FRAME_GRACE:
            self.stale_result_drops += 1
            return None
        if not res_key:
            return None
        self.ack_ir(res_key, worker)
        values = dict(payload.get("values", {}) or {})
        if values:
            self.result_accepts += 1
            self.last_valid.setdefault(res_key, {}).update(values)
            self.last_valid_frame[res_key] = frame_id
        frame_map = self.results.setdefault(res_key, {})
        frame_map.setdefault(frame_id, {}).update(values)
        for old in sorted(frame_map)[: -self.KEPT_RESULT_FRAMES]:
            frame_map.pop(old, None)
        self.failures.pop(res_key, None)
        return None

    def _on_job_failure(self, fail_key: str, exc: BaseException):
        self.job_exceptions += 1
        fail_count = self.failures.get(fail_key, 0) + 1
        self.failures[fail_key] = fail_count
        if isinstance(exc, BrokenProcessPool):
            return fail_key, f"worker crash: {exc}"
        if fail_count <= 1 and self.pool.restart_budget_left():
            return fail_key, f"worker exception: {exc}"
        self.disabled_sources.add(fail_key)
        utils.log_throttled(
            f"parallel_job_error:{fail_key}",
            f"Parallel job failed for '{fail_key}', reusing last-valid offload values: {exc}",
            "WARNING",
            min_interval=0.0,
        )
        return None

    def _trim_results(self, threshold: int):
        for graph_key, frame_map in list(self.results.items()):
            for old in [fid for fid in frame_map if int(fid) < threshold]:
                frame_map.pop(old, None)
            if not frame_map:
                self.results.pop(graph_key, None)

    # Values

    def _within_reuse_window(self, graph_key: str, frame_index: int) -> bool:
        fps = max(float(self.engine._current_scene_fps), 1.0)
        window = max(1, int(self.STALE_REUSE_MS * fps / 1000.0 + 0.5))
        return int(frame_index) - self.last_valid_frame.get(graph_key, _NO_FRAME) <= window

    def values_for_frame(self, graph: ACEAudioGraph, graph_key: str, frame_index: int) -> Dict[str, Any]:
        """Worker values for every offloaded output: this frame's, else recent ones, else zero."""
        offload_keys = list(getattr(graph, "parallel_offload_attr_keys", []) or [])
        if not offload_keys:
            return {}
        frame_map = self.results.get(graph_key, {})
        fresh = dict(frame_map.pop(int(frame_index), {}) or {})
        if not frame_map:
            self.results.pop(graph_key, None)
        if fresh:
            self.last_valid.setdefault(graph_key, {}).update(fresh)
            self.last_valid_frame[graph_key] = int(frame_index)
            base = self.last_valid[graph_key]
        elif self._within_reuse_window(graph_key, frame_index):
            base = dict(self.last_valid.get(graph_key, {}) or {})
        else:
            base = {}
        frame = self.engine.frame
        return {
            key: frame._normalize_output_value(key, base[key]) if key in base else frame._zero_value_for_attr(key)
            for key in offload_keys
        }

    def has_usable_values(self, graph_key: str, frame_index: int) -> bool:
        """Fresh values for this frame, or recent ones within the reuse window."""
        key = str(graph_key or "")
        if self.results.get(key, {}).get(int(frame_index)):
            return True
        return bool(self.last_valid.get(key)) and self._within_reuse_window(key, frame_index)
