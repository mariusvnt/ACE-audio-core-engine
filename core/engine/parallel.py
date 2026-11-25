"""Worker-process DSP offload: pool, graph IR, payload slots, submit and poll."""

import bpy
import concurrent.futures
import hashlib
import json
import multiprocessing
import numpy as np
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional
from ... import utils
from ..abstract_tree import ACEAbstractNode, ACEAbstractTree
from ..tasks.chunk import SAMPLE_RATE
from ..tasks.graph import ACEAudioGraph
from ..tasks.source import SourceTask
from ..worker.job import evaluate_parallel_job, warm_worker, worker_pid
from .common import BrokenProcessPool, _SHARED_MEMORY_AVAILABLE, _main_script_hidden_from_spawn, engine_settings, mp_shared_memory


class ParallelOffload:
    """Worker-process DSP offload: pool, graph IR, payload slots, submit and poll."""

    _PARALLEL_OFFLOAD_STREAM_TYPES = frozenset(
        {
            "rms",
            "lufs",
            "peak",
            "pitch",
            "spectral_centroid",
            "spectral_flux",
            "onset",
            "zero_crossing_rate",
            "stereo_width",
            "stereo_balance",
            "stereo_vector",
        }
    )
    _PARALLEL_MAX_INFLIGHT_PER_SOURCE = 2
    _PARALLEL_ATTRS_PER_JOB = 8
    _PARALLEL_WORKER_HEADROOM_CORES = 1
    # Hard realtime stale-job budget: jobs older than this are dropped instead
    # of being allowed to resolve late and contaminate frame determinism.
    _PARALLEL_STALE_JOB_MAX_MS = 66
    _PARALLEL_RESTART_LIMIT = 3  # Allow up to 3 restarts before disabling; exponential backoff between attempts.
    # Backoff delays (seconds) indexed by restart attempt number (0-based).
    # Attempt 0 → 0.5 s, attempt 1 → 1.0 s, attempt 2 → 2.0 s.
    _PARALLEL_RESTART_BACKOFF_S: tuple = (0.5, 1.0, 2.0)
    _PARALLEL_STALE_FRAME_GRACE = 2
    # Time-based stale-reuse window (FPS-invariant). At 24 fps this is 1 frame;
    # at 30 fps it is 1 frame; at 60 fps it is 3 frames — reduced from 66 ms (4 frames
    # at 60 fps) to tighten the stale-result contract closer to the ≤2-frame spec.
    _PARALLEL_STALE_REUSE_MS = 50
    _PARALLEL_CANCEL_STALE_FRAME_DELTA = 4

    def __init__(self, engine):
        self.engine = engine
        self._parallel_generation_id: int = 0
        self._parallel_job_seq: int = 0
        self._parallel_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._parallel_inflight: Dict[str, deque] = {}
        self._parallel_future_meta: Dict[Any, Dict[str, Any]] = {}
        self._parallel_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._parallel_last_valid: Dict[str, Dict[str, Any]] = {}
        self._parallel_last_valid_frame: Dict[str, int] = {}
        self._parallel_failures: Dict[str, int] = {}
        self._parallel_disabled_sources: set[str] = set()
        self._parallel_graph_ir_required: set[str] = set()
        self._parallel_restart_count: int = 0
        self._parallel_restart_backoff_until: float = 0.0
        self._parallel_pool_failed: bool = False
        self._parallel_payload_slots: Dict[str, Dict[str, Any]] = {}
        self._parallel_payload_slot_refs: Dict[str, int] = {}
        self._parallel_payload_slot_counters: Dict[str, int] = {}
        self._parallel_payload_free_slots: Dict[str, deque[str]] = {}
        self._parallel_frame_payload_slots: Dict[tuple[str, int, str], str] = {}
        self._parallel_queue_pressure_drops: int = 0
        self._parallel_job_timeouts: int = 0
        self._parallel_job_exceptions: int = 0
        self._parallel_stale_job_drops: int = 0
        self._parallel_stale_result_drops: int = 0
        self._parallel_result_accepts: int = 0
        # Parallel submit tracking for pause-drain SHM reclaim.
        self._parallel_last_submit_frame: int = -10_000_000
        # Untracked jobs still running in a worker, with the payload slots they read.
        self._parallel_retired: Dict[Any, List[str]] = {}
        # None: warm-up not started; empty: pool warm.
        self._parallel_warmup_futures: Optional[list] = None
        # Worker pids that hold the current graph IR, per graph.
        self._parallel_ir_acks: Dict[str, set[int]] = {}

    @classmethod
    def _parallel_mode(cls, scene_or_engine_settings=None) -> str:
        settings = engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_parallel_mode", "processes_dsp") or "processes_dsp").strip().lower()
            return val if val in {"processes_dsp", "threads_io", "off"} else "processes_dsp"
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._parallel_mode", exc)
            return "processes_dsp"

    @staticmethod
    def _parallel_shard_attr_keys(
        attr_keys: List[str],
        source_groups: Optional[Dict[str, str]] = None,
    ) -> List[List[str]]:
        """Partition attr_keys into per-worker shards.

        When source_groups is provided (a dict mapping each attr_key to its
        primary source node ID), keys are bucketed by source so that each
        source's DSP chain runs in a dedicated worker job.  This allows
        multi-source scenes to exploit all available worker processes in
        parallel rather than serialising through a single job.

        Single-source graphs (or graphs where source grouping is unknown)
        fall back to the legacy single-shard path to avoid extra scheduling
        overhead.
        """
        keys = [str(v) for v in list(attr_keys or []) if str(v)]
        if not keys:
            return []
        if not source_groups or len(set(source_groups.values())) <= 1:
            # Single-source (or unknown) graph: keep legacy single-shard behavior.
            return [keys]
        # Shard by source ancestry so each source group runs in a separate worker job.
        buckets: Dict[str, List[str]] = {}
        ungrouped: List[str] = []
        for k in keys:
            group = source_groups.get(k)
            if group:
                buckets.setdefault(str(group), []).append(k)
            else:
                ungrouped.append(k)
        shards = [v for v in buckets.values() if v]
        if ungrouped:
            shards.append(ungrouped)
        return shards if shards else [keys]

    def _source_task_requires_local_frontend(self, task: Optional[SourceTask]) -> bool:
        if not isinstance(task, SourceTask):
            return True
        target_graph = None
        for graph in self.engine.graphs.values():
            node_id = str(getattr(task, "node_id", "") or "")
            if graph.tasks.get(node_id) is task:
                target_graph = graph
                break
        if target_graph is None:
            return True

        scene = getattr(getattr(bpy, "context", None), "scene", None)
        parallel_mode = self._parallel_mode(scene)
        parallel_offload_graph = (
            parallel_mode == "processes_dsp"
            and bool(getattr(target_graph, "parallel_ir", {}))
            and bool(getattr(target_graph, "parallel_offload_attr_keys", []))
        )
        if not parallel_offload_graph:
            return True
        local_keys = list(getattr(target_graph, "parallel_local_attr_keys", []) or [])
        return bool(local_keys)

    @classmethod
    def _is_parallel_offload_attr_candidate(cls, node: Optional[ACEAbstractNode]) -> bool:
        """
        Runtime offload policy:
        - offload all numeric Layer 2 outputs so the full upstream chain,
          including source preprocessing, can be evaluated in worker jobs.
        """
        if node is None:
            return False
        category = str(getattr(node, "category", "") or "")
        node_type = str(getattr(node, "type", "") or "")
        if category == "utilities" and node_type == "src_specs":
            return True
        if category == "utilities" and node_type == "stream_accumulator":
            # Stateful running-sum nodes must remain local. Their output should
            # never jump when the engine temporarily falls back between worker
            # and main-thread execution paths.
            return False
        if category != "stream":
            return False
        if node_type in cls._PARALLEL_OFFLOAD_STREAM_TYPES:
            return True
        # Future stream operators should default to worker offload.
        return True

    @staticmethod
    def _sanitize_ir_value(value: Any):
        if value is None:
            return None
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [ParallelOffload._sanitize_ir_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ParallelOffload._sanitize_ir_value(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _parallel_graph_uid(graph: ACEAudioGraph) -> str:
        key = (
            str(getattr(graph, "source_uid", "") or "")
            or str(getattr(getattr(graph, "source_context", None), "stable_id", "") or "")
            or str(getattr(graph, "proxy_name", "") or "")
        )
        return key if key else f"graph:{id(graph)}"

    def _parallel_worker_count(self, scene) -> int:
        """Return the target process-pool worker count for the current machine.

        Respects the ``ACE_PARALLEL_WORKERS`` environment variable override.
        Otherwise uses ``os.cpu_count() - _PARALLEL_WORKER_HEADROOM_CORES`` so
        one core stays free for Blender's main thread and render pipeline.
        """
        try:
            override = int(os.environ.get("ACE_PARALLEL_WORKERS", "0") or 0)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._parallel_worker_count", exc)
            override = 0
        cpu_count = max(int(os.cpu_count() or 1), 1)
        if override > 0:
            return max(1, min(int(override), cpu_count))
        reserve = int(self._PARALLEL_WORKER_HEADROOM_CORES) if cpu_count > 1 else 0
        return max(1, cpu_count - max(reserve, 0))

    def _parallel_inflight_limit(self, scene=None) -> int:
        # Always honour the spec constant regardless of worker count.
        # The previous implementation returned max(..., 3) for ≥4 workers,
        # silently exceeding _PARALLEL_MAX_INFLIGHT_PER_SOURCE = 2 and causing
        # SHM slot exhaustion under high-core-count machines.
        return max(min(int(self._PARALLEL_MAX_INFLIGHT_PER_SOURCE), 2), 1)

    def _parallel_forget_jobs(self):
        """Stop tracking every job; jobs still running keep their payload slots until done."""
        for future_obj in list(self._parallel_future_meta.keys()):
            self._parallel_drop_future(future_obj)
        self._parallel_inflight.clear()
        self._parallel_future_meta.clear()
        self._parallel_results.clear()
        self._parallel_failures.clear()
        self._parallel_disabled_sources.clear()
        self._parallel_graph_ir_required.clear()
        self._parallel_ir_acks.clear()
        self._parallel_frame_payload_slots.clear()
        self._parallel_last_valid.clear()
        self._parallel_last_valid_frame.clear()

    def _parallel_require_ir(self, graph_key: str):
        self._parallel_graph_ir_required.add(graph_key)
        self._parallel_ir_acks[graph_key] = set()

    def _parallel_ack_ir(self, graph_key: str, pid: int):
        acks = self._parallel_ir_acks.setdefault(graph_key, set())
        if pid > 0:
            acks.add(pid)
        if len(acks) >= self._parallel_process_count():
            self._parallel_graph_ir_required.discard(graph_key)

    def _parallel_process_count(self) -> int:
        return max(len(getattr(self._parallel_executor, "_processes", None) or {}), 1)

    def _mark_parallel_ir_required_for_all_graphs(self):
        self._parallel_graph_ir_required.clear()
        for graph in self.engine.graphs.values():
            graph_key = str(getattr(graph, "parallel_graph_uid", "") or self._parallel_graph_uid(graph))
            if not graph_key:
                continue
            if bool(getattr(graph, "parallel_ir", {})) and bool(getattr(graph, "parallel_offload_attr_keys", [])):
                self._parallel_require_ir(graph_key)

    def _shutdown_parallel_pool(self, wait: bool = False):
        executor = self._parallel_executor
        self._parallel_executor = None
        wait_effective = bool(wait)
        if not wait_effective:
            # Realtime paths must stay non-blocking. Best-effort cancel pending jobs.
            for future_obj in list(self._parallel_future_meta.keys()):
                with utils.logged_errors("audio_engine.ACEAudioEngine._shutdown_parallel_pool"):
                    future_obj.cancel()
            # Terminate worker processes immediately so they release SHM file
            # descriptors and CPU. Without this, already-running workers outlive
            # the executor shutdown and hold /dev/shm blocks until completion.
            with utils.logged_errors("audio_engine.ACEAudioEngine._shutdown_parallel_pool#2"):
                procs = dict(getattr(executor, "_processes", {}) or {})
                for proc in procs.values():
                    with utils.logged_errors("audio_engine.ACEAudioEngine._shutdown_parallel_pool#3"):
                        proc.terminate()
        if executor is not None:
            with utils.logged_errors("audio_engine.ACEAudioEngine._shutdown_parallel_pool#4"):
                executor.shutdown(wait=wait_effective, cancel_futures=True)
        self._parallel_forget_jobs()
        self._parallel_retired.clear()
        self._parallel_warmup_futures = None
        self._parallel_release_payload_slots()
        self._parallel_payload_slot_refs.clear()
        self._parallel_payload_slot_counters.clear()

    def _parallel_reset_generation(self):
        """Start a new job generation on the running pool; workers drop older graphs themselves."""
        self._parallel_generation_id = int(self._parallel_generation_id) + 1
        self._parallel_job_seq = 0
        self._parallel_pool_failed = False
        self._parallel_forget_jobs()
        self._parallel_destroy_idle_payload_slots()

    def _ensure_parallel_pool(self, scene) -> bool:
        """True when the pool runs and its workers are warm."""
        if self._parallel_mode(scene) != "processes_dsp":
            self._shutdown_parallel_pool(wait=False)
            return False
        if self._parallel_pool_failed:
            return False
        if self._parallel_executor is not None:
            return self._parallel_pool_warm(scene)
        worker_count = self._parallel_worker_count(scene)
        try:
            spawn_ctx = multiprocessing.get_context("spawn")
            self._parallel_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max(int(worker_count), 1),
                mp_context=spawn_ctx,
                initializer=warm_worker,
            )
            self._mark_parallel_ir_required_for_all_graphs()
            return self._parallel_pool_warm(scene)
        except Exception as e:
            self._parallel_executor = None
            self._parallel_pool_failed = True
            utils.log_throttled(
                "parallel_init",
                f"Processes-DSP unavailable, reusing last-valid offload values: {e}",
                "WARNING",
                min_interval=0.0,
            )
            return False

    def _parallel_warm_demand(self) -> int:
        shards = 0
        for graph in self.engine.graphs.values():
            if getattr(graph, "parallel_ir", {}) and getattr(graph, "parallel_offload_attr_keys", []):
                shards += max(len(getattr(graph, "parallel_output_shards", []) or []), 1)
        return shards * self._parallel_inflight_limit()

    def _parallel_pool_warm(self, scene) -> bool:
        """Start one warm-up job per needed worker, then report whether all have answered."""
        if self._parallel_warmup_futures is None:
            demand = self._parallel_warm_demand()
            if demand <= 0:
                return False
            count = min(demand, self._parallel_worker_count(scene))
            with _main_script_hidden_from_spawn():
                self._parallel_warmup_futures = [self._parallel_executor.submit(worker_pid) for _ in range(count)]
        pending = self._parallel_warmup_futures
        if not all(f.done() for f in pending):
            return False
        failed = next((f.exception() for f in pending if f.exception() is not None), None)
        self._parallel_warmup_futures = []
        if failed is not None:
            self._restart_parallel_pool(scene, f"warm-up failed: {failed}")
            return False
        return True

    def _restart_parallel_pool(self, scene, reason: str, source_key: str = "") -> bool:
        if self._parallel_restart_count >= self._PARALLEL_RESTART_LIMIT:
            if source_key:
                self._parallel_disabled_sources.add(source_key)
            return False
        # Enforce exponential backoff between restart attempts to absorb transient OS
        # errors (OOM spikes, process spawn latency) without burning the restart budget.
        now = time.monotonic()
        if now < self._parallel_restart_backoff_until:
            # Still in the backoff window — defer this attempt; reuse last-valid values.
            return False
        self._parallel_restart_count += 1
        backoff_idx = min(self._parallel_restart_count - 1, len(self._PARALLEL_RESTART_BACKOFF_S) - 1)
        delay_s = float(self._PARALLEL_RESTART_BACKOFF_S[backoff_idx])
        self._parallel_restart_backoff_until = now + delay_s
        utils.log_throttled(
            "parallel_restart",
            f"Restarting Processes-DSP pool ({reason}), "
            f"attempt {self._parallel_restart_count}/{self._PARALLEL_RESTART_LIMIT}, "
            f"backoff {delay_s:.1f}s.",
            "WARNING",
            min_interval=0.0,
        )
        self._shutdown_parallel_pool(wait=False)
        self._parallel_pool_failed = False
        ok = self._ensure_parallel_pool(scene)
        if not ok and source_key:
            self._parallel_disabled_sources.add(source_key)
        return ok

    def _prepare_parallel_graph_ir(self, graph: ACEAudioGraph, abstract_tree: ACEAbstractTree):
        graph.parallel_graph_uid = self._parallel_graph_uid(graph)
        graph.parallel_ir = {}
        graph.parallel_ir_digest = ""
        graph.parallel_offload_attr_keys = []
        graph.parallel_output_shards = []
        graph.parallel_local_attr_keys = list(getattr(graph, "output_attr_keys", []) or [])
        self._parallel_graph_ir_required.discard(str(getattr(graph, "parallel_graph_uid", "") or ""))

        if abstract_tree is None or self.engine.writer is None:
            return

        offload_attr_keys: list[str] = []
        offload_node_ids: set[str] = set()
        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            node_id, _, sock_idx_str = attr_key.partition(":")
            node = abstract_tree.nodes.get(node_id)
            if not node:
                continue
            if not self._is_parallel_offload_attr_candidate(node):
                continue
            try:
                _ = int(sock_idx_str) if sock_idx_str else 0
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._prepare_parallel_graph_ir", exc)
                continue
            offload_attr_keys.append(attr_key)
            offload_node_ids.add(node_id)

        if not offload_attr_keys:
            return

        reverse_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            reverse_adj.setdefault(str(link.target_node_id), set()).add(str(link.source_node_id))

        required_node_ids: set[str] = set()
        stack = list(offload_node_ids)
        graph_task_ids = set(getattr(graph, "tasks", {}).keys())
        while stack:
            node_id = str(stack.pop())
            if node_id in required_node_ids:
                continue
            required_node_ids.add(node_id)
            for src_id in reverse_adj.get(node_id, set()):
                if src_id in graph_task_ids and src_id not in required_node_ids:
                    stack.append(src_id)

        nodes_ir: Dict[str, Dict[str, Any]] = {}
        source_node_ids: list[str] = []
        source_ir: Dict[str, Dict[str, Any]] = {}
        for node_id in sorted(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            params = {
                str(k): self._sanitize_ir_value(v)
                for k, v in dict(getattr(node, "parameters", {}) or {}).items()
            }
            # The worker's SourceSpecs replica needs BPM/FPS for beat phase; each
            # graph takes them from its own source.
            if node.category == "utilities" and node.type == "src_specs":
                _specs_ctx = getattr(graph, "source_context", None)
                params["scene_bpm"] = float(getattr(_specs_ctx, "bpm", 120.0) or 120.0)
                params["scene_fps"] = float(getattr(self.engine, "_current_scene_fps", 24.0) or 24.0)
                params["scene_beats_per_bar"] = float(getattr(_specs_ctx, "beats_per_bar", 4.0) or 4.0)
            nodes_ir[node_id] = {
                "category": str(getattr(node, "category", "") or ""),
                "type": str(getattr(node, "type", "") or ""),
                "parameters": params,
            }
            if node.category == "source_mngmt" and node.type == "src":
                source_node_ids.append(node_id)
                src_task = graph.tasks.get(node_id)
                src_ctx = getattr(src_task, "source_context", None) if isinstance(src_task, SourceTask) else None
                sr_hint = int(
                    getattr(src_task, "sample_rate", 0) if isinstance(src_task, SourceTask) else 0
                    or getattr(src_ctx, "sample_rate", 0)
                    or SAMPLE_RATE
                )
                source_ir[node_id] = {
                    "sample_rate": int(max(sr_hint, 1)),
                    "lineage": [node_id],
                    "spectrum_token": [node_id],
                }

        links_ir: list[tuple[str, int, str, int]] = []
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if src_id not in nodes_ir or dst_id not in nodes_ir:
                continue
            links_ir.append(
                (
                    src_id,
                    int(getattr(link, "source_socket_idx", 0) or 0),
                    dst_id,
                    int(getattr(link, "target_socket_idx", 0) or 0),
                )
            )

        output_specs: Dict[str, Dict[str, Any]] = {}
        for attr_key in offload_attr_keys:
            node_id, _, sock_idx_str = attr_key.partition(":")
            if node_id not in nodes_ir:
                continue
            info = self.engine.writer.output_defs.get(attr_key, {}) if self.engine.writer else {}
            output_specs[attr_key] = {
                "node_id": node_id,
                "socket_idx": int(sock_idx_str) if sock_idx_str else 0,
                "attr_type": str(info.get("type", "FLOAT") or "FLOAT"),
            }

        if not output_specs:
            return

        graph.parallel_ir = {
            "nodes": nodes_ir,
            "links": links_ir,
            "source_node_ids": sorted(set(source_node_ids)),
            "source_ir": source_ir,
            "output_specs": output_specs,
        }
        graph.parallel_offload_attr_keys = sorted(output_specs.keys())
        # Build per-source grouping: trace each attr_key's node back to its
        # primary source ancestor so multi-source graphs can be sharded across
        # multiple worker processes (one shard per source lineage).
        attr_key_source_group: Dict[str, str] = {}
        _src_id_set = set(source_node_ids)
        for attr_key in graph.parallel_offload_attr_keys:
            nid, _, _ = attr_key.partition(":")
            visited_s: set[str] = set()
            stack_s = [nid]
            found_src = ""
            while stack_s and not found_src:
                cur = stack_s.pop()
                if cur in visited_s:
                    continue
                visited_s.add(cur)
                if cur in _src_id_set:
                    found_src = cur
                    break
                for parent in reverse_adj.get(cur, set()):
                    if parent not in visited_s:
                        stack_s.append(parent)
            if found_src:
                attr_key_source_group[attr_key] = found_src
        graph.parallel_output_shard_groups = attr_key_source_group
        graph.parallel_output_shards = self._parallel_shard_attr_keys(
            graph.parallel_offload_attr_keys,
            source_groups=attr_key_source_group if len(_src_id_set) > 1 else None,
        )
        offload_set = set(graph.parallel_offload_attr_keys)
        graph.parallel_local_attr_keys = [
            key for key in list(getattr(graph, "output_attr_keys", []) or [])
            if key not in offload_set
        ]
        graph.parallel_ir_digest = self._parallel_ir_digest(graph.parallel_ir)
        if graph.parallel_graph_uid:
            self._parallel_require_ir(str(graph.parallel_graph_uid))

    @staticmethod
    def _parallel_ir_digest(parallel_ir: Dict[str, Any]) -> str:
        try:
            payload = json.dumps(parallel_ir, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            utils.note_swallowed("engine_parallel.ParallelOffload._parallel_ir_digest", exc)
            payload = str(parallel_ir).encode("utf-8", "ignore")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()

    def _refresh_specs_ir_bpm(self) -> None:
        """Patch scene_bpm / scene_beats_per_bar / scene_fps in each graph's
        parallel IR so _WorkerSourceSpecsPassthroughTask always sees the current
        per-source BPM without requiring a full graph recompile.

        Called each frame from execute_next_frame() after _current_scene_fps is
        updated but before _maybe_compile_dirty() so any compile triggered this
        frame starts with correct values already in place.
        """
        fps = float(getattr(self.engine, "_current_scene_fps", 24.0) or 24.0)
        for graph in self.engine.graphs.values():
            parallel_ir = getattr(graph, "parallel_ir", None)
            if not parallel_ir:
                continue
            src_ctx = getattr(graph, "source_context", None)
            if src_ctx is None:
                continue
            bpm = float(getattr(src_ctx, "bpm", 120.0) or 120.0)
            bpb = float(getattr(src_ctx, "beats_per_bar", 4.0) or 4.0)
            changed = False
            nodes_ir = parallel_ir.get("nodes", {})
            for node_info in nodes_ir.values():
                if (
                    node_info.get("category") != "utilities"
                    or node_info.get("type") != "src_specs"
                ):
                    continue
                params = node_info.get("parameters")
                if not isinstance(params, dict):
                    continue
                old_bpm = float(params.get("scene_bpm", 120.0) or 120.0)
                old_bpb = float(params.get("scene_beats_per_bar", 4.0) or 4.0)
                old_fps = float(params.get("scene_fps", 24.0) or 24.0)
                if abs(old_bpm - bpm) > 1e-6 or abs(old_bpb - bpb) > 1e-6 or abs(old_fps - fps) > 1e-6:
                    params["scene_bpm"] = bpm
                    params["scene_fps"] = fps
                    params["scene_beats_per_bar"] = bpb
                    changed = True
            if changed:
                graph.parallel_ir_digest = self._parallel_ir_digest(parallel_ir)
                graph_key = str(getattr(graph, "parallel_graph_uid", "") or "")
                if graph_key:
                    self._parallel_require_ir(graph_key)

    def _build_parallel_source_payload(self, chunk, block_size: int, sample_rate: int) -> Dict[str, Any]:
        """Serialise an AudioChunk into a picklable worker payload dict.

        Pads or trims the mono signal to exactly *block_size* samples.  Stereo
        is normalised to channels-first layout (shape ``(C, N)``).  Optional
        fields (``frontend_config``, ``band_indices``, ``band_hz_bounds``) are
        included only when present on the chunk.

        Args:
            chunk: Source AudioChunk to serialise.
            block_size: Expected number of mono samples in the worker.
            sample_rate: Sample rate to record in the payload.

        Returns:
            Dict suitable for passing to ``evaluate_parallel_job``.
        """
        bs = max(int(block_size), 1)
        sr = max(int(sample_rate), 1)
        mono_raw = getattr(chunk, "mono", chunk)
        mono = np.asarray(mono_raw, dtype=np.float32).reshape(-1)
        if mono.shape[0] < bs:
            padded = np.zeros(bs, dtype=np.float32)
            if mono.shape[0] > 0:
                padded[: mono.shape[0]] = mono
            mono = padded
        elif mono.shape[0] > bs:
            mono = mono[:bs]

        stereo = getattr(chunk, "stereo", None)
        stereo_payload = None
        if isinstance(stereo, np.ndarray):
            arr = np.asarray(stereo, dtype=np.float32)
            if arr.ndim == 2:
                if arr.shape[1] == mono.shape[0]:
                    stereo_payload = arr.astype(np.float32, copy=False)
                elif arr.shape[0] == mono.shape[0]:
                    stereo_payload = arr.T.astype(np.float32, copy=False)

        payload: Dict[str, Any] = {
            "mono": mono.astype(np.float32, copy=False),
            "stereo": stereo_payload,
            "sample_rate": sr,
            "latency_ms": float(getattr(chunk, "latency_ms", 0.0) or 0.0),
        }
        frontend_config = getattr(chunk, "frontend_config", None)
        if isinstance(frontend_config, (list, tuple)) and len(frontend_config) >= 4:
            with utils.logged_errors("audio_engine.ACEAudioEngine._build_parallel_source_payload"):
                payload["frontend_config"] = (
                    float(frontend_config[0] or 0.0),
                    float(frontend_config[1] or 0.0),
                    float(frontend_config[2] or 0.0),
                    float(frontend_config[3] or 0.0),
                )
        band_indices = getattr(chunk, "band_indices", None)
        if band_indices is not None:
            with utils.logged_errors("audio_engine.ACEAudioEngine._build_parallel_source_payload#2"):
                band_arr = np.asarray(band_indices, dtype=np.int64).reshape(-1)
                payload["band_indices"] = band_arr
                payload["band_index_fft_size"] = int(
                    max(int(getattr(chunk, "band_index_fft_size", 0) or 0), 1)
                )
        band_hz_bounds = getattr(chunk, "band_hz_bounds", None)
        if isinstance(band_hz_bounds, (list, tuple)) and len(band_hz_bounds) >= 2:
            with utils.logged_errors("audio_engine.ACEAudioEngine._build_parallel_source_payload#3"):
                low_hz = float(band_hz_bounds[0])
                high_hz = float(band_hz_bounds[1])
                if high_hz > low_hz:
                    payload["band_hz_bounds"] = (low_hz, high_hz)
        return payload

    def _parallel_drop_future(self, future_obj):
        meta = self._parallel_future_meta.pop(future_obj, None)
        if not meta:
            return
        queue_key = str(meta.get("queue_key", "") or meta.get("graph_key", "") or "")
        pending = self._parallel_inflight.get(queue_key)
        if pending:
            try:
                pending.remove(future_obj)
            except ValueError:
                pass
            if not pending:
                self._parallel_inflight.pop(queue_key, None)
        slot_keys = [str(v) for v in meta.get("payload_slot_keys", []) or []]
        if future_obj.done() or future_obj.cancel():
            self._parallel_release_slot_keys(slot_keys)
        elif slot_keys:
            self._parallel_retired[future_obj] = slot_keys

    def _parallel_release_slot_keys(self, slot_keys: List[str]):
        for slot_key in slot_keys:
            self._parallel_release_payload_slot(slot_key)

    def _parallel_reap_retired(self):
        for future_obj, slot_keys in list(self._parallel_retired.items()):
            if future_obj.done():
                del self._parallel_retired[future_obj]
                self._parallel_release_slot_keys(slot_keys)

    def _parallel_poll_futures(self, scene, current_frame: int):
        self._parallel_reap_retired()
        if not self._parallel_future_meta:
            return
        now = time.time()
        restart_source = ""
        restart_reason = ""

        for future_obj in list(self._parallel_future_meta.keys()):
            meta = dict(self._parallel_future_meta.get(future_obj, {}) or {})
            graph_key = str(meta.get("graph_key", "") or "")
            owner_graph_key = str(meta.get("owner_graph_key", "") or graph_key)
            submit_ts = float(meta.get("submit_ts", now) or now)
            undo_generation = int(meta.get("undo_generation", getattr(self.engine, "undo_generation", 0)) or 0)
            if undo_generation != int(getattr(self.engine, "undo_generation", 0) or 0):
                self._parallel_drop_future(future_obj)
                continue
            stale_budget_s = max(float(self._PARALLEL_STALE_JOB_MAX_MS) / 1000.0, 0.001)
            age_s = max(float(now - submit_ts), 0.0)

            if (not future_obj.done()) and age_s > stale_budget_s:
                # Hard-realtime contract: never keep over-budget jobs alive in
                # frame tracking. Drop immediately, regardless of cancel result.
                if not bool(meta.get("stale_job", False)):
                    meta["stale_job"] = True
                    self._parallel_future_meta[future_obj] = dict(meta)
                    self._parallel_stale_job_drops += 1
                with utils.logged_errors("audio_engine.ACEAudioEngine._parallel_poll_futures"):
                    future_obj.cancel()
                self._parallel_drop_future(future_obj)
                continue

            if future_obj.done():
                if bool(meta.get("stale_job", False)):
                    self._parallel_drop_future(future_obj)
                    continue
                try:
                    payload = future_obj.result(timeout=0.0)
                except Exception as e:
                    self._parallel_job_exceptions += 1
                    self._parallel_drop_future(future_obj)
                    fail_key = owner_graph_key or "unknown"
                    fail_count = int(self._parallel_failures.get(fail_key, 0)) + 1
                    self._parallel_failures[fail_key] = fail_count
                    if isinstance(e, BrokenProcessPool):
                        restart_source = fail_key
                        restart_reason = f"worker crash: {e}"
                        break
                    if fail_count <= 1 and self._parallel_restart_count < self._PARALLEL_RESTART_LIMIT:
                        restart_source = fail_key
                        restart_reason = f"worker exception: {e}"
                        break
                    if fail_key:
                        self._parallel_disabled_sources.add(fail_key)
                    utils.log_throttled(
                        f"parallel_job_error:{fail_key}",
                        f"Parallel job failed for '{fail_key}', reusing last-valid offload values: {e}",
                        "WARNING",
                        min_interval=0.0,
                    )
                    continue

                self._parallel_drop_future(future_obj)
                if not isinstance(payload, dict):
                    continue
                generation_id = utils.as_int(payload.get("generation_id"), -1)
                if generation_id != int(self._parallel_generation_id):
                    continue
                res_key = str(payload.get("graph_uid", "") or graph_key)
                worker = utils.as_int(payload.get("pid"), 0)
                if bool(payload.get("needs_graph_ir", False)):
                    if res_key:
                        self._parallel_graph_ir_required.add(res_key)
                        self._parallel_ir_acks.get(res_key, set()).discard(worker)
                    continue
                worker_elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
                if worker_elapsed_s > 0.0:
                    with utils.logged_errors("audio_engine.ACEAudioEngine._parallel_poll_futures#2"):
                        self.engine.diagnostics._perf_add("dsp", worker_elapsed_s)
                frame_id = utils.as_int(payload.get("frame_id"), -1)
                if frame_id < (int(current_frame) - int(self._PARALLEL_STALE_FRAME_GRACE)):
                    self._parallel_stale_result_drops += 1
                    continue
                values = dict(payload.get("values", {}) or {})
                if not res_key:
                    continue
                self._parallel_ack_ir(res_key, worker)
                if values:
                    self._parallel_result_accepts += 1
                    merged_last = dict(self._parallel_last_valid.get(res_key, {}) or {})
                    merged_last.update(values)
                    self._parallel_last_valid[res_key] = merged_last
                    self._parallel_last_valid_frame[res_key] = int(frame_id)
                frame_map = self._parallel_results.setdefault(res_key, {})
                merged_frame = dict(frame_map.get(frame_id, {}) or {})
                merged_frame.update(values)
                frame_map[frame_id] = merged_frame
                if len(frame_map) > 4:
                    for old_frame in sorted(frame_map.keys())[:-4]:
                        frame_map.pop(old_frame, None)
                self._parallel_failures.pop(res_key, None)
                continue

        if restart_reason:
            self._restart_parallel_pool(scene, restart_reason, source_key=restart_source)

        # Pause-drain: when Blender is paused (no new submits for several frames),
        # drain all completed futures so SHM ref-counts reach zero and blocks are freed.
        _PAUSE_DRAIN_FRAMES = 3
        last_submit = int(self._parallel_last_submit_frame)
        if (int(current_frame) - last_submit) > _PAUSE_DRAIN_FRAMES and self._parallel_future_meta:
            for future_obj in list(self._parallel_future_meta.keys()):
                if future_obj.done():
                    self._parallel_drop_future(future_obj)

        if self._parallel_results:
            threshold = int(current_frame) - 4
            for graph_key, frame_map in list(self._parallel_results.items()):
                for old_frame in [fid for fid in frame_map.keys() if int(fid) < threshold]:
                    frame_map.pop(old_frame, None)
                if not frame_map:
                    self._parallel_results.pop(graph_key, None)

    def _parallel_submit_graph_job(
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
        if owner_graph_key in self._parallel_disabled_sources:
            return False
        if not getattr(graph, "parallel_ir", {}):
            return False
        requested_output_keys = [str(v) for v in list(output_keys or []) if str(v)]
        if not requested_output_keys:
            requested_output_keys = list(getattr(graph, "parallel_offload_attr_keys", []) or [])
        if not requested_output_keys:
            return False
        if not self._ensure_parallel_pool(scene):
            return False
        if self._parallel_executor is None:
            return False

        queue_key = str(queue_key or graph_key)
        inflight_limit = self._parallel_inflight_limit(scene)
        pending = self._parallel_inflight.get(queue_key)
        if pending is None or int(getattr(pending, "maxlen", 0) or 0) != int(inflight_limit):
            pending = deque(list(pending or []), maxlen=max(int(inflight_limit), 1))
            self._parallel_inflight[queue_key] = pending
        for fut in list(pending):
            if fut.done():
                self._parallel_drop_future(fut)
        pending = self._parallel_inflight.get(queue_key)
        if pending is None or int(getattr(pending, "maxlen", 0) or 0) != int(inflight_limit):
            pending = deque(list(pending or []), maxlen=max(int(inflight_limit), 1))
            self._parallel_inflight[queue_key] = pending

        if len(pending) >= int(inflight_limit):
            oldest = pending[0]
            oldest_meta = dict(self._parallel_future_meta.get(oldest, {}) or {})
            try:
                oldest_frame = int(oldest_meta.get("frame_id", frame_index) or frame_index)
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._parallel_submit_graph_job", exc)
                oldest_frame = int(frame_index)
            frame_delta = int(frame_index) - int(oldest_frame)
            if frame_delta <= int(self._PARALLEL_CANCEL_STALE_FRAME_DELTA):
                self._parallel_queue_pressure_drops += 1
                return False
            if oldest.cancel():
                self._parallel_drop_future(oldest)
            else:
                self._parallel_queue_pressure_drops += 1
                return False

        chunks_by_id = {
            str(src_id): chunk
            for src_id, chunk in dict(source_chunks or {}).items()
            if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate")
        }
        if not chunks_by_id:
            return False

        block_size = int(getattr(graph, "block_size", 0) or 0)
        if block_size <= 0:
            block_size = max(
                int(getattr(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)), "shape", [1])[-1] or 1)
                for chunk in chunks_by_id.values()
            )
        block_size = int(max(block_size, 1))
        sample_rate = max(
            int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            for chunk in chunks_by_id.values()
        )
        include_graph_ir_flag = bool(
            (graph_key in self._parallel_graph_ir_required)
            if include_graph_ir is None
            else include_graph_ir
        )
        source_payloads: Dict[str, Dict[str, Any]] = {}
        for src_id, chunk in chunks_by_id.items():
            src_sr = int(getattr(chunk, "sample_rate", sample_rate) or sample_rate)
            source_payloads[src_id] = self._build_parallel_source_payload(
                chunk,
                block_size=block_size,
                sample_rate=int(max(src_sr, 1)),
            )
        self._parallel_job_seq = int(self._parallel_job_seq) + 1
        job_key = (
            f"g{int(self._parallel_generation_id)}:"
            f"q{str(queue_key)}:"
            f"f{int(frame_index)}:"
            f"j{int(self._parallel_job_seq)}"
        )
        source_payload_ipc, payload_slot_keys = self._parallel_payload_to_ipc(
            source_payloads,
            graph_key=graph_key,
            frame_index=frame_index,
        )
        payload = {
            "generation_id": int(self._parallel_generation_id),
            "graph_uid": str(getattr(graph, "parallel_graph_uid", "") or owner_graph_key),
            "frame_id": int(frame_index),
            "block_size": int(block_size),
            "sample_rate": int(max(sample_rate, 1)),
            "graph_ir_digest": str(getattr(graph, "parallel_ir_digest", "") or ""),
            "output_keys": list(requested_output_keys),
            "job_key": str(job_key),
        }
        payload.update(source_payload_ipc)
        if include_graph_ir_flag:
            payload["graph_ir"] = dict(getattr(graph, "parallel_ir", {}) or {})

        try:
            with _main_script_hidden_from_spawn():
                future_obj = self._parallel_executor.submit(evaluate_parallel_job, payload)
        except Exception as e:
            for slot_key in payload_slot_keys:
                self._parallel_release_payload_slot(slot_key)
            fail_count = int(self._parallel_failures.get(owner_graph_key, 0)) + 1
            self._parallel_failures[owner_graph_key] = fail_count
            if fail_count <= 1 and self._parallel_restart_count < self._PARALLEL_RESTART_LIMIT:
                self._restart_parallel_pool(scene, f"submit failure: {e}", source_key=owner_graph_key)
            else:
                self._parallel_disabled_sources.add(owner_graph_key)
            utils.log_throttled(
                f"parallel_submit:{owner_graph_key}",
                f"Parallel submit failed for '{owner_graph_key}', reusing last-valid offload values: {e}",
                "WARNING",
                min_interval=0.0,
            )
            return False

        pending.append(future_obj)
        self._parallel_future_meta[future_obj] = {
            "graph_key": graph_key,
            "queue_key": queue_key,
            "owner_graph_key": owner_graph_key,
            "frame_id": int(frame_index),
            "generation_id": int(self._parallel_generation_id),
            "undo_generation": int(getattr(self.engine, "undo_generation", 0) or 0),
            "submit_ts": time.time(),
            "payload_slot_keys": list(payload_slot_keys),
        }
        self._parallel_last_submit_frame = int(frame_index)
        return True

    def _parallel_values_for_graph_frame(self, graph: ACEAudioGraph, graph_key: str, frame_index: int) -> Dict[str, Any]:
        """Retrieve the best available parallel worker results for *frame_index*.

        Priority: exact frame match > stale-reuse window > zero values.
        The stale window is derived from ``_PARALLEL_STALE_REUSE_MS`` converted
        to frames at the current scene FPS so behaviour is FPS-invariant.

        Returns:
            Dict mapping each offload attr_key to its normalised output value.
        """
        offload_keys = list(getattr(graph, "parallel_offload_attr_keys", []) or [])
        if not offload_keys:
            return {}
        frame_map = self._parallel_results.get(graph_key, {}) or {}
        frame_values = dict(frame_map.pop(int(frame_index), {}) or {})
        if isinstance(frame_map, dict) and not frame_map:
            self._parallel_results.pop(graph_key, None)
        if frame_values:
            merged_last = dict(self._parallel_last_valid.get(graph_key, {}) or {})
            merged_last.update(frame_values)
            self._parallel_last_valid[graph_key] = merged_last
            self._parallel_last_valid_frame[graph_key] = int(frame_index)
            base = merged_last
        else:
            last_frame = int(self._parallel_last_valid_frame.get(graph_key, -10_000_000) or -10_000_000)
            fps_now = max(float(self.engine._current_scene_fps), 1.0)
            stale_window_frames = max(1, int(self._PARALLEL_STALE_REUSE_MS * fps_now / 1000.0 + 0.5))
            if (int(frame_index) - last_frame) <= stale_window_frames:
                base = dict(self._parallel_last_valid.get(graph_key, {}) or {})
            else:
                base = {}
        resolved: Dict[str, Any] = {}
        for attr_key in offload_keys:
            if attr_key in base:
                resolved[attr_key] = self.engine.frame._normalize_output_value(attr_key, base[attr_key])
            else:
                resolved[attr_key] = self.engine.frame._zero_value_for_attr(attr_key)
        return resolved

    def _parallel_has_usable_values_for_graph_frame(self, graph_key: str, frame_index: int) -> bool:
        """Return True when Stage-B has usable data for *graph_key* at *frame_index*.

        "Usable" means either fresh values for the current frame or non-empty
        last-valid values still inside the stale reuse window.
        """
        key = str(graph_key or "")
        frame_map = dict(self._parallel_results.get(key, {}) or {})
        if dict(frame_map.get(int(frame_index), {}) or {}):
            return True

        last_vals = dict(self._parallel_last_valid.get(key, {}) or {})
        if not last_vals:
            return False

        last_frame = int(self._parallel_last_valid_frame.get(key, -10_000_000) or -10_000_000)
        fps_now = max(float(self.engine._current_scene_fps), 1.0)
        stale_window_frames = max(1, int(self._PARALLEL_STALE_REUSE_MS * fps_now / 1000.0 + 0.5))
        return (int(frame_index) - last_frame) <= stale_window_frames

    def _parallel_destroy_idle_payload_slots(self):
        for slot_key in list(self._parallel_payload_slots.keys()):
            if int(self._parallel_payload_slot_refs.get(slot_key, 0) or 0) <= 0:
                self._parallel_destroy_payload_slot(slot_key)

    @staticmethod
    def _parallel_close_unlink_shm(shm_obj):
        if shm_obj is None:
            return
        with utils.logged_errors("audio_engine.ACEAudioEngine._parallel_close_unlink_shm"):
            shm_obj.close()
        try:
            shm_obj.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._parallel_close_unlink_shm#2", exc)

    def _parallel_destroy_payload_slot(self, slot_key: str):
        key = str(slot_key)
        self._parallel_payload_slot_refs.pop(key, None)
        prefix = key.split("#", 1)[0]
        free_list = self._parallel_payload_free_slots.get(prefix)
        if isinstance(free_list, deque):
            self._parallel_payload_free_slots[prefix] = deque(v for v in free_list if str(v) != key)
            if not self._parallel_payload_free_slots[prefix]:
                self._parallel_payload_free_slots.pop(prefix, None)
        for frame_key, mapped_slot in list(self._parallel_frame_payload_slots.items()):
            if str(mapped_slot) == key:
                self._parallel_frame_payload_slots.pop(frame_key, None)
        slot = self._parallel_payload_slots.pop(key, None)
        if not isinstance(slot, dict):
            return
        self._parallel_close_unlink_shm(slot.get("mono_shm"))
        self._parallel_close_unlink_shm(slot.get("stereo_shm"))

    def _parallel_clear_payload_slot_field(self, slot_key: str, field: str):
        entry = self._parallel_payload_slots.get(str(slot_key))
        if not isinstance(entry, dict):
            return
        shm_key = f"{field}_shm"
        bytes_key = f"{field}_bytes"
        shape_key = f"{field}_shape"
        self._parallel_close_unlink_shm(entry.pop(shm_key, None))
        entry.pop(bytes_key, None)
        entry.pop(shape_key, None)

    def _parallel_payload_slot_descriptor(self, slot_key: str, field: str) -> Optional[Dict[str, Any]]:
        entry = self._parallel_payload_slots.get(str(slot_key))
        if not isinstance(entry, dict):
            return None
        shm_obj = entry.get(f"{field}_shm")
        shape = tuple(int(v) for v in tuple(entry.get(f"{field}_shape", ()) or ()))
        if shm_obj is None or not shape:
            return None
        return {
            "name": str(shm_obj.name),
            "shape": shape,
            "dtype": "float32",
        }

    def _parallel_release_payload_slot(self, slot_key: str):
        key = str(slot_key)
        if key not in self._parallel_payload_slots:
            return
        ref_count = max(int(self._parallel_payload_slot_refs.get(key, 0) or 0) - 1, 0)
        self._parallel_payload_slot_refs[key] = int(ref_count)
        if ref_count > 0:
            return
        for frame_key, mapped_slot in list(self._parallel_frame_payload_slots.items()):
            if str(mapped_slot) == key:
                self._parallel_frame_payload_slots.pop(frame_key, None)
        prefix = key.split("#", 1)[0]
        free_list = self._parallel_payload_free_slots.setdefault(prefix, deque())
        if key not in free_list:
            free_list.append(key)

    def _parallel_release_payload_slots(self):
        if not self._parallel_payload_slots:
            return
        slot_keys = list(self._parallel_payload_slots.keys())
        for slot_key in slot_keys:
            self._parallel_destroy_payload_slot(slot_key)
        self._parallel_payload_free_slots.clear()

    def _parallel_reserve_payload_slot(
        self,
        graph_key: str,
        frame_index: int,
        src_id: str,
    ) -> tuple[str, bool]:
        graph_key = str(graph_key or "")
        src_key = str(src_id or "")
        frame_key = (graph_key, int(frame_index), src_key)
        existing = self._parallel_frame_payload_slots.get(frame_key)
        if existing and existing in self._parallel_payload_slots:
            self._parallel_payload_slot_refs[existing] = int(self._parallel_payload_slot_refs.get(existing, 0) or 0) + 1
            return str(existing), False

        slot_prefix = f"{graph_key}:{src_key}"
        free = self._parallel_payload_free_slots.get(slot_prefix)
        chosen_slot = ""
        if free:
            while free and not chosen_slot:
                candidate = str(free.popleft())
                if candidate in self._parallel_payload_slots:
                    chosen_slot = candidate
            if free is not None and not free:
                self._parallel_payload_free_slots.pop(slot_prefix, None)
        if not chosen_slot:
            slot_index = int(self._parallel_payload_slot_counters.get(slot_prefix, 0) or 0)
            chosen_slot = f"{slot_prefix}#{slot_index}"
            self._parallel_payload_slot_counters[slot_prefix] = int(slot_index) + 1
            self._parallel_payload_slots.setdefault(chosen_slot, {})
            self._parallel_payload_slot_refs[chosen_slot] = 0

        self._parallel_frame_payload_slots[frame_key] = chosen_slot
        self._parallel_payload_slot_refs[chosen_slot] = int(self._parallel_payload_slot_refs.get(chosen_slot, 0) or 0) + 1
        return chosen_slot, True

    def _parallel_ensure_payload_slot(
        self,
        slot_key: str,
        field: str,
        array: np.ndarray,
    ) -> Dict[str, Any]:
        key = str(slot_key)
        entry = self._parallel_payload_slots.setdefault(key, {})
        shm_key = f"{field}_shm"
        bytes_key = f"{field}_bytes"
        shape_key = f"{field}_shape"

        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        required_bytes = max(int(arr.nbytes), 4)
        shm_obj = entry.get(shm_key)
        current_bytes = int(entry.get(bytes_key, 0) or 0)
        if shm_obj is None or current_bytes < required_bytes:
            self._parallel_close_unlink_shm(shm_obj)
            shm_obj = mp_shared_memory.SharedMemory(create=True, size=required_bytes)
            entry[shm_key] = shm_obj
            entry[bytes_key] = int(required_bytes)
        shape = tuple(int(v) for v in arr.shape)
        entry[shape_key] = shape
        view = np.ndarray(shape, dtype=np.float32, buffer=shm_obj.buf)
        view[:] = arr
        return {
            "name": str(shm_obj.name),
            "shape": shape,
            "dtype": "float32",
        }

    def _parallel_payload_to_ipc(
        self,
        source_payloads: Dict[str, Dict[str, Any]],
        graph_key: str,
        frame_index: int,
    ) -> tuple[Dict[str, Any], list[str]]:
        payload_chunks = {
            str(src_id): dict(src_payload or {})
            for src_id, src_payload in dict(source_payloads or {}).items()
        }
        if not _SHARED_MEMORY_AVAILABLE:
            return {"source_chunks": payload_chunks}, []

        slot_keys: list[str] = []
        shm_map: Dict[str, Dict[str, Any]] = {}
        try:
            for src_id, src_payload in payload_chunks.items():
                mono = src_payload.get("mono")
                if not isinstance(mono, np.ndarray):
                    continue

                slot_key, needs_upload = self._parallel_reserve_payload_slot(
                    graph_key=str(graph_key or ""),
                    frame_index=int(frame_index),
                    src_id=str(src_id),
                )
                slot_keys.append(slot_key)
                mono_arr = np.ascontiguousarray(np.asarray(mono, dtype=np.float32))
                if needs_upload:
                    src_meta: Dict[str, Any] = {
                        "mono": self._parallel_ensure_payload_slot(slot_key, "mono", mono_arr),
                    }
                else:
                    mono_meta = self._parallel_payload_slot_descriptor(slot_key, "mono")
                    if mono_meta is None:
                        src_meta = {
                            "mono": self._parallel_ensure_payload_slot(slot_key, "mono", mono_arr),
                        }
                    else:
                        src_meta = {"mono": mono_meta}

                stereo = src_payload.get("stereo")
                if isinstance(stereo, np.ndarray):
                    stereo_arr = np.ascontiguousarray(np.asarray(stereo, dtype=np.float32))
                    if needs_upload:
                        src_meta["stereo"] = self._parallel_ensure_payload_slot(slot_key, "stereo", stereo_arr)
                    else:
                        stereo_meta = self._parallel_payload_slot_descriptor(slot_key, "stereo")
                        if stereo_meta is not None:
                            src_meta["stereo"] = stereo_meta
                        else:
                            src_meta["stereo"] = self._parallel_ensure_payload_slot(slot_key, "stereo", stereo_arr)
                elif needs_upload:
                    self._parallel_clear_payload_slot_field(slot_key, "stereo")

                src_payload.pop("mono", None)
                src_payload.pop("stereo", None)
                shm_map[src_id] = src_meta

            packed = {"source_chunks": payload_chunks}
            if shm_map:
                packed["source_chunks_shm"] = shm_map
            return packed, slot_keys
        except Exception as e:
            for slot_key in slot_keys:
                self._parallel_release_payload_slot(slot_key)
            utils.log_throttled(
                "parallel_shm_ipc",
                f"Shared-memory IPC setup failed, falling back to pickle transport: {e}",
                "WARNING",
                min_interval=4.0,
            )
            return {"source_chunks": payload_chunks}, []
