"""Process-pool entry points: job evaluation, warm-up and per-process caches."""

from __future__ import annotations
import os
import time
from collections import OrderedDict as _OrderedDict
from typing import Any, Dict, Optional, Tuple
import numpy as np
try:
    from multiprocessing import shared_memory as mp_shared_memory
    _SHARED_MEMORY_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime/platform dependent
    mp_shared_memory = None
    _SHARED_MEMORY_AVAILABLE = False
from ... import utils
from ..tasks.chunk import SAMPLE_RATE
from .graph import _CompiledWorkerGraph, _compile_worker_graph, _normalize_worker_value


_WORKER_CACHE_MAX = 64


# Keyed by (generation_id, graph_uid, block_size, graph_ir_digest).
_WORKER_GRAPH_CACHE: "_OrderedDict[Tuple[int, str, int, str], _CompiledWorkerGraph]" = _OrderedDict()


_WORKER_LAST_GENERATION: Optional[int] = None


# Per-source shared-memory attachment cache; keyed by "<src_id>:mono" / "<src_id>:stereo".
_WORKER_SHM_CACHE: Dict[str, Dict[str, Any]] = {}


def _trim_worker_cache():
    while len(_WORKER_GRAPH_CACHE) > _WORKER_CACHE_MAX:
        _key, graph = _WORKER_GRAPH_CACHE.popitem(last=False)
        if graph is not None:
            graph.close()


def _close_worker_shm_cache_entry(cache_key: str):
    entry = _WORKER_SHM_CACHE.pop(str(cache_key), None)
    if not isinstance(entry, dict):
        return
    shm_obj = entry.get("shm")
    if shm_obj is None:
        return
    with utils.logged_errors("job._close_worker_shm_cache_entry"):
        shm_obj.close()


def _close_worker_shm_cache():
    for cache_key in list(_WORKER_SHM_CACHE.keys()):
        _close_worker_shm_cache_entry(cache_key)


def _prune_worker_cache_for_generation(generation_id: int):
    stale_keys = [key for key in list(_WORKER_GRAPH_CACHE.keys()) if int(key[0]) != int(generation_id)]
    for key in stale_keys:
        graph = _WORKER_GRAPH_CACHE.pop(key, None)
        if graph is not None:
            graph.close()
    _close_worker_shm_cache()


def _resolve_worker_shm_array(cache_key: str, spec: Dict[str, Any]) -> Optional[np.ndarray]:
    """Attach to or retrieve a named shared-memory segment as a NumPy view."""
    if not _SHARED_MEMORY_AVAILABLE:
        return None
    name = str(spec.get("name", "") or "")
    shape = tuple(int(v) for v in tuple(spec.get("shape", ()) or ()))
    if not name or not shape:
        return None

    key = str(cache_key)
    entry = _WORKER_SHM_CACHE.get(key)
    if isinstance(entry, dict):
        entry_name = str(entry.get("name", "") or "")
        entry_shape = tuple(int(v) for v in tuple(entry.get("shape", ()) or ()))
        arr = entry.get("array")
        if entry_name == name and entry_shape == shape and isinstance(arr, np.ndarray):
            return arr
        _close_worker_shm_cache_entry(key)

    shm_obj = mp_shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.float32, buffer=shm_obj.buf)
    _WORKER_SHM_CACHE[key] = {
        "name": name,
        "shape": shape,
        "shm": shm_obj,
        "array": arr,
    }
    return arr


def _resolve_source_payloads(job: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract per-source audio payloads from a job dict, resolving SHM references."""
    source_chunks_raw = dict(job.get("source_chunks", {}) or {})
    if not source_chunks_raw and isinstance(job.get("source_chunk"), dict):
        source_chunks_raw = {"__default__": dict(job.get("source_chunk", {}) or {})}
    source_chunks = {
        str(src_id): dict(payload or {})
        for src_id, payload in source_chunks_raw.items()
    }
    if not _SHARED_MEMORY_AVAILABLE:
        return source_chunks

    shm_map = job.get("source_chunks_shm")
    if not isinstance(shm_map, dict):
        return source_chunks

    resolved = {str(src_id): dict(payload or {}) for src_id, payload in source_chunks.items()}
    try:
        for src_id, src_meta_raw in shm_map.items():
            src_key = str(src_id or "")
            if not src_key:
                continue
            src_meta = dict(src_meta_raw or {})
            src_payload = resolved.setdefault(src_key, {})

            mono_spec = dict(src_meta.get("mono", {}) or {})
            mono_arr = _resolve_worker_shm_array(f"{src_key}:mono", mono_spec)
            if isinstance(mono_arr, np.ndarray):
                src_payload["mono"] = mono_arr

            stereo_spec_raw = src_meta.get("stereo")
            if isinstance(stereo_spec_raw, dict):
                stereo_spec = dict(stereo_spec_raw)
                stereo_arr = _resolve_worker_shm_array(f"{src_key}:stereo", stereo_spec)
                if isinstance(stereo_arr, np.ndarray):
                    src_payload["stereo"] = stereo_arr
            else:
                _close_worker_shm_cache_entry(f"{src_key}:stereo")
        return resolved
    except Exception as e:
        _close_worker_shm_cache()
        raise RuntimeError(f"source payload shared-memory resolve failed: {e}")


def evaluate_parallel_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool entrypoint."""
    t0 = time.perf_counter() if "time" in globals() else None
    generation_id = int(job.get("generation_id", 0) or 0)
    graph_uid = str(job.get("graph_uid", "") or "")
    job_key = str(job.get("job_key", "") or "")
    frame_id = int(job.get("frame_id", 0) or 0)
    block_size = max(int(job.get("block_size", 0) or 0), 1)
    sample_rate = max(int(job.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE), 1)
    graph_ir = dict(job.get("graph_ir", {}) or {})
    graph_ir_digest = str(job.get("graph_ir_digest", "") or "")
    output_keys = tuple(
        str(v)
        for v in tuple(job.get("output_keys", ()) or ())
        if str(v)
    )

    if not graph_uid:
        raise ValueError("parallel job missing graph_uid")

    global _WORKER_LAST_GENERATION
    if _WORKER_LAST_GENERATION is None:
        _WORKER_LAST_GENERATION = generation_id
    elif int(_WORKER_LAST_GENERATION) != int(generation_id):
        _prune_worker_cache_for_generation(generation_id)
        _WORKER_LAST_GENERATION = generation_id

    cache_key = (generation_id, graph_uid, block_size, graph_ir_digest)
    graph = _WORKER_GRAPH_CACHE.get(cache_key)
    if graph is not None:
        _WORKER_GRAPH_CACHE.move_to_end(cache_key)
    else:
        if not graph_ir:
            # Worker does not have this graph yet: ask main process to resend IR once.
            return {
                "generation_id": generation_id,
                "graph_uid": graph_uid,
                "job_key": job_key,
                "frame_id": frame_id,
                "values": {},
                "needs_graph_ir": True,
                "pid": os.getpid(),
            }
        graph = _compile_worker_graph(graph_ir=graph_ir, sample_rate=sample_rate, block_size=block_size)
        _WORKER_GRAPH_CACHE[cache_key] = graph
        _trim_worker_cache()

    # Linked parameter values change from frame to frame without a new graph.
    for node_id, params in dict(job.get("param_overrides", {}) or {}).items():
        task = graph.tasks.get(str(node_id))
        if task is not None:
            task.parameters.update(params)

    resolved_payloads = _resolve_source_payloads(job)
    graph.set_source_payloads(
        payloads=resolved_payloads,
        frame_id=frame_id,
        block_size=block_size,
    )

    values: Dict[str, Any] = {}
    if output_keys:
        output_iter = [(key, graph.output_specs.get(key)) for key in output_keys]
    else:
        output_iter = list(graph.output_specs.items())
    for attr_key, spec in output_iter:
        if not isinstance(spec, dict):
            continue
        node_id = str(spec.get("node_id", "") or "")
        socket_idx = int(spec.get("socket_idx", 0) or 0)
        attr_type = str(spec.get("attr_type", "FLOAT") or "FLOAT")
        task = graph.tasks.get(node_id)
        if task is None:
            continue
        try:
            raw_val = task.get_output(frame_id, socket_idx)
        except Exception as exc:
            utils.note_swallowed("job.evaluate_parallel_job", exc)
            continue
        if raw_val is None:
            continue
        values[str(attr_key)] = _normalize_worker_value(attr_type, raw_val)

    return {
        "generation_id": generation_id,
        "graph_uid": graph_uid,
        "job_key": job_key,
        "frame_id": frame_id,
        "values": values,
        "elapsed_s": float(max(time.perf_counter() - t0, 0.0)) if t0 is not None else 0.0,
        "pid": os.getpid(),
    }


# Every stream task class, as (node type, mode), so each one warms up.
_WARMUP_STREAM_TYPES = (
    ("loudness", "rms"), ("loudness", "lufs"), ("loudness", "peak"), ("pitch", ""), ("brightness", ""),
    ("transient", ""), ("stereo", "width"), ("stereo", "balance"), ("stereo_vector", ""),
)


def warm_worker():
    """Pool initializer: load the DSP stack before the process takes jobs."""
    try:
        _warm_evaluation()
    except Exception as exc:  # noqa: BLE001 boundary: a failed warm-up must not break the pool
        utils.note_swallowed("job.warm_worker", exc)


def _warm_evaluation():
    sample_rate, block_size = 48000, 2000
    nodes = {"src": {"category": "source_mngmt", "type": "src", "parameters": {}}}
    nodes["split"] = {"category": "delta", "type": "frequency_splitter", "parameters": {}}
    links = [("src", 0, "split", 0)]
    output_specs = {}
    wiring = [("src", t) for t in _WARMUP_STREAM_TYPES] + [("split", ("loudness", "rms"))]
    for parent, (node_type, mode) in wiring:
        node_id = f"{parent}_{node_type}_{mode}"
        nodes[node_id] = {"category": "stream", "type": node_type, "parameters": {"mode": mode} if mode else {}}
        links.append((parent, 0, node_id, 0))
        output_specs[node_id] = {"node_id": node_id, "socket_idx": 0, "attr_type": "FLOAT"}
    graph = _compile_worker_graph(
        {"nodes": nodes, "links": links, "source_node_ids": ["src"], "source_ir": {}, "output_specs": output_specs},
        sample_rate=sample_rate,
        block_size=block_size,
    )
    try:
        t = np.arange(block_size, dtype=np.float32) / sample_rate
        mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        payload = {"mono": mono, "stereo": np.stack([mono, mono]), "sample_rate": sample_rate}
        for frame_id in range(2):
            graph.set_source_payloads({"src": payload}, frame_id=frame_id, block_size=block_size)
            for spec in output_specs.values():
                with utils.logged_errors("job._warm_evaluation"):
                    graph.tasks[spec["node_id"]].get_output(frame_id, 0)
    finally:
        graph.close()


def worker_pid() -> int:
    return os.getpid()
