"""
Worker-side DSP evaluation helpers for ACE process offload.

This module is intentionally Blender-agnostic:
- no bpy imports
- pure Python/NumPy/audioflux task evaluation only
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict as _OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from multiprocessing import shared_memory as mp_shared_memory
    _SHARED_MEMORY_AVAILABLE = True
except Exception:  # pragma: no cover - runtime/platform dependent
    mp_shared_memory = None
    _SHARED_MEMORY_AVAILABLE = False

from .audio_tasks_registry import (
    SAMPLE_RATE,
    AudioChunk,
    AudioFluxAnalyzer,
    ACEAudioTask,
    SourceFrontendProcessor,
    FrequencySplitterTask,
    MidsideSplitterTask,
    AttackReleaseSplitterTask,
    ChromagramTask,
    AutoComponentSplitterTask,
    RMSTask,
    LUFSTask,
    PeakTask,
    PitchTask,
    SpectralCentroidTask,
    SpectralFluxTask,
    OnsetTask,
    ZeroCrossTask,
    StereoWidthTask,
    StereoBalanceTask,
    StreamAccumulatorTask,
    LogicTask,
    AudioMathTask,
    RollingAvgTask,
    SmoothOperatorTask,
    UnifiedLoudnessTask,
)


_WORKER_CACHE_MAX = 64
# Keyed by (generation_id, graph_uid, block_size, graph_ir_digest).
_WORKER_GRAPH_CACHE: "_OrderedDict[Tuple[int, str, int, str], _CompiledWorkerGraph]" = _OrderedDict()
_WORKER_LAST_GENERATION: Optional[int] = None
# Per-source shared-memory attachment cache; keyed by "<src_id>:mono" / "<src_id>:stereo".
_WORKER_SHM_CACHE: Dict[str, Dict[str, Any]] = {}


def _coerce_worker_audio_chunk(data: Any, block_size: int, sample_rate: int = SAMPLE_RATE) -> AudioChunk:
    """Coerce arbitrary payload data into a typed AudioChunk.

    If *data* is already an AudioChunk it is returned unchanged.  Otherwise
    the value is cast to a float32 NumPy array, zero-padded or truncated to
    *block_size*, and wrapped in a minimal AudioChunk with default metadata.

    Args:
        data: Raw audio payload — may be None, an ndarray, a list, or an
            existing AudioChunk.
        block_size: Required number of samples in the returned chunk.
        sample_rate: Sample rate to attach to the new chunk.

    Returns:
        An AudioChunk of exactly *block_size* samples.
    """
    if isinstance(data, AudioChunk):
        return data

    bs = max(int(block_size), 1)
    sr = max(int(sample_rate), 1)
    if data is None:
        mono = np.zeros(bs, dtype=np.float32)
    else:
        mono = np.asarray(data, dtype=np.float32).reshape(-1)
        if mono.shape[0] < bs:
            padded = np.zeros(bs, dtype=np.float32)
            if mono.shape[0] > 0:
                padded[: mono.shape[0]] = mono
            mono = padded
        elif mono.shape[0] > bs:
            mono = mono[:bs]
    return AudioChunk(mono, None, sr, frame_index=-1, lineage=())


class _WorkerSourceSpecsPassthroughTask(ACEAudioTask):
    """
    Worker-safe replacement for SourceSpecsTask.

    Reads scene_bpm, scene_fps, and scene_beats_per_bar from self.parameters,
    which are injected by ACEAudioEngine._prepare_parallel_graph_ir() at
    graph-IR compilation time.  Falls back to neutral defaults (120 BPM,
    4/4 time, 24 FPS) only when the graph IR has not been updated yet so
    that the worker never crashes on a missing key.
    """

    def process(self, inputs):
        chunk_in = _coerce_worker_audio_chunk(inputs.get(0), self.block_size)
        # Read live scene values injected during graph IR compilation.
        # Fallback constants are intentionally kept identical to the previous
        # hard-coded values to avoid silent downstream breakage when the IR
        # has not been populated yet (e.g. first frame after pool start).
        bpm           = float(self.parameters.get("scene_bpm", 120.0) or 120.0)
        beats_per_bar = float(self.parameters.get("scene_beats_per_bar", 4.0) or 4.0)
        fps           = float(self.parameters.get("scene_fps", 24.0) or 24.0)
        beats_per_frame = float(bpm / max(60.0 * fps, 1e-6))
        frames_per_beat = float(1.0 / beats_per_frame) if beats_per_frame > 1e-6 else 0.0
        frame_idx = int(getattr(chunk_in, "frame_index", 0) or 0)
        pos_in_measure = (float(frame_idx) * beats_per_frame) % beats_per_bar
        beat_phase = float(pos_in_measure / beats_per_bar) if beats_per_bar > 0.0 else 0.0
        return (float(frames_per_beat), float(beat_phase), chunk_in)


class _WorkerSourceProvider:
    """Light provider shim exposing `compute_feature` for stream tasks.

    Wraps a borrowed AudioFluxAnalyzer and re-borrows it whenever the
    sample-rate or block-size changes so every source bridge task gets a
    correctly configured analyzer without allocating one per task.
    """

    def __init__(self, sample_rate: int, block_size: int):
        self.sample_rate = max(int(sample_rate), 1)
        self.block_size = max(int(block_size), 1)
        self.analysis = AudioFluxAnalyzer.borrow(self.sample_rate, self.block_size)

    def ensure_context(self, sample_rate: int, block_size: int):
        """Re-borrow the analyzer when audio parameters change.

        Args:
            sample_rate: New sample rate in Hz.
            block_size: New block size in samples.
        """
        sr = max(int(sample_rate), 1)
        bs = max(int(block_size), 1)
        if sr == self.sample_rate and bs == self.block_size and self.analysis is not None:
            return
        if self.analysis is not None:
            AudioFluxAnalyzer.release(self.analysis)
        self.sample_rate = sr
        self.block_size = bs
        self.analysis = AudioFluxAnalyzer.borrow(self.sample_rate, self.block_size)

    def compute_feature(self, name: str, chunk: AudioChunk):
        """Delegate feature computation to the underlying analyzer.

        Args:
            name: Feature name (e.g. "rms", "centroid").
            chunk: The audio chunk to analyze.

        Returns:
            The computed feature value, or None if no analyzer is available.
        """
        if self.analysis is None:
            return None
        return self.analysis.compute_feature(name, chunk)

    def close(self):
        """Release the borrowed analyzer back to the pool."""
        if self.analysis is not None:
            AudioFluxAnalyzer.release(self.analysis)
            self.analysis = None


class _WorkerSourceBridgeTask:
    """
    Source placeholder used in worker graphs.
    The main thread injects source chunk payloads each frame.
    """

    def __init__(
        self,
        node_id: str,
        provider: _WorkerSourceProvider,
        block_size: int,
        source_ir: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.provider = provider
        self.block_size = max(int(block_size), 1)
        self._current_frame = -1
        self._current_chunk: Optional[AudioChunk] = None
        self._silence = np.zeros(self.block_size, dtype=np.float32)
        self._frontend_processor = SourceFrontendProcessor()
        meta = dict(source_ir or {})
        self.default_sample_rate = max(int(meta.get("sample_rate", provider.sample_rate) or provider.sample_rate), 1)
        lineage_raw = meta.get("lineage", (self.node_id,))
        if isinstance(lineage_raw, (list, tuple)):
            lineage = tuple(str(v) for v in lineage_raw if str(v))
        else:
            lineage = (self.node_id,)
        self.default_lineage = lineage if lineage else (self.node_id,)
        spectrum_token = meta.get("spectrum_token", self.default_lineage)
        if isinstance(spectrum_token, (list, tuple)):
            self.default_spectrum_token = tuple(spectrum_token)
        else:
            self.default_spectrum_token = spectrum_token

    def set_block_size(self, block_size: int):
        bs = max(int(block_size), 1)
        if bs == self.block_size:
            return
        self.block_size = bs
        self._silence = np.zeros(self.block_size, dtype=np.float32)

    def set_chunk(self, payload: Dict[str, Any], frame_id: int, block_size: int):
        self.set_block_size(block_size)
        sr = max(int(payload.get("sample_rate", self.default_sample_rate) or self.default_sample_rate), 1)
        self.provider.ensure_context(sample_rate=sr, block_size=self.block_size)

        mono = np.asarray(payload.get("mono", ()), dtype=np.float32).reshape(-1)
        if mono.shape[0] < self.block_size:
            padded = np.zeros(self.block_size, dtype=np.float32)
            if mono.shape[0] > 0:
                padded[: mono.shape[0]] = mono
            mono = padded
        elif mono.shape[0] > self.block_size:
            mono = mono[: self.block_size]

        stereo_raw = payload.get("stereo")
        stereo = None
        if stereo_raw is not None:
            arr = np.asarray(stereo_raw, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == mono.shape[0]:
                stereo = arr.astype(np.float32, copy=False)
            elif arr.ndim == 2 and arr.shape[0] == mono.shape[0]:
                stereo = arr.T.astype(np.float32, copy=False)
            else:  # ndim != 2 OR both shape axes mismatch mono length
                # utils.log() cannot be used here: this module is intentionally
                # bpy-free (Stage-B worker process). print() to stderr is the only
                # safe fallback — this is an explicit deviation from project style.
                print(
                    f"[ACE WARNING] Worker stereo shape mismatch: stereo={arr.shape}, "
                    f"mono_len={mono.shape[0]}, frame={frame_id}. Dropping stereo.",
                    file=sys.stderr,
                )

        band_indices = payload.get("band_indices")
        if band_indices is not None:
            try:
                band_indices = np.asarray(band_indices, dtype=np.int64).reshape(-1)
            except Exception:
                band_indices = None

        band_hz_bounds = payload.get("band_hz_bounds")
        if isinstance(band_hz_bounds, (list, tuple)) and len(band_hz_bounds) >= 2:
            try:
                band_hz_bounds = (float(band_hz_bounds[0]), float(band_hz_bounds[1]))
            except Exception:
                band_hz_bounds = None
        else:
            band_hz_bounds = None

        frontend_config = payload.get("frontend_config")
        if isinstance(frontend_config, (list, tuple)) and len(frontend_config) >= 4:
            try:
                frontend_config = (
                    float(frontend_config[0] or 0.0),
                    float(frontend_config[1] or 0.0),
                    float(frontend_config[2] or 0.0),
                    float(frontend_config[3] or 0.0),
                )
            except Exception:
                frontend_config = None
        else:
            frontend_config = None

        analysis_ops_raw = payload.get("analysis_ops", ()) or ()
        analysis_ops: Tuple[Tuple[str, str], ...] = ()
        if isinstance(analysis_ops_raw, (list, tuple)):
            normalized = []
            for item in analysis_ops_raw:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                kind = str(item[0] or "").strip().lower()
                mode = str(item[1] or "").strip().lower()
                if kind and mode:
                    normalized.append((kind, mode))
            analysis_ops = tuple(normalized)

        analysis_mono = None
        analysis_stereo = None
        if frontend_config is not None:
            analysis_mono, analysis_stereo = self._frontend_processor.process(
                mono=mono,
                stereo=stereo,
                sample_rate=sr,
                frontend_config=frontend_config,
            )

        chunk = AudioChunk(
            mono=mono,
            stereo=stereo,
            sample_rate=sr,
            provider=self.provider,
            frame_index=int(frame_id),
            analysis_mono=analysis_mono,
            analysis_stereo=analysis_stereo,
            frontend_config=frontend_config,
            band_indices=band_indices,
            band_index_fft_size=payload.get("band_index_fft_size"),
            band_hz_bounds=band_hz_bounds,
            lineage=self.default_lineage,
            spectrum_token=self.default_spectrum_token,
            analysis_ops=analysis_ops if analysis_ops else (),
            latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
        )
        self._current_chunk = chunk
        self._current_frame = int(frame_id)

    def get_output(self, frame_index: int, output_socket_idx: int = 0):  # noqa: ARG002
        if self._current_chunk is None:
            return self._silence
        if int(frame_index) != self._current_frame:
            return self._silence
        return self._current_chunk


class _CompiledWorkerGraph:
    """A fully instantiated, wired task graph ready for per-frame evaluation.

    Created once per (generation_id, graph_uid, block_size, graph_ir_digest)
    cache key and reused across frames.  Source chunks are injected each frame
    via set_source_payloads(); output values are read via get_output() calls on
    individual tasks.
    """

    def __init__(
        self,
        tasks: Dict[str, Any],
        source_tasks: Dict[str, _WorkerSourceBridgeTask],
        output_specs: Dict[str, Dict[str, Any]],
        source_providers: Dict[str, _WorkerSourceProvider],
        block_size: int,
    ):
        self.tasks = tasks
        self.source_tasks = source_tasks
        self.output_specs = output_specs
        self.source_providers = source_providers
        self.block_size = max(int(block_size), 1)

    def set_source_payloads(self, payloads: Dict[str, Dict[str, Any]], frame_id: int, block_size: int):
        """Inject per-source audio chunks for the current frame.

        Args:
            payloads: Mapping from source node ID to a payload dict containing
                at minimum ``"mono"`` and optionally ``"stereo"`` arrays.
            frame_id: The current frame index used to validate stale chunks.
            block_size: Expected number of samples per block.
        """
        bs = max(int(block_size), 1)
        if bs != self.block_size:
            self.block_size = bs
            for task in self.tasks.values():
                if hasattr(task, "block_size"):
                    task.block_size = bs
                if hasattr(task, "silence_buffer"):
                    task.silence_buffer = np.zeros(bs, dtype=np.float32)
        source_payloads = {str(k): dict(v or {}) for k, v in dict(payloads or {}).items()}
        for src_task in self.source_tasks.values():
            src_payload = source_payloads.get(str(src_task.node_id))
            if not isinstance(src_payload, dict):
                src_payload = {}
            src_task.set_chunk(src_payload, frame_id=frame_id, block_size=bs)

    def close(self):
        """Release all task and provider resources held by this graph."""
        for task in self.tasks.values():
            close_fn = getattr(task, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        closed = set()
        for provider in self.source_providers.values():
            key = id(provider)
            if key in closed:
                continue
            closed.add(key)
            provider.close()


def _create_worker_task(node_id: str, category: str, node_type: str, params: Dict[str, Any], block_size: int):
    """Instantiate the correct ACEAudioTask subclass for a given node IR entry.

    Falls back to a bare ACEAudioTask for unknown category/type combinations
    so unrecognised nodes produce silence rather than crashing the worker.

    Args:
        node_id: Unique node identifier string from the graph IR.
        category: ACE node category (e.g. "stream", "delta", "utilities").
        node_type: ACE node type within the category (e.g. "rms", "pitch").
        params: Parameter dict from the node IR record.
        block_size: Analysis block size in samples.

    Returns:
        An ACEAudioTask (or subclass) instance.
    """
    cat = str(category or "")
    typ = str(node_type or "")
    bs = max(int(block_size), 1)

    if cat == "utilities":
        if typ == "src_specs":
            return _WorkerSourceSpecsPassthroughTask(node_id, params, bs)
        if typ == "audio_math":
            return AudioMathTask(node_id, params, bs)
        if typ == "stream_accumulator":
            return StreamAccumulatorTask(node_id, params, bs)
        if typ == "logic":
            return LogicTask(node_id, params, bs)
        if typ == "util_rolling_avg":
            return RollingAvgTask(node_id, params, bs)
        if typ == "util_smooth_operator":
            return SmoothOperatorTask(node_id, params, bs)

    if cat == "delta":
        if typ == "frequency_splitter":
            return FrequencySplitterTask(node_id, params, bs)
        if typ == "midside_splitter":
            return MidsideSplitterTask(node_id, params, bs)
        if typ == "attackrelease_splitter":
            return AttackReleaseSplitterTask(node_id, params, bs)
        if typ == "chromagram":
            return ChromagramTask(node_id, params, bs)
        if typ == "auto_component_splitter":
            return AutoComponentSplitterTask(node_id, params, bs)

    if cat == "stream":
        if typ == "rms":
            return RMSTask(node_id, params, bs)
        if typ == "lufs":
            return LUFSTask(node_id, params, bs)
        if typ == "peak":
            return PeakTask(node_id, params, bs)
        if typ == "pitch":
            return PitchTask(node_id, params, bs)
        if typ == "spectral_centroid":
            return SpectralCentroidTask(node_id, params, bs)
        if typ == "spectral_flux":
            return SpectralFluxTask(node_id, params, bs)
        if typ == "onset":
            return OnsetTask(node_id, params, bs)
        if typ == "zero_crossing_rate":
            return ZeroCrossTask(node_id, params, bs)
        if typ == "stereo_width":
            return StereoWidthTask(node_id, params, bs)
        if typ == "stereo_balance":
            return StereoBalanceTask(node_id, params, bs)
        if typ == "loudness":
            return UnifiedLoudnessTask(node_id, params, bs)

    return ACEAudioTask(node_id, params, bs)


def _compile_worker_graph(graph_ir: Dict[str, Any], sample_rate: int, block_size: int) -> _CompiledWorkerGraph:
    """Build a _CompiledWorkerGraph from a serialised graph IR dict.

    Creates source bridge tasks, regular analysis tasks, and wires all links
    from the IR before returning the ready-to-evaluate graph.

    Args:
        graph_ir: Serialised graph IR as produced by
            ACEAudioEngine._prepare_parallel_graph_ir().
        sample_rate: Default sample rate to assign to source providers.
        block_size: Initial analysis block size in samples.

    Returns:
        A fully wired _CompiledWorkerGraph instance.
    """
    bs = max(int(block_size), 1)
    sr = max(int(sample_rate), 1)
    nodes = dict(graph_ir.get("nodes", {}) or {})
    links = list(graph_ir.get("links", ()) or ())
    source_ids = list(graph_ir.get("source_node_ids", ()) or ())
    source_ir = dict(graph_ir.get("source_ir", {}) or {})
    output_specs = dict(graph_ir.get("output_specs", {}) or {})

    tasks: Dict[str, Any] = {}
    source_tasks: Dict[str, _WorkerSourceBridgeTask] = {}
    source_providers: Dict[str, _WorkerSourceProvider] = {}

    for src_id in source_ids:
        src_key = str(src_id or "")
        if not src_key:
            continue
        src_meta = dict(source_ir.get(src_key, {}) or {})
        src_sr = max(int(src_meta.get("sample_rate", sr) or sr), 1)
        provider = _WorkerSourceProvider(sample_rate=src_sr, block_size=bs)
        source_providers[src_key] = provider
        src_task = _WorkerSourceBridgeTask(src_key, provider=provider, block_size=bs, source_ir=src_meta)
        source_tasks[src_key] = src_task
        tasks[src_key] = src_task

    for node_id, rec in nodes.items():
        nid = str(node_id or "")
        if not nid or nid in tasks:
            continue
        category = str(rec.get("category", "") or "")
        node_type = str(rec.get("type", "") or "")
        params = dict(rec.get("parameters", {}) or {})
        tasks[nid] = _create_worker_task(
            node_id=nid,
            category=category,
            node_type=node_type,
            params=params,
            block_size=bs,
        )

    for raw in links:
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        src_id = str(raw[0] or "")
        src_sock = int(raw[1] or 0)
        dst_id = str(raw[2] or "")
        dst_sock = int(raw[3] or 0)
        src_task = tasks.get(src_id)
        dst_task = tasks.get(dst_id)
        if src_task is None or dst_task is None:
            continue
        connect_fn = getattr(dst_task, "connect_input", None)
        if callable(connect_fn):
            connect_fn(input_idx=dst_sock, target_task=src_task, target_output_idx=src_sock)

    return _CompiledWorkerGraph(
        tasks=tasks,
        source_tasks=source_tasks,
        output_specs=output_specs,
        source_providers=source_providers,
        block_size=bs,
    )


def _normalize_worker_value(attr_type: str, value: Any):
    """Convert a raw task output to the scalar or tuple form expected by the main process.

    Args:
        attr_type: Blender attribute type string — "FLOAT", "FLOAT_VECTOR", or
            "FLOAT_COLOR".
        value: Raw value from a task's get_output() call.

    Returns:
        A plain float for FLOAT, or a tuple of 3 or 4 floats for vector/color types.
    """
    attr = str(attr_type or "FLOAT")
    if attr == "FLOAT":
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            return float(arr[0]) if arr.size > 0 else 0.0
        return 0.0

    target_len = 4 if attr == "FLOAT_COLOR" else 3
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        n = int(flat.size)
        padded = [float(flat[i]) if i < n else 0.0 for i in range(target_len)]
    elif isinstance(value, (int, float, np.integer, np.floating)):
        padded = [float(value)] + [0.0] * (target_len - 1)
    else:
        padded = [0.0] * target_len
    if attr == "FLOAT_COLOR" and padded[3] == 0.0:
        padded[3] = 1.0
    return tuple(padded)


def _trim_worker_cache():
    """Evict least-recently-used entries from _WORKER_GRAPH_CACHE."""
    while len(_WORKER_GRAPH_CACHE) > _WORKER_CACHE_MAX:
        _key, graph = _WORKER_GRAPH_CACHE.popitem(last=False)
        if graph is not None:
            graph.close()


def _close_worker_shm_cache_entry(cache_key: str):
    """Close and remove one shared-memory attachment from _WORKER_SHM_CACHE."""
    entry = _WORKER_SHM_CACHE.pop(str(cache_key), None)
    if not isinstance(entry, dict):
        return
    shm_obj = entry.get("shm")
    if shm_obj is None:
        return
    try:
        shm_obj.close()
    except Exception:
        pass


def _close_worker_shm_cache():
    """Close and release all shared-memory attachments in _WORKER_SHM_CACHE."""
    for cache_key in list(_WORKER_SHM_CACHE.keys()):
        _close_worker_shm_cache_entry(cache_key)


def _prune_worker_cache_for_generation(generation_id: int):
    """Drop all cached graphs from previous generations and flush the SHM cache.

    Called when the worker detects a generation change, indicating that the
    main process has recompiled and all previously cached IRs are stale.

    Args:
        generation_id: The new active generation ID.
    """
    stale_keys = [key for key in list(_WORKER_GRAPH_CACHE.keys()) if int(key[0]) != int(generation_id)]
    for key in stale_keys:
        graph = _WORKER_GRAPH_CACHE.pop(key, None)
        if graph is not None:
            graph.close()
    _close_worker_shm_cache()


def _resolve_worker_shm_array(cache_key: str, spec: Dict[str, Any]) -> Optional[np.ndarray]:
    """Attach to or retrieve a named shared-memory segment as a NumPy view.

    Caches the attachment in _WORKER_SHM_CACHE to avoid re-opening the same
    segment every frame.  Returns None when shared memory is unavailable or
    the spec is incomplete.

    Args:
        cache_key: A stable string key used for the internal SHM attachment
            cache (e.g. "src0:mono").
        spec: Dict with at least ``"name"`` (shared-memory block name) and
            ``"shape"`` (tuple of ints).

    Returns:
        A float32 NumPy array backed by the shared-memory segment, or None on
        any failure.
    """
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
    """Extract per-source audio payloads from a job dict, resolving SHM references.

    Shared-memory arrays are attached in-process so callers receive plain
    NumPy views with no extra copies.

    Args:
        job: The raw job dict received by evaluate_parallel_job().

    Returns:
        Mapping from source node ID to resolved payload dict.

    Raises:
        RuntimeError: If shared-memory segment resolution fails.
    """
    source_chunks_raw = dict(job.get("source_chunks", {}) or {})
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
    """
    Process-pool entrypoint.
    Accepts only plain dict/list/number/NumPy payloads.
    """
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
            }
        graph = _compile_worker_graph(graph_ir=graph_ir, sample_rate=sample_rate, block_size=block_size)
        _WORKER_GRAPH_CACHE[cache_key] = graph
        _trim_worker_cache()

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
        except Exception:
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
    }


def task_serialize_audio_cache(
    audio_array: np.ndarray,
    cache_dir: str,
    sample_rate: int,
    channels: int,
) -> Dict[str, Any]:
    """Serialize captured live-audio chunks to a WAV file on disk.

    Designed to run in a daemon thread or subprocess worker — intentionally
    bpy-free.  All arguments are plain Python / NumPy types.

    Prefers ``scipy.io.wavfile`` (float32 WAV) when available; falls back to
    stdlib ``wave`` with int16 conversion so the file is always written even
    in minimal Python environments.

    Args:
        audio_array: (N, channels) float32 array of raw captured audio.
        cache_dir:   Directory where the output file will be created.
        sample_rate: Sample rate of the audio in Hz.
        channels:    Number of channels in *audio_array*.

    Returns:
        ``{"frames_written": int, "output_path": str}`` on success.

    Raises:
        OSError: if the directory cannot be created or the file cannot be written.
        ValueError: if *audio_array* has an unexpected number of dimensions.
    """
    import os as _os
    import time as _time

    arr = np.asarray(audio_array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, max(1, int(channels)))
    elif arr.ndim != 2:
        raise ValueError(
            f"task_serialize_audio_cache: expected 1-D or 2-D array, got shape {arr.shape}"
        )

    frames_written = int(arr.shape[0])
    _os.makedirs(cache_dir, exist_ok=True)

    # Millisecond timestamp to avoid collisions on repeated bakes within the same second.
    ts_ms = int(_time.time() * 1000)
    out_path = _os.path.join(cache_dir, f"ace_audio_cache_{ts_ms}.wav")

    try:
        from scipy.io import wavfile as _wavfile  # type: ignore
        # scipy.io.wavfile.write accepts float32 directly and produces a valid WAV.
        _wavfile.write(out_path, int(sample_rate), arr)
    except ImportError:
        # Fallback: stdlib wave — must convert to int16 (no float WAV support).
        import wave as _wave
        arr_i16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        n_ch = int(arr_i16.shape[1]) if arr_i16.ndim == 2 else 1
        with _wave.open(out_path, "w") as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(sample_rate))
            wf.writeframes(arr_i16.tobytes())

    return {"frames_written": frames_written, "output_path": out_path}
