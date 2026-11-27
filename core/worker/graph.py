"""Worker-side task graph compiled from the graph IR."""

from __future__ import annotations
import sys
from typing import Any, Dict, Optional
import numpy as np
from ... import utils
from ...vars import ACE_NODES
from ..tasks.base import ACEAudioTask
from ..tasks.chunk import AudioChunk, SAMPLE_RATE
from ..tasks.delta import FrequencySplitterTask
from ..tasks.stream import create_stream_task
from ..tasks.utility import StreamAccumulatorTask


def _coerce_worker_audio_chunk(data: Any, block_size: int, sample_rate: int = SAMPLE_RATE) -> AudioChunk:
    """Coerce arbitrary payload data into a typed AudioChunk."""
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


class _WorkerSourceProvider:
    """The source's audio parameters, as the chunks of a worker graph carry them."""

    def __init__(self, sample_rate: int, block_size: int):
        self.sample_rate = max(int(sample_rate), 1)
        self.block_size = max(int(block_size), 1)

    def ensure_context(self, sample_rate: int, block_size: int):
        self.sample_rate = max(int(sample_rate), 1)
        self.block_size = max(int(block_size), 1)

    def close(self):
        pass


class _WorkerSourceBridgeTask:
    """Source placeholder used in worker graphs."""

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
        history_len = min(max(int(payload.get("history_len", 0) or 0), 0), mono.shape[0])
        # Copied: the arrays may view shared memory, which the next frame overwrites.
        history_mono = np.array(mono[:history_len], dtype=np.float32) if history_len else None
        history_stereo = None
        mono = mono[history_len:]
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
            if history_len and arr.ndim == 2 and arr.shape[1] > history_len:
                history_stereo = np.array(arr[:, :history_len], dtype=np.float32)
                arr = arr[:, history_len:]
            if arr.ndim == 2 and arr.shape[1] == mono.shape[0]:
                stereo = arr.astype(np.float32, copy=False)
            elif arr.ndim == 2 and arr.shape[0] == mono.shape[0]:
                stereo = arr.T.astype(np.float32, copy=False)
            else:  # ndim != 2 OR both shape axes mismatch mono length
                # This module is bpy-free, so utils.log is unavailable.
                print(
                    f"[ACE WARNING] Worker stereo shape mismatch: stereo={arr.shape}, "
                    f"mono_len={mono.shape[0]}, frame={frame_id}. Dropping stereo.",
                    file=sys.stderr,
                )

        band_indices = payload.get("band_indices")
        if band_indices is not None:
            try:
                band_indices = np.asarray(band_indices, dtype=np.int64).reshape(-1)
            except Exception as exc:
                utils.note_swallowed("graph._WorkerSourceBridgeTask.set_chunk", exc)
                band_indices = None

        band_hz_bounds = payload.get("band_hz_bounds")
        if isinstance(band_hz_bounds, (list, tuple)) and len(band_hz_bounds) >= 2:
            try:
                band_hz_bounds = (float(band_hz_bounds[0]), float(band_hz_bounds[1]))
            except Exception as exc:
                utils.note_swallowed("graph._WorkerSourceBridgeTask.set_chunk#2", exc)
                band_hz_bounds = None
        else:
            band_hz_bounds = None

        chunk = AudioChunk(
            mono=mono,
            stereo=stereo,
            sample_rate=sr,
            provider=self.provider,
            frame_index=int(frame_id),
            band_indices=band_indices,
            band_index_fft_size=payload.get("band_index_fft_size"),
            band_hz_bounds=band_hz_bounds,
            lineage=self.default_lineage,
            spectrum_token=self.default_spectrum_token,
            latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
            history_mono=history_mono,
            history_stereo=history_stereo,
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
    """A fully instantiated, wired task graph ready for per-frame evaluation."""

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
        """Inject per-source audio chunks for the current frame."""
        bs = max(int(block_size), 1)
        if bs != self.block_size:
            self.block_size = bs
            for task in self.tasks.values():
                if hasattr(task, "block_size"):
                    task.block_size = bs
                if hasattr(task, "silence_buffer"):
                    task.silence_buffer = np.zeros(bs, dtype=np.float32)
        source_payloads = {str(k): dict(v or {}) for k, v in dict(payloads or {}).items()}
        legacy_default = source_payloads.get("__default__")
        for src_task in self.source_tasks.values():
            src_payload = source_payloads.get(str(src_task.node_id))
            if not isinstance(src_payload, dict):
                src_payload = dict(legacy_default or {})
            src_task.set_chunk(src_payload, frame_id=frame_id, block_size=bs)

    def close(self):
        """Release all task and provider resources held by this graph."""
        for task in self.tasks.values():
            close_fn = getattr(task, "close", None)
            if callable(close_fn):
                with utils.logged_errors("graph._CompiledWorkerGraph.close"):
                    close_fn()
        closed = set()
        for provider in self.source_providers.values():
            key = id(provider)
            if key in closed:
                continue
            closed.add(key)
            provider.close()


def _create_worker_task(node_id: str, category: str, node_type: str, params: Dict[str, Any], block_size: int):
    """Instantiate the correct ACEAudioTask subclass for a given node IR entry."""
    cat = str(category or "")
    typ = str(node_type or "")
    bs = max(int(block_size), 1)

    if cat == "utilities":
        if typ == "stream_accumulator":
            return StreamAccumulatorTask(node_id, params, bs)

    if cat == "delta":
        if typ == "frequency_splitter":
            return FrequencySplitterTask(node_id, params, bs)

    if cat == "stream":
        task = create_stream_task(node_id, typ, params, bs)
        if task is not None:
            return task

    return ACEAudioTask(node_id, params, bs)


# Node types whose output count depends on the instance (splitter band count).
_VARIABLE_OUTPUT_TYPES = {("delta", "frequency_splitter")}


def _declared_output_count(category: str, node_type: str) -> Optional[int]:
    if not category or (category, node_type) in _VARIABLE_OUTPUT_TYPES:
        return None
    node_def = ACE_NODES.get(category, {}).get("nodes", {}).get(node_type)
    if node_def is None:
        return None
    return len(node_def.get("outputs", []))


def _compile_worker_graph(graph_ir: Dict[str, Any], sample_rate: int, block_size: int) -> _CompiledWorkerGraph:
    """Build a _CompiledWorkerGraph from a serialised graph IR dict."""
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
            raise ValueError(f"Malformed graph IR link: {raw!r}")
        src_id = str(raw[0] or "")
        src_sock = int(raw[1] or 0)
        dst_id = str(raw[2] or "")
        dst_sock = int(raw[3] or 0)
        src_task = tasks.get(src_id)
        dst_task = tasks.get(dst_id)
        if src_task is None or dst_task is None:
            missing = src_id if src_task is None else dst_id
            raise ValueError(f"Graph IR link {src_id}[{src_sock}] -> {dst_id}[{dst_sock}] references unknown node '{missing}'")
        src_rec = dict(nodes.get(src_id, {}) or {})
        declared = _declared_output_count(str(src_rec.get("category", "") or ""), str(src_rec.get("type", "") or ""))
        if declared is not None and not 0 <= src_sock < declared:
            raise ValueError(
                f"Graph IR link {src_id}[{src_sock}] -> {dst_id}[{dst_sock}]: "
                f"'{src_rec.get('type')}' has {declared} output(s)"
            )
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
    """Convert a raw task output to the scalar or tuple form expected by the main process."""
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
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        arr = np.asarray([float(value)], dtype=np.float32)
    else:
        arr = np.zeros(0, dtype=np.float32)

    padded = np.zeros(target_len, dtype=np.float32)
    if arr.size > 0:
        padded[: min(target_len, arr.size)] = arr[:target_len]
    if attr == "FLOAT_COLOR" and arr.size < 4:
        padded[3] = 1.0
    return tuple(float(x) for x in padded.tolist())
