"""Preview routing: picks preview sinks and pushes chunks to the monitor."""

import numpy as np
import threading
import time
from typing import Any, Optional
from .... import utils
from ...abstract_tree import ACEAbstractNode, ACEAbstractTree
from ...tasks.block_size import resolve_preview_block_size
from ...tasks.chunk import AudioChunk, SAMPLE_RATE
from ...tasks.graph import ACEAudioGraph
from ...tasks.preview import PreviewSinkTask
from ...tasks.source import SourceTask
from ..common import engine_settings
from .monitor import _ACEPreviewAudioMonitor


class PreviewRouter:
    """Audio preview routing: chunk collection, mixing and the monitor feeder."""

    _preview_monitor = _ACEPreviewAudioMonitor()
    _PREVIEW_MIN_INTERVAL_S = 0.003

    def __init__(self, engine):
        self.engine = engine
        self._preview_worker_stop: bool = False
        self._preview_worker_cond: threading.Condition = threading.Condition()
        self._preview_worker_queue: list = []
        self._preview_worker_thread: Optional[threading.Thread] = None

    @classmethod
    def preview_monitor_available(cls) -> bool:
        """Return True if the sounddevice backend loaded successfully."""
        try:
            return bool(cls._preview_monitor._resolve_backend())
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.preview_monitor_available", exc)
            return False

    @classmethod
    def preview_monitor_consecutive_errors(cls) -> int:
        """Return the current consecutive DSP error count from the prepare worker."""
        try:
            return int(cls._preview_monitor._preview_consecutive_errors)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.preview_monitor_consecutive_errors", exc)
            return 0

    @classmethod
    def _preview_block_size(cls, scene_or_engine_settings=None) -> int:
        settings = engine_settings(scene_or_engine_settings)
        raw = 256
        try:
            raw = int(getattr(settings, "ace_preview_block_size", 256) or 256)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._preview_block_size", exc)
            raw = 256
        return int(resolve_preview_block_size(raw))

    @staticmethod
    def _resample_signal_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
        """Linear-interpolation resample of a 1-D float32 signal to *target_len* samples."""
        src = np.asarray(signal, dtype=np.float32).reshape(-1)
        n = int(src.shape[0] or 0)
        t_len = int(max(target_len, 1))
        if n == t_len:
            return src.astype(np.float32, copy=False)
        if n <= 0:
            return np.zeros(t_len, dtype=np.float32)
        if n == 1:
            return np.full(t_len, float(src[0]), dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, t_len, endpoint=False, dtype=np.float64)
        out = np.interp(x_new, x_old, src.astype(np.float64, copy=False))
        return out.astype(np.float32, copy=False)

    @classmethod
    def _fit_preview_chunk_block(cls, chunk: AudioChunk, preview_block_size: int) -> AudioChunk:
        """Resample a chunk's mono and stereo signals to *preview_block_size* samples.

        Returns the original chunk unchanged if it already has the correct size.
        """
        target = int(max(preview_block_size, 1))
        mono = np.asarray(getattr(chunk, "mono", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if mono.shape[0] == target:
            return chunk
        mono_resized = cls._resample_signal_1d(mono, target)

        stereo_raw = getattr(chunk, "stereo", None)
        stereo_resized = None
        if isinstance(stereo_raw, np.ndarray):
            stereo_arr = np.asarray(stereo_raw, dtype=np.float32)
            if stereo_arr.ndim == 2:
                if stereo_arr.shape[1] == mono.shape[0]:
                    stereo_resized = np.stack(
                        [cls._resample_signal_1d(stereo_arr[ch], target) for ch in range(int(stereo_arr.shape[0]))],
                        axis=0,
                    ).astype(np.float32, copy=False)
                elif stereo_arr.shape[0] == mono.shape[0]:
                    transposed = stereo_arr.T.astype(np.float32, copy=False)
                    resized_t = np.stack(
                        [cls._resample_signal_1d(transposed[ch], target) for ch in range(int(transposed.shape[0]))],
                        axis=0,
                    ).astype(np.float32, copy=False)
                    stereo_resized = resized_t

        return AudioChunk(
            mono_resized,
            stereo_resized,
            int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE),
            provider=getattr(chunk, "provider", None),
            frame_index=utils.as_int(getattr(chunk, "frame_index", None), -1),
            band_indices=getattr(chunk, "band_indices", None),
            band_index_fft_size=getattr(chunk, "band_index_fft_size", None),
            band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
            lineage=tuple(getattr(chunk, "lineage", ()) or ()),
            spectrum_token=getattr(chunk, "spectrum_token", None),
            analysis_ops=tuple(getattr(chunk, "analysis_ops", ()) or ()),
            latency_ms=float(getattr(chunk, "latency_ms", 0.0) or 0.0),
        )

    @staticmethod
    def _mix_preview_chunks(chunks: list[AudioChunk]) -> Optional[AudioChunk]:
        """Average-mix a list of stereo AudioChunks into one.

        Returns None for an empty list, the single chunk unchanged for a list of
        one, or a new AudioChunk with the equal-power average mix for multiple.
        A final peak limiter is applied to the mixed result to prevent clipping.
        """
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]

        frames_list: list[np.ndarray] = []
        sample_rate = int(getattr(chunks[0], "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        max_frames = 0
        for chunk in chunks:
            frames = _ACEPreviewAudioMonitor._chunk_to_stereo_frames(chunk)
            if frames is None:
                continue
            frames_list.append(np.asarray(frames, dtype=np.float32))
            max_frames = max(max_frames, int(frames.shape[0]))

        if not frames_list or max_frames <= 0:
            return None

        mixed = np.zeros((max_frames, 2), dtype=np.float32)
        mix_count = 0
        for frames in frames_list:
            frame_n = int(frames.shape[0] or 0)
            if frame_n <= 0:
                continue
            mixed[:frame_n, :] += frames[:frame_n, :]
            mix_count += 1

        if mix_count <= 0:
            return None
        if mix_count > 1:
            mixed *= (1.0 / float(mix_count))
        # Instant-attack vectorized limiter: match PreviewSinkTask behavior.
        ceiling = 0.98
        n_frames = int(mixed.shape[0])
        peaks = np.max(np.abs(mixed.reshape(n_frames, -1)), axis=1).astype(np.float64)
        desired = np.where(
            peaks > 1e-9,
            np.minimum(1.0, ceiling / np.where(peaks > 1e-9, peaks, 1.0)),
            1.0,
        )
        gains = np.minimum.accumulate(desired).astype(np.float32)
        mixed *= gains.reshape(-1, 1)
        np.clip(mixed, -ceiling, ceiling, out=mixed)

        stereo = mixed.T.astype(np.float32, copy=False)
        mono = (0.5 * (mixed[:, 0] + mixed[:, 1])).astype(np.float32, copy=False)
        latency_ms = max((float(getattr(chunk, "latency_ms", 0.0) or 0.0) for chunk in chunks), default=0.0)
        return AudioChunk(
            mono,
            stereo,
            sample_rate,
            lineage=("preview_mix",),
            analysis_ops=(),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _chunk_is_unmasked_audio(chunk: Any) -> bool:
        """Return True if *chunk* is a plain, unprocessed audio block.

        Returns False for frequency-masked, band-limited, or analysis-ops-tagged
        chunks because these should not be played back directly through the
        preview monitor.
        """
        if not hasattr(chunk, "mono") or not hasattr(chunk, "sample_rate"):
            return False
        if getattr(chunk, "band_hz_bounds", None) is not None:
            return False
        idx = getattr(chunk, "band_indices", None)
        if idx is not None:
            try:
                if np.asarray(idx).size > 0:
                    return False
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._chunk_is_unmasked_audio", exc)
                return False
        try:
            if bool(getattr(chunk, "analysis_ops", ())):
                return False
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._chunk_is_unmasked_audio#2", exc)
            return False
        return True

    @staticmethod
    def _task_current_chunk_for_frame(task: Any, frame_index: int):
        """Return the task's cached current chunk if it was produced for *frame_index*.

        Returns None if the task has no cached chunk or the chunk belongs to a
        different frame.
        """
        try:
            chunk_frame = utils.as_int(getattr(task, "_current_chunk_frame", None), -1)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._task_current_chunk_for_frame", exc)
            chunk_frame = -1
        if chunk_frame != int(frame_index):
            return None
        chunk = getattr(task, "_current_chunk", None)
        if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate"):
            return chunk
        return None

    def _collect_preview_chunk(self, graph: ACEAudioGraph, frame_index: int, preview_block_size: int):
        """Collect and resample preview audio for *graph* at *frame_index*.

        Prefers direct SourceTask passthrough (zero DSP overhead) when no manual
        Preview Sink nodes are present.  Falls back to evaluating PreviewSinkTask
        outputs when biquad processing is needed.  Returns the mixed/resampled
        chunk, or None if no preview audio is available.
        """
        manual_ids = getattr(graph, "manual_preview_task_ids", []) or []
        auto_ids = getattr(graph, "auto_preview_task_ids", []) or []
        passthrough_specs = getattr(graph, "auto_preview_passthrough_specs", []) or []
        candidate_ids = manual_ids if manual_ids else (auto_ids or (getattr(graph, "preview_task_ids", []) or []))

        chunks: list[AudioChunk] = []
        if (not manual_ids) and passthrough_specs:
            for source_task_id, source_sock_idx in passthrough_specs:
                source_task = graph.tasks.get(str(source_task_id))
                if not isinstance(source_task, SourceTask):
                    continue
                try:
                    t0 = time.perf_counter()
                    direct_chunk = source_task.get_output(frame_index, int(source_sock_idx))
                    self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine._collect_preview_chunk", exc)
                    continue
                if direct_chunk is not None and self._chunk_is_unmasked_audio(direct_chunk):
                    chunks.append(self._fit_preview_chunk_block(direct_chunk, preview_block_size))

        for node_id in candidate_ids:
            task = graph.tasks.get(node_id)
            if not task:
                continue
            if 0 not in getattr(task, "inputs", {}):
                continue
            is_enabled_fn = getattr(task, "is_enabled", None)
            if callable(is_enabled_fn) and not is_enabled_fn():
                continue
            # Fast path: when auto-preview taps a direct source output branch with no
            # masking/analysis metadata, bypass PreviewSink reconstruction entirely.
            if isinstance(task, PreviewSinkTask):
                conn = getattr(task, "inputs", {}).get(0)
                upstream = getattr(conn, "task", None) if conn is not None else None
                if isinstance(upstream, SourceTask):
                    try:
                        t0 = time.perf_counter()
                        direct_chunk = upstream.get_output(frame_index, int(getattr(conn, "socket_index", 0) or 0))
                        self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
                    except Exception as exc:
                        utils.note_swallowed("audio_engine.ACEAudioEngine._collect_preview_chunk#2", exc)
                        direct_chunk = None
                    if direct_chunk is not None and self._chunk_is_unmasked_audio(direct_chunk):
                        chunks.append(self._fit_preview_chunk_block(direct_chunk, preview_block_size))
                        continue
            try:
                t0 = time.perf_counter()
                chunk = task.get_output(frame_index, 0)
                self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._collect_preview_chunk#3", exc)
                continue
            if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate"):
                chunks.append(self._fit_preview_chunk_block(chunk, preview_block_size))
        return PreviewRouter._mix_preview_chunks(chunks)

    def _start_preview_worker(self):
        """Start the preview DSP thread. Idempotent: no-op if already running."""
        if self._preview_worker_thread is not None and self._preview_worker_thread.is_alive():
            return
        with self._preview_worker_cond:
            self._preview_worker_stop = False
        t = threading.Thread(
            target=self._preview_worker_loop, daemon=True, name="ACEPreviewWorker"
        )
        self._preview_worker_thread = t
        t.start()

    def _stop_preview_worker(self):
        """Signal the preview worker to stop and wait for it to exit."""
        with self._preview_worker_cond:
            self._preview_worker_stop = True
            self._preview_worker_queue.clear()
            self._preview_worker_cond.notify_all()
        t = self._preview_worker_thread
        self._preview_worker_thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    def _preview_worker_loop(self):
        """Background thread: run PreviewSinkTask.process() and push to monitor."""
        monitor = self.__class__._preview_monitor
        while True:
            with self._preview_worker_cond:
                while not self._preview_worker_queue and not self._preview_worker_stop:
                    self._preview_worker_cond.wait(timeout=0.1)
                if self._preview_worker_stop:
                    break
                if not self._preview_worker_queue:
                    continue
                # Take the LATEST item only; drop any older submissions.
                item = self._preview_worker_queue[-1]
                self._preview_worker_queue.clear()
            task, upstream_chunk = item
            with utils.logged_errors("audio_engine.ACEAudioEngine._preview_worker_loop"):
                result_chunk = task.process({0: upstream_chunk})
                if result_chunk is not None and hasattr(result_chunk, "mono"):
                    monitor.push_chunk(result_chunk)

    def _submit_preview_work(self, task, upstream_chunk) -> bool:
        """Enqueue a PreviewSinkTask for background DSP processing.

        Returns True if the work was accepted, False if the worker is stopped.
        Only the most-recent submission per frame is retained; older items are
        dropped to prevent queue buildup during heavy load.
        """
        with self._preview_worker_cond:
            if self._preview_worker_stop:
                return False
            self._preview_worker_queue.append((task, upstream_chunk))
            if len(self._preview_worker_queue) > 1:
                self._preview_worker_queue[:] = [self._preview_worker_queue[-1]]
            self._preview_worker_cond.notify()
        return True

    @staticmethod
    def _preview_node_sort_key(
        node: Optional[ACEAbstractNode],
        abstract_tree: Optional[ACEAbstractTree],
    ) -> tuple[int, int, str, str]:
        if node is None:
            return (-1, -1, "", "")
        frontends = dict(getattr(abstract_tree, "frontends", {}) or {}) if abstract_tree is not None else {}
        frontend_uid = str(getattr(node, "frontend_uid", "") or "")
        frontend = frontends.get(frontend_uid)
        discovery_index = int(getattr(frontend, "discovery_index", 0) or 0)
        params = dict(getattr(node, "parameters", {}) or {})
        frontend_node_index = utils.as_int(params.get("frontend_node_index"), -1)
        return (
            discovery_index,
            frontend_node_index,
            frontend_uid,
            str(getattr(node, "id", "") or ""),
        )

    def _resolve_global_manual_preview_winner(self, abstract_tree: Optional[ACEAbstractTree]) -> tuple[str, list[str]]:
        candidates: list[str] = []
        if abstract_tree is None:
            return "", candidates
        for node_id, node in (getattr(abstract_tree, "nodes", {}) or {}).items():
            if str(getattr(node, "category", "") or "") != "utilities":
                continue
            if str(getattr(node, "type", "") or "") != "preview_sink":
                continue
            candidates.append(str(node_id))
        if not candidates:
            return "", []
        winner_id = max(
            candidates,
            key=lambda node_id: self._preview_node_sort_key(
                abstract_tree.nodes.get(str(node_id)),
                abstract_tree,
            ),
        )
        return str(winner_id or ""), sorted(
            candidates,
            key=lambda node_id: self._preview_node_sort_key(
                abstract_tree.nodes.get(str(node_id)),
                abstract_tree,
            ),
        )
