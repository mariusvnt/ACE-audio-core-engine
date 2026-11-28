"""Preview routing: picks preview sinks and pushes chunks to the monitor."""

import numpy as np
import threading
from typing import Any, Optional
from .... import utils
from ...abstract_tree import ACEAbstractNode, ACEAbstractTree
from ...tasks.block_size import resolve_preview_block_size
from ...tasks.chunk import AudioChunk, SAMPLE_RATE
from ..common import engine_settings
from ....vars import PREVIEW_KIND
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
            utils.note_swallowed("router.PreviewRouter.preview_monitor_available", exc)
            return False

    @classmethod
    def preview_monitor_consecutive_errors(cls) -> int:
        """Return the current consecutive DSP error count from the prepare worker."""
        try:
            return int(cls._preview_monitor._preview_consecutive_errors)
        except Exception as exc:
            utils.note_swallowed("router.PreviewRouter.preview_monitor_consecutive_errors", exc)
            return 0

    @classmethod
    def _preview_block_size(cls, scene_or_engine_settings=None) -> int:
        settings = engine_settings(scene_or_engine_settings)
        raw = 256
        try:
            raw = int(getattr(settings, "ace_preview_block_size", 256) or 256)
        except Exception as exc:
            utils.note_swallowed("router.PreviewRouter._preview_block_size", exc)
            raw = 256
        return int(resolve_preview_block_size(raw))

    @staticmethod
    def _mix_preview_chunks(chunks: list[AudioChunk]) -> Optional[AudioChunk]:
        """Average-mix a list of stereo AudioChunks into one."""
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
            latency_ms=latency_ms,
        )

    @staticmethod
    def _chunk_is_unmasked_audio(chunk: Any) -> bool:
        """Return True if *chunk* is a plain, unprocessed audio block."""
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
                utils.note_swallowed("router.PreviewRouter._chunk_is_unmasked_audio", exc)
                return False
        return True

    @staticmethod
    def _task_current_chunk_for_frame(task: Any, frame_index: int):
        try:
            chunk_frame = utils.as_int(getattr(task, "_current_chunk_frame", None), -1)
        except Exception as exc:
            utils.note_swallowed("router.PreviewRouter._task_current_chunk_for_frame", exc)
            chunk_frame = -1
        if chunk_frame != int(frame_index):
            return None
        chunk = getattr(task, "_current_chunk", None)
        if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate"):
            return chunk
        return None

    def _start_preview_worker(self):
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
            with utils.logged_errors("router.PreviewRouter._preview_worker_loop"):
                result_chunk = task.process({0: upstream_chunk})
                if result_chunk is not None and hasattr(result_chunk, "mono"):
                    monitor.push_chunk(result_chunk)

    def _submit_preview_work(self, task, upstream_chunk) -> bool:
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
            if (str(getattr(node, "category", "") or ""), str(getattr(node, "type", "") or "")) != PREVIEW_KIND:
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
