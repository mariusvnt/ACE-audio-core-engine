"""Block size per graph, from sample rate and fps, kept stable across small changes."""

import numpy as np
from .... import utils
from ...tasks.base import ACEAudioTask
from ...tasks.block_size import resolve_context_block_size
from ...tasks.source import SourceTask


class BlockSizing:
    def adjust_graph_block_size_for_task(self, task: ACEAudioTask, new_block_size: int):
        """Update the block size for all tasks in the graph that owns *task*.

        SourceTask instances are updated via ``_set_block_size_from_engine``
        (which handles prefetch buffer resize).  All other tasks have their
        ``block_size``, ``silence_buffer``, and cached array attributes reset
        directly so stale workspace allocations are not reused.
        """
        target_graph = None
        for graph in self.engine.graphs.values():
            if task in graph.tasks.values():
                target_graph = graph
                break
        if not target_graph or new_block_size <= 0:
            return
        current_size = getattr(target_graph, "block_size", None)
        if current_size == new_block_size:
            return
        target_graph.block_size = new_block_size
        for t in target_graph.tasks.values():
            if isinstance(t, SourceTask):
                try:
                    t._set_block_size_from_engine(new_block_size)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine.adjust_graph_block_size_for_task", exc)
                    t.block_size = new_block_size
                    t.silence_buffer = np.zeros(new_block_size, dtype=np.float32)
                    t._ensure_frontend_workspace(new_block_size)
                continue

            t.block_size = new_block_size
            t.silence_buffer = np.zeros(new_block_size, dtype=np.float32)
            if hasattr(t, "_mask_cache_key"):
                t._mask_cache_key = None
            if hasattr(t, "_mask_cache"):
                t._mask_cache = None
            if hasattr(t, "_cached_masks"):
                t._cached_masks = None
            if hasattr(t, "_cached_freq_bins"):
                t._cached_freq_bins = None
            if hasattr(t, "_work_fft"):
                t._work_fft = None
            if hasattr(t, "_work_fft_stereo"):
                t._work_fft_stereo = None

    @staticmethod
    def _resolve_block_size(
        sample_rate: int,
        fps: float,
        source_type: str = "FILE",
        timeslice_mode: str = "fixed",
        delta_frames: float = 1.0,
    ) -> int:
        try:
            return int(
                resolve_context_block_size(
                    sample_rate,
                    fps,
                    source_type=source_type,
                    timeslice_mode=timeslice_mode,
                    delta_frames=delta_frames,
                )
            )
        except TypeError:
            # Backward compatibility for older helper signatures.
            return int(resolve_context_block_size(sample_rate, fps, source_type=source_type))

    def _stabilize_graph_block_size(
        self,
        graph_key: str,
        current_size: int,
        proposed_size: int,
        mode: str = "fixed",
    ) -> int:
        """Apply hysteresis to block-size changes in adaptive timeslice mode.

        In ``fixed`` mode the proposed size is returned immediately.  In
        ``adaptive`` mode small changes must persist for
        ``_BLOCK_SIZE_HYSTERESIS_FRAMES`` consecutive frames before being
        accepted; large changes (≥ ``_BLOCK_SIZE_IMMEDIATE_DELTA`` samples)
        are always applied immediately to handle timeline scrubs.

        Args:
            graph_key: Stable key used to track per-graph hysteresis state.
            current_size: Currently active block size for this graph.
            proposed_size: Block size calculated from the current frame delta.
            mode: ``"fixed"`` or ``"adaptive"``.

        Returns:
            The block size to use for this frame.
        """
        curr = int(max(current_size or 0, 0))
        prop = int(max(proposed_size or 0, 0))
        timeslice_mode = str(mode or "fixed").strip().lower()

        if prop <= 0:
            return curr if curr > 0 else prop
        if timeslice_mode != "adaptive":
            self._block_size_state.pop(graph_key, None)
            return prop
        if curr <= 0 or prop == curr:
            self._block_size_state.pop(graph_key, None)
            return prop if prop > 0 else curr
        if abs(prop - curr) >= self._BLOCK_SIZE_IMMEDIATE_DELTA:
            # Large jumps usually come from timeline scrubs: adapt immediately.
            self._block_size_state.pop(graph_key, None)
            return prop

        state = self._block_size_state.get(graph_key)
        if state is None:
            self._block_size_state[graph_key] = {"candidate": prop, "count": 1}
            return curr
        if int(state.get("candidate", -1)) != prop:
            self._block_size_state[graph_key] = {"candidate": prop, "count": 1}
            return curr

        count = int(state.get("count", 1)) + 1
        if count < self._BLOCK_SIZE_HYSTERESIS_FRAMES:
            state["count"] = count
            return curr

        self._block_size_state.pop(graph_key, None)
        return prop
