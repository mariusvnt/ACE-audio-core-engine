"""Source specs task: frames per beat and beat phase for a source."""

import threading
from queue import Queue
from typing import Dict, Any, Optional, Tuple
from ... import utils
from .base import ACEAudioTask
from .chunk import SAMPLE_RATE, _ensure_audio_chunk
from .specs_analysis import LongTermAnalysis
from .tempo import TempoEstimation


class SourceSpecsTask(TempoEstimation, LongTermAnalysis, ACEAudioTask):
    """Outputs static metadata about the source with lightweight long-term analysis."""
    _GLOBAL_LONG_TERM_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}
    _GLOBAL_ACTIVE_ANALYSIS: Dict[Tuple[str, float], int] = {}
    _GLOBAL_CACHE_TTL = 120.0  # seconds
    _GLOBAL_CACHE_MAX = 128
    _LONG_TERM_MAX_SECONDS = 15.0
    _LONG_TERM_TICK_INTERVAL_S = 0.05
    _GLOBAL_CACHE_LOCK = threading.Lock()
    _BG_QUEUE_MAX = 16
    _BG_TASK_QUEUE: Queue = Queue(maxsize=_BG_QUEUE_MAX)
    _BG_PENDING: set[Tuple[str, float]] = set()
    _BG_QUEUED: set[Tuple[str, float]] = set()
    _BG_THREAD_LOCK = threading.Lock()
    _BG_STOP_EVENT = threading.Event()
    _BG_THREAD: Optional[threading.Thread] = None
    # Pending ctx writes: (target_key, bpm, beats_per_bar). Drained on main thread.
    _PENDING_CTX_WRITES: "Queue[Tuple[Tuple[str, float], float, int]]" = Queue()

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._static_specs: Optional[Tuple[float, int, float, float, float, float]] = None
        self._long_term_cache: Dict[str, Any] = {}
        self._long_term_target: Optional[Tuple[str, float]] = None
        self._long_term_done: bool = False
        self._auto_bpm_key = "ace_bpm_auto"
        self._auto_meter_key = "ace_meter_auto"
        self._analysis_next_tick_ts: float = 0.0

    def process(self, inputs):
        # Flush any detected-tempo results from the background worker.
        # Guarded to main-thread execution only so worker/bake threads never mutate bpy state.
        if threading.current_thread() is threading.main_thread():
            self.flush_pending_ctx_writes(
                getattr(getattr(self, "engine_ref", None), "_source_specs_task_registry", {})
            )
        # Preserve the audio path
        chunk_in = _ensure_audio_chunk(inputs.get(0), self.block_size)
        ctx = self.source_context
        if ctx:
            # If the user changed BPM or meter after an automatic estimate, honor
            # that override and stop auto-updates for this source.
            with utils.logged_errors("specs.SourceSpecsTask.process"):
                auto_val = ctx.get(self._auto_bpm_key, None) if hasattr(ctx, "get") else None
                if auto_val is not None:
                    current = float(getattr(ctx, "bpm", 120.0))
                    if abs(current - float(auto_val)) > 1e-3:
                        ctx["ace_bpm_user_override"] = True
                        self._long_term_done = True
                        self._reset_long_term_analysis_state(clear_target=False)
            with utils.logged_errors("specs.SourceSpecsTask.process#2"):
                auto_meter = ctx.get(self._auto_meter_key, None) if hasattr(ctx, "get") else None
                if auto_meter is not None:
                    current_meter = int(getattr(ctx, "beats_per_bar", 4) or 4)
                    if current_meter != int(auto_meter):
                        ctx["ace_meter_user_override"] = True
                        self._long_term_done = True
                        self._reset_long_term_analysis_state(clear_target=False)
        fps = getattr(self, "scene_fps", 24.0)
        self._maybe_run_long_term_analysis(ctx)

        bpm = float(self._long_term_cache.get("bpm", getattr(ctx, "bpm", 120.0) if ctx else 120.0) or 120.0)
        beats_per_bar = int(getattr(ctx, "beats_per_bar", 4) or 4) if ctx else 4
        sample_rate = float(getattr(ctx, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE) if ctx else float(SAMPLE_RATE)
        duration = float(getattr(ctx, "duration", 0.0) or 0.0) if ctx else 0.0

        fps_val = float(fps) if float(fps) > 0.0 else 24.0
        if (
            self._static_specs is None
            or self._static_specs[2] != sample_rate
            or self._static_specs[1] != beats_per_bar
            or self._static_specs[4] != bpm
            or self._static_specs[5] != fps_val
        ):
            beats_per_frame = float(bpm / max(60.0 * fps_val, 1e-6))
            self._static_specs = (beats_per_frame, beats_per_bar, sample_rate, duration, bpm, fps_val)
        else:
            beats_per_frame, beats_per_bar, sample_rate, duration, _, _ = self._static_specs

        frames_per_beat = float(1.0 / beats_per_frame) if beats_per_frame > 1e-6 else 0.0
        engine = getattr(self, "engine_ref", None)
        current_frame = getattr(engine, "current_frame", 0) if engine else 0
        measure_len_frames = float(beats_per_bar) / beats_per_frame if beats_per_frame > 0 else 0.0
        if measure_len_frames > 0.0:
            pos_in_measure = (float(current_frame) * beats_per_frame) % float(beats_per_bar)
        else:
            pos_in_measure = 0.0

        beat_phase = pos_in_measure / float(beats_per_bar) if beats_per_bar > 0 else 0.0
        # Output order matches ACE_NODES: Frames per Beat, Beat Phase, Audio passthrough
        return (frames_per_beat, beat_phase, chunk_in)

    def close(self):
        self._reset_long_term_analysis_state(clear_target=True)

    @classmethod
    def _user_overrode_tempo(cls, ctx: Optional[Any]) -> bool:
        if not ctx:
            return False
        getter = getattr(ctx, "get", None)
        if callable(getter):
            try:
                if getter("ace_bpm_user_override", False) or getter("ace_meter_user_override", False):
                    return True
            except Exception as exc:
                utils.note_swallowed("specs.SourceSpecsTask._user_overrode_tempo", exc)
                return False
        return False
