"""Source specs task: frames per beat and beat phase for a source."""

from .base import ACEAudioTask
from .chunk import _ensure_audio_chunk


class SourceSpecsTask(ACEAudioTask):
    """Frames per beat and beat phase from the source's BPM and Beats per Bar.

    The tempo itself is detected per file source in the background
    (core/tasks/tempo.py) and written to the source on the main thread.
    """

    def process(self, inputs):
        chunk_in = _ensure_audio_chunk(inputs.get(0), self.block_size)
        ctx = self.source_context
        bpm = float(getattr(ctx, "bpm", 120.0) or 120.0) if ctx else 120.0
        beats_per_bar = int(getattr(ctx, "beats_per_bar", 4) or 4) if ctx else 4
        fps = float(getattr(self, "scene_fps", 24.0) or 0.0)
        fps = fps if fps > 0.0 else 24.0

        beats_per_frame = float(bpm / max(60.0 * fps, 1e-6))
        frames_per_beat = float(1.0 / beats_per_frame) if beats_per_frame > 1e-6 else 0.0
        engine = getattr(self, "engine_ref", None)
        current_frame = getattr(engine, "current_frame", 0) if engine else 0
        pos_in_measure = (float(current_frame) * beats_per_frame) % float(beats_per_bar)
        beat_phase = pos_in_measure / float(beats_per_bar)
        # Output order matches ACE_NODES: Frames per Beat, Beat Phase, Audio passthrough
        return (frames_per_beat, beat_phase, chunk_in)
