"""Auto preview (Group Output audio) hands the monitor the whole frame of audio."""

import types
import unittest

import fake_bpy
from module_loader import load_module

try:
    import numpy as np
except Exception:
    np = None

try:
    fake_bpy.install()
    runner_mod = load_module("core.engine.frame.runner") if np is not None else None
    router_mod = load_module("core.engine.preview.router") if np is not None else None
    source_mod = load_module("core.tasks.source") if np is not None else None
    chunk_mod = load_module("core.tasks.chunk") if np is not None else None
except Exception:
    runner_mod = None


@unittest.skipIf(runner_mod is None or np is None, "numpy/engine modules unavailable in this environment")
class TestPreviewPassthrough(unittest.TestCase):
    def test_source_audio_reaches_the_monitor_at_full_length(self):
        # One 24 fps frame at 48 kHz: 2000 samples, far above any preview block size.
        frame_len = 2000
        t = np.arange(frame_len, dtype=np.float32) / 48000.0
        mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        chunk = chunk_mod.AudioChunk(mono, np.stack([mono, mono]), 48000, lineage=("src",))

        source_task = source_mod.SourceTask.__new__(source_mod.SourceTask)
        source_task._current_chunk = chunk
        source_task._current_chunk_frame = 7

        router = router_mod.PreviewRouter
        engine = types.SimpleNamespace(
            current_frame=7,
            preview=types.SimpleNamespace(
                _start_preview_worker=lambda: None,
                _task_current_chunk_for_frame=router._task_current_chunk_for_frame,
                _chunk_is_unmasked_audio=router._chunk_is_unmasked_audio,
            ),
        )
        graph = types.SimpleNamespace(
            manual_preview_task_ids=[],
            auto_preview_task_ids=[],
            auto_preview_passthrough_specs=[("src", 0)],
            preview_task_ids=[],
            tasks={"src": source_task},
        )
        ctx = types.SimpleNamespace(preview_block_size=256, preview_chunks=[], preview_pushed=False)

        runner = runner_mod.FrameRunner.__new__(runner_mod.FrameRunner)
        runner.engine = engine
        runner._collect_graph_preview(graph, ctx)

        self.assertEqual(len(ctx.preview_chunks), 1)
        pushed = ctx.preview_chunks[0]
        # Squeezing the frame into the preview block played it ~8x too high, with gaps.
        self.assertEqual(int(pushed.mono.shape[0]), frame_len)
        self.assertTrue(np.array_equal(pushed.mono, mono))


if __name__ == "__main__":
    unittest.main()
