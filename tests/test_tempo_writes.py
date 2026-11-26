"""A detected tempo is written to the source once per change, not on every frame."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import load_module  # noqa: E402

import fake_bpy  # noqa: E402


class TempoWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.task_cls = load_module("core.tasks.specs").SourceSpecsTask

    def setUp(self):
        self.queue = self.task_cls._PENDING_CTX_WRITES
        self._drain()
        self.addCleanup(self._drain)
        self.task = self.task_cls("specs", {}, 2000)
        self.task._long_term_target = ("loop.wav", 1.0)
        self.ctx = {"bpm": 0.0}

    def _drain(self):
        while not self.queue.empty():
            self.queue.get_nowait()

    def test_cached_tempo_applied_every_frame_is_queued_once(self):
        for _ in range(5):
            self.task._apply_detected_tempo(self.ctx, 120.0, 4)
        self.assertEqual(self.queue.qsize(), 1)
        self.task._apply_detected_tempo(self.ctx, 121.0, 4)
        self.assertEqual(self.queue.qsize(), 2)

    def test_tempo_returns_after_the_override_is_removed(self):
        self.task._apply_detected_tempo(self.ctx, 120.0, 4)
        self.ctx["ace_bpm_user_override"] = True
        self.task._apply_detected_tempo(self.ctx, 120.0, 4)
        self.assertEqual(self.queue.qsize(), 1)
        del self.ctx["ace_bpm_user_override"]
        self.task._apply_detected_tempo(self.ctx, 120.0, 4)
        self.assertEqual(self.queue.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
