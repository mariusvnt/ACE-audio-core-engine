"""A file source's tempo is detected on its own, written once per file version, and never over a typed one."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import load_module  # noqa: E402

import fake_bpy  # noqa: E402


class _Source(dict):
    """A FILE source: attributes as RNA properties, items as ID properties."""

    def __init__(self, filepath):
        super().__init__()
        self.source_type = "FILE"
        self.filepath = filepath
        self.writes = 0
        self._bpm = 120.0
        self._meter = 4

    # Writing BPM or Beats per Bar runs the update callback that marks a user override.
    @property
    def bpm(self):
        return self._bpm

    @bpm.setter
    def bpm(self, value):
        self._bpm = value
        self.writes += 1
        self["ace_bpm_user_override"] = True

    @property
    def beats_per_bar(self):
        return self._meter

    @beats_per_bar.setter
    def beats_per_bar(self, value):
        self._meter = value
        self["ace_meter_user_override"] = True


class TempoWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.service = load_module("core.tasks.tempo").TempoService
        cls.probes = load_module("core.engine.probes").SourceProbes(engine=None)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "loop.wav")
        Path(self.path).write_bytes(b"")
        self.source = _Source(self.path)
        self.requests = []

        def request(cls, target):  # queued, never run
            self.requests.append(target)
            cls._waiting.add(target)
            return True

        original = self.service.request
        self.service.request = classmethod(request)
        self.addCleanup(setattr, self.service, "request", original)
        self.addCleanup(self.service._waiting.clear)
        self.service.clear_cache()
        self.addCleanup(self.service.clear_cache)

    def _finish(self, bpm=128.0, beats_per_bar=3):
        for target in self.requests:
            self.service._waiting.discard(target)
            self.service._store(target, {"bpm": bpm, "beats_per_bar": beats_per_bar, "downbeat_s": 0.0})

    def _frames(self, count=3):
        for _ in range(count):
            self.probes.flush_source_tempo_detection_updates([self.source])

    def test_a_new_file_is_detected_once_and_written_once(self):
        self._frames()
        self.assertEqual(len(self.requests), 1)
        self._finish()
        self._frames()
        self.assertEqual((self.source.bpm, self.source.beats_per_bar, self.source.writes), (128.0, 3, 1))
        self.assertFalse(self.source["ace_bpm_user_override"])
        self.assertEqual(self.source["ace_tempo_detect_status"], "ready")
        self.assertEqual(len(self.requests), 1)

    def test_a_typed_tempo_is_kept(self):
        self.source["ace_bpm_user_override"] = True
        self._frames()
        self.assertEqual(self.requests, [])
        # The Auto Detect button still replaces it.
        self.probes.request_source_tempo_detection(self.source)
        self._finish()
        self._frames()
        self.assertEqual(self.source.bpm, 128.0)
        self.assertFalse(self.source["ace_bpm_user_override"])

    def test_a_tempo_typed_during_detection_wins(self):
        self._frames()
        self.source.bpm = 99.0
        self._finish()
        self._frames()
        self.assertEqual(self.source.bpm, 99.0)
        self.assertEqual(self.source["ace_tempo_detect_status"], "")

    def test_a_changed_file_is_detected_again(self):
        self._frames()
        self._finish()
        self._frames()
        os.utime(self.path, (1_000_000_000, 1_000_000_000))
        self._frames()
        self.assertEqual(len(self.requests), 2)

    def test_a_detection_lost_to_a_reset_is_asked_again(self):
        self._frames()
        self.service._waiting.clear()  # the thread was stopped: nothing waits, no result came
        self._frames(1)
        self.assertEqual(len(self.requests), 2)

    def test_a_file_without_a_beat_keeps_the_tempo(self):
        self._frames()
        for target in self.requests:
            self.service._waiting.discard(target)
            self.service._store(target, {"error": "No steady beat found."})
        self._frames()
        self.assertEqual((self.source.bpm, self.source.writes), (120.0, 0))
        self.assertEqual(self.source["ace_tempo_detect_status"], "failed")
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
