"""Takes (core/takes.py): the audio a live input delivered, and the values each frame showed."""

import os
import tempfile
import types
import unittest
import wave

from module_loader import load_module

import fake_bpy

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy is required")
class TakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.takes = load_module("core.takes")

    def test_the_wav_holds_every_block_after_silence_up_to_the_session_start(self):
        folder = tempfile.mkdtemp(prefix="ace_take_test_")
        scene = types.SimpleNamespace(name="Scene", frame_start=1)
        recorder = self.takes.TakeRecorder(engine=None, scene=scene, start_frame=25, fps=24.0)
        track = recorder.add_track("S1", "Mic 1", sample_rate=1000, channels=2, folder=folder)
        first = np.full((300, 2), 0.5, dtype=np.float32)
        second = np.full((200, 2), -0.25, dtype=np.float32)
        track.push(first)
        track.push(second)
        track.finish()
        self.assertEqual(track.error, "")
        self.assertEqual(track.pad_frames, 1000, "frames 1 to 24 at 24 fps: one second of silence")
        with wave.open(track.path, "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getframerate()), (2, 1000))
            data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, 2)
        self.assertEqual(data.shape[0], 1000 + 300 + 200)
        self.assertFalse(data[:1000].any())
        self.assertTrue(np.all(data[1000:1300] == int(0.5 * 32767)))
        self.assertTrue(np.all(data[1300:] == int(-0.25 * 32767)))
        os.remove(track.path)

    def test_the_values_of_each_frame_make_a_curve_and_a_missing_frame_holds_the_last(self):
        take = self.takes.TakeValues()
        take.add(10, {"a:0": 0.1, "b:0": 1.0})
        take.add(11, {"a:0": 0.2, "b:0": 2.0})
        take.add(13, {"a:0": 0.4})
        first, last, series = take.series()
        self.assertEqual((first, last), (10, 13))
        np.testing.assert_allclose(series["a:0"], [0.1, 0.2, 0.2, 0.4], rtol=1e-6)
        np.testing.assert_allclose(series["b:0"], [1.0, 2.0, 2.0, 2.0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
