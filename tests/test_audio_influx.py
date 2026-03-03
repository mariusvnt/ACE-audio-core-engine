import unittest

from module_loader import load_module

try:
    import numpy as np
except Exception:
    np = None

try:
    ai = load_module("audio_influx") if np is not None else None
except Exception:
    ai = None


@unittest.skipIf(ai is None or np is None, "numpy/audio_influx unavailable in this environment")
class TestAudioInflux(unittest.TestCase):
    def test_decode_wave_bytes_sign_extends_24bit_pcm(self):
        # Little-endian packed 24-bit PCM: -1, 0, +1
        frames = bytes([
            0xFF, 0xFF, 0xFF,
            0x00, 0x00, 0x00,
            0x01, 0x00, 0x00,
        ])
        mono, stereo = ai.AudioReception._decode_wave_bytes(frames, 3, 1)
        self.assertIsNone(stereo)
        self.assertAlmostEqual(float(mono[0]), -1.0 / float(1 << 23), places=8)
        self.assertEqual(float(mono[1]), 0.0)
        self.assertAlmostEqual(float(mono[2]), 1.0 / float(1 << 23), places=8)

    def test_decode_wave_bytes_16bit_stereo_outputs_stereo_and_mono(self):
        pcm = np.asarray(
            [
                -32768, 32767,  # frame 0: L, R
                0, 0,           # frame 1: L, R
            ],
            dtype="<i2",
        )
        mono, stereo = ai.AudioReception._decode_wave_bytes(pcm.tobytes(), 2, 2)
        self.assertIsNotNone(stereo)
        self.assertEqual(tuple(stereo.shape), (2, 2))
        self.assertAlmostEqual(float(stereo[0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(stereo[1, 0]), 32767.0 / 32768.0, places=6)
        self.assertAlmostEqual(float(mono[1]), 0.0, places=6)

    def test_decode_wave_bytes_24bit_ignores_incomplete_triplet(self):
        frames = bytes([
            0x01, 0x00, 0x00,  # +1 sample
            0xFF, 0xFF,        # trailing incomplete bytes should be dropped
        ])
        mono, stereo = ai.AudioReception._decode_wave_bytes(frames, 3, 1)
        self.assertIsNone(stereo)
        self.assertEqual(int(mono.shape[0]), 1)
        self.assertAlmostEqual(float(mono[0]), 1.0 / float(1 << 23), places=8)

    def test_decode_wave_bytes_multichannel_truncates_incomplete_frame(self):
        # 3 mono bytes at 8-bit decoded with channels=2 -> one full stereo frame + tail drop.
        frames = bytes([0, 255, 128])
        mono, stereo = ai.AudioReception._decode_wave_bytes(frames, 1, 2)
        self.assertIsNotNone(stereo)
        self.assertEqual(int(mono.shape[0]), 1)
        self.assertEqual(tuple(stereo.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
