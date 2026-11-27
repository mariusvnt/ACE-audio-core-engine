"""File decoding into memory and block reads over it (the wave path: aud exists only inside Blender)."""

import os
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import load_module  # noqa: E402

import fake_bpy  # noqa: E402


def _write_wav(path, ints, sample_rate, sampwidth=2):
    ints = np.asarray(ints)
    if ints.ndim == 1:
        ints = ints[:, None]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(ints.shape[1]))
        w.setsampwidth(sampwidth)
        w.setframerate(int(sample_rate))
        if sampwidth == 2:
            w.writeframes(ints.astype("<i2").tobytes())
        else:  # 24-bit packed
            raw = ints.astype("<i4").reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3]
            w.writeframes(raw.tobytes())


class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.decode = load_module("core.input.decode")
        cls.resampler = load_module("core.input.resampler")

    def setUp(self):
        self.decode.clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_outside_blender_only_wav_is_offered(self):
        self.assertFalse(self.decode.AUD_AVAILABLE)
        self.assertEqual(self.decode.supported_extensions(), (".wav", ".wave"))

    def test_stereo_16_bit(self):
        ints = np.stack([np.arange(-500, 500), -np.arange(-500, 500)], axis=1) * 30
        path = self.dir / "s16.wav"
        _write_wav(path, ints, 44100)
        result = self.decode.decode(str(path))
        self.assertEqual(result.samples.shape, (1000, 2))
        self.assertEqual((result.sample_rate, result.native_sample_rate, result.native_frames), (44100, 44100, 1000))
        np.testing.assert_array_equal(result.samples, ints.astype(np.float32) / 32768.0)
        self.assertFalse(result.samples.flags.writeable, "shared samples must be read-only")
        self.assertEqual(self.decode.probe(str(path)), self.decode.FileInfo(44100, 2, 1000))

    def test_mono_24_bit(self):
        ints = np.array([0, 1, -1, 8388607, -8388608, 4194304])
        path = self.dir / "m24.wav"
        _write_wav(path, ints, 48000, sampwidth=3)
        result = self.decode.decode(str(path))
        np.testing.assert_allclose(result.samples[:, 0], ints / float(1 << 23), atol=1e-7)

    def test_downsampling_to_the_working_rate(self):
        ramp = np.arange(0, 9600)  # a ramp is interpolated exactly
        path = self.dir / "ramp96k.wav"
        _write_wav(path, ramp, 96000)
        result = self.decode.decode(str(path), target_sample_rate=48000)
        self.assertEqual((result.sample_rate, result.native_sample_rate, result.native_frames), (48000, 96000, 9600))
        self.assertEqual(result.samples.shape, (4800, 1))
        np.testing.assert_allclose(result.samples[:, 0], np.arange(0, 9600, 2) / 32768.0, atol=1e-7)
        # A higher target leaves the file at its own rate.
        self.assertEqual(self.decode.decode(str(path), target_sample_rate=192000).sample_rate, 96000)

    def test_resample_ends_on_the_last_input_frame(self):
        out = self.resampler.resample_linear(np.arange(10, dtype=np.float32)[:, None], 44100, 48000)
        step = 44100 / 48000
        self.assertEqual(out.shape[0], int(np.floor(9 / step)) + 1)
        np.testing.assert_allclose(out[:, 0], np.arange(out.shape[0]) * step, atol=1e-5)

    def test_one_decode_per_file_until_it_changes(self):
        path = self.dir / "cached.wav"
        _write_wav(path, np.arange(100), 48000)
        first = self.decode.decode(str(path))
        self.assertIs(self.decode.decode(str(path)), first)
        time.sleep(0.01)
        _write_wav(path, np.arange(200), 48000)
        os.utime(path, None)
        second = self.decode.decode(str(path))
        self.assertIsNot(second, first)
        self.assertEqual(second.samples.shape[0], 200)

    def test_unreadable_files(self):
        junk = self.dir / "junk.wav"
        junk.write_bytes(b"not audio" * 50)
        with self.assertRaises(self.decode.DecodeError):
            self.decode.decode(str(junk))
        with self.assertRaises(self.decode.DecodeError):
            self.decode.probe(str(junk))
        mp3 = self.dir / "song.mp3"
        mp3.write_bytes(b"\x00" * 100)
        with self.assertRaises(self.decode.DecodeError):
            self.decode.decode(str(mp3))
        with self.assertRaises(FileNotFoundError):
            self.decode.decode(str(self.dir / "missing.wav"))


class FileReceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.reception = load_module("core.input.reception")
        cls.decode = load_module("core.input.decode")

    def setUp(self):
        self.decode.clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tone.wav"
        self.ints = np.stack([np.arange(250) * 10, np.arange(250) * -10], axis=1)
        _write_wav(self.path, self.ints, 48000)

    def _open(self):
        rec = self.reception.AudioReception("FILE", filepath=str(self.path))
        self.assertTrue(rec.open(block_size=100), rec.last_error)
        return rec

    def test_blocks_padding_and_end_of_file(self):
        rec = self._open()
        self.assertEqual((rec.kind, rec.channels, rec.sample_rate, rec.native_sample_rate), ("memory", 2, 48000, 48000))
        self.assertAlmostEqual(rec.duration, 250 / 48000)
        expected = self.ints.astype(np.float32) / 32768.0
        mono, stereo, sr = rec.read_block(100)
        np.testing.assert_allclose(mono, expected[:100].mean(axis=1), atol=1e-7)
        np.testing.assert_array_equal(stereo, expected[:100].T)
        self.assertTrue(stereo.flags.writeable, "blocks are copies the caller may change")
        rec.read_block(100)
        mono, stereo, _ = rec.read_block(100)  # 50 left: padded with silence
        self.assertEqual((mono.shape, stereo.shape), ((100,), (2, 100)))
        self.assertTrue(np.all(mono[50:] == 0.0))
        self.assertTrue(rec._eof)
        self.assertIsNone(rec.read_block(100))

    def test_seek_and_reset(self):
        rec = self._open()
        self.assertTrue(rec.seek(200))
        mono, _, _ = rec.read_block(10)
        np.testing.assert_allclose(mono, (self.ints[200:210] / 32768.0).mean(axis=1), atol=1e-7)
        rec.reset()
        mono, _, _ = rec.read_block(10)
        np.testing.assert_allclose(mono, (self.ints[:10] / 32768.0).mean(axis=1), atol=1e-7)
        self.assertFalse(rec.seek(-1))

    def test_readers_of_one_file_have_their_own_cursor(self):
        a, b = self._open(), self._open()
        a.read_block(100)
        mono_b, _, _ = b.read_block(10)
        np.testing.assert_allclose(mono_b, (self.ints[:10] / 32768.0).mean(axis=1), atol=1e-7)

    def test_errors_and_support(self):
        reception = self.reception.AudioReception
        self.assertTrue(reception.is_supported_file_path(str(self.path)))
        junk = Path(self.tmp.name) / "junk.wav"
        junk.write_bytes(b"junk" * 40)
        self.assertFalse(reception.is_supported_file_path(str(junk)))
        rec = reception("FILE", filepath=str(junk))
        self.assertFalse(rec.open())
        self.assertEqual(rec.error_kind, "open_failed")
        rec = reception("FILE", filepath=str(Path(self.tmp.name) / "song.flac"))
        Path(rec.filepath).write_bytes(b"\x00" * 10)
        self.assertFalse(rec.open())
        self.assertEqual(rec.error_kind, "unsupported_format")


if __name__ == "__main__":
    unittest.main()
