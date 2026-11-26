"""The shared spectrum cache never hands one setup's spectrum to another.

The cache lets the splitter bands of one chunk share one transform. Its key
once held only the chunk's lineage and frame index, which stay the same when
the file, sample rate or frame rate changes: the next frames got the old
spectra, and with another FFT size the band masks indexed past their end.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import load_module  # noqa: E402


class SpectrumCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spectral = load_module("core.tasks.spectral")
        cls.chunk_mod = load_module("core.tasks.chunk")

    def setUp(self):
        self.spectral.clear_spectrum_cache()
        self.addCleanup(self.spectral.clear_spectrum_cache)
        real_rfft = np.fft.rfft
        self.calls = 0

        def counting_rfft(*args, **kwargs):
            self.calls += 1
            return real_rfft(*args, **kwargs)

        patcher = mock.patch.object(self.spectral.np.fft, "rfft", counting_rfft)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _chunk(self, samples, sample_rate=48000, frame_index=1, **kwargs):
        kwargs.setdefault("lineage", ("src",))
        return self.chunk_mod.AudioChunk(np.asarray(samples, dtype=np.float32), None, sample_rate, frame_index=frame_index, **kwargs)

    def test_bands_of_one_chunk_share_one_transform(self):
        block = np.sin(np.arange(2000) / 7.0)
        low = self._chunk(block, lineage=("src", "split:0"), spectrum_token=("src",), band_hz_bounds=(0.0, 1000.0))
        high = self._chunk(block, lineage=("src", "split:1"), spectrum_token=("src",), band_hz_bounds=(1000.0, 24000.0))
        self.assertIs(self.spectral._frame_spectrum(low, 2048), self.spectral._frame_spectrum(high, 2048))
        self.assertEqual(self.calls, 1)
        self.assertGreater(self.spectral._band_centroid(high), self.spectral._band_centroid(low))

    def test_other_frame_rate_gets_its_own_spectrum(self):
        """60 fps (block 800, FFT 1024) then 24 fps (block 2000, FFT 2048) at the same frame."""
        signal = np.sin(np.arange(2000) / 7.0)
        self.spectral._band_centroid(self._chunk(signal[:800]))
        mag = self.spectral._frame_spectrum(self._chunk(signal), 2048)
        self.assertEqual(mag.shape[0], 1025)

    def test_other_file_gets_its_own_spectrum(self):
        first = self.spectral._frame_spectrum(self._chunk(np.sin(np.arange(2000) / 7.0)), 2048)
        second = self.spectral._frame_spectrum(self._chunk(np.sin(np.arange(2000) / 3.0)), 2048)
        self.assertFalse(np.array_equal(first, second))

    def test_other_analysis_ops_get_their_own_spectrum(self):
        block = np.sin(np.arange(2000) / 7.0)
        shared = ("src", "shared")
        mid = self._chunk(block, spectrum_token=shared, analysis_ops=(("midside", "mid"),))
        side = self._chunk(block, spectrum_token=shared, analysis_ops=(("midside", "side"),))
        self.spectral._frame_spectrum(mid, 2048)
        self.spectral._frame_spectrum(side, 2048)
        self.assertEqual(self.calls, 2)


if __name__ == "__main__":
    unittest.main()
