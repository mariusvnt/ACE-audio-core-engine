"""The shared spectrum cache never hands one setup's spectrum to another.

The cache is shared by every analyzer so splitter bands of one chunk take one
transform. Its key once held only the chunk's lineage and frame index, which
stay the same when the file, sample rate or frame rate changes: the next frames
got the old spectra, and with another FFT size the band masks indexed past
their end (IndexError). A fake BFT stands in for AudioFlux.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import load_module  # noqa: E402


class _FakeBFT:
    """Magnitude spectrum of the block, sized like the analyzer's FFT."""

    is_temporal = False

    def __init__(self, fft_length):
        self.fft_length = fft_length
        self.calls = 0

    def bft(self, data):
        self.calls += 1
        return np.abs(np.fft.rfft(np.asarray(data, dtype=np.float64), n=self.fft_length))[:, None]


class _FakeSpectral:
    def set_time_length(self, _n):
        pass


class SpectrumCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer_mod = load_module("core.tasks.analyzer")
        cls.chunk_mod = load_module("core.tasks.chunk")

    def setUp(self):
        self.cls = self.analyzer_mod.AudioFluxAnalyzer
        self.cls._GLOBAL_SPECTRUM_CACHE.clear()
        patcher = mock.patch.object(self.analyzer_mod, "AUDIOFLUX_AVAILABLE", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.cls._GLOBAL_SPECTRUM_CACHE.clear)

    def _analyzer(self, sample_rate, block_size):
        # The real constructor builds AudioFlux objects; set the fields _ensure_spectrum reads by hand.
        analyzer = self.cls.__new__(self.cls)
        analyzer.sample_rate = sample_rate
        analyzer.block_size = block_size
        analyzer.radix2_exp = max(6, int(np.ceil(np.log2(block_size))))
        analyzer.fft_length = 1 << analyzer.radix2_exp
        analyzer.num_bins = analyzer.fft_length // 2 + 1
        analyzer.bft = _FakeBFT(analyzer.fft_length)
        analyzer.spectral = _FakeSpectral()
        analyzer.cache = {}
        analyzer.cache_frame = -1
        analyzer.prev_masked_mag = {}
        analyzer._last_error_log = {}
        return analyzer

    def _chunk(self, samples, sample_rate, frame_index=1):
        return self.chunk_mod.AudioChunk(
            np.asarray(samples, dtype=np.float32), None, sample_rate, frame_index=frame_index, lineage=("src",),
        )

    def test_same_chunk_and_frame_share_one_transform(self):
        block = np.sin(np.arange(2000) / 7.0)
        a, b = self._analyzer(48000, 2000), self._analyzer(48000, 2000)
        first = a._ensure_spectrum(self._chunk(block, 48000))
        second = b._ensure_spectrum(self._chunk(block, 48000))
        self.assertIs(first, second)
        self.assertEqual(a.bft.calls + b.bft.calls, 1)

    def test_other_frame_rate_gets_its_own_spectrum(self):
        """60 fps (block 800, FFT 1024) then 24 fps (block 2000, FFT 2048) at the same frame."""
        signal = np.sin(np.arange(2000) / 7.0)
        fast = self._analyzer(48000, 800)
        fast._ensure_spectrum(self._chunk(signal[:800], 48000))
        slow = self._analyzer(48000, 2000)
        mag = slow._ensure_spectrum(self._chunk(signal, 48000))
        self.assertEqual(mag.shape[0], slow.num_bins)

    def test_other_file_gets_its_own_spectrum(self):
        a, b = self._analyzer(48000, 2000), self._analyzer(48000, 2000)
        first = a._ensure_spectrum(self._chunk(np.sin(np.arange(2000) / 7.0), 48000))
        second = b._ensure_spectrum(self._chunk(np.sin(np.arange(2000) / 3.0), 48000))
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
