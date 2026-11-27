"""Analysis over the audio before a block: file history, workers, pitch and tempo."""

import os
import tempfile
import unittest
import wave

import numpy as np

import fake_bpy
from module_loader import forget_modules, load_module

SR = 48000
BLOCK = 800  # 60 fps


def _write_wav(path: str, samples: np.ndarray) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _tone(freq: float, n: int) -> np.ndarray:
    return (0.5 * np.sin(2.0 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


class AnalysisHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()  # the payload slots import the engine package
        forget_modules("core.worker", "core.tasks", "core.input", "core.engine.parallel")
        cls.chunk = load_module("core.tasks.chunk")
        cls.stream = load_module("core.tasks.stream")
        cls.spectral = load_module("core.tasks.spectral")
        cls.tempo = load_module("core.tasks.tempo")
        cls.reception = load_module("core.input.reception")
        cls.graph = load_module("core.worker.graph")
        cls.slots = load_module("core.engine.parallel.slots")

    def _chunk(self, signal: np.ndarray, end: int, **kwargs):
        """The block ending at *end*, with what came before it."""
        history = np.zeros(self.chunk.HISTORY_SAMPLES, dtype=np.float32)
        before = signal[max(end - BLOCK - history.size, 0):end - BLOCK]
        if before.size:
            history[-before.size:] = before
        return self.chunk.AudioChunk(signal[end - BLOCK:end], None, SR, lineage=("src",), history_mono=history, **kwargs)

    def test_file_history_is_the_audio_before_the_next_block(self):
        ramp = np.arange(3000, dtype=np.float32) / 4000.0
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "ramp.wav")
            _write_wav(path, ramp)
            reception = self.reception.AudioReception("FILE", filepath=path)
            self.assertTrue(reception.open())
            first, _ = reception.history_before(1000)
            self.assertFalse(first.any(), "nothing comes before the start")
            block = reception.read_block(800)[0]
            history, stereo = reception.history_before(1000)
            reception.close()
        self.assertIsNone(stereo)
        self.assertFalse(history[:200].any())
        np.testing.assert_array_equal(history[200:], block)

    def test_pitch_reads_tones_from_blocks_shorter_than_their_analysis(self):
        for freq in (30.0, 55.0, 440.0, 3000.0):
            signal = _tone(freq, 8000)
            value = self.stream.PitchTask("p", {}, BLOCK).process({0: self._chunk(signal, 8000)})
            self.assertAlmostEqual(value / freq, 1.0, delta=0.005, msg=f"{freq} Hz read as {value}")

    def test_pitch_of_noise_and_silence_is_zero(self):
        noise = (0.3 * np.random.default_rng(1).standard_normal(8000)).astype(np.float32)
        for signal in (noise, np.zeros(8000, dtype=np.float32)):
            self.assertEqual(self.stream.PitchTask("p", {}, BLOCK).process({0: self._chunk(signal, 8000)}), 0.0)

    def test_pitch_stays_inside_the_band(self):
        signal = _tone(110.0, 8000) + _tone(1500.0, 8000)
        chunk = self._chunk(signal, 8000, band_hz_bounds=(800.0, 4000.0))
        value = self.stream.PitchTask("p", {}, BLOCK).process({0: chunk})
        self.assertAlmostEqual(value, 1500.0, delta=15.0)

    def _stereo_chunk(self, left: np.ndarray, right: np.ndarray, end: int, **kwargs):
        """A stereo block ending at *end*, with what came before it in both channels."""
        stereo = np.stack([left, right]).astype(np.float32)
        mono = stereo.mean(axis=0)
        history = np.zeros((2, self.chunk.HISTORY_SAMPLES), dtype=np.float32)
        before = stereo[:, max(end - BLOCK - history.shape[1], 0):end - BLOCK]
        if before.shape[1]:
            history[:, -before.shape[1]:] = before
        return self.chunk.AudioChunk(
            mono[end - BLOCK:end], stereo[:, end - BLOCK:end], SR, lineage=("src",),
            history_mono=history.mean(axis=0), history_stereo=history, **kwargs,
        )

    def test_stereo_measures_inside_the_band(self):
        left, right = _tone(100.0, 8000), _tone(3000.0, 8000)
        low = self._stereo_chunk(left, right, 8000, band_hz_bounds=(20.0, 1000.0))
        high = self._stereo_chunk(left, right, 8000, band_hz_bounds=(1000.0, 20000.0))
        for chunk, side in ((low, -1.0), (high, 1.0)):
            balance = self.stream.StereoBalanceTask("b", {}, BLOCK).process({0: chunk})
            self.assertAlmostEqual(balance, side, delta=0.02)
            l_rms, r_rms, vec_balance = self.stream.StereoVectorTask("v", {}, BLOCK).process({0: chunk})
            self.assertAlmostEqual(vec_balance, balance, places=9)
            self.assertAlmostEqual(max(l_rms, r_rms), 0.5 / np.sqrt(2.0), delta=0.01)
        # Without a splitter both tones count (the block holds 1.67 periods of 100 Hz, so not exactly 0).
        full = self._stereo_chunk(left, right, 8000)
        self.assertAlmostEqual(self.stream.StereoBalanceTask("b", {}, BLOCK).process({0: full}), 0.0, delta=0.05)

    def test_band_stereo_levels_add_up_and_a_centred_band_has_no_width(self):
        noise = np.random.default_rng(4).standard_normal((2, 8000)).astype(np.float32) * 0.2
        parts = [
            self.spectral._band_stereo_levels(self._stereo_chunk(noise[0], noise[1], 8000, band_hz_bounds=bounds))
            for bounds in ((0.0, 1000.0), (1000.0, 1e9))
        ]
        whole = self.spectral._band_stereo_levels(self._stereo_chunk(noise[0], noise[1], 8000, band_hz_bounds=(0.0, 1e9)))
        np.testing.assert_allclose(np.add(parts[0], parts[1]), whole, rtol=1e-9)
        centred = self._stereo_chunk(noise[0], noise[0], 8000, band_hz_bounds=(200.0, 2000.0))
        self.assertAlmostEqual(self.stream.StereoWidthTask("w", {}, BLOCK).process({0: centred}), 0.0, places=6)
        self.assertAlmostEqual(self.stream.StereoBalanceTask("b", {}, BLOCK).process({0: centred}), 0.0, places=6)

    def test_brightness_of_the_full_band_is_the_hann_centroid_of_the_frame(self):
        signal = _tone(300.0, 8000) + 0.5 * _tone(4000.0, 8000)
        value = self.stream.BrightnessTask("c", {}, BLOCK).process({0: self._chunk(signal, 8000)})
        frame = signal[8000 - 1024:8000].astype(np.float64)  # the 1024-point FFT ends with the block
        mag = np.abs(np.fft.rfft(frame * (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1024) / 1024))))
        freqs = np.fft.rfftfreq(1024, d=1.0 / SR)
        self.assertAlmostEqual(value, float(np.sum(freqs * mag) / np.sum(mag)), delta=1e-3)

    def test_brightness_stays_inside_the_band(self):
        signal = _tone(300.0, 8000) + _tone(4000.0, 8000)
        for bounds in ((20.0, 1000.0), (1000.0, 20000.0), (2000.0, 8000.0)):
            chunk = self._chunk(signal, 8000, band_hz_bounds=bounds)
            value = self.stream.BrightnessTask("c", {}, BLOCK).process({0: chunk})
            self.assertGreaterEqual(value, bounds[0], bounds)
            self.assertLess(value, bounds[1], bounds)

    def test_brightness_of_a_band_without_a_bin_is_zero_not_the_full_band(self):
        # A 1024-point FFT at 48 kHz has bins every 46.9 Hz: none lies in 100..120 Hz.
        chunk = self._chunk(_tone(110.0, 8000) + _tone(3000.0, 8000), 8000, band_hz_bounds=(100.0, 120.0))
        self.assertEqual(self.stream.BrightnessTask("c", {}, BLOCK).process({0: chunk}), 0.0)
        silent = self._chunk(np.zeros(8000, dtype=np.float32), 8000)
        self.assertEqual(self.stream.BrightnessTask("c", {}, BLOCK).process({0: silent}), 0.0)

    def test_worker_gets_the_same_history_as_the_main_process(self):
        signal = _tone(55.0, 8000) + _tone(2000.0, 8000)
        local = self._chunk(signal, 8000)
        local.history_stereo = np.stack([local.history_mono, 0.5 * local.history_mono])
        local.stereo = np.stack([local.mono, 0.5 * local.mono])
        payload = self.slots.source_payload(local, BLOCK, SR)
        provider = self.graph._WorkerSourceProvider(sample_rate=SR, block_size=BLOCK)
        bridge = self.graph._WorkerSourceBridgeTask("src", provider=provider, block_size=BLOCK)
        bridge.set_chunk(payload, frame_id=1, block_size=BLOCK)
        remote = bridge.get_output(1)
        provider.close()
        np.testing.assert_array_equal(remote.mono, local.mono)
        np.testing.assert_array_equal(remote.stereo, local.stereo)
        np.testing.assert_array_equal(remote.history_mono, local.history_mono)
        np.testing.assert_array_equal(remote.history_stereo, local.history_stereo)

    def test_tempo_of_a_loop_with_off_beats_is_the_beat_not_the_eighth(self):
        for bpm in (90.0, 120.0, 140.0):
            signal = np.zeros(8 * SR, dtype=np.float32)
            rng = np.random.default_rng(2)
            t = np.arange(int(0.2 * SR)) / SR
            kick = 0.7 * np.sin(2.0 * np.pi * 60.0 * t) * np.exp(-t / 0.08)
            hat = 0.15 * rng.uniform(-1.0, 1.0, 1600) * np.exp(-np.arange(1600) / 400.0)
            beat = 60.0 / bpm
            for k in range(int(8.0 / beat) - 1):
                for start, sound in ((k * beat, kick), ((k + 0.5) * beat, hat)):
                    i = int(round(start * SR))
                    signal[i:i + sound.size] += sound[:signal.size - i].astype(np.float32)
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "loop.wav")
                _write_wav(path, signal)
                detected = self.tempo.estimate_file_tempo(path)
            self.assertIsNotNone(detected, f"{bpm} BPM not detected")
            self.assertAlmostEqual(detected["bpm"] / bpm, 1.0, delta=0.01, msg=f"{bpm} read as {detected['bpm']}")

    def test_tempo_of_a_signal_without_a_beat_is_not_guessed(self):
        noise = (0.2 * np.random.default_rng(3).standard_normal(6 * SR)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "noise.wav")
            _write_wav(path, noise)
            self.assertIsNone(self.tempo.estimate_file_tempo(path))

    @staticmethod
    def _accented_loop(bpm: float, beats_per_bar: int, seconds: float = 8.0) -> np.ndarray:
        """Kick on every beat, hats on the off-beats, a crash on every downbeat."""
        rng = np.random.default_rng(2)
        t = np.arange(int(0.25 * SR)) / SR
        kick = 0.7 * np.sin(2.0 * np.pi * np.cumsum(50.0 + 70.0 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.12)
        hat = 0.15 * rng.uniform(-1.0, 1.0, 1600) * np.exp(-np.arange(1600) / 400.0)
        crash = 0.3 * rng.uniform(-1.0, 1.0, 4800) * np.exp(-np.arange(4800) / 1500.0)
        signal = np.zeros(int(seconds * SR), dtype=np.float32)
        beat = 60.0 / bpm
        for k in range(int(seconds / beat) - 1):
            hits = [(k * beat, kick), ((k + 0.5) * beat, hat)] + ([(k * beat, crash)] if k % beats_per_bar == 0 else [])
            for start, sound in hits:
                i = int(round(start * SR))
                signal[i:i + sound.size] += sound[:signal.size - i].astype(np.float32)
        return signal

    def test_beats_per_bar_follow_the_accents(self):
        with tempfile.TemporaryDirectory() as folder:
            for bpm in (90.0, 120.0, 140.0):
                for beats_per_bar in (3, 4):
                    path = os.path.join(folder, f"loop_{int(bpm)}_{beats_per_bar}.wav")
                    _write_wav(path, self._accented_loop(bpm, beats_per_bar))
                    detected = self.tempo.estimate_file_tempo(path)
                    label = f"{bpm} BPM in {beats_per_bar}"
                    self.assertIsNotNone(detected, label)
                    self.assertAlmostEqual(detected["bpm"] / bpm, 1.0, delta=0.01, msg=f"{label}: {detected}")
                    self.assertEqual(detected["beats_per_bar"], beats_per_bar, f"{label}: {detected}")
                    # The crashes start the bars: the downbeat lies on a multiple of the bar.
                    bar_s = beats_per_bar * 60.0 / bpm
                    offset = detected["downbeat_s"] % bar_s
                    self.assertLess(min(offset, bar_s - offset), 0.05, f"{label}: {detected}")

    def test_beats_per_bar_without_accents_is_four(self):
        """Every beat alike, or a click track: nothing to count bars by."""
        step = 0.01
        period = 50.0
        envelope = np.zeros(1500)
        envelope[np.arange(0, 1500, int(period))] = 10.0
        self.assertEqual(self.tempo._meter(envelope, period), (4, None))
        rng = np.random.default_rng(7)
        envelope[np.arange(0, 1500, int(period))] = rng.uniform(5.0, 10.0, 30)
        self.assertEqual(self.tempo._meter(envelope, period)[0], 4)
        accented = np.zeros(1500)
        accented[np.arange(0, 1500, int(period))] = 5.0
        accented[np.arange(100, 1500, 3 * int(period))] = 10.0
        self.assertEqual(self.tempo._meter(accented, period), (3, 100.0))
        self.assertAlmostEqual(self.tempo._tempo_of_envelope(accented, step)["bpm"], 120.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
