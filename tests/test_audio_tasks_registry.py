import types
import unittest
import warnings

from module_loader import load_module, load_tasks

try:
    import numpy as np
except Exception:
    np = None

try:
    atr = load_tasks() if np is not None else None
    tasks_source = load_module("core.tasks.source") if np is not None else None
    tasks_spectral = load_module("core.tasks.spectral") if np is not None else None
except Exception:
    atr = None


@unittest.skipIf(atr is None or np is None, "numpy/audio_tasks_registry unavailable in this environment")
class TestAudioTasksRegistry(unittest.TestCase):
    def test_frequency_splitter_emits_bounds_and_lineage(self):
        task = atr.FrequencySplitterTask("splitter_node", {}, block_size=256)
        chunk = atr.AudioChunk(
            np.ones(256, dtype=np.float32),
            None,
            48000,
            lineage=("src_node",),
        )

        out = task.process({0: chunk, 1: 1000.0, 2: 4000.0})
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].lineage[-1], "splitter_node:0")
        self.assertEqual(out[1].lineage[-1], "splitter_node:1")
        self.assertEqual(out[2].lineage[-1], "splitter_node:2")
        self.assertEqual(out[0].band_hz_bounds[0], 20.0)
        self.assertAlmostEqual(out[0].band_hz_bounds[1], 1000.0)
        self.assertAlmostEqual(out[1].band_hz_bounds[0], 1000.0)
        self.assertAlmostEqual(out[1].band_hz_bounds[1], 4000.0)

    def test_frequency_splitter_nested_defaults_are_parent_relative(self):
        task = atr.FrequencySplitterTask("splitter_node", {}, block_size=256)
        parent = atr.AudioChunk(
            np.ones(256, dtype=np.float32),
            None,
            48000,
            band_hz_bounds=(1000.0, 4000.0),
            lineage=("src_node", "parent"),
        )
        # Legacy global-Hz defaults should be remapped into the parent band.
        out = task.process({0: parent, 1: 200.0, 2: 2000.0})
        self.assertEqual(len(out), 3)
        for band in out:
            self.assertGreaterEqual(band.band_hz_bounds[0], 1000.0)
            self.assertLessEqual(band.band_hz_bounds[1], 4000.0)
        self.assertGreater(out[0].band_hz_bounds[1], 1000.0)
        self.assertLess(out[0].band_hz_bounds[1], out[1].band_hz_bounds[1])

    def test_frequency_splitter_supports_normalized_nested_mode(self):
        task = atr.FrequencySplitterTask("splitter_node", {}, block_size=256)
        parent = atr.AudioChunk(
            np.ones(256, dtype=np.float32),
            None,
            48000,
            band_hz_bounds=(1000.0, 5000.0),
            lineage=("src_node", "parent"),
        )
        out = task.process({0: parent, 1: 0.25, 2: 0.75})
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0].band_hz_bounds[1], 2000.0, places=3)
        self.assertAlmostEqual(out[1].band_hz_bounds[0], 2000.0, places=3)
        self.assertAlmostEqual(out[1].band_hz_bounds[1], 4000.0, places=3)

    def test_band_bins_follow_hz_bounds_in_the_fft_domain(self):
        chunk = atr.AudioChunk(np.zeros(750, dtype=np.float32), None, 48000, band_hz_bounds=(1000.0, 5000.0), lineage=("src", "band"))
        freqs = np.fft.rfftfreq(1024, d=1.0 / 48000.0)[atr._band_bins(chunk, 1024, 750)]
        self.assertGreater(freqs.size, 0)
        self.assertGreaterEqual(float(freqs.min()), 1000.0)
        self.assertLess(float(freqs.max()), 5000.0)

    def test_band_bins_intersect_indices_and_bounds(self):
        base_indices = np.array([8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
        chunk = atr.AudioChunk(
            np.zeros(256, dtype=np.float32), None, 48000,
            band_indices=base_indices, band_hz_bounds=(2000.0, 2500.0), lineage=("src", "pre_masked"),
        )
        idx = np.nonzero(atr._band_bins(chunk, 256, 256))[0]
        self.assertEqual(idx.tolist(), [11, 12, 13])

    def test_band_bins_remap_indices_of_another_fft_size(self):
        src_freqs = np.fft.rfftfreq(750, d=1.0 / 48000.0)
        src_idx = np.array([int(np.argmin(np.abs(src_freqs - f))) for f in (1000.0, 3500.0)], dtype=np.int64)
        chunk = atr.AudioChunk(
            np.zeros(750, dtype=np.float32), None, 48000,
            band_indices=src_idx, band_index_fft_size=750, lineage=("src",),
        )
        out_freqs = np.fft.rfftfreq(1024, d=1.0 / 48000.0)[atr._band_bins(chunk, 1024, 750)]
        self.assertTrue(np.any(np.abs(out_freqs - 1000.0) < 40.0))
        self.assertTrue(np.any(np.abs(out_freqs - 3500.0) < 40.0))

    def test_frequency_splitter_preserves_parent_spectrum_token(self):
        splitter = atr.FrequencySplitterTask("splitter_node", {}, block_size=256)
        source = atr.AudioChunk(
            np.ones(256, dtype=np.float32),
            None,
            48000,
            lineage=("src_node",),
            spectrum_token=("src_node", "frame"),
        )
        out = splitter.process({0: source, 1: 1000.0, 2: 4000.0})
        self.assertEqual(len(out), 3)
        self.assertTrue(all(c.spectrum_token == source.spectrum_token for c in out))

    def test_frequency_splitter_keeps_an_empty_band_for_coinciding_cuts(self):
        n, sr = 2048, 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        source = atr.AudioChunk(mono, np.stack([mono, mono]), sr, lineage=("src",), spectrum_token=("src", "frame"))

        for cuts, empty in (((1000.0, 1000.0), 1), ((1000.0, 1000.4), 1), ((0.0, 10.0), 0), ((0.0, 10.0), 1)):
            out = atr.FrequencySplitterTask("split", {}, block_size=n).process({0: source, 1: cuts[0], 2: cuts[1]})
            self.assertEqual(len(out), 3, cuts)
            band = out[empty]
            self.assertFalse(np.any(band.mono), cuts)
            self.assertFalse(np.any(band.stereo), cuts)
            self.assertEqual(band.lineage[-1], f"split:{empty}")
            for task_cls in (atr.RMSTask, atr.RMSLoudnessTask, atr.LUFSTask, atr.PeakTask):
                self.assertEqual(float(task_cls("m", {}, block_size=n).process({0: band})), 0.0, task_cls.__name__)
            heard = atr.PreviewSinkTask("p", {"preview_enabled": True}, block_size=n).process({0: band})
            self.assertFalse(np.any(heard.mono), cuts)

        # The bands around the empty one are unchanged.
        low, none, high = atr.FrequencySplitterTask("split", {}, block_size=n).process({0: source, 1: 1000.0, 2: 1000.0})
        self.assertEqual((low.band_hz_bounds, high.band_hz_bounds), ((20.0, 1000.0), (1000.0, 20000.0)))
        np.testing.assert_allclose(high.mono, mono)

    def test_frequency_splitter_leaves_an_uncomputable_link_as_an_empty_band(self):
        source = atr.AudioChunk(np.ones(256, dtype=np.float32), None, 48000, lineage=("src",))
        task = atr.FrequencySplitterTask("split", {"unknown_inputs": [1]}, block_size=256)
        low, none, high = task.process({0: source, 2: 4000.0})
        self.assertEqual((low.band_hz_bounds, high.band_hz_bounds), ((20.0, 4000.0), (4000.0, 20000.0)))
        self.assertFalse(np.any(none.mono))
        # An audio value arriving through the link is used.
        low, mid, _high = task.process({0: source, 1: 500.0, 2: 4000.0})
        self.assertEqual((low.band_hz_bounds, mid.band_hz_bounds), ((20.0, 500.0), (500.0, 4000.0)))

    def test_source_task_forwards_live_latency_to_reception(self):
        captured = {}

        class DummyReception:
            def __init__(self, source_type, filepath="", flow_id="", live_latency_ms=50):
                captured["source_type"] = source_type
                captured["filepath"] = filepath
                captured["flow_id"] = flow_id
                captured["live_latency_ms"] = live_latency_ms
                self.channels = 2
                self.sample_rate = 44100
                self.duration = 0.0
                self.last_error = "offline"
                self.kind = "live_stub"

            def open(self, block_size=None, target_sample_rate=None):
                captured["block_size"] = block_size
                captured["target_sample_rate"] = target_sample_rate
                return False

            def close(self):
                return None

        original_reception = tasks_source.AudioReception
        tasks_source.AudioReception = DummyReception
        try:
            class DummySource:
                source_type = "LIVE"
                filepath = ""
                flow_identifier = "3"
                live_latency_ms = 123
                live_status = ""

            task = atr.SourceTask("src_node", {}, block_size=128)
            task.source_context = DummySource()
            task.scene_fps = 24.0
            task._load_source()
            task.close()
        finally:
            tasks_source.AudioReception = original_reception

        self.assertEqual(captured["source_type"], "LIVE")
        self.assertEqual(captured["flow_id"], "3")
        self.assertEqual(captured["live_latency_ms"], 123)

    def test_source_task_keeps_last_live_error_visible_without_reception(self):
        class DummySource:
            source_type = "LIVE"
            filepath = ""
            flow_identifier = ""
            live_latency_ms = 50
            sample_rate = 0
            duration = 0.0
            live_transport = "SOUNDDEVICE"
            live_device_selector = "__AUTO__"
            live_udp_port = 7001
            live_udp_channels = 2
            live_udp_sample_rate = 44100
            live_status = ""

        task = atr.SourceTask("src_node", {}, block_size=128)
        task.source_context = DummySource()
        task.scene_fps = 24.0
        task.initialized = True
        task.source_ready = True
        task.reception = None
        task.last_error = "SoundDevice live stream failed to open: mono device only"
        task._last_config_signature = task._config_signature()
        task._last_live_retry_ts = 10.0 ** 12

        chunk = task._read_next_block_sync()

        self.assertIsNone(chunk)
        self.assertEqual(task.source_context.live_status, task.last_error)

    def test_source_outputs_its_tempo_after_the_audio(self):
        """Outputs 1 and 2 of a Source node are BPM and Beats per Bar, for links into value nodes."""
        task = atr.SourceTask("src", {}, 256)
        task.source_context = types.SimpleNamespace(bpm=128.0, beats_per_bar=3)
        self.assertEqual(task._extract_result(1), 128.0)
        self.assertEqual(task._extract_result(2), 3)
        task.source_context = None
        self.assertEqual((task._extract_result(1), task._extract_result(2)), (120.0, 4))

    def test_rolling_average_covers_a_time_window(self):
        task = atr.RollingAvgTask("rolling_node", {"in_1": 0.3}, block_size=128)  # 3 frames at 10 fps
        task.scene_fps = 10.0

        def at(frame, value, window=None):
            task._current_frame_request = frame
            inputs = {0: value} if window is None else {0: value, 1: window}
            return float(task.process(inputs))

        self.assertAlmostEqual(at(1, 1.0), 1.0)
        self.assertAlmostEqual(at(2, 2.0), 1.5)
        self.assertAlmostEqual(at(3, 3.0), 2.0)
        self.assertAlmostEqual(at(4, 9.0), (2.0 + 3.0 + 9.0) / 3.0)
        self.assertAlmostEqual(at(4, 12.0), (2.0 + 3.0 + 12.0) / 3.0)  # the same frame again
        self.assertAlmostEqual(at(6, 6.0), (12.0 + 6.0) / 2.0)  # a skipped frame: time still counts
        self.assertAlmostEqual(at(7, 5.0, window=0.05), 5.0)  # shorter than a frame: this one only
        self.assertAlmostEqual(at(2, 4.0), 4.0)  # rewound

    def test_stream_accumulator_persists_and_resets_on_timeline_rewind(self):
        task = atr.StreamAccumulatorTask("accum_node", {}, block_size=128)
        task.scene_frame_start = 1

        task._current_frame_request = 1
        self.assertAlmostEqual(float(task.process({0: 1.25})), 1.25, places=6)
        task._current_frame_request = 2
        self.assertAlmostEqual(float(task.process({0: 2.75})), 4.0, places=6)
        task._current_frame_request = 3
        self.assertAlmostEqual(float(task.process({0: -1.0})), 3.0, places=6)

        # Rewind to frame 1: accumulator should reset before adding the new value.
        task._current_frame_request = 1
        self.assertAlmostEqual(float(task.process({0: 5.0})), 5.0, places=6)

    def test_stream_accumulator_supports_vector_inputs(self):
        task = atr.StreamAccumulatorTask("accum_vec_node", {}, block_size=64)
        task.scene_frame_start = 1

        task._current_frame_request = 1
        out_a = task.process({0: (1.0, 2.0, 3.0)})
        self.assertTrue(np.allclose(np.asarray(out_a, dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32)))

        task._current_frame_request = 2
        out_b = task.process({0: np.array([0.5, -1.0, 2.0], dtype=np.float32)})
        self.assertTrue(np.allclose(np.asarray(out_b, dtype=np.float32), np.array([1.5, 1.0, 5.0], dtype=np.float32)))

    def test_stream_accumulator_state_roundtrip_preserves_running_sum(self):
        task_a = atr.StreamAccumulatorTask("accum_state_node", {}, block_size=64)
        task_a.scene_frame_start = 1
        task_a._current_frame_request = 1
        self.assertAlmostEqual(float(task_a.process({0: 1.0})), 1.0, places=6)
        task_a._current_frame_request = 2
        self.assertAlmostEqual(float(task_a.process({0: 2.0})), 3.0, places=6)

        snapshot = dict(task_a.export_state() or {})
        self.assertIn("accumulator", snapshot)

        task_b = atr.StreamAccumulatorTask("accum_state_node", {}, block_size=64)
        task_b.scene_frame_start = 1
        restored = task_b.import_state(snapshot)
        self.assertTrue(restored)
        task_b._current_frame_request = 3
        self.assertAlmostEqual(float(task_b.process({0: 0.5})), 3.5, places=6)

    def test_stream_accumulator_state_roundtrip_preserves_vector_sum(self):
        task_a = atr.StreamAccumulatorTask("accum_state_vec_node", {}, block_size=64)
        task_a.scene_frame_start = 1
        task_a._current_frame_request = 1
        _ = task_a.process({0: np.array([1.0, 2.0], dtype=np.float32)})
        task_a._current_frame_request = 2
        _ = task_a.process({0: np.array([0.5, -1.0], dtype=np.float32)})

        task_b = atr.StreamAccumulatorTask("accum_state_vec_node", {}, block_size=64)
        task_b.scene_frame_start = 1
        self.assertTrue(task_b.import_state(dict(task_a.export_state() or {})))
        task_b._current_frame_request = 3
        out = np.asarray(task_b.process({0: np.array([0.25, 0.75], dtype=np.float32)}), dtype=np.float32)
        self.assertTrue(np.allclose(out, np.array([1.75, 1.75], dtype=np.float32)))

    def test_band_rms_of_a_tone_inside_the_band_equals_its_rms(self):
        sr, n = 48000, 2000
        t = np.arange(3 * n, dtype=np.float64) / sr
        signal = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        task = atr.RMSTask("rms_node", {}, block_size=n)
        expected = 0.5 / np.sqrt(2.0)
        for history in (None, signal[:2 * n]):
            chunk = atr.AudioChunk(
                signal[2 * n:], None, sr, band_hz_bounds=(500.0, 2000.0), lineage=("src",), history_mono=history,
            )
            self.assertAlmostEqual(float(task.process({0: chunk})), expected, delta=0.01 * expected)

    def test_band_rms_counts_a_click_on_a_block_boundary_once(self):
        sr, n = 48000, 2000
        signal = np.zeros(4 * n, dtype=np.float32)
        click = (0.8 * np.sin(2.0 * np.pi * 2000.0 * np.arange(96) / sr)).astype(np.float32)
        task = atr.RMSTask("rms_node", {}, block_size=n)
        for start in (2 * n, 2 * n - 48):  # just after a boundary, and across it
            signal[:] = 0.0
            signal[start:start + click.size] = click
            energy = 0.0
            for k in range(1, 4):
                chunk = atr.AudioChunk(
                    signal[k * n:(k + 1) * n], None, sr, band_hz_bounds=(20.0, 20000.0), lineage=("src",),
                    history_mono=signal[:k * n], frame_index=k,
                )
                energy += float(task.process({0: chunk})) ** 2 * n
            self.assertAlmostEqual(energy, float(np.sum(click ** 2)), delta=0.02 * float(np.sum(click ** 2)))

    def test_rms_paths_avoid_overflow_for_hot_float32_chunks(self):
        hot = np.full(256, 1.0e20, dtype=np.float32)
        stereo = np.stack([hot, hot], axis=0)
        chunk = atr.AudioChunk(hot, stereo, 48000, lineage=("src",))

        rms_task = atr.RMSTask("rms_node", {}, block_size=256)
        bal_task = atr.StereoBalanceTask("balance_node", {}, block_size=256)
        vec_task = atr.StereoVectorTask("vector_node", {}, block_size=256)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            rms_val = float(rms_task.process({0: chunk}))
            bal_val = float(bal_task.process({0: chunk}))
            vec_val = vec_task.process({0: chunk})

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning) and "overflow encountered in square" in str(w.message)
        ]
        self.assertFalse(overflow_warnings)
        self.assertTrue(np.isfinite(rms_val))
        self.assertGreater(rms_val, 0.0)
        self.assertTrue(np.isfinite(bal_val))
        self.assertEqual(len(vec_val), 3)
        self.assertTrue(np.isfinite(float(vec_val[0])))
        self.assertTrue(np.isfinite(float(vec_val[1])))
        self.assertTrue(np.isfinite(float(vec_val[2])))

    def test_lufs_task_reports_floor_for_silence(self):
        task = atr.LUFSTask("lufs_node", {}, block_size=512)
        chunk = atr.AudioChunk(np.zeros(512, dtype=np.float32), None, 48000, lineage=("src",))
        loudness = task.process({0: chunk})
        self.assertAlmostEqual(float(loudness), 0.0, places=6)

    def test_finite_copy_matches_nan_to_num_and_always_copies(self):
        src = np.array([1.0, np.nan, np.inf, -np.inf, -2.5], dtype=np.float32)
        out = atr._finite_copy(src)
        self.assertTrue(np.array_equal(out, np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)))
        self.assertFalse(np.shares_memory(out, src))
        clean = np.array([0.5, -0.5], dtype=np.float32)
        self.assertFalse(np.shares_memory(atr._finite_copy(clean), clean))
        self.assertEqual(atr._finite_copy(np.array([1.0], dtype=np.float64), dtype=None).dtype, np.float64)
        self.assertEqual(atr._finite_copy([1, 2]).dtype, np.float32)

    def test_lufs_task_increases_with_signal_level(self):
        n = 2048
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        low = 0.05 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        high = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        task = atr.LUFSTask("lufs_node", {}, block_size=n)

        task._current_frame_request = 1
        low_lufs = task.process({0: atr.AudioChunk(low, None, sr, lineage=("src", "low"))})
        task._current_frame_request = 2
        high_lufs = task.process({0: atr.AudioChunk(high, None, sr, lineage=("src", "high"))})

        self.assertGreater(float(high_lufs), float(low_lufs))

    def test_lufs_follows_a_level_change_on_the_next_block(self):
        n = 1024  # about one frame at 48 kHz / 47 fps
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        task = atr.LUFSTask("lufs_node", {}, block_size=n)
        values = []
        for frame in range(1, 9):
            level = 0.05 if frame <= 4 else 0.5
            task._current_frame_request = frame
            values.append(float(task.process({0: atr.AudioChunk(level * tone, None, sr, lineage=("src", frame))})))
        # Each block after the jump moves the output; nothing is held for a hop.
        self.assertGreater(values[4], values[3])
        self.assertGreater(values[5], values[4])
        self.assertGreater(values[6], values[5])

    def test_lufs_task_accepts_window_input(self):
        n = 1024
        sr = 44100
        t = np.arange(n, dtype=np.float32) / float(sr)
        signal = 0.25 * np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        task = atr.LUFSTask("lufs_node", {}, block_size=n)
        loudness = task.process(
            {
                0: atr.AudioChunk(signal, None, sr, lineage=("src",)),
                1: 0.8,   # window seconds
            }
        )
        self.assertTrue(np.isfinite(float(loudness)))
        self.assertGreaterEqual(float(loudness), 0.0)
        self.assertLessEqual(float(loudness), 1.0)

    def test_peak_task_reports_floor_for_silence(self):
        task = atr.PeakTask("peak_node", {}, block_size=512)
        chunk = atr.AudioChunk(np.zeros(512, dtype=np.float32), None, 48000, lineage=("src",))
        peak = task.process({0: chunk})
        self.assertAlmostEqual(float(peak), 0.0, places=6)

    def test_peak_task_increases_with_signal_level(self):
        n = 2048
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        low = 0.05 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        high = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        task = atr.PeakTask("peak_node", {}, block_size=n)

        task._current_frame_request = 1
        low_peak = task.process({0: atr.AudioChunk(low, None, sr, lineage=("src", "low"))})
        task._current_frame_request = 2
        high_peak = task.process({0: atr.AudioChunk(high, None, sr, lineage=("src", "high"))})

        self.assertGreater(float(high_peak), float(low_peak))
        self.assertGreaterEqual(float(high_peak), 0.0)
        self.assertLessEqual(float(high_peak), 1.0)

    def test_preview_sink_audibly_masks_frequency_band(self):
        sr = 48000
        n = 1024
        t = np.arange(n, dtype=np.float32) / float(sr)
        low = np.sin(2.0 * np.pi * 200.0 * t).astype(np.float32)
        high = np.sin(2.0 * np.pi * 5000.0 * t).astype(np.float32)
        mixed = (low + high).astype(np.float32)
        chunk = atr.AudioChunk(mixed, None, sr, band_hz_bounds=(120.0, 400.0), lineage=("src",))

        task = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        out = task.process({0: chunk})

        spec = np.abs(np.fft.rfft(out.mono))
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
        low_idx = int(np.argmin(np.abs(freqs - 200.0)))
        high_idx = int(np.argmin(np.abs(freqs - 5000.0)))
        self.assertGreater(float(spec[low_idx]), float(spec[high_idx]) * 5.0)

    def test_preview_sink_sparse_indices_uses_masked_fft(self):
        sr = 48000
        n = 1024
        t = np.arange(n, dtype=np.float32) / float(sr)
        tone_low = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        tone_mid = np.sin(2.0 * np.pi * 1100.0 * t).astype(np.float32)
        tone_high = np.sin(2.0 * np.pi * 5200.0 * t).astype(np.float32)
        mixed = (tone_low + tone_mid + tone_high).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
        idx_mid = int(np.argmin(np.abs(freqs - 1100.0)))
        idx_high = int(np.argmin(np.abs(freqs - 5200.0)))
        chunk = atr.AudioChunk(
            mixed,
            None,
            sr,
            band_indices=np.array([idx_mid, idx_high], dtype=np.int64),
            lineage=("src",),
        )

        task = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        out = task.process({0: chunk})

        spec = np.abs(np.fft.rfft(out.mono))
        low_idx = int(np.argmin(np.abs(freqs - 220.0)))
        self.assertGreater(float(spec[idx_mid]), float(spec[low_idx]) * 1.8)
        self.assertGreater(float(spec[idx_high]), float(spec[low_idx]) * 1.8)

    def test_preview_sink_gain_staging_avoids_hard_clipping(self):
        sr = 48000
        n = 1024
        t = np.arange(n, dtype=np.float32) / float(sr)
        # Hot signal into a narrow band to exercise preview make-up gain.
        low = (0.95 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        high = (0.95 * np.sin(2.0 * np.pi * 3500.0 * t)).astype(np.float32)
        mixed = (low + high).astype(np.float32)
        chunk = atr.AudioChunk(mixed, None, sr, band_hz_bounds=(180.0, 260.0), lineage=("src",))

        task = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        out = task.process({0: chunk})
        peak = float(np.max(np.abs(out.mono))) if out.mono.size > 0 else 0.0
        self.assertLessEqual(peak, 0.985)

    def test_source_frontend_signed_smoothing_preserves_polarity_and_reduces_jitter(self):
        processor = atr.SourceFrontendProcessor()
        mono = np.array([1.0, -1.0] * 64, dtype=np.float32)
        analysis_mono, analysis_stereo = processor.process(
            mono=mono,
            stereo=None,
            sample_rate=48000,
            frontend_config=(0.0, 0.0, 5.0, 50.0),
        )

        self.assertIsNone(analysis_stereo)
        self.assertGreater(float(np.max(analysis_mono)), 0.0)
        self.assertLess(float(np.min(analysis_mono)), 0.0)
        self.assertLess(
            float(np.mean(np.abs(np.diff(analysis_mono)))),
            float(np.mean(np.abs(np.diff(mono)))),
        )
        self.assertTrue(np.all(np.sign(analysis_mono[1:]) != np.sign(analysis_mono[:-1])))

    def test_preview_sink_uses_raw_audio_track_when_analysis_frontend_is_present(self):
        n = 256
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        raw_stereo = np.stack(
            [
                np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32),
                np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32),
            ],
            axis=0,
        )
        raw_mono = np.mean(raw_stereo, axis=0).astype(np.float32)
        processor = atr.SourceFrontendProcessor()
        analysis_mono, analysis_stereo = processor.process(
            mono=raw_mono,
            stereo=raw_stereo,
            sample_rate=sr,
            frontend_config=(120.0, 800.0, 10.0, 80.0),
        )
        chunk = atr.AudioChunk(
            raw_mono,
            raw_stereo,
            sr,
            analysis_mono=analysis_mono,
            analysis_stereo=analysis_stereo,
            frontend_config=(120.0, 800.0, 10.0, 80.0),
            lineage=("src",),
            spectrum_token=("src", "frame"),
        )

        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        preview_out = preview.process({0: chunk})
        analysis_view, analysis_stereo_view = atr._analysis_signals_normalized(chunk)

        self.assertTrue(np.allclose(preview_out.mono, raw_mono, atol=1e-6))
        self.assertFalse(np.allclose(analysis_view, raw_mono, atol=1e-3))
        self.assertIsNotNone(analysis_stereo_view)
        self.assertLess(float(np.min(analysis_view)), -0.01)

    def test_source_task_hot_reload_on_runtime_config_change(self):
        load_calls = []

        class DummyReception:
            def __init__(self, source_type, filepath="", flow_id="", live_latency_ms=50):
                load_calls.append((str(source_type), str(filepath), str(flow_id), int(live_latency_ms)))
                self.source_type = source_type
                self.filepath = filepath
                self.flow_id = flow_id
                self.live_latency_ms = live_latency_ms
                self.channels = 2
                self.sample_rate = 48000
                self.duration = 0.0
                self.last_error = ""
                self.kind = "file_reader"
                self._cursor_sample = 0
                self._eof = False
                self.reader = object()

            def open(self, block_size=None, target_sample_rate=None):  # noqa: ARG002
                return True

            def read_block(self, block_size):
                n = int(block_size)
                mono = np.ones(n, dtype=np.float32)
                self._cursor_sample += n
                return mono, None, int(self.sample_rate)

            def seek(self, sample):
                self._cursor_sample = int(sample)
                return True

            def close(self):
                return None

        class DummyCtx:
            source_type = "FILE"
            filepath = "A.wav"
            flow_identifier = ""
            live_latency_ms = 50
            sample_rate = 48000
            duration = 1.0
            live_transport = "SOUNDDEVICE"
            live_source_kind = "DAW"
            live_device_selector = "__AUTO__"
            live_udp_port = 7001
            live_udp_channels = 2
            live_udp_sample_rate = 48000

            def __setitem__(self, key, value):  # noqa: ARG002
                return None

        original_reception = tasks_source.AudioReception
        tasks_source.AudioReception = DummyReception
        task = None
        try:
            task = atr.SourceTask("src_node", {}, block_size=1024)
            task.scene_fps = 24.0
            task.source_context = DummyCtx()
            out_a = task._read_next_block()
            self.assertIsNotNone(out_a)
            self.assertGreaterEqual(len(load_calls), 1)

            task.source_context.filepath = "B.wav"
            out_b = task._read_next_block()
            self.assertIsNotNone(out_b)
            self.assertGreaterEqual(len(load_calls), 2)
            self.assertEqual(load_calls[-1][1], "B.wav")
        finally:
            try:
                task.close()
            except Exception:
                pass
            tasks_source.AudioReception = original_reception

    def test_resolve_context_block_size_adaptive_handles_frame_jumps(self):
        sr = 48000
        fps = 24.0
        block_1x = atr.resolve_context_block_size(sr, fps, source_type="FILE", timeslice_mode="adaptive", delta_frames=1.0)
        block_2x = atr.resolve_context_block_size(sr, fps, source_type="FILE", timeslice_mode="adaptive", delta_frames=2.0)
        block_0x = atr.resolve_context_block_size(sr, fps, source_type="FILE", timeslice_mode="adaptive", delta_frames=0.0)
        block_rev = atr.resolve_context_block_size(sr, fps, source_type="FILE", timeslice_mode="adaptive", delta_frames=-1.0)

        self.assertGreaterEqual(block_1x, atr.MIN_BLOCK_SIZE)
        self.assertLessEqual(block_2x, atr.MAX_BLOCK_SIZE)
        self.assertGreaterEqual(block_2x, block_1x)
        self.assertEqual(block_0x, block_1x)
        self.assertEqual(block_rev, block_1x)

        live_1x = atr.resolve_context_block_size(sr, fps, source_type="LIVE", timeslice_mode="adaptive", delta_frames=1.0)
        live_2x = atr.resolve_context_block_size(sr, fps, source_type="LIVE", timeslice_mode="adaptive", delta_frames=2.0)
        self.assertGreaterEqual(live_1x, atr.MIN_BLOCK_SIZE)
        self.assertGreaterEqual(live_2x, live_1x)
        self.assertLessEqual(live_2x, atr.MAX_BLOCK_SIZE)

    def test_preview_sink_sparse_mask_remaps_band_index_fft_size(self):
        sr = 48000
        src_n = 1024
        dst_n = 960
        src_freqs = np.fft.rfftfreq(src_n, d=1.0 / float(sr))
        dst_freqs = np.fft.rfftfreq(dst_n, d=1.0 / float(sr))
        target_hz = 1500.0
        src_idx = int(np.argmin(np.abs(src_freqs - target_hz)))
        chunk = atr.AudioChunk(
            np.random.randn(dst_n).astype(np.float32),
            None,
            sr,
            band_indices=np.array([src_idx], dtype=np.int64),
            band_index_fft_size=src_n,
            lineage=("src",),
        )
        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=dst_n)
        mask = preview._resolve_sparse_mask(chunk, signal_length=dst_n)
        self.assertIsNotNone(mask)
        mapped_idx = int(np.argmax(mask.astype(np.int32)))
        self.assertLess(abs(float(dst_freqs[mapped_idx]) - target_hz), 80.0)

    def test_preview_sink_passthrough_without_masks(self):
        n = 512
        chunk = atr.AudioChunk(np.linspace(-1.0, 1.0, n, dtype=np.float32), None, 48000, lineage=("src",))
        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        out = preview.process({0: chunk})
        self.assertIs(out, chunk)


if __name__ == "__main__":
    unittest.main()
