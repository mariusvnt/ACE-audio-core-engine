import unittest
import warnings

from module_loader import load_module

try:
    import numpy as np
except Exception:
    np = None

try:
    atr = load_module("audio_tasks_registry") if np is not None else None
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

    def test_analyzer_resolves_hz_bounds_in_fft_domain(self):
        analyzer = atr.AudioFluxAnalyzer(sample_rate=48000, block_size=750)
        chunk = atr.AudioChunk(
            np.zeros(750, dtype=np.float32),
            None,
            48000,
            band_hz_bounds=(1000.0, 5000.0),
            lineage=("src", "band"),
        )
        idx = analyzer._resolve_chunk_mask_indices(chunk)
        self.assertIsNotNone(idx)
        self.assertGreater(idx.size, 0)
        freq = analyzer.freq_bins[idx]
        self.assertGreaterEqual(float(freq.min()), 1000.0)
        self.assertLess(float(freq.max()), 5000.0)

    def test_analyzer_intersects_existing_indices_and_bounds(self):
        analyzer = atr.AudioFluxAnalyzer(sample_rate=48000, block_size=256)
        base_indices = np.array([8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
        chunk = atr.AudioChunk(
            np.zeros(256, dtype=np.float32),
            None,
            48000,
            band_indices=base_indices,
            band_hz_bounds=(2000.0, 2500.0),
            lineage=("src", "pre_masked"),
        )
        idx = analyzer._resolve_chunk_mask_indices(chunk)
        self.assertIsNotNone(idx)
        self.assertGreater(idx.size, 0)
        self.assertTrue(set(idx.tolist()).issubset(set(base_indices.tolist())))
        freq = analyzer.freq_bins[idx]
        self.assertGreaterEqual(float(freq.min()), 2000.0)
        self.assertLess(float(freq.max()), 2500.0)

    def test_analyzer_remaps_foreign_fft_domain_indices(self):
        analyzer = atr.AudioFluxAnalyzer(sample_rate=48000, block_size=750)
        src_fft_size = 750
        src_freqs = np.fft.rfftfreq(src_fft_size, d=1.0 / 48000.0)
        src_idx = np.array([
            int(np.argmin(np.abs(src_freqs - 1000.0))),
            int(np.argmin(np.abs(src_freqs - 3500.0))),
        ], dtype=np.int64)
        chunk = atr.AudioChunk(
            np.zeros(750, dtype=np.float32),
            None,
            48000,
            band_indices=src_idx,
            band_index_fft_size=src_fft_size,
            lineage=("src",),
        )
        idx = analyzer._resolve_chunk_mask_indices(chunk)
        self.assertIsNotNone(idx)
        self.assertGreater(idx.size, 0)
        out_freqs = analyzer.freq_bins[idx]
        self.assertTrue(np.any(np.abs(out_freqs - 1000.0) < 40.0))
        self.assertTrue(np.any(np.abs(out_freqs - 3500.0) < 40.0))

    def test_chunk_token_uses_lineage(self):
        analyzer = atr.AudioFluxAnalyzer(sample_rate=44100, block_size=128)
        mono = np.ones(128, dtype=np.float32)
        a = atr.AudioChunk(mono, None, 44100, lineage=("src", "a"))
        b = atr.AudioChunk(mono, None, 44100, lineage=("src", "b"))
        token_a = analyzer._chunk_token(a)
        token_b = analyzer._chunk_token(b)
        self.assertNotEqual(token_a, token_b)

    def test_chunk_token_prefers_explicit_spectrum_token(self):
        analyzer = atr.AudioFluxAnalyzer(sample_rate=44100, block_size=128)
        mono = np.ones(128, dtype=np.float32)
        a = atr.AudioChunk(mono, None, 44100, lineage=("src", "a"), spectrum_token=("src", "shared"))
        b = atr.AudioChunk(mono, None, 44100, lineage=("src", "b"), spectrum_token=("src", "shared"))
        self.assertEqual(analyzer._chunk_token(a), analyzer._chunk_token(b))

    def test_chunk_token_distinguishes_analysis_ops_for_shared_token(self):
        analyzer = atr.AudioFluxAnalyzer(sample_rate=44100, block_size=128)
        mono = np.ones(128, dtype=np.float32)
        shared = ("src", "shared")
        mid_chunk = atr.AudioChunk(
            mono,
            None,
            44100,
            lineage=("src", "mid"),
            spectrum_token=shared,
            analysis_ops=(("midside", "mid"),),
        )
        side_chunk = atr.AudioChunk(
            mono,
            None,
            44100,
            lineage=("src", "side"),
            spectrum_token=shared,
            analysis_ops=(("midside", "side"),),
        )
        self.assertNotEqual(analyzer._chunk_token(mid_chunk), analyzer._chunk_token(side_chunk))

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

    def test_midside_splitter_emits_metadata_only_branches(self):
        n = 256
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        left = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        right = (-left).astype(np.float32)
        stereo = np.stack([left, right], axis=0)
        mono = np.mean(stereo, axis=0).astype(np.float32)
        source = atr.AudioChunk(
            mono,
            stereo,
            sr,
            lineage=("src",),
            spectrum_token=("src", "frame"),
            band_hz_bounds=(100.0, 2000.0),
        )

        splitter = atr.MidsideSplitterTask("ms_node", {}, block_size=n)
        mid_chunk, side_chunk = splitter.process({0: source})

        self.assertTrue(np.array_equal(mid_chunk.mono, source.mono))
        self.assertTrue(np.array_equal(side_chunk.mono, source.mono))
        self.assertEqual(mid_chunk.spectrum_token, source.spectrum_token)
        self.assertEqual(side_chunk.spectrum_token, source.spectrum_token)
        self.assertEqual(mid_chunk.band_hz_bounds, source.band_hz_bounds)
        self.assertEqual(side_chunk.band_hz_bounds, source.band_hz_bounds)
        self.assertEqual(mid_chunk.analysis_ops[-1], ("midside", "mid"))
        self.assertEqual(side_chunk.analysis_ops[-1], ("midside", "side"))

    def test_attackrelease_splitter_emits_metadata_only_branches(self):
        n = 64
        sr = 48000
        mono = np.array([0.0, 0.8, 0.2, 0.9, 0.3, 0.1, 0.7, 0.4], dtype=np.float32)
        mono = np.tile(mono, n // mono.shape[0]).astype(np.float32)
        source = atr.AudioChunk(
            mono,
            None,
            sr,
            lineage=("src",),
            spectrum_token=("src", "frame"),
            band_indices=np.array([1, 2, 3], dtype=np.int64),
        )

        splitter = atr.AttackReleaseSplitterTask("ar_node", {}, block_size=n)
        attack_chunk, release_chunk = splitter.process({0: source})

        self.assertTrue(np.array_equal(attack_chunk.mono, source.mono))
        self.assertTrue(np.array_equal(release_chunk.mono, source.mono))
        self.assertEqual(attack_chunk.spectrum_token, source.spectrum_token)
        self.assertEqual(release_chunk.spectrum_token, source.spectrum_token)
        self.assertTrue(np.array_equal(attack_chunk.band_indices, source.band_indices))
        self.assertTrue(np.array_equal(release_chunk.band_indices, source.band_indices))
        self.assertEqual(attack_chunk.analysis_ops[-1], ("attackrelease", "attack"))
        self.assertEqual(release_chunk.analysis_ops[-1], ("attackrelease", "release"))

    def test_audio_math_add_mode(self):
        n = 8
        a = atr.AudioChunk(np.full(n, 1.0, dtype=np.float32), None, 48000, lineage=("src", "a"))
        b = atr.AudioChunk(np.full(n, 2.0, dtype=np.float32), None, 48000, lineage=("src", "b"))
        task = atr.AudioMathTask("math_node", {"op": "add"}, block_size=n)
        out = task.process({0: a, 1: b})
        self.assertTrue(np.allclose(out.mono, np.full(n, 3.0, dtype=np.float32)))

    def test_audio_math_subtract_mode(self):
        n = 8
        a = atr.AudioChunk(np.full(n, 5.0, dtype=np.float32), None, 48000, lineage=("src", "a"))
        b = atr.AudioChunk(np.full(n, 2.0, dtype=np.float32), None, 48000, lineage=("src", "b"))
        c = atr.AudioChunk(np.full(n, 1.0, dtype=np.float32), None, 48000, lineage=("src", "c"))
        task = atr.AudioMathTask("math_node", {"op": "subtract"}, block_size=n)
        out = task.process({0: a, 1: b, 2: c})
        self.assertTrue(np.allclose(out.mono, np.full(n, 2.0, dtype=np.float32)))

    def test_audio_math_multiply_mode(self):
        n = 8
        a = atr.AudioChunk(np.full(n, 3.0, dtype=np.float32), None, 48000, lineage=("src", "a"))
        b = atr.AudioChunk(np.full(n, 2.0, dtype=np.float32), None, 48000, lineage=("src", "b"))
        task = atr.AudioMathTask("math_node", {"op": "multiply"}, block_size=n)
        out = task.process({0: a, 1: b})
        self.assertTrue(np.allclose(out.mono, np.full(n, 6.0, dtype=np.float32)))

    def test_audio_math_add_mode_preserves_stereo(self):
        n = 16
        left_a = np.linspace(0.0, 1.0, n, dtype=np.float32)
        right_a = np.linspace(1.0, 0.0, n, dtype=np.float32)
        stereo_a = np.stack([left_a, right_a], axis=0).astype(np.float32, copy=False)
        mono_a = np.mean(stereo_a, axis=0).astype(np.float32, copy=False)

        left_b = np.linspace(0.5, -0.5, n, dtype=np.float32)
        right_b = np.linspace(-0.25, 0.75, n, dtype=np.float32)
        stereo_b = np.stack([left_b, right_b], axis=0).astype(np.float32, copy=False)
        mono_b = np.mean(stereo_b, axis=0).astype(np.float32, copy=False)

        a = atr.AudioChunk(mono_a, stereo_a, 48000, lineage=("src", "a"))
        b = atr.AudioChunk(mono_b, stereo_b, 48000, lineage=("src", "b"))

        task = atr.AudioMathTask("math_node", {"op": "add"}, block_size=n)
        out = task.process({0: a, 1: b})

        self.assertIsNotNone(out.stereo)
        self.assertEqual(out.stereo.shape, (2, n))
        self.assertTrue(np.allclose(out.stereo[0], left_a + left_b))
        self.assertTrue(np.allclose(out.stereo[1], right_a + right_b))
        self.assertTrue(np.allclose(out.mono, np.mean(out.stereo, axis=0)))

    def test_audio_math_merges_mask_metadata_from_all_inputs(self):
        n = 256
        a = atr.AudioChunk(
            np.ones(n, dtype=np.float32),
            None,
            48000,
            lineage=("src", "a"),
            band_indices=np.array([2, 4, 6], dtype=np.int64),
            band_index_fft_size=n,
            band_hz_bounds=(100.0, 1200.0),
        )
        b = atr.AudioChunk(
            np.ones(n, dtype=np.float32) * 2.0,
            None,
            48000,
            lineage=("src", "b"),
            band_indices=np.array([6, 8, 10], dtype=np.int64),
            band_index_fft_size=n,
            band_hz_bounds=(900.0, 3200.0),
        )
        task = atr.AudioMathTask("math_node", {"op": "add"}, block_size=n)
        out = task.process({0: a, 1: b})

        self.assertTrue(np.array_equal(out.band_indices, np.array([2, 4, 6, 8, 10], dtype=np.int64)))
        self.assertEqual(out.band_index_fft_size, n)
        self.assertEqual(out.band_hz_bounds, (100.0, 3200.0))

    def test_audio_math_handles_mismatched_input_lengths(self):
        a = atr.AudioChunk(np.full(512, 1.0, dtype=np.float32), None, 48000, lineage=("src", "a"))
        b = atr.AudioChunk(np.full(1024, 2.0, dtype=np.float32), None, 48000, lineage=("src", "b"))
        task = atr.AudioMathTask("math_node", {"op": "add"}, block_size=1024)
        out = task.process({0: a, 1: b})

        self.assertEqual(out.mono.shape[0], 1024)
        self.assertTrue(np.allclose(out.mono[:512], 3.0))
        self.assertTrue(np.allclose(out.mono[512:], 2.0))

    def test_zero_cross_task_uses_numpy_fallback(self):
        signal = np.array([1.0, -1.0] * 16, dtype=np.float32)
        chunk = atr.AudioChunk(signal, None, 48000, lineage=("src",))
        task = atr.ZeroCrossTask("zcr_node", {}, block_size=signal.shape[0])
        out = task.process({0: chunk})
        self.assertGreater(out, 0.9)

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

        original_reception = atr.AudioReception
        atr.AudioReception = DummyReception
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
            atr.AudioReception = original_reception

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

    def test_source_specs_outputs_frames_per_beat(self):
        class DummyCtx:
            source_type = "FILE"
            filepath = ""
            bpm = 120.0
            beats_per_bar = 4
            sample_rate = 48000
            duration = 10.0

        class DummyEngine:
            current_frame = 12

        task = atr.SourceSpecsTask("specs_node", {}, block_size=256)
        task.source_context = DummyCtx()
        task.scene_fps = 24.0
        task.engine_ref = DummyEngine()

        chunk = atr.AudioChunk(np.ones(256, dtype=np.float32), None, 48000, lineage=("src",))
        frames_per_beat, beat_phase, _ = task.process({0: chunk})

        self.assertAlmostEqual(float(frames_per_beat), (60.0 * 24.0) / 120.0, places=6)
        self.assertAlmostEqual(float(beat_phase), 0.25, places=6)

    def test_rolling_average_uses_fixed_frame_count(self):
        task = atr.RollingAvgTask("rolling_node", {"in_1": 3}, block_size=128)

        self.assertAlmostEqual(float(task.process({0: 1.0})), 1.0, places=6)
        self.assertAlmostEqual(float(task.process({0: 2.0})), 1.5, places=6)
        self.assertAlmostEqual(float(task.process({0: 3.0})), 2.0, places=6)
        self.assertAlmostEqual(float(task.process({0: 9.0})), (2.0 + 3.0 + 9.0) / 3.0, places=6)
        self.assertAlmostEqual(float(task.process({0: 5.0, 1: 2})), 7.0, places=6)

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

    def test_masked_rms_prefers_provider_feature_path(self):
        calls = []

        class DummyProvider:
            def compute_feature(self, name, chunk):  # noqa: ARG002
                calls.append(name)
                if name == "rms":
                    return 0.1234
                return None

        chunk = atr.AudioChunk(
            np.ones(256, dtype=np.float32),
            None,
            48000,
            provider=DummyProvider(),
            band_hz_bounds=(100.0, 2000.0),
            lineage=("src",),
        )
        task = atr.RMSTask("rms_node", {}, block_size=256)
        out = task.process({0: chunk})
        self.assertAlmostEqual(float(out), 0.1234, places=6)
        self.assertIn("rms", calls)

    def test_rms_paths_avoid_overflow_for_hot_float32_chunks(self):
        hot = np.full(256, 1.0e20, dtype=np.float32)
        stereo = np.stack([hot, hot], axis=0)
        chunk = atr.AudioChunk(hot, stereo, 48000, lineage=("src",))

        rms_task = atr.RMSTask("rms_node", {}, block_size=256)
        bal_task = atr.StereoBalanceTask("balance_node", {}, block_size=256)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            rms_val = float(rms_task.process({0: chunk}))
            bal_val = float(bal_task.process({0: chunk}))

        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning) and "overflow encountered in square" in str(w.message)
        ]
        self.assertFalse(overflow_warnings)
        self.assertTrue(np.isfinite(rms_val))
        self.assertGreater(rms_val, 0.0)
        self.assertTrue(np.isfinite(bal_val))

    def test_logic_task_thresholds_scalar_stream(self):
        task = atr.LogicTask("logic_node", {}, block_size=128)
        self.assertEqual(float(task.process({0: 0.25, 1: 0.5})), 0.0)
        self.assertEqual(float(task.process({0: 0.5, 1: 0.5})), 1.0)
        self.assertEqual(float(task.process({0: 0.75, 1: 0.5})), 1.0)

    def test_lufs_task_reports_floor_for_silence(self):
        task = atr.LUFSTask("lufs_node", {}, block_size=512)
        chunk = atr.AudioChunk(np.zeros(512, dtype=np.float32), None, 48000, lineage=("src",))
        loudness = task.process({0: chunk})
        self.assertAlmostEqual(float(loudness), 0.0, places=6)

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

    def test_lufs_task_accepts_window_and_overlap_inputs(self):
        n = 1024
        sr = 44100
        t = np.arange(n, dtype=np.float32) / float(sr)
        signal = 0.25 * np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        task = atr.LUFSTask("lufs_node", {}, block_size=n)
        loudness = task.process(
            {
                0: atr.AudioChunk(signal, None, sr, lineage=("src",)),
                1: 0.8,   # window seconds
                2: 0.2,   # overlap ratio (0..1)
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

    def test_splitter_then_chromagram_preview_respects_parent_band_with_variable_chunk_size(self):
        sr = 48000
        chunk_n = 960
        task_block = 1024
        t = np.arange(chunk_n, dtype=np.float32) / float(sr)
        tone_low = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)   # A
        tone_high = np.sin(2.0 * np.pi * 1760.0 * t).astype(np.float32)  # A one octave stack
        mixed = (tone_low + tone_high).astype(np.float32)

        src = atr.AudioChunk(mixed, None, sr, lineage=("src",))
        splitter = atr.FrequencySplitterTask("split_node", {"in_1": 400.0}, block_size=task_block)
        low_band = splitter.process({0: src})[0]

        chroma = atr.ChromagramTask("chroma_node", {}, block_size=task_block)
        chroma_a = chroma.process({0: low_band})[9]  # A pitch class

        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=task_block)
        out = preview.process({0: chroma_a})

        freqs = np.fft.rfftfreq(chunk_n, d=1.0 / float(sr))
        spec = np.abs(np.fft.rfft(out.mono))
        low_idx = int(np.argmin(np.abs(freqs - 220.0)))
        high_idx = int(np.argmin(np.abs(freqs - 1760.0)))
        self.assertGreater(float(spec[low_idx]), float(spec[high_idx]) * 3.0)

    def test_chromagram_fallback_keeps_sparse_mask_non_empty_in_coarse_band(self):
        sr = 48000
        chunk_n = 960
        task_block = 1024
        t = np.arange(chunk_n, dtype=np.float32) / float(sr)
        signal = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        src = atr.AudioChunk(signal, None, sr, lineage=("src",))
        splitter = atr.FrequencySplitterTask("split_node", {"in_1": 400.0}, block_size=task_block)
        low_band = splitter.process({0: src})[0]

        chroma = atr.ChromagramTask("chroma_node", {}, block_size=task_block)
        chroma_a = chroma.process({0: low_band})[9]
        self.assertIsNotNone(chroma_a.band_indices)
        self.assertGreater(int(chroma_a.band_indices.size), 0)

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

    def test_preview_sink_reconstructs_midside_branches_from_analysis_ops(self):
        sr = 48000
        n = 1024
        t = np.arange(n, dtype=np.float32) / float(sr)
        left = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        right = (-left).astype(np.float32)
        stereo = np.stack([left, right], axis=0)
        mono = np.mean(stereo, axis=0).astype(np.float32)
        src = atr.AudioChunk(mono, stereo, sr, lineage=("src",), spectrum_token=("src", "frame"))

        splitter = atr.MidsideSplitterTask("ms_node", {}, block_size=n)
        mid_chunk, side_chunk = splitter.process({0: src})
        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)

        mid_out = preview.process({0: mid_chunk})
        side_out = preview.process({0: side_chunk})

        self.assertLess(float(np.max(np.abs(mid_out.mono))), 1e-4)
        self.assertGreater(float(np.max(np.abs(side_out.mono))), 0.2)
        self.assertEqual(mid_out.analysis_ops, ())
        self.assertEqual(side_out.analysis_ops, ())

    def test_preview_sink_reconstructs_attackrelease_branches_from_analysis_ops(self):
        n = 512
        sr = 48000
        pattern = np.array([0.0, 1.0, 0.3, 0.2, 0.9, 0.1, 0.8, 0.0], dtype=np.float32)
        mono = np.tile(pattern, n // pattern.shape[0]).astype(np.float32)
        src = atr.AudioChunk(mono, None, sr, lineage=("src",), spectrum_token=("src", "frame"))

        splitter = atr.AttackReleaseSplitterTask("ar_node", {}, block_size=n)
        attack_chunk, release_chunk = splitter.process({0: src})
        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)

        attack_out = preview.process({0: attack_chunk})
        release_out = preview.process({0: release_chunk})

        recombined = attack_out.mono + release_out.mono
        self.assertTrue(np.allclose(recombined, mono, atol=1e-6))
        self.assertEqual(attack_out.analysis_ops, ())
        self.assertEqual(release_out.analysis_ops, ())

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
        self.assertGreater(atr._zero_crossing_rate(analysis_mono), 0.9)

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

        original_reception = atr.AudioReception
        atr.AudioReception = DummyReception
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
            atr.AudioReception = original_reception

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

    def test_preview_sink_passthrough_without_masks_or_ops(self):
        n = 512
        chunk = atr.AudioChunk(np.linspace(-1.0, 1.0, n, dtype=np.float32), None, 48000, lineage=("src",))
        preview = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        out = preview.process({0: chunk})
        self.assertIs(out, chunk)

    def test_audio_math_caches_analysis_extraction_per_input(self):
        original_extract = atr._analysis_signal_for_chunk
        calls = {"count": 0}

        def _counting_extract(chunk):
            calls["count"] += 1
            return original_extract(chunk)

        atr._analysis_signal_for_chunk = _counting_extract
        try:
            n = 128
            task = atr.AudioMathTask("math_node", {"op": "add"}, block_size=n)
            a = atr.AudioChunk(np.full(n, 1.0, dtype=np.float32), None, 48000, lineage=("src", "a"))
            b = atr.AudioChunk(np.full(n, 2.0, dtype=np.float32), None, 48000, lineage=("src", "b"))
            c = atr.AudioChunk(np.full(n, 3.0, dtype=np.float32), None, 48000, lineage=("src", "c"))
            out = task.process({0: a, 1: b, 2: c})
            self.assertEqual(out.mono.shape[0], n)
            self.assertEqual(calls["count"], 3)
        finally:
            atr._analysis_signal_for_chunk = original_extract


if __name__ == "__main__":
    unittest.main()
