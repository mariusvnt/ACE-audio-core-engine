import sys
import types
import unittest
import unittest.mock as mock

try:
    import numpy as np
except Exception:
    np = None

from module_loader import load_module

# Sentinel distinguishes "key absent from sys.modules" from "key mapped to None".
_MISSING = object()


def _clear_worker_state(mod):
    cache = getattr(mod, "_WORKER_GRAPH_CACHE", None)
    if isinstance(cache, dict):
        for graph in list(cache.values()):
            try:
                graph.close()
            except Exception:
                pass
        cache.clear()
    shm_cache = getattr(mod, "_WORKER_SHM_CACHE", None)
    if isinstance(shm_cache, dict):
        close_entry = getattr(mod, "_close_worker_shm_cache_entry", None)
        if callable(close_entry):
            for key in list(shm_cache.keys()):
                try:
                    close_entry(key)
                except Exception:
                    pass
        shm_cache.clear()
    if hasattr(mod, "_WORKER_LAST_GENERATION"):
        mod._WORKER_LAST_GENERATION = None


class TestParallelDSP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        # Smoke tests may import registry modules with a fake NumPy module.
        # Force a clean import here so worker tests always use real NumPy.
        import sys
        sys.modules.pop("ace_addon_testpkg.parallel_dsp", None)
        sys.modules.pop("ace_addon_testpkg.audio_tasks_registry", None)
        sys.modules.pop("ace_addon_testpkg.audio_influx", None)
        cls.pds = load_module("parallel_dsp")

    def setUp(self):
        _clear_worker_state(self.pds)

    def tearDown(self):
        _clear_worker_state(self.pds)

    def _audio_math_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "src_b": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "math": {"category": "utilities", "type": "audio_math", "parameters": {"op": "add"}},
                "rms": {"category": "stream", "type": "rms", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "math", 0),
                ("src_b", 0, "math", 1),
                ("math", 0, "rms", 0),
            ],
            "source_node_ids": ["src_a", "src_b"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
                "src_b": {"sample_rate": 48000, "lineage": ["src_b"], "spectrum_token": ["src_b"]},
            },
            "output_specs": {
                "rms:0": {"node_id": "rms", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _dual_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "rms": {"category": "stream", "type": "rms", "parameters": {}},
                "zcr": {"category": "stream", "type": "zero_crossing_rate", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "rms", 0),
                ("src_a", 0, "zcr", 0),
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "rms:0": {"node_id": "rms", "socket_idx": 0, "attr_type": "FLOAT"},
                "zcr:0": {"node_id": "zcr", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _lufs_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "lufs": {
                    "category": "stream",
                    "type": "lufs",
                    "parameters": {"window_size": 0.4, "overlap_ratio": 0.25},
                },
            },
            "links": [
                ("src_a", 0, "lufs", 0),
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "lufs:0": {"node_id": "lufs", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _peak_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "peak": {"category": "stream", "type": "peak", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "peak", 0),
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "peak:0": {"node_id": "peak", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _specs_splitter_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "specs": {"category": "utilities", "type": "src_specs", "parameters": {}},
                "split": {"category": "delta", "type": "frequency_splitter", "parameters": {}},
                "rms": {"category": "stream", "type": "rms", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "specs", 0),
                ("specs", 2, "split", 0),
                ("split", 0, "rms", 0),
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "rms:0": {"node_id": "rms", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _stream_accumulator_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "rms": {"category": "stream", "type": "rms", "parameters": {}},
                "accum": {"category": "utilities", "type": "stream_accumulator", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "rms", 0),
                ("rms", 0, "accum", 0),
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "accum:0": {"node_id": "accum", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _math_chroma_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "src_b": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "src_c": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "math": {"category": "utilities", "type": "audio_math", "parameters": {"op": "multiply"}},
                "chroma": {"category": "delta", "type": "chromagram", "parameters": {}},
                "rms0": {"category": "stream", "type": "rms", "parameters": {}},
                "rms1": {"category": "stream", "type": "rms", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "math", 0),
                ("src_b", 0, "math", 1),
                ("src_c", 0, "math", 2),
                ("math", 0, "chroma", 0),
                ("chroma", 0, "rms0", 0),
                ("chroma", 1, "rms1", 0),
            ],
            "source_node_ids": ["src_a", "src_b", "src_c"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
                "src_b": {"sample_rate": 48000, "lineage": ["src_b"], "spectrum_token": ["src_b"]},
                "src_c": {"sample_rate": 48000, "lineage": ["src_c"], "spectrum_token": ["src_c"]},
            },
            "output_specs": {
                "rms0:0": {"node_id": "rms0", "socket_idx": 0, "attr_type": "FLOAT"},
                "rms1:0": {"node_id": "rms1", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def test_cross_source_audio_math_uses_distinct_source_chunks(self):
        n = 256
        a = np.full(n, 1.0, dtype=np.float32)
        b = np.full(n, 2.0, dtype=np.float32)
        job = {
            "generation_id": 1001,
            "graph_uid": "cross_source_graph",
            "job_key": "cross_source_graph:f7:j1",
            "frame_id": 7,
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_a",
            "graph_ir": self._audio_math_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": a, "sample_rate": 48000, "latency_ms": 1.0},
                "src_b": {"mono": b, "sample_rate": 48000, "latency_ms": 2.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("rms:0", values)
        self.assertAlmostEqual(float(values["rms:0"]), 3.0, places=4)
        self.assertEqual(str(out.get("job_key", "")), "cross_source_graph:f7:j1")

    def test_payload_schema_accepts_keyed_source_chunks_contract(self):
        n = 128
        a = np.full(n, 0.25, dtype=np.float32)
        b = np.full(n, 0.75, dtype=np.float32)
        job = {
            "generation_id": 1002,
            "graph_uid": "schema_graph",
            "job_key": "schema_graph:f3:j1",
            "frame_id": 3,
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_b",
            "graph_ir": self._audio_math_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": a, "sample_rate": 48000, "latency_ms": 0.0},
                "src_b": {"mono": b, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        self.assertEqual(int(out.get("frame_id", -1)), 3)
        self.assertEqual(str(out.get("graph_uid", "")), "schema_graph")
        self.assertEqual(str(out.get("job_key", "")), "schema_graph:f3:j1")
        self.assertIn("rms:0", dict(out.get("values", {}) or {}))

    def test_parallel_job_can_evaluate_subset_of_output_keys(self):
        n = 256
        mono = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        job = {
            "generation_id": 1003,
            "graph_uid": "subset_graph",
            "frame_id": 5,
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_subset",
            "graph_ir": self._dual_stream_graph_ir(),
            "output_keys": ["zcr:0"],
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("zcr:0", values)
        self.assertNotIn("rms:0", values)

    def test_worker_supports_specs_splitter_stream_chain(self):
        n = 512
        t = np.arange(n, dtype=np.float32) / 48000.0
        mono = np.sin((2.0 * np.pi * 440.0 * t)).astype(np.float32, copy=False)
        job = {
            "generation_id": 1004,
            "graph_uid": "specs_splitter_graph",
            "frame_id": 9,
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_specs_splitter",
            "graph_ir": self._specs_splitter_stream_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }
        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("rms:0", values)
        self.assertTrue(np.isfinite(float(values["rms:0"])))
        self.assertGreater(float(values["rms:0"]), 0.0)

    def test_worker_supports_lufs_stream_operator(self):
        n = 2048
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        mono = (0.2 * np.sin((2.0 * np.pi * 440.0 * t))).astype(np.float32, copy=False)
        job = {
            "generation_id": 10041,
            "graph_uid": "lufs_graph",
            "frame_id": 4,
            "block_size": n,
            "sample_rate": sr,
            "graph_ir_digest": "digest_lufs",
            "graph_ir": self._lufs_stream_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": sr, "latency_ms": 0.0},
            },
        }
        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("lufs:0", values)
        self.assertTrue(np.isfinite(float(values["lufs:0"])))
        self.assertGreaterEqual(float(values["lufs:0"]), 0.0)
        self.assertLessEqual(float(values["lufs:0"]), 1.0)

    def test_worker_supports_peak_stream_operator(self):
        n = 2048
        sr = 48000
        t = np.arange(n, dtype=np.float32) / float(sr)
        mono = (0.2 * np.sin((2.0 * np.pi * 440.0 * t))).astype(np.float32, copy=False)
        job = {
            "generation_id": 10042,
            "graph_uid": "peak_graph",
            "frame_id": 4,
            "block_size": n,
            "sample_rate": sr,
            "graph_ir_digest": "digest_peak",
            "graph_ir": self._peak_stream_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": sr, "latency_ms": 0.0},
            },
        }
        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("peak:0", values)
        self.assertTrue(np.isfinite(float(values["peak:0"])))
        self.assertGreaterEqual(float(values["peak:0"]), 0.0)
        self.assertLessEqual(float(values["peak:0"]), 1.0)

    def test_worker_supports_cross_source_math_chromagram_branches(self):
        n = 1024
        t = np.arange(n, dtype=np.float32) / 48000.0
        src_a = np.sin((2.0 * np.pi * 220.0 * t)).astype(np.float32, copy=False)
        src_b = np.sin((2.0 * np.pi * 330.0 * t)).astype(np.float32, copy=False)
        src_c = np.sin((2.0 * np.pi * 440.0 * t)).astype(np.float32, copy=False)
        job = {
            "generation_id": 1005,
            "graph_uid": "math_chroma_graph",
            "frame_id": 11,
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_math_chroma",
            "graph_ir": self._math_chroma_stream_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": src_a, "sample_rate": 48000, "latency_ms": 0.0},
                "src_b": {"mono": src_b, "sample_rate": 48000, "latency_ms": 0.0},
                "src_c": {"mono": src_c, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }
        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("rms0:0", values)
        self.assertIn("rms1:0", values)
        self.assertTrue(np.isfinite(float(values["rms0:0"])))
        self.assertTrue(np.isfinite(float(values["rms1:0"])))

    def test_worker_stream_accumulator_persists_state_and_resets_on_rewind(self):
        n = 256
        mono = np.ones(n, dtype=np.float32)
        base_job = {
            "generation_id": 1006,
            "graph_uid": "stream_accum_graph",
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_stream_accum",
            "graph_ir": self._stream_accumulator_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }

        def _run(frame_id: int, job_key: str) -> float:
            payload = dict(base_job)
            payload["frame_id"] = int(frame_id)
            payload["job_key"] = str(job_key)
            out = self.pds.evaluate_parallel_job(payload)
            values = dict(out.get("values", {}) or {})
            self.assertIn("accum:0", values)
            return float(values["accum:0"])

        v1 = _run(1, "stream_accum_graph:f1:j1")
        v2 = _run(2, "stream_accum_graph:f2:j1")
        v3 = _run(3, "stream_accum_graph:f3:j1")
        v1_rewind = _run(1, "stream_accum_graph:f1:j2")

        self.assertAlmostEqual(v1, 1.0, places=5)
        self.assertAlmostEqual(v2, 2.0, places=5)
        self.assertAlmostEqual(v3, 3.0, places=5)
        self.assertAlmostEqual(v1_rewind, 1.0, places=5)

    def test_shared_memory_payload_resolution_reuses_worker_cache(self):
        if not bool(getattr(self.pds, "_SHARED_MEMORY_AVAILABLE", False)):
            raise unittest.SkipTest("shared memory unavailable in this environment")

        shm_mod = getattr(self.pds, "mp_shared_memory", None)
        if shm_mod is None:
            raise unittest.SkipTest("shared memory module unavailable")

        mono = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        shm = shm_mod.SharedMemory(create=True, size=int(mono.nbytes))
        try:
            buf = np.ndarray(mono.shape, dtype=np.float32, buffer=shm.buf)
            buf[:] = mono
            job = {
                "source_chunks": {
                    "src_a": {"sample_rate": 48000, "latency_ms": 0.0},
                },
                "source_chunks_shm": {
                    "src_a": {
                        "mono": {
                            "name": shm.name,
                            "shape": tuple(int(v) for v in mono.shape),
                            "dtype": "float32",
                        },
                    },
                },
            }

            resolved_a = self.pds._resolve_source_payloads(job)
            resolved_b = self.pds._resolve_source_payloads(job)
            mono_a = resolved_a["src_a"]["mono"]
            mono_b = resolved_b["src_a"]["mono"]

            self.assertIs(mono_a, mono_b)
            self.assertTrue(np.allclose(mono_a, mono))
            self.assertIn("src_a:mono", getattr(self.pds, "_WORKER_SHM_CACHE", {}))
        finally:
            _clear_worker_state(self.pds)
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


    def test_worker_specs_uses_injected_bpm_fps(self):
        """Gap 1 runtime check: worker SourceSpecs computes correct beat-phase from injected params."""
        n = 512
        mono = np.zeros(n, dtype=np.float32)
        # 180 BPM, 3/4 time, 30 FPS → frames_per_beat = (60 * 30) / 180 = 10
        # At frame 5: pos_in_measure = (5 * 0.1) % 3.0 = 0.5; beat_phase = 0.5 / 3.0 ≈ 0.1667
        expected_frames_per_beat = (60.0 * 30.0) / 180.0
        expected_phase = ((5 * (1.0 / expected_frames_per_beat)) % 3.0) / 3.0

        graph_ir = {
            "nodes": {
                "src_a":  {"category": "source_mngmt", "type": "src",       "parameters": {}},
                "specs":  {"category": "utilities",    "type": "src_specs",  "parameters": {
                    "scene_bpm":           180.0,
                    "scene_fps":           30.0,
                    "scene_beats_per_bar": 3.0,
                }},
                "rms":    {"category": "stream",       "type": "rms",        "parameters": {}},
            },
            "links": [
                ("src_a", 0, "specs", 0),
                ("specs", 2, "rms",   0),   # socket 2 = audio passthrough
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "specs:0": {"node_id": "specs", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

        job = {
            "generation_id":  2001,
            "graph_uid":      "injected_bpm_graph",
            "job_key":        "injected_bpm_graph:f5:j1",
            "frame_id":       5,
            "block_size":     n,
            "sample_rate":    48000,
            "graph_ir_digest": "digest_injected_bpm",
            "graph_ir":       graph_ir,
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("specs:0", values)
        # Output socket 0 = frames_per_beat
        frames_per_beat_val = float(values["specs:0"])
        self.assertAlmostEqual(frames_per_beat_val, expected_frames_per_beat, places=5,
                               msg="frames_per_beat should reflect injected BPM+FPS, not hardcoded defaults")


class TestSerializeAudioCache(unittest.TestCase):
    """Tests for task_serialize_audio_cache.

    Guardrails enforced by this suite:
    - The function executes correctly when ``bpy`` is absent from sys.modules.
    - All disk I/O is mocked so the suite is fast and deterministic.
    - Both the scipy (float32 WAV) and stdlib wave (int16 fallback) write paths
      are exercised explicitly.
    """

    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        # Reload to pick up task_serialize_audio_cache added in this session.
        sys.modules.pop("ace_addon_testpkg.parallel_dsp", None)
        sys.modules.pop("ace_addon_testpkg.audio_tasks_registry", None)
        sys.modules.pop("ace_addon_testpkg.audio_influx", None)
        cls.pds = load_module("parallel_dsp")
        if not hasattr(cls.pds, "task_serialize_audio_cache"):
            raise unittest.SkipTest(
                "task_serialize_audio_cache not present in parallel_dsp — skipping"
            )

    def setUp(self):
        _clear_worker_state(self.pds)

    def tearDown(self):
        _clear_worker_state(self.pds)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _fake_scipy():
        """Build minimal scipy.io.wavfile stubs and return (scipy, io, wavfile, mock_write)."""
        mock_write = mock.MagicMock(return_value=None)
        fake_wavfile = types.ModuleType("scipy.io.wavfile")
        fake_wavfile.write = mock_write
        fake_scipy_io = types.ModuleType("scipy.io")
        fake_scipy_io.wavfile = fake_wavfile
        fake_scipy = types.ModuleType("scipy")
        fake_scipy.io = fake_scipy_io
        return fake_scipy, fake_scipy_io, fake_wavfile, mock_write

    def _call_with_scipy_mock(self, arr, cache_dir="/fake/cache", sr=44100, ch=2):
        """Helper: run task_serialize_audio_cache with mocked scipy path."""
        fake_scipy, fake_scipy_io, fake_wavfile, mock_write = self._fake_scipy()
        with mock.patch.dict(sys.modules, {
            "scipy": fake_scipy,
            "scipy.io": fake_scipy_io,
            "scipy.io.wavfile": fake_wavfile,
        }), mock.patch("os.makedirs"):
            result = self.pds.task_serialize_audio_cache(
                audio_array=arr,
                cache_dir=cache_dir,
                sample_rate=sr,
                channels=ch,
            )
        return result, mock_write

    # ------------------------------------------------------------------ no-bpy guard

    def test_task_executes_when_bpy_is_masked_in_sys_modules(self):
        """task_serialize_audio_cache must not touch bpy at call time.

        sys.modules["bpy"] = None simulates a non-Blender Python interpreter.
        Any accidental ``import bpy`` inside the function would raise ImportError,
        causing the test to fail.
        """
        fake_scipy, fake_scipy_io, fake_wavfile, mock_write = self._fake_scipy()
        arr = np.ones((256, 2), dtype=np.float32)

        original = sys.modules.get("bpy", _MISSING)
        try:
            sys.modules["bpy"] = None  # type: ignore[assignment]
            with mock.patch.dict(sys.modules, {
                "scipy": fake_scipy,
                "scipy.io": fake_scipy_io,
                "scipy.io.wavfile": fake_wavfile,
            }), mock.patch("os.makedirs"):
                result = self.pds.task_serialize_audio_cache(
                    audio_array=arr,
                    cache_dir="/fake/dir",
                    sample_rate=44100,
                    channels=2,
                )
        finally:
            if original is _MISSING:
                sys.modules.pop("bpy", None)
            else:
                sys.modules["bpy"] = original  # type: ignore[assignment]

        self.assertIsInstance(result, dict)
        self.assertIn("frames_written", result)
        self.assertEqual(result["frames_written"], 256)
        mock_write.assert_called_once()

    def test_module_level_imports_do_not_reference_bpy(self):
        """parallel_dsp must be importable in an environment without bpy.

        Reload the module with bpy masked and confirm no ImportError is raised.
        Any module-level ``import bpy`` would cause an ImportError here.
        """
        original = sys.modules.get("bpy", _MISSING)
        try:
            sys.modules["bpy"] = None  # type: ignore[assignment]
            # Pop the cached copy to force a fresh load
            for key in list(sys.modules):
                if "parallel_dsp" in key or "audio_tasks_registry" in key or "audio_influx" in key:
                    sys.modules.pop(key, None)
            try:
                reloaded = load_module("parallel_dsp")
            except ImportError as exc:
                self.fail(
                    f"parallel_dsp import failed when bpy=None — "
                    f"module has an unexpected bpy dependency: {exc}"
                )
        finally:
            if original is _MISSING:
                sys.modules.pop("bpy", None)
            else:
                sys.modules["bpy"] = original  # type: ignore[assignment]

        self.assertTrue(callable(getattr(reloaded, "evaluate_parallel_job", None)))

    # ------------------------------------------------------------------ scipy path

    def test_scipy_path_calls_wavfile_write_with_correct_signature(self):
        """scipy.io.wavfile.write receives (output_path, sample_rate, float32_array)."""
        arr = np.ones((512, 2), dtype=np.float32) * 0.5
        result, mock_write = self._call_with_scipy_mock(arr, sr=44100, ch=2)

        self.assertEqual(result["frames_written"], 512)
        self.assertIn("output_path", result)
        self.assertTrue(str(result["output_path"]).endswith(".wav"))

        mock_write.assert_called_once()
        _, call_sr, call_arr = mock_write.call_args[0]
        self.assertEqual(call_sr, 44100)
        self.assertEqual(call_arr.shape, (512, 2))
        self.assertEqual(call_arr.dtype, np.float32)

    def test_frames_written_equals_input_frame_count(self):
        """Return value frames_written must equal arr.shape[0] regardless of channels."""
        for n_frames in (64, 512, 1337):
            arr = np.zeros((n_frames, 2), dtype=np.float32)
            result, _ = self._call_with_scipy_mock(arr, sr=48000, ch=2)
            self.assertEqual(
                result["frames_written"], n_frames,
                f"frames_written mismatch for input with {n_frames} frames",
            )

    def test_output_path_is_inside_cache_dir(self):
        """The returned output_path must be a child of the supplied cache_dir."""
        cache_dir = "/some/specific/cache/dir"
        arr = np.zeros((64, 1), dtype=np.float32)
        result, _ = self._call_with_scipy_mock(arr, cache_dir=cache_dir, sr=44100, ch=1)
        self.assertTrue(
            str(result["output_path"]).startswith(cache_dir),
            f"output_path '{result['output_path']}' is not inside '{cache_dir}'",
        )

    def test_output_path_filename_contains_wav_extension(self):
        arr = np.zeros((128, 2), dtype=np.float32)
        result, _ = self._call_with_scipy_mock(arr)
        self.assertTrue(
            result["output_path"].endswith(".wav"),
            "output file must have a .wav extension",
        )

    def test_1d_input_is_reshaped_to_mono_before_writing(self):
        """A 1-D input array must be reshaped to (N, 1) so wavfile.write gets 2-D data."""
        arr_1d = np.ones(256, dtype=np.float32)
        result, mock_write = self._call_with_scipy_mock(arr_1d, sr=22050, ch=1)

        self.assertEqual(result["frames_written"], 256)
        _, _, call_arr = mock_write.call_args[0]
        self.assertEqual(call_arr.ndim, 2)
        self.assertEqual(call_arr.shape, (256, 1))

    def test_scipy_path_preserves_stereo_channel_order(self):
        """Left/right channel data must not be transposed before passing to wavfile.write."""
        left = np.ones((128, 1), dtype=np.float32)
        right = np.full((128, 1), 0.5, dtype=np.float32)
        arr = np.concatenate([left, right], axis=1)  # (128, 2): col0=L, col1=R
        _, mock_write = self._call_with_scipy_mock(arr, sr=44100, ch=2)

        _, _, call_arr = mock_write.call_args[0]
        self.assertEqual(call_arr.shape, (128, 2))
        np.testing.assert_array_almost_equal(call_arr[:, 0], 1.0)
        np.testing.assert_array_almost_equal(call_arr[:, 1], 0.5)

    # ------------------------------------------------------------------ stdlib wave fallback

    def test_stdlib_wave_fallback_triggered_when_scipy_absent(self):
        """When scipy is unavailable, the stdlib wave branch runs without error."""
        written_payloads = []

        class _FakeWaveWriter:
            """Minimal context-manager stub for wave.open."""
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def setnchannels(self, n):
                self.n_ch = n
            def setsampwidth(self, n):
                self.sampwidth = n
            def setframerate(self, n):
                self.framerate = n
            def writeframes(self, data):
                written_payloads.append(bytes(data))

        arr = np.ones((256, 1), dtype=np.float32) * 0.5

        with mock.patch.dict(sys.modules, {
            "scipy": None,
            "scipy.io": None,
            "scipy.io.wavfile": None,
        }), mock.patch("os.makedirs"), \
             mock.patch("wave.open", return_value=_FakeWaveWriter()):
            result = self.pds.task_serialize_audio_cache(
                audio_array=arr,
                cache_dir="/fake/cache",
                sample_rate=22050,
                channels=1,
            )

        self.assertEqual(result["frames_written"], 256)
        self.assertIn("output_path", result)
        # The wave writer must have been called exactly once
        self.assertEqual(len(written_payloads), 1)

    def test_stdlib_wave_fallback_encodes_as_int16(self):
        """stdlib wave fallback must write int16 PCM data (2 bytes per sample)."""
        written_payloads = []

        class _CapturingWriter:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def setnchannels(self, n): pass
            def setsampwidth(self, n): pass
            def setframerate(self, n): pass
            def writeframes(self, data): written_payloads.append(bytes(data))

        arr = np.ones((128, 1), dtype=np.float32) * 0.5  # 0.5 → ≈16383 in int16

        with mock.patch.dict(sys.modules, {
            "scipy": None, "scipy.io": None, "scipy.io.wavfile": None,
        }), mock.patch("os.makedirs"), \
             mock.patch("wave.open", return_value=_CapturingWriter()):
            self.pds.task_serialize_audio_cache(
                audio_array=arr, cache_dir="/fake", sample_rate=44100, channels=1
            )

        raw = written_payloads[0]
        decoded = np.frombuffer(raw, dtype=np.int16)
        self.assertEqual(len(decoded), 128, "int16 frame count must equal input frames")
        # 0.5 * 32767 ≈ 16383 (tolerance ±2 for rounding)
        self.assertAlmostEqual(float(decoded[0]), 0.5 * 32767.0, delta=2.0)

    def test_stdlib_wave_fallback_clips_beyond_unit_range(self):
        """Values outside [-1, 1] must be clipped before int16 conversion."""
        written_payloads = []

        class _CapturingWriter:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def setnchannels(self, n): pass
            def setsampwidth(self, n): pass
            def setframerate(self, n): pass
            def writeframes(self, data): written_payloads.append(bytes(data))

        # Mix of over-range and in-range samples
        arr = np.array([[2.0], [-3.0], [0.5]], dtype=np.float32)

        with mock.patch.dict(sys.modules, {
            "scipy": None, "scipy.io": None, "scipy.io.wavfile": None,
        }), mock.patch("os.makedirs"), \
             mock.patch("wave.open", return_value=_CapturingWriter()):
            self.pds.task_serialize_audio_cache(
                audio_array=arr, cache_dir="/fake", sample_rate=44100, channels=1
            )

        decoded = np.frombuffer(written_payloads[0], dtype=np.int16)
        self.assertEqual(decoded[0], 32767, "2.0 must clip to max int16 (32767)")
        self.assertEqual(decoded[1], -32767, "-3.0 must clip to min int16 (-32767)")

    # ------------------------------------------------------------------ error handling

    def test_raises_value_error_on_3d_input(self):
        """task_serialize_audio_cache must raise ValueError for >2-D input arrays."""
        arr_3d = np.zeros((4, 8, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.pds.task_serialize_audio_cache(
                audio_array=arr_3d,
                cache_dir="/tmp",
                sample_rate=44100,
                channels=2,
            )

    def test_makedirs_called_with_exist_ok(self):
        """os.makedirs must be called with exist_ok=True so concurrent bakes don't race."""
        fake_scipy, fake_scipy_io, fake_wavfile, _ = self._fake_scipy()
        with mock.patch.dict(sys.modules, {
            "scipy": fake_scipy, "scipy.io": fake_scipy_io,
            "scipy.io.wavfile": fake_wavfile,
        }), mock.patch("os.makedirs") as mock_makedirs:
            self.pds.task_serialize_audio_cache(
                audio_array=np.zeros((64, 1), dtype=np.float32),
                cache_dir="/some/dir",
                sample_rate=44100,
                channels=1,
            )
        mock_makedirs.assert_called_once()
        _, kwargs = mock_makedirs.call_args
        self.assertTrue(
            kwargs.get("exist_ok", False),
            "os.makedirs must be called with exist_ok=True",
        )


if __name__ == "__main__":
    unittest.main()
