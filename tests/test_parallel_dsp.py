import unittest

try:
    import numpy as np
except Exception:
    np = None

from module_loader import load_module


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
                ("specs", 3, "split", 0),
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

            resolved_a, attached_a = self.pds._resolve_source_payloads(job)
            resolved_b, attached_b = self.pds._resolve_source_payloads(job)
            mono_a = resolved_a["src_a"]["mono"]
            mono_b = resolved_b["src_a"]["mono"]

            self.assertEqual(attached_a, [])
            self.assertEqual(attached_b, [])
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


if __name__ == "__main__":
    unittest.main()
