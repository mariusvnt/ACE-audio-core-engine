import unittest

try:
    import numpy as np
except Exception:
    np = None

from module_loader import forget_modules, load_module


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
        forget_modules("core.worker", "core.tasks", "core.input")
        cls.pds = load_module("core.worker.job")

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
                "bright": {"category": "stream", "type": "brightness", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "rms", 0),
                ("src_a", 0, "bright", 0),
            ],
            "source_node_ids": ["src_a"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
            },
            "output_specs": {
                "rms:0": {"node_id": "rms", "socket_idx": 0, "attr_type": "FLOAT"},
                "bright:0": {"node_id": "bright", "socket_idx": 0, "attr_type": "FLOAT"},
            },
        }

    def _lufs_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "lufs": {
                    "category": "stream",
                    "type": "lufs",
                    "parameters": {"window_size": 0.4},
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
        a = np.full(n, 0.1, dtype=np.float32)
        b = np.full(n, 0.2, dtype=np.float32)
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
        # RMS reads on the Loudness scale: 0.1 + 0.2 = 0.3, i.e. -10.46 dB over the -70 dB floor.
        expected = (20.0 * np.log10(0.3) + 70.0) / 70.0
        self.assertAlmostEqual(float(values["rms:0"]), expected, places=4)
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
            "output_keys": ["bright:0"],
            "source_chunks": {
                "src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        self.assertIn("bright:0", values)
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

    def test_worker_applies_per_frame_parameter_overrides(self):
        n = 2048
        t = np.arange(n, dtype=np.float32) / 48000.0
        mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        ir = self._specs_splitter_stream_graph_ir()
        ir["nodes"]["split"]["parameters"] = {"in_1": 100.0}
        job = {
            "generation_id": 1099, "graph_uid": "driven_split_graph", "frame_id": 1, "block_size": n,
            "sample_rate": 48000, "graph_ir_digest": "digest_driven", "graph_ir": ir,
            "source_chunks": {"src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0}},
        }
        below = float(self.pds.evaluate_parallel_job(job)["values"]["rms:0"])
        # Same graph, no new IR: the split frequency now sits above the tone.
        job.update(frame_id=2, param_overrides={"split": {"in_1": 5000.0}})
        job.pop("graph_ir")
        above = float(self.pds.evaluate_parallel_job(job)["values"]["rms:0"])
        # Loudness scale (0..1): only spectral leakage reaches the low band, then the whole tone.
        self.assertLess(below, 0.7)
        self.assertGreater(above, 0.9)

    def _job_for(self, graph_ir, digest):
        n = 512
        mono = np.zeros(n, dtype=np.float32)
        return {
            "generation_id": 7, "graph_uid": "bad_graph", "frame_id": 1, "block_size": n,
            "sample_rate": 48000, "graph_ir_digest": digest, "graph_ir": graph_ir,
            "source_chunks": {"src_a": {"mono": mono, "sample_rate": 48000, "latency_ms": 0.0}},
        }

    def test_worker_rejects_out_of_range_output_socket(self):
        ir = self._specs_splitter_stream_graph_ir()
        ir["links"][1] = ("specs", 3, "split", 0)  # Source Specs has outputs 0..2
        with self.assertRaisesRegex(ValueError, "has 3 output"):
            self.pds.evaluate_parallel_job(self._job_for(ir, "digest_bad_socket"))

    def test_worker_rejects_link_to_unknown_node(self):
        ir = self._specs_splitter_stream_graph_ir()
        ir["links"].append(("ghost", 0, "rms", 0))
        with self.assertRaisesRegex(ValueError, "unknown node 'ghost'"):
            self.pds.evaluate_parallel_job(self._job_for(ir, "digest_unknown_node"))

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


if __name__ == "__main__":
    unittest.main()
