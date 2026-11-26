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

    def _two_source_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "src_b": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "rms_a": {"category": "stream", "type": "rms", "parameters": {}},
                "rms_b": {"category": "stream", "type": "rms", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "rms_a", 0),
                ("src_b", 0, "rms_b", 0),
            ],
            "source_node_ids": ["src_a", "src_b"],
            "source_ir": {
                "src_a": {"sample_rate": 48000, "lineage": ["src_a"], "spectrum_token": ["src_a"]},
                "src_b": {"sample_rate": 48000, "lineage": ["src_b"], "spectrum_token": ["src_b"]},
            },
            "output_specs": {
                "rms_a:0": {"node_id": "rms_a", "socket_idx": 0, "attr_type": "FLOAT"},
                "rms_b:0": {"node_id": "rms_b", "socket_idx": 0, "attr_type": "FLOAT"},
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

    def _splitter_stream_graph_ir(self):
        return {
            "nodes": {
                "src_a": {"category": "source_mngmt", "type": "src", "parameters": {}},
                "split": {"category": "delta", "type": "frequency_splitter", "parameters": {}},
                "rms": {"category": "stream", "type": "rms", "parameters": {}},
            },
            "links": [
                ("src_a", 0, "split", 0),
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

    def test_each_source_of_a_graph_gets_its_own_chunk(self):
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
            "graph_ir": self._two_source_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": a, "sample_rate": 48000, "latency_ms": 1.0},
                "src_b": {"mono": b, "sample_rate": 48000, "latency_ms": 2.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        values = dict(out.get("values", {}) or {})
        # RMS reads on the Loudness scale: dB over the -70 dB floor.
        for key, level in (("rms_a:0", 0.1), ("rms_b:0", 0.2)):
            self.assertAlmostEqual(float(values[key]), (20.0 * np.log10(level) + 70.0) / 70.0, places=4)
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
            "graph_ir": self._two_source_graph_ir(),
            "source_chunks": {
                "src_a": {"mono": a, "sample_rate": 48000, "latency_ms": 0.0},
                "src_b": {"mono": b, "sample_rate": 48000, "latency_ms": 0.0},
            },
        }

        out = self.pds.evaluate_parallel_job(job)
        self.assertEqual(int(out.get("frame_id", -1)), 3)
        self.assertEqual(str(out.get("graph_uid", "")), "schema_graph")
        self.assertEqual(str(out.get("job_key", "")), "schema_graph:f3:j1")
        self.assertIn("rms_a:0", dict(out.get("values", {}) or {}))

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

    def test_worker_supports_splitter_stream_chain(self):
        n = 512
        t = np.arange(n, dtype=np.float32) / 48000.0
        mono = np.sin((2.0 * np.pi * 440.0 * t)).astype(np.float32, copy=False)
        job = {
            "generation_id": 1004,
            "graph_uid": "splitter_graph",
            "frame_id": 9,
            "block_size": n,
            "sample_rate": 48000,
            "graph_ir_digest": "digest_splitter",
            "graph_ir": self._splitter_stream_graph_ir(),
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
        ir = self._splitter_stream_graph_ir()
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
        ir = self._splitter_stream_graph_ir()
        ir["links"][0] = ("src_a", 3, "split", 0)  # Source has outputs 0..2
        with self.assertRaisesRegex(ValueError, "has 3 output"):
            self.pds.evaluate_parallel_job(self._job_for(ir, "digest_bad_socket"))

    def test_worker_rejects_link_to_unknown_node(self):
        ir = self._splitter_stream_graph_ir()
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



if __name__ == "__main__":
    unittest.main()
