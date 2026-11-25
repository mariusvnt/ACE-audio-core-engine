"""
Architectural rules from README "Strict Development Guardrails", checked by
behaviour rather than by reading source text.
"""

import os
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import fake_bpy
from module_loader import PKG_NAME, load_module

try:
    import numpy as np
except Exception:  # pragma: no cover - environment dependent
    np = None

TESTS_DIR = Path(__file__).resolve().parent


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


class TestWorkerIsBpyFree(unittest.TestCase):
    def test_parallel_dsp_imports_without_bpy(self):
        # A fresh interpreter, so nothing else has put bpy in sys.modules.
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(TESTS_DIR)!r})
            from module_loader import load_module
            load_module("parallel_dsp")
            leaked = sorted(m for m in sys.modules if m == "bpy" or m.startswith("bpy."))
            print("LEAKED=" + ",".join(leaked))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("LEAKED=\n", result.stdout + "\n", msg=result.stdout)


@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestPreviewIsTimeDomain(unittest.TestCase):
    """Layer 3 never reconstructs audio from the frequency domain."""

    @classmethod
    def setUpClass(cls):
        cls.atr = load_module("audio_tasks_registry")

    def _forbid_inverse_transforms(self):
        def _boom(*_args, **_kwargs):
            raise AssertionError("inverse FFT called on the preview path")

        return mock.patch.multiple(np.fft, irfft=_boom, ifft=_boom, irfftn=_boom, ifftn=_boom)

    def test_band_limited_preview_uses_no_inverse_fft(self):
        sr = 48000
        n = 1024
        t = np.arange(n, dtype=np.float32) / float(sr)
        signal = (np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
        chunk = self.atr.AudioChunk(signal, None, sr, band_hz_bounds=(120.0, 400.0), lineage=("src",))
        task = self.atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        with self._forbid_inverse_transforms():
            out = task.process({0: chunk})
        self.assertEqual(len(out.mono), n)

    def test_sparse_band_preview_uses_no_inverse_fft(self):
        sr = 48000
        n = 1024
        t = np.arange(n, dtype=np.float32) / float(sr)
        signal = (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
        idx = int(np.argmin(np.abs(freqs - 880.0)))
        chunk = self.atr.AudioChunk(
            signal,
            None,
            sr,
            band_indices=np.array([idx], dtype=np.int64),
            band_index_fft_size=n,
            lineage=("src",),
        )
        task = self.atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=n)
        with self._forbid_inverse_transforms():
            out = task.process({0: chunk})
        self.assertEqual(len(out.mono), n)


@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestEngineInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        cls.bpy = fake_bpy.install()
        fake_bpy.reset(cls.bpy)
        cls.ae = load_module("audio_engine")
        cls.handlers_mod = load_module("engine_handlers")
        cls.Engine = cls.ae.ACEAudioEngine

    def setUp(self):
        fake_bpy.reset(self.bpy)

    def test_registered_handlers_are_persistent(self):
        # Without @persistent Blender drops a handler on every file load.
        with mock.patch.object(self.Engine, "ensure_instance", classmethod(lambda cls, _scene=None: None)):
            self.handlers_mod.register_handlers()
        try:
            registered = [
                fn
                for name in fake_bpy.HANDLER_NAMES
                for fn in getattr(self.bpy.app.handlers, name)
            ]
            self.assertGreaterEqual(len(registered), len(fake_bpy.HANDLER_NAMES))
            for fn in registered:
                self.assertTrue(hasattr(fn, "_bpy_persistent"), msg=f"{fn.__name__} is not @persistent")
            for timer in self.bpy.app.timers.registered:
                self.assertTrue(self.bpy.app.timers.registered[timer][1], msg=f"{timer.__name__} timer not persistent")
        finally:
            self.handlers_mod.unregister_handlers()
        for name in fake_bpy.HANDLER_NAMES:
            self.assertEqual(getattr(self.bpy.app.handlers, name), [], msg=f"{name} left registered")

    def test_worker_count_follows_cpus_not_sources(self):
        engine = self.Engine.__new__(self.Engine)
        engine.graphs = {f"src_{i}": object() for i in range(3)}
        with mock.patch.dict(os.environ, {"ACE_PARALLEL_WORKERS": ""}), mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(engine._parallel_worker_count(None), 16 - self.Engine._PARALLEL_WORKER_HEADROOM_CORES)
        with mock.patch.dict(os.environ, {"ACE_PARALLEL_WORKERS": "4"}), mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(engine._parallel_worker_count(None), 4)
        with mock.patch.dict(os.environ, {"ACE_PARALLEL_WORKERS": ""}), mock.patch("os.cpu_count", return_value=1):
            self.assertEqual(engine._parallel_worker_count(None), 1)

    def test_inflight_jobs_per_source_stay_within_two(self):
        self.assertLessEqual(self.Engine._PARALLEL_MAX_INFLIGHT_PER_SOURCE, 2)
        engine = self.Engine.__new__(self.Engine)
        self.assertLessEqual(engine._parallel_inflight_limit(None), 2)

    def test_shards_split_by_source_group(self):
        shard = self.Engine._parallel_shard_attr_keys
        self.assertEqual(shard(["a", "b"]), [["a", "b"]])
        self.assertEqual(shard(["a", "b"], {"a": "s1", "b": "s1"}), [["a", "b"]])
        self.assertEqual(
            sorted(shard(["a", "b", "c"], {"a": "s1", "b": "s2"})),
            sorted([["a"], ["b"], ["c"]]),
        )
        self.assertEqual(shard([]), [])

    def test_frame_writes_blocked_only_while_rendering_without_lock(self):
        locked = types.SimpleNamespace(render=types.SimpleNamespace(use_lock_interface=True))
        unlocked = types.SimpleNamespace(render=types.SimpleNamespace(use_lock_interface=False))
        with mock.patch.object(self.bpy.app, "is_job_running", lambda job: job == "RENDER"):
            self.assertTrue(self.Engine._unsafe_for_handler_writes(unlocked))
            self.assertFalse(self.Engine._unsafe_for_handler_writes(locked))
        with mock.patch.object(self.bpy.app, "is_job_running", lambda _job: False):
            self.assertFalse(self.Engine._unsafe_for_handler_writes(unlocked))


@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestAmbiguousCompilePruning(unittest.TestCase):
    """A node fed by two sources' branches is dropped from each source's graph."""

    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        fake_bpy.install()
        cls.abs_mod = load_module("abstract_tree")
        cls.Engine = load_module("audio_engine").ACEAudioEngine

    def _tree(self):
        node = self.abs_mod.ACEAbstractNode
        link = self.abs_mod.ACEAbstractLink
        nodes = {
            "src_a": node("src_a", "A", "source_mngmt", "src", root_sources={"a"}),
            "src_b": node("src_b", "B", "source_mngmt", "src", root_sources={"b"}),
            "rms_a": node("rms_a", "RMS A", "stream", "rms", root_sources={"a"}),
            "mix": node("mix", "Mix", "utilities", "audio_math", root_sources={"a", "b"}),
            "rms_mix": node("rms_mix", "RMS Mix", "stream", "rms", root_sources={"a", "b"}),
        }
        links = [
            link("src_a", "rms_a", 0, 0),
            link("src_a", "mix", 0, 0),
            link("src_b", "mix", 0, 1),
            link("mix", "rms_mix", 0, 0),
        ]
        return self.abs_mod.ACEAbstractTree(nodes=nodes, links=links)

    def test_mixed_node_is_pruned_and_own_branch_kept(self):
        tree = self._tree()
        required = set(tree.nodes)
        kept = self.Engine._prune_ambiguous_required_node_ids(
            tree, required, source_uid="a", proxy_name="", single_source_mode=False
        )
        self.assertNotIn("mix", kept)
        self.assertTrue({"src_a", "rms_a"}.issubset(kept))

    def test_single_source_mode_keeps_everything(self):
        tree = self._tree()
        required = set(tree.nodes)
        kept = self.Engine._prune_ambiguous_required_node_ids(
            tree, required, source_uid="a", proxy_name="", single_source_mode=True
        )
        self.assertEqual(kept, required)


class _Socket:
    def __init__(self, name, data_type, is_output):
        self.name = name
        self.identifier = name
        self.bl_idname = data_type
        self.type = data_type
        self.is_output = is_output
        self.is_linked = False
        self.links = []
        self.default_value = None
        self.hide_value = False


class _Node(fake_bpy.TruthyDict):
    def __init__(self, name, node_tree, inputs, outputs):
        super().__init__()
        self.type = "GROUP"
        self.name = name
        self.label = name
        self.node_tree = node_tree
        self.inputs = [_Socket(n, t, False) for n, t in inputs]
        self.outputs = [_Socket(n, t, True) for n, t in outputs]

    def as_pointer(self):
        return id(self)


class TestNodeIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        fake_bpy.install()
        cls.dummy_tree = load_module("dummy_tree")

    def test_deterministic_seed_ignores_node_name(self):
        group = types.SimpleNamespace(name=".ACE_stream_rms")
        tree = types.SimpleNamespace(name="ACE Scene", nodes=[], links=[], bl_idname="GeometryNodeTree")
        reader = self.dummy_tree.ACEDummyTreeReader(tree, frontend_uid="frontend_a")
        sockets = ([("Audio", "NodeSocketBundle")], [("RMS", "NodeSocketFloat")])
        node_a = _Node("RMS", group, *sockets)
        node_b = _Node("RMS.001 renamed by the user", group, *sockets)
        seed_a = reader._build_deterministic_node_seed(node_a, "stream", "rms", root_id="src_a")
        seed_b = reader._build_deterministic_node_seed(node_b, "stream", "rms", root_id="src_a")
        self.assertEqual(seed_a, seed_b)
        seed_other_root = reader._build_deterministic_node_seed(node_a, "stream", "rms", root_id="src_b")
        self.assertNotEqual(seed_a, seed_other_root)


class TestFileFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.influx = load_module("audio_influx")

    def test_wav_is_always_supported_and_text_is_not(self):
        reception = self.influx.AudioReception
        exts = reception.supported_file_extensions()
        self.assertIn(".wav", exts)
        self.assertNotIn(".txt", exts)
        self.assertFalse(reception.is_supported_file_path(str(TESTS_DIR / "module_loader.py")))
        self.assertFalse(reception.is_supported_file_path(str(TESTS_DIR / "missing.wav")))


if __name__ == "__main__":
    unittest.main()
