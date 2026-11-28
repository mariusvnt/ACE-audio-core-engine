"""Architectural rules (ARCHITECTURE.md §1, §6, §7), checked by behaviour."""

import os
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import fake_bpy
from module_loader import PKG_NAME, load_module, load_tasks

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
            load_module("core.worker.job")
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
        cls.atr = load_tasks()

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
        cls.ae = load_module("core.engine.engine")
        cls.handlers_mod = load_module("core.engine.handlers")
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
        engine = self.Engine(0)
        engine.graphs = {f"src_{i}": object() for i in range(3)}
        with mock.patch.dict(os.environ, {"ACE_PARALLEL_WORKERS": ""}), mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(engine.parallel.pool.worker_count(), 16 - engine.parallel.pool.HEADROOM_CORES)
        with mock.patch.dict(os.environ, {"ACE_PARALLEL_WORKERS": "4"}), mock.patch("os.cpu_count", return_value=16):
            self.assertEqual(engine.parallel.pool.worker_count(), 4)
        with mock.patch.dict(os.environ, {"ACE_PARALLEL_WORKERS": ""}), mock.patch("os.cpu_count", return_value=1):
            self.assertEqual(engine.parallel.pool.worker_count(), 1)

    def test_inflight_jobs_per_source_stay_within_two(self):
        self.assertLessEqual(self.ae.ParallelOffload.MAX_INFLIGHT_PER_SOURCE, 2)
        engine = self.Engine(0)
        self.assertLessEqual(engine.parallel.inflight_limit(), 2)

    def test_shards_split_by_source_group(self):
        shard = self.ae.ParallelOffload.shard_attr_keys
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
        cls.ae = load_module("core.engine.engine")
        cls.Engine = cls.ae.ACEAudioEngine

    def _tree(self):
        node = self.abs_mod.ACEAbstractNode
        link = self.abs_mod.ACEAbstractLink
        nodes = {
            "src_a": node("src_a", "A", "input", "src", root_sources={"a"}),
            "src_b": node("src_b", "B", "input", "src", root_sources={"b"}),
            "rms_a": node("rms_a", "RMS A", "audio", "rms", root_sources={"a"}),
            "mix": node("mix", "Mix", "audio", "frequency_splitter", root_sources={"a", "b"}),
            "rms_mix": node("rms_mix", "RMS Mix", "audio", "rms", root_sources={"a", "b"}),
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
        kept = self.ae.GraphCompiler._prune_ambiguous_required_node_ids(
            tree, required, source_uid="a", proxy_name="", single_source_mode=False
        )
        self.assertNotIn("mix", kept)
        self.assertTrue({"src_a", "rms_a"}.issubset(kept))

    def test_single_source_mode_keeps_everything(self):
        tree = self._tree()
        required = set(tree.nodes)
        kept = self.ae.GraphCompiler._prune_ambiguous_required_node_ids(
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


class _Tree:
    def __init__(self, uid):
        self.session_uid = uid
        self.nodes = []


class TestNodeIdentity(unittest.TestCase):
    """A node's kind comes from its group, its id from its tree and name; nothing is stored on it."""

    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        fake_bpy.install()
        cls.identity = load_module("nodes.identity")

    def _node(self, name, group_name, tree):
        node = _Node(name, types.SimpleNamespace(name=group_name, get=lambda _k, d=None: d), [], [])
        node.id_data = tree
        tree.nodes.append(node)
        return node

    def test_kind_comes_from_the_group_name_and_variants(self):
        tree = _Tree(1)
        cases = {
            ".ACE_audio_loudness": ("audio", "loudness"),
            ".ACE_SHADER_audio_stereo": ("audio", "stereo"),
            ".ACE_audio_stereo_vector": ("audio", "stereo_vector"),
            ".ACE_audio_frequency_splitter_4": ("audio", "frequency_splitter"),
            ".ACE_audio_loudness.001": ("audio", "loudness"),
            "My Group": ("", ""),
        }
        for group_name, kind in cases.items():
            self.assertEqual(self.identity.node_kind(self._node("N", group_name, tree)), kind, group_name)

    def test_id_follows_tree_and_name_and_nothing_is_written(self):
        tree_a, tree_b = _Tree(1), _Tree(2)
        node = self._node("Loudness", ".ACE_audio_loudness", tree_a)
        twin = self._node("Loudness", ".ACE_audio_loudness", tree_b)
        node_id = self.identity.node_id(node)
        self.assertTrue(node_id.startswith("ACE_audio_loudness_"))
        self.assertEqual(node_id, self.identity.node_id(node))
        self.assertNotEqual(node_id, self.identity.node_id(twin))
        self.assertIs(self.identity.find_node(tree_a, node_id), node)
        node.name = "Kick"
        self.assertNotEqual(node_id, self.identity.node_id(node))
        self.assertEqual(dict(node), {})

    def test_new_sink_names_avoid_taken_ones(self):
        taken = set()
        for _ in range(50):
            name = self.identity.new_sink_name("RMS", taken)
            self.assertTrue(name.startswith("ACE_rms_"))
            self.assertNotIn(name, taken)
            taken.add(name)
        self.assertFalse(self.identity.usable_sink_name("ACE_undefined"))
        self.assertFalse(self.identity.usable_sink_name(""))


class TestFileFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.influx = load_module("core.input.reception")

    def test_wav_is_always_supported_and_text_is_not(self):
        reception = self.influx.AudioReception
        exts = reception.supported_file_extensions()
        self.assertIn(".wav", exts)
        self.assertNotIn(".txt", exts)
        self.assertFalse(reception.is_supported_file_path(str(TESTS_DIR / "module_loader.py")))
        self.assertFalse(reception.is_supported_file_path(str(TESTS_DIR / "missing.wav")))


if __name__ == "__main__":
    unittest.main()
