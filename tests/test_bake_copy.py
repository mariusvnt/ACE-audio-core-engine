"""The bake streams through a fresh copy of a source graph, never through the live tasks."""

import sys
import types
import unittest

from module_loader import PKG_NAME, load_module, load_tasks

import fake_bpy

try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    numpy = None


def _install_properties_stub():
    properties_mod = types.ModuleType(f"{PKG_NAME}.properties")

    class ACEAudioSource:  # pragma: no cover - import placeholder
        pass

    properties_mod.ACEAudioSource = ACEAudioSource
    sys.modules[f"{PKG_NAME}.properties"] = properties_mod


def _node(node_id, category, node_type, parameters=None):
    return types.SimpleNamespace(id=node_id, category=category, type=node_type, parameters=dict(parameters or {}))


@unittest.skipIf(numpy is None, "numpy is required")
class FreshGraphCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        _install_properties_stub()
        cls.tasks_mod = load_tasks()
        cls.engine_mod = load_module("core.engine.engine")
        cls.graph_mod = load_module("core.tasks.graph")
        cls.source_mod = load_module("core.tasks.source")
        cls.stream_mod = load_module("core.tasks.stream")

    def _live_graph(self):
        engine = self.engine_mod.ACEAudioEngine(0)
        engine.abstract_tree_ref = types.SimpleNamespace(
            nodes={
                "src": _node("src", "source_mngmt", "src"),
                "lufs": _node("lufs", "stream", "loudness", {"mode": "lufs"}),
            }
        )
        source_ctx = types.SimpleNamespace(source_type="FILE", filepath="tone.wav", stable_id="S1", live_latency_ms=5)
        graph = self.graph_mod.ACEAudioGraph(source_ctx, "ACE_proxy")
        graph.source_uid = "S1"
        graph.block_size = 2000
        graph.sample_rate = 48000
        graph.signal_signature = "sig"
        src = self.source_mod.SourceTask("src", {}, 2000)
        src.source_context = source_ctx
        src.scene_fps = 24.0
        src.scene_frame_start = 1
        lufs = self.stream_mod.create_stream_task("lufs", "loudness", {"mode": "lufs", "Window (s)": 0.4}, 2000)
        lufs.connect_input(input_idx=0, target_task=src, target_output_idx=0)
        lufs.parameters["Window (s)"] = 1.5  # changed at runtime, after the compile
        graph.tasks = {"src": src, "lufs": lufs}
        graph.output_attr_keys = ["lufs:0"]
        return engine, graph

    def test_the_copy_has_new_tasks_wired_the_same_way(self):
        engine, graph = self._live_graph()
        copy = engine.compiler.fresh_graph_copy(graph)
        self.assertEqual(sorted(copy.tasks), ["lufs", "src"])
        for node_id in copy.tasks:
            self.assertIsNot(copy.tasks[node_id], graph.tasks[node_id])
            self.assertIs(type(copy.tasks[node_id]), type(graph.tasks[node_id]))
        self.assertIs(copy.tasks["lufs"].inputs[0].task, copy.tasks["src"])
        self.assertEqual(copy.output_attr_keys, ["lufs:0"])
        self.assertEqual([spec[0] for spec in copy.output_slot_specs], ["lufs:0"])
        self.assertEqual((copy.signal_signature, copy.block_size, copy.source_uid), ("sig", 2000, "S1"))
        self.assertEqual(copy.tasks["lufs"].parameters["Window (s)"], 1.5, "the copy takes the live parameters")

    def test_the_copy_reads_snapshots_and_starts_no_prefetch(self):
        engine, graph = self._live_graph()
        copy = engine.compiler.fresh_graph_copy(graph)
        src = copy.tasks["src"]
        self.assertFalse(src.prefetch_allowed)
        self.assertIsNot(src.source_context, graph.source_context)
        self.assertEqual(src.source_context.filepath, "tone.wav")
        src.source_context.sample_rate = 44100  # what a task writes back lands on the snapshot
        self.assertFalse(hasattr(graph.source_context, "sample_rate"))
        self.assertEqual(copy.source_task_ids, ["src"])
        self.assertEqual(copy.file_source_task_ids, ["src"])


if __name__ == "__main__":
    unittest.main()
