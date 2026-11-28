"""Signals: node uses computing the same thing are one task writing every use's sinks."""

import types
import unittest

from module_loader import load_module


class TestSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = load_module("abstract_tree")
        cls.signals = load_module("core.engine.compiler.signals")
        cls.graph_mod = load_module("core.tasks.graph")

    def _node(self, node_id, category, node_type, params=None, root="SRC_A", kind="GEOMETRY_NODES", inputs=()):
        at = self.at
        return at.ACEAbstractNode(
            id=node_id, name=node_id, category=category, type=node_type,
            parameters=dict(params or {}),
            inputs=[at.ACEAbstractSocket(id=f"{node_id}_in_{i}", name=name, socket_idx=i, is_output=False, data_type=dt)
                    for i, (name, dt) in enumerate(inputs)],
            outputs=[at.ACEAbstractSocket(id=f"{node_id}_out_0", name="Value", socket_idx=0, is_output=True,
                                          data_type="NodeSocketFloat")],
            root_sources={root} if root else set(),
            frontend_kind=kind,
            source_binding_id=root if (category, node_type) == ("input", "src") else "",
        )

    def _tree(self, nodes, links):
        at = self.at
        tree = at.ACEAbstractTree()
        for node in nodes:
            tree.add_node(node)
        for src, dst in links:
            tree.add_link(at.ACEAbstractLink(source_node_id=src, target_node_id=dst, source_socket_idx=0, target_socket_idx=0))
        return tree

    def _loudness(self, node_id, window=0.4, sink="", root="SRC_A", kind="GEOMETRY_NODES"):
        params = {"mode": "rms", "in_1": window, "frontend_node_index": len(node_id)}
        if sink:
            params["attr_id_out_0"] = sink
            params["attr_id_custom_out_0"] = True
        return self._node(node_id, "audio", "loudness", params, root=root, kind=kind,
                          inputs=[("Audio", "NodeSocketBundle"), ("Window (s)", "NodeSocketFloat")])

    def _scene(self, *uses):
        """Two Source nodes of one source, each feeding the given Loudness uses in turn."""
        sources = [self._node("src_1", "input", "src", {"in_0": ("object", "proxy")}),
                   self._node("src_2", "input", "src", {"frontend_node_index": 9})]
        links = [(sources[i % 2].id, use.id) for i, use in enumerate(uses)]
        return self._tree([*sources, *uses], links)

    def test_equal_computations_are_one_signal_with_every_sink(self):
        tree = self._scene(self._loudness("loud_a", sink="ACE_rms_a"),
                           self._loudness("loud_b", sink="ACE_rms_b", kind="SHADER_NODES"))
        result = self.signals.build_signals(tree, {"SRC_A": (48000, 2000, 24.0)})

        self.assertEqual(result.signal_of["src_1"], result.signal_of["src_2"], "one source read once")
        self.assertEqual(result.signal_of["loud_a"], result.signal_of["loud_b"])
        (loud,) = [n for n in result.tree.nodes.values() if n.type == "loudness"]
        self.assertEqual(loud.instance_ids, ["loud_a", "loud_b"])
        self.assertEqual(loud.instance_sinks[0], [("loud_a", "ACE_rms_a", "GEOMETRY_NODES"),
                                                  ("loud_b", "ACE_rms_b", "SHADER_NODES")])
        self.assertEqual(len(result.tree.links), 1)

    def test_a_parameter_or_a_source_setting_makes_another_signal(self):
        tree = self._scene(self._loudness("loud_a"), self._loudness("loud_b", window=0.8))
        result = self.signals.build_signals(tree, {"SRC_A": (48000, 2000, 24.0)})
        self.assertNotEqual(result.signal_of["loud_a"], result.signal_of["loud_b"])

        tree = self._scene(self._loudness("loud_a"))
        at_48k = self.signals.build_signals(tree, {"SRC_A": (48000, 2000, 24.0)}).signal_of["loud_a"]
        at_44k = self.signals.build_signals(tree, {"SRC_A": (44100, 1838, 24.0)}).signal_of["loud_a"]
        self.assertNotEqual(at_48k, at_44k, "a key reaches down to how its source is read")

    def test_keys_are_stable_and_ignore_sink_names_and_editor_order(self):
        first = self.signals.build_signals(self._scene(self._loudness("loud_a", sink="ACE_rms_a")),
                                           {"SRC_A": (48000, 2000, 24.0)})
        renamed = self._loudness("loud_a", sink="ACE_rms_z")
        renamed.parameters["frontend_node_index"] = 42
        again = self.signals.build_signals(self._scene(renamed), {"SRC_A": (48000, 2000, 24.0)})
        self.assertEqual(first.signal_of, again.signal_of)

    def test_preview_sinks_keep_their_own_ids(self):
        preview = self._node("preview_1", "output", "preview_sink", {"preview_enabled": True})
        tree = self._tree([self._node("src_1", "input", "src"), preview], [("src_1", "preview_1")])
        result = self.signals.build_signals(tree, {})
        self.assertEqual(result.signal_of["preview_1"], "preview_1")

    def test_two_signals_on_one_sink_keep_the_first_use(self):
        binding = self.graph_mod.ACESinkBinding
        graph = types.SimpleNamespace(sink_bindings_by_attr={
            "S_one:0": [binding("PROXY_ATTRIBUTE", "proxy", "ACE_rms_a", instance_id="use_b")],
            "S_two:0": [binding("PROXY_ATTRIBUTE", "proxy", "ACE_rms_a", instance_id="use_a"),
                        binding("PROXY_ATTRIBUTE", "proxy", "ACE_other", instance_id="use_a")],
        })
        conflicts = self.signals.resolve_sink_conflicts({"SRC_A": graph})

        self.assertEqual(conflicts, [{"sink_kind": "PROXY_ATTRIBUTE", "target": "proxy", "sink_name": "ACE_rms_a",
                                      "kept": "use_a", "dropped": ["use_b"]}])
        self.assertEqual(graph.sink_bindings_by_attr["S_one:0"], [])
        self.assertEqual([b.target_data_name for b in graph.sink_bindings_by_attr["S_two:0"]], ["ACE_rms_a", "ACE_other"])


if __name__ == "__main__":
    unittest.main()
