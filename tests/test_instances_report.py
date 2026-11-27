"""The Active ACE Instances report: definitions, their uses, shared signals and conflicts."""

import types
import unittest

from module_loader import load_module


class TestInstancesReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = load_module("abstract_tree")
        cls.instances = load_module("ui.panel.instances")

    def _engine(self, compiled_at=1.0):
        at = self.at
        tree = at.ACEAbstractTree()
        tree.add_frontend(at.ACEFrontendRef(uid="fe_cube", kind="GEOMETRY_NODES", tree_name="Main",
                                            tree_bl_idname="GeometryNodeTree", owner_keys=("Cube:GeometryNodes",)))
        tree.add_frontend(at.ACEFrontendRef(uid="fe_mat", kind="SHADER_NODES", tree_name="Mat",
                                            tree_bl_idname="ShaderNodeTree", owner_keys=("Cube",), material_name="Mat",
                                            discovery_index=1))
        for node_id, frontend, tree_name, chain, source in (
            ("use_a", "fe_cube", "Group", ("Group A",), "SRC_1"),
            ("use_b", "fe_cube", "Group", ("Group B",), "SRC_2"),
            ("use_c", "fe_mat", "Mat", (), "SRC_1"),
        ):
            tree.add_node(at.ACEAbstractNode(
                id=node_id, name="Loudness" if tree_name == "Group" else "RMS", category="stream", type="loudness",
                root_sources={source}, frontend_uid=frontend, definition_tree=tree_name, use_chain=chain,
            ))
        tree.add_node(at.ACEAbstractNode(id="viewer", name="Viewer", category="utilities", type="preview_sink",
                                         parameters={"viewer": True}, frontend_uid="fe_cube", definition_tree="Main"))
        return types.SimpleNamespace(
            instance_tree_ref=tree,
            signal_of={"use_a": "S_1", "use_b": "S_2", "use_c": "S_1", "viewer": "viewer"},
            sink_conflicts=[{"sink_name": "ACE_rms", "kept": "use_a", "dropped": ["use_b"]}],
            compiler=types.SimpleNamespace(_last_compile_time=compiled_at),
        )

    def test_each_definition_lists_its_uses_with_owner_chain_source_and_sharing(self):
        report = self.instances.instance_report(self._engine())
        self.assertEqual([(d.tree_name, d.node_name) for d in report], [("Group", "Loudness"), ("Mat", "RMS")])
        group, mat = report
        self.assertEqual([(u.owners, u.chain, u.sources, u.shared_with) for u in group.uses], [
            (["Cube › GeometryNodes"], ("Group A",), ["SRC_1"], 1),  # the material's RMS computes the same
            (["Cube › GeometryNodes"], ("Group B",), ["SRC_2"], 0),
        ])
        self.assertEqual([c["sink_name"] for c in group.conflicts], ["ACE_rms"])
        self.assertEqual((mat.uses[0].owners, mat.conflicts), (["Cube › Mat"], []))

    def test_owner_labels(self):
        label = self.instances.owner_label
        self.assertEqual(label("world:World"), "World › World")
        self.assertEqual(label("light:Key"), "Light › Key")
        self.assertEqual(label("Cube:Nodes"), "Cube › Nodes")
        self.assertEqual(label("Cube", "Mat"), "Cube › Mat")

    def test_source_uses_lists_the_owners_reading_each_source(self):
        uses = self.instances.source_uses(self._engine(compiled_at=2.0))
        self.assertEqual({k: [f.uid for f in v] for k, v in uses.items()},
                         {"SRC_1": ["fe_cube", "fe_mat"], "SRC_2": ["fe_cube"]})

    def test_nothing_compiled_is_an_empty_report(self):
        self.assertEqual(self.instances.instance_report(None), [])
        self.assertEqual(self.instances.source_uses(types.SimpleNamespace(instance_tree_ref=None)), {})


if __name__ == "__main__":
    unittest.main()
