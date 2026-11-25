"""
Index 0 is a real index: ``int(x or -1)`` used to turn it into "none".
"""

import sys
import types
import unittest

import fake_bpy
from module_loader import PKG_NAME, load_module


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


class TestAsInt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils = load_module("utils")

    def test_zero_stays_zero(self):
        self.assertEqual(self.utils.as_int(0, -1), 0)
        self.assertEqual(self.utils.as_float(0.0, 1.5), 0.0)

    def test_missing_or_invalid_falls_back(self):
        for value in (None, "", "abc", object()):
            self.assertEqual(self.utils.as_int(value, -1), -1)
            self.assertEqual(self.utils.as_float(value, 2.5), 2.5)

    def test_numeric_strings_convert(self):
        self.assertEqual(self.utils.as_int("3", -1), 3)
        self.assertEqual(self.utils.as_float("0.25", 1.0), 0.25)


class _Source:
    def __init__(self, source_type="FILE"):
        self.source_type = source_type
        self.stable_id = "src_a"


class TestBakePollFirstSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        fake_bpy.install()
        cls.ops = load_module("n_panel_operators")

    def _context(self, index, sources):
        ace = types.SimpleNamespace(active_audio_source_index=index, audio_sources=sources)
        return types.SimpleNamespace(scene=types.SimpleNamespace(audiocore=ace))

    def test_bake_is_available_for_the_first_source(self):
        poll = self.ops.ACE_OT_bake_active_source.poll
        self.assertTrue(poll(self._context(0, [_Source()])))

    def test_bake_needs_a_file_source_in_range(self):
        poll = self.ops.ACE_OT_bake_active_source.poll
        self.assertFalse(poll(self._context(0, [_Source("LIVE")])))
        self.assertFalse(poll(self._context(1, [_Source()])))
        self.assertFalse(poll(self._context(-1, [_Source()])))


class _HelperNode(fake_bpy.TruthyDict):
    bl_idname = "ShaderNodeAttribute"

    def __init__(self):
        super().__init__()
        self.name = ""
        self.label = ""
        self.location = (0.0, 0.0)
        self.parent = None
        self.attribute_name = ""
        self.attribute_type = "GEOMETRY"
        self.hide = False
        self.outputs = []
        self.inputs = []


class _Nodes(list):
    def new(self, _bl_idname):
        node = _HelperNode()
        self.append(node)
        return node

    def remove(self, node):
        list.remove(self, node)


class TestShaderHelperOutputZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        fake_bpy.install()
        cls.factory = load_module("shader_tree").ACEShaderNodeFactory

    def test_helper_for_output_zero_is_found_and_not_rewritten(self):
        tree = types.SimpleNamespace(nodes=_Nodes(), links=[])
        owner = types.SimpleNamespace(location=(0.0, 0.0), name="owner", parent=None)
        node, _ = self.factory._ensure_helper_attr_node(tree, owner, "node_a", 0, "RMS")
        self.assertIsNotNone(node)
        found = self.factory._find_helper_attr_nodes(tree, "node_a")
        self.assertIs(found.get(0), node)
        again, changed = self.factory._ensure_helper_attr_node(tree, owner, "node_a", 0, "RMS")
        self.assertIs(again, node)
        self.assertFalse(changed)
        self.assertEqual(len(tree.nodes), 1)


if __name__ == "__main__":
    unittest.main()
