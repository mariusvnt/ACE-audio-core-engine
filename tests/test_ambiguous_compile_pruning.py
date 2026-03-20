import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_SUBPACKAGE_PATHS = {
    "audio_engine.py": "core/audio_engine.py",
    "dummy_tree.py": "nodes/dummy_tree.py",
}


def _read(name: str) -> str:
    return (ROOT / _SUBPACKAGE_PATHS.get(name, name)).read_text(encoding="utf-8")


class TestMixedSourceCompile(unittest.TestCase):
    def test_required_sets_use_mixed_graph_partitioning(self):
        src = _read("audio_engine.py")
        self.assertIn("def _compute_mixed_graph_specs(", src)
        self.assertIn("if not single_source_mode and self._node_is_mixed_source(node):", src)
        self.assertIn('mixed_graph_uid = "__ace_mixed__:"', src)
        self.assertIn("graph.is_mixed = True", src)
        self.assertIn("graph.root_source_tokens = sorted(root_tokens)", src)

    def test_dummy_tree_does_not_fallback_to_node_ace_id_as_root(self):
        src = _read("dummy_tree.py")
        self.assertIn("def _resolve_root_id(", src)
        self.assertNotIn("root_id = bl_node.get('ace_id'", src)


if __name__ == "__main__":
    unittest.main()
