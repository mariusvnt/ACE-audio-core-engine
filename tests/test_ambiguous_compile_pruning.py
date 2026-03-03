import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class TestAmbiguousCompilePruning(unittest.TestCase):
    def test_required_sets_use_ambiguity_pruning(self):
        src = _read("audio_engine.py")
        self.assertIn("def _prune_ambiguous_required_node_ids(", src)
        self.assertIn("return self._prune_ambiguous_required_node_ids(", src)
        self.assertIn("if len(upstream_roots) >= 2:", src)
        self.assertIn("if src_id in ambiguous or dst_id in ambiguous:", src)
        self.assertIn("return bool(roots.intersection(source_aliases))", src)

    def test_dummy_tree_does_not_fallback_to_node_ace_id_as_root(self):
        src = _read("dummy_tree.py")
        self.assertIn("def _resolve_root_id(", src)
        self.assertNotIn("root_id = bl_node.get('ace_id'", src)


if __name__ == "__main__":
    unittest.main()
