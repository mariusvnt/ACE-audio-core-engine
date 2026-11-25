"""Folded operator tree of the Active Operators list."""

import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "ui" / "panel" / "operator_tree.py"
_spec = importlib.util.spec_from_file_location("ace_operator_tree", _PATH)
ot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ot)  # type: ignore[union-attr]

N = ot.GraphNode
L = ot.GraphLink


def _example():
    """Source -> LUFS -> Accumulator; Splitter (Low -> RMS; High -> LUFS, Width); Mid/Side (Side -> Width)."""
    nodes = [
        N("src", "Source", 0),
        N("lufs_a", "Loudness (LUFS)", 1),
        N("acc", "Accumulator", 2),
        N("split", "Splitter", 3, band_names=("Low", "High")),
        N("rms", "Loudness (RMS)", 4),
        N("lufs_b", "Loudness (LUFS)", 5),
        N("width_a", "Stereo (Width)", 6),
        N("ms", "Mid/Side", 7, band_names=("Mid", "Side")),
        N("width_b", "Stereo (Width)", 8),
    ]
    links = [
        L("src", 0, "lufs_a"), L("lufs_a", 0, "acc"),
        L("src", 0, "split"), L("split", 0, "rms"), L("split", 1, "lufs_b"), L("split", 1, "width_a"),
        L("src", 0, "ms"), L("ms", 1, "width_b"),
    ]
    return nodes, links


def _render(rows, shown=None):
    out = []
    for idx, row in enumerate(rows):
        if shown is not None and not shown[idx]:
            continue
        names = " › ".join(("↳ " if s.kind == ot.SEGMENT_REF else "") + s.label for s in row.segments)
        out.append(("  " * row.depth) + ("▾ " if row.has_children else "") + names)
    return out


class TestOperatorTree(unittest.TestCase):
    def test_example_folds_to_eight_rows(self):
        nodes, links = _example()
        rows = ot.build_rows(nodes, links, ["src"])
        self.assertEqual(_render(rows), [
            "▾ Source",
            "  Loudness (LUFS) › Accumulator",
            "  ▾ Splitter",
            "    Low › Loudness (RMS)",
            "    ▾ High",
            "      Loudness (LUFS)",
            "      Stereo (Width)",
            "  Mid/Side › Side › Stereo (Width)",
        ])

    def test_band_segments_select_their_node(self):
        nodes, links = _example()
        rows = ot.build_rows(nodes, links, ["src"])
        low = rows[3].segments[0]
        self.assertEqual((low.kind, low.node_id, low.key), (ot.SEGMENT_BAND, "split", "split:0"))
        self.assertEqual(rows[4].branch_key, "split:1")

    def test_merge_is_placed_once_and_referenced_elsewhere(self):
        nodes = [N("src", "Source", 0), N("a", "A", 1), N("b", "B", 2), N("math", "Audio Math", 3), N("out", "Loudness", 4)]
        links = [L("src", 0, "a"), L("src", 0, "b"), L("a", 0, "math"), L("b", 0, "math"), L("math", 0, "out")]
        rows = ot.build_rows(nodes, links, ["src"])
        self.assertEqual(_render(rows), [
            "▾ Source",
            "  A › Audio Math › Loudness",
            "  B › ↳ Audio Math",
        ])
        self.assertEqual(ot.row_of_node(rows, "math"), 1)

    def test_children_follow_creation_order(self):
        nodes = [N("src", "Source", 0), N("late", "Late", 9), N("early", "Early", 1)]
        rows = ot.build_rows(nodes, [L("src", 0, "late"), L("src", 0, "early")], ["src"])
        self.assertEqual(_render(rows)[1:], ["  Early", "  Late"])

    def test_collapse_hides_descendants_only(self):
        nodes, links = _example()
        rows = ot.build_rows(nodes, links, ["src"])
        shown = ot.visible_rows(rows, collapsed={"split"})
        self.assertEqual(_render(rows, shown), [
            "▾ Source",
            "  Loudness (LUFS) › Accumulator",
            "  ▾ Splitter",
            "  Mid/Side › Side › Stereo (Width)",
        ])
        self.assertEqual(sum(ot.visible_rows(rows, collapsed={"src"})), 1)

    def test_search_keeps_ancestors_and_ignores_collapse(self):
        nodes, links = _example()
        rows = ot.build_rows(nodes, links, ["src"])
        shown = ot.visible_rows(rows, collapsed={"split", "src"}, search="rms")
        self.assertEqual(_render(rows, shown), ["▾ Source", "  ▾ Splitter", "    Low › Loudness (RMS)"])

    def test_revealing_a_node_expands_its_ancestors(self):
        nodes, links = _example()
        rows = ot.build_rows(nodes, links, ["src"])
        idx = ot.row_of_node(rows, "width_a")
        self.assertEqual(ot.ancestor_keys(rows, idx), ["split:1", "split", "src"])

    def test_unknown_roots_and_dangling_links_are_ignored(self):
        rows = ot.build_rows([N("src", "Source")], [L("src", 0, "ghost")], ["src", "missing"])
        self.assertEqual(_render(rows), ["Source"])


if __name__ == "__main__":
    unittest.main()
