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


class TestSplitBands(unittest.TestCase):
    def _rows(self, prefix="Split"):
        nodes = [
            N("src", "Source", 0),
            N("split", "Splitter (4 Bands)", 1, band_names=("Low", "Band 2", "Band 3", "High"), band_prefix=prefix),
            N("a", "A", 2),
            N("b", "B", 3),
        ]
        links = [L("src", 0, "split"), L("split", 0, "a"), L("split", 1, "b")]
        return ot.build_rows(nodes, links, ["src"])

    def test_the_bands_stand_for_the_splitter(self):
        rows = self._rows()
        self.assertEqual([[seg.label for seg in row.segments] for row in rows],
                         [["Source"], ["Split Low", "A"], ["Split 2", "B"]])
        self.assertEqual([row.depth for row in rows], [0, 1, 1])
        self.assertEqual(ot.row_of_node(rows, "split"), 1)
        self.assertEqual({seg.kind for row in rows[1:] for seg in row.segments[:1]}, {ot.SEGMENT_SPLIT})

    def test_a_given_name_replaces_split(self):
        self.assertEqual(self._rows("Drums")[1].segments[0].label, "Drums Low")

    def test_an_unconnected_splitter_keeps_its_own_name(self):
        nodes = [N("src", "Source", 0), N("split", "Splitter (2 Bands)", 1, ("Low", "High"), "Split")]
        rows = ot.build_rows(nodes, [L("src", 0, "split")], ["src"])
        self.assertEqual([seg.label for seg in rows[0].segments], ["Source", "Splitter (2 Bands)"])


class TestColumnWidths(unittest.TestCase):
    def test_columns_fit_the_widest_cell_that_has_another_after_it(self):
        widths = ot.column_widths([(1.0, [4.0, 6.0, 9.0]), (2.0, [3.0, 20.0])], available=60.0)
        self.assertEqual(widths, [5.0, 6.0])  # the first column includes the indent

    def test_middle_names_shrink_before_the_first_one(self):
        row = [(1.0, [5.0, 10.0, 10.0, 8.0])]
        slightly = ot.column_widths(row, available=32.0)
        self.assertEqual(slightly[0], 6.0)
        self.assertAlmostEqual(sum(slightly), 32.0 - 3.0 - 8.0)
        self.assertAlmostEqual(slightly[1], slightly[2], delta=0.5)
        tight = ot.column_widths(row, available=10.0)
        self.assertEqual(tight, [5.0, 3.0, 3.0])  # floors; the last name takes what is left

    def test_single_names_need_no_columns(self):
        self.assertEqual(ot.column_widths([(1.0, [4.0])], available=10.0), [])


if __name__ == "__main__":
    unittest.main()
