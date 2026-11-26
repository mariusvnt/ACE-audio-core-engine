"""Folded tree of an operator graph for the Active Operators list (no bpy).

Chains without branching fold into one row of names (``Low › Loudness (RMS) › Accumulator``);
branch points start indented rows, so common path parts are shown once. A node reached
from a second parent is placed once, under the first parent met, and shown as a
reference everywhere else, so nothing downstream is repeated.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional


SEGMENT_NODE = "node"
SEGMENT_BAND = "band"
SEGMENT_REF = "ref"
SEGMENT_SPLIT = "split"  # a band that stands for its splitter, which has no name of its own
# Kinds that are the node itself (selecting the node selects them).
NODE_KINDS = (SEGMENT_NODE, SEGMENT_SPLIT)


@dataclass(frozen=True)
class GraphNode:
    """One operator of the graph, as the tree needs it."""

    id: str
    label: str
    order: int = 0
    # Output names of a node that splits audio into bands (splitter, mid/side, ...);
    # empty for single-output nodes.
    band_names: tuple[str, ...] = ()
    # Set for the frequency splitter: its connected bands replace it, as "<prefix> <band>".
    band_prefix: str = ""


@dataclass(frozen=True)
class GraphLink:
    source_id: str
    source_socket: int
    target_id: str


@dataclass(frozen=True)
class Segment:
    node_id: str
    label: str
    kind: str = SEGMENT_NODE
    # Identifies the tree position this segment stands for (a node, or a node's band).
    key: str = ""


@dataclass
class Row:
    depth: int
    segments: list[Segment] = field(default_factory=list)
    has_children: bool = False
    # Key of the last segment: collapsing it hides the rows below.
    branch_key: str = ""
    parent: int = -1


def band_key(node_id: str, socket_idx: int) -> str:
    return f"{node_id}:{int(socket_idx)}"


class _Builder:
    def __init__(self, nodes: Iterable[GraphNode], links: Iterable[GraphLink]):
        self.nodes = {n.id: n for n in nodes}
        self.out_links: dict[str, list[GraphLink]] = {}
        seen: set[tuple[str, int, str]] = set()
        for link in links:
            if link.source_id not in self.nodes or link.target_id not in self.nodes:
                continue
            key = (link.source_id, int(link.source_socket), link.target_id)
            if key in seen:
                continue
            seen.add(key)
            self.out_links.setdefault(link.source_id, []).append(link)
        for node_id, node_links in self.out_links.items():
            node_links.sort(key=lambda l: (int(l.source_socket), self.nodes[l.target_id].order, l.target_id))
        self.placed: set[str] = set()
        self.rows: list[Row] = []

    def _node_segment(self, node_id: str) -> Segment:
        node = self.nodes[node_id]
        if node_id in self.placed:
            return Segment(node_id, node.label, SEGMENT_REF, f"ref:{node_id}")
        self.placed.add(node_id)
        return Segment(node_id, node.label, SEGMENT_NODE, node_id)

    def _bands(self, node_id: str) -> list[Segment]:
        """One segment per connected output of a band node."""
        node = self.nodes[node_id]
        segments = []
        for sock in sorted({int(l.source_socket) for l in self.out_links.get(node_id, [])}):
            name = node.band_names[sock] if sock < len(node.band_names) else f"Band {sock + 1}"
            if node.band_prefix:
                label = f"{node.band_prefix} {name.removeprefix('Band ')}"
                segments.append(Segment(node_id, label, SEGMENT_SPLIT, band_key(node_id, sock)))
            else:
                segments.append(Segment(node_id, name, SEGMENT_BAND, band_key(node_id, sock)))
        return segments

    def _entries(self, node_id: str) -> list[Segment]:
        """How a node appears under its parent: its name, or its bands for a prefixed splitter."""
        if self.nodes[node_id].band_prefix and self.out_links.get(node_id) and node_id not in self.placed:
            self.placed.add(node_id)
            return self._bands(node_id)
        return [self._node_segment(node_id)]

    def _children(self, seg: Segment) -> list[Segment]:
        """Segments directly below *seg*; a node is placed the first time it is met."""
        if seg.kind == SEGMENT_REF:
            return []
        links = self.out_links.get(seg.node_id, [])
        if seg.kind in (SEGMENT_BAND, SEGMENT_SPLIT):
            sock = int(seg.key.rsplit(":", 1)[1])
            links = [l for l in links if int(l.source_socket) == sock]
        elif self.nodes[seg.node_id].band_names:
            return self._bands(seg.node_id)
        children: list[Segment] = []
        listed: set[str] = set()
        for link in links:
            if link.target_id in listed:
                continue
            listed.add(link.target_id)
            children.extend(self._entries(link.target_id))
        return children

    def emit(self, first: Segment, depth: int, parent: int) -> None:
        segments = [first]
        kids = self._children(first)
        while len(kids) == 1:
            segments.append(kids[0])
            kids = self._children(kids[0])
        row_idx = len(self.rows)
        self.rows.append(
            Row(
                depth=depth,
                segments=segments,
                has_children=bool(kids),
                branch_key=segments[-1].key,
                parent=parent,
            )
        )
        for kid in kids:
            self.emit(kid, depth + 1, row_idx)


def column_widths(rows, available: float, separator: float = 1.0, min_middle: float = 3.0,
                  min_first: float = 5.0) -> list[float]:
    """Width of each segment column, so that the separators of every row line up.

    *rows* is [(indent, [natural width per segment])], in UI units; column 0 includes the indent.
    A column is as wide as its widest cell that has another after it (a row's last cell takes
    whatever is left). When a row is longer than *available*, the middle columns shrink first,
    then the first one, so the first and last names keep their room.
    """
    columns = [0.0] * max((len(widths) - 1 for _indent, widths in rows), default=0)
    for indent, widths in rows:
        for k, width in enumerate(widths[:-1]):
            columns[k] = max(columns[k], width + (indent if k == 0 else 0.0))

    def excess(indent: float, widths: list[float]) -> float:
        before_last = len(widths) - 1
        used = sum(columns[:before_last]) + separator * before_last + widths[-1]
        return used + (indent if before_last == 0 else 0.0) - available

    for indent, widths in sorted(rows, key=lambda row: -excess(*row)):
        before_last = len(widths) - 1
        for shrinkable, floor in ((range(1, before_last), min_middle), (range(0, min(before_last, 1)), min_first)):
            while excess(indent, widths) > 1e-6:
                wide = [k for k in shrinkable if columns[k] > floor + 1e-6]
                if not wide:
                    break
                k = max(wide, key=columns.__getitem__)
                columns[k] -= min(excess(indent, widths), columns[k] - floor, 0.5)
    return columns


def build_rows(nodes: Iterable[GraphNode], links: Iterable[GraphLink], roots: Iterable[str]) -> list[Row]:
    """Rows of the folded tree under *roots*, parents before their children."""
    builder = _Builder(nodes, links)
    for root_id in roots:
        if root_id in builder.nodes and root_id not in builder.placed:
            builder.emit(builder._node_segment(root_id), 0, -1)
    return builder.rows


def visible_rows(rows: list[Row], collapsed: Iterable[str] = (), search: str = "") -> list[bool]:
    """Which rows show: rows under a collapsed key are hidden.

    With a search string, rows whose names match and their ancestors show, and collapse is ignored.
    """
    needle = str(search or "").strip().lower()
    if needle:
        shown = [False] * len(rows)
        for idx, row in enumerate(rows):
            if any(needle in seg.label.lower() for seg in row.segments):
                walk = idx
                while walk >= 0 and not shown[walk]:
                    shown[walk] = True
                    walk = rows[walk].parent
        return shown
    collapsed_keys = set(collapsed or ())
    shown = []
    for row in rows:
        if row.parent < 0:
            shown.append(True)
            continue
        parent = rows[row.parent]
        shown.append(shown[row.parent] and parent.branch_key not in collapsed_keys)
    return shown


def row_of_node(rows: list[Row], node_id: str) -> int:
    """Row holding the placed (non-reference) segment of *node_id*, or -1."""
    for idx, row in enumerate(rows):
        for seg in row.segments:
            if seg.node_id == node_id and seg.kind in NODE_KINDS:
                return idx
    return -1


def ancestor_keys(rows: list[Row], row_idx: int) -> list[str]:
    """Branch keys to expand so row *row_idx* shows."""
    keys = []
    walk = rows[row_idx].parent if 0 <= row_idx < len(rows) else -1
    while walk >= 0:
        keys.append(rows[walk].branch_key)
        walk = rows[walk].parent
    return keys


def find_segment(rows: list[Row], node_id: str) -> Optional[Segment]:
    idx = row_of_node(rows, node_id)
    if idx < 0:
        return None
    return next(seg for seg in rows[idx].segments if seg.node_id == node_id and seg.kind in NODE_KINDS)
