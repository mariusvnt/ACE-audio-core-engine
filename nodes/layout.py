"""Readable layout for the trees ACE generates: data flows left to right, one column per step.

Each node goes in the column after the last node it reads from (Group Input first, Group Output
last, a Simulation Zone's input before what it holds, its output after). Within a column, nodes
follow the rows of what feeds them, so links run as straight as they can. Nodes are spaced by
their size, estimated from their visible sockets and settings, since a tree that was never drawn
has no dimensions yet.
"""

COLUMN_GAP = 70.0
ROW_GAP = 28.0
SOCKET_ROW = 22.0
HEADER = 36.0

# Rows of settings drawn in the node's body, above its sockets.
_SETTING_ROWS = {
    "ShaderNodeMath": 2, "ShaderNodeVectorMath": 1, "FunctionNodeCompare": 2, "FunctionNodeBooleanMath": 1,
    "GeometryNodeSwitch": 1, "GeometryNodeMenuSwitch": 1, "GeometryNodeInputNamedAttribute": 1,
    "GeometryNodeSampleIndex": 3, "GeometryNodeStoreNamedAttribute": 2, "GeometryNodeObjectInfo": 1,
    "ShaderNodeAttribute": 2, "GeometryNodeMeshLine": 2, "NodeCombineBundle": 0, "NodeSeparateBundle": 0,
    "GeometryNodeSimulationInput": 0, "GeometryNodeSimulationOutput": 0, "GeometryNodeInputSceneTime": 0,
    "NodeGroupInput": 0, "NodeGroupOutput": 0,
}


def _visible(sockets) -> int:
    return sum(1 for s in sockets if getattr(s, "enabled", True) and not getattr(s, "hide", False))


def node_size(node) -> tuple[float, float]:
    """(width, height) the node is drawn at, estimated."""
    if node.bl_idname == "NodeReroute":
        return 16.0, 16.0
    rows = _visible(node.outputs) + _visible(node.inputs) + _SETTING_ROWS.get(node.bl_idname, 1)
    return float(getattr(node, "width", 140.0) or 140.0), HEADER + SOCKET_ROW * rows


def _flow(nodes, links):
    """Who feeds whom: {node: set of nodes it reads}, a Simulation Zone's output reading its input."""
    feeds = {n: set() for n in nodes}
    for link in links:
        if link.from_node in feeds and link.to_node in feeds and link.from_node != link.to_node:
            feeds[link.to_node].add(link.from_node)
    for node in nodes:
        paired = getattr(node, "paired_output", None) if node.bl_idname == "GeometryNodeSimulationInput" else None
        if paired is not None and paired in feeds:
            feeds[paired].add(node)
    return feeds


def _columns(nodes, feeds) -> dict:
    column: dict = {}
    pending = list(nodes)
    for _ in range(len(nodes) + 1):  # longest path; a cycle (never built by ACE) just stops it
        changed = False
        for node in pending:
            col = 1 + max((column.get(src, 0) for src in feeds[node]), default=-1)
            if node.bl_idname == "NodeGroupInput":
                col = 0
            if column.get(node) != col:
                column[node] = col
                changed = True
        if not changed:
            break
    last = max(column.values(), default=0)
    for node in nodes:
        if node.bl_idname == "NodeGroupOutput":
            column[node] = max(last, column[node])
        elif node.bl_idname == "NodeGroupInput":
            column[node] = 0
    # Pull a node with nothing upstream (a constant, Scene Time) next to what reads it.
    readers: dict = {}
    for node, sources in feeds.items():
        for src in sources:
            readers.setdefault(src, []).append(node)
    for node in nodes:
        if not feeds[node] and node.bl_idname != "NodeGroupInput" and readers.get(node):
            column[node] = max(min(column[r] for r in readers[node]) - 1, 0)
    return column


def arrange(ng, origin=(0.0, 0.0)) -> None:
    """Lay out every node of *ng* except frames, left to right along the flow of data."""
    nodes = [n for n in ng.nodes if n.bl_idname != "NodeFrame"]
    if not nodes:
        return
    feeds = _flow(nodes, ng.links)
    column = _columns(nodes, feeds)
    by_column: dict = {}
    for index, node in enumerate(nodes):
        by_column.setdefault(column[node], []).append((index, node))
    rank: dict = {}
    for col in sorted(by_column):
        entries = by_column[col]

        def key(entry):
            index, node = entry
            sources = [rank[s] for s in feeds[node] if s in rank]
            return (sum(sources) / len(sources) if sources else float(index), index)

        entries.sort(key=key)
        # Ranks centred on 0, as the columns are drawn: comparable from one column to the next.
        for pos, (_index, node) in enumerate(entries):
            rank[node] = pos - (len(entries) - 1) / 2.0
    x = float(origin[0])
    for col in sorted(by_column):
        entries = by_column[col]
        sizes = [node_size(node) for _i, node in entries]
        width = max(w for w, _h in sizes)
        total = sum(h for _w, h in sizes) + ROW_GAP * (len(sizes) - 1)
        y = float(origin[1]) + total / 2.0
        for (_index, node), (w, h) in zip(entries, sizes):
            node.location = (x + (width - w) / 2.0, y)
            y -= h + ROW_GAP
        x += width + COLUMN_GAP
