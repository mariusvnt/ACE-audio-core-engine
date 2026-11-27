"""Frequency splitter split frequencies: edited by the panel's sliders, kept in ascending order."""

import bpy  # type: ignore
from .identity import node_kind
from .. import utils
from . import driven_values



MAX_HZ = 20000.0


def _split_freq_sockets(node) -> list:
    return list(node.inputs)[1:]


def get_split_freq(node, index: int) -> float:
    sockets = _split_freq_sockets(node)
    return float(sockets[index].default_value) if index < len(sockets) else 0.0


def set_split_freq(node, index: int, value: float) -> bool:
    """Write one split frequency, without reordering: a drag may still be going. True if it changed."""
    sockets = _split_freq_sockets(node)
    if index >= len(sockets):
        return False
    value = min(max(float(value), 0.0), MAX_HZ)
    if abs(sockets[index].default_value - value) < 0.01:  # float32 rounding is not a change
        return False
    sockets[index].default_value = value
    return True


def split_freq_states(node, scene=None) -> list[tuple]:
    """(socket, Hz or None, state) per split frequency, as the engine will use it.

    state: "own" (unlinked), "linked" (computed from the link) or "unreadable" (it cannot be
    computed, so there is no value; the link still wins over the socket's own value).
    """
    states = []
    for sock in _split_freq_sockets(node):
        value, state = float(sock.default_value), "own"
        if sock.is_linked:
            value = driven_values.linked_value(sock, scene)
            state = "linked" if value is not None else "unreadable"
        states.append((sock, value, state))
    return states


def capture_driven_splits(node, params: dict, chain=(), modifier=None, root_key: str = "") -> None:
    """Give the engine the values of linked split frequencies, and where to read them again.

    *chain* is the group nodes the walk entered to reach *node* and *modifier* the one at the
    root, so a link from a group's input reads what that use of the group receives.
    """
    linked = [i for i, sock in enumerate(node.inputs) if i and sock.is_linked]
    if not linked:
        return
    context = driven_values.WalkContext(chain=tuple(chain or ()), modifier=modifier, root_key=root_key)
    unknown = []
    for i in linked:
        value = driven_values.linked_value(node.inputs[i], bpy.context.scene, context)
        # The link wins over the socket's own value, even when it cannot be computed here (an
        # ACE output reaches the engine through its own link).
        params.pop(f"in_{i}", None)
        if value is None:
            unknown.append(i)
        else:
            params[f"in_{i}"] = value
    params["unknown_inputs"] = unknown
    params["driven_inputs"] = linked
    params["driven_node"] = driven_values.node_ref(node)
    params["driven_context"] = context.ref()


def refresh_driven_splits(graphs, scene) -> None:
    """Per frame: read the linked split frequencies again, without recompiling."""
    for graph in graphs:
        overrides: dict[str, dict] = {}
        for node_id, task in graph.tasks.items():
            params = getattr(task, "parameters", None)
            if not params or not params.get("driven_inputs"):
                continue
            node = driven_values.node_from_ref(params.get("driven_node"))
            if node is None:
                continue
            context = driven_values.WalkContext.from_ref(params.get("driven_context"))
            for i in params["driven_inputs"]:
                if i >= len(node.inputs) or not node.inputs[i].is_linked:
                    continue
                value = driven_values.linked_value(node.inputs[i], scene, context)
                if value is not None:
                    params[f"in_{i}"] = value
                    overrides.setdefault(str(node_id), {})[f"in_{i}"] = value
        graph.param_overrides = overrides  # sent to the worker processes with each frame


def sort_split_freqs(node) -> None:
    """Put the split frequencies in ascending order (linked ones keep their socket)."""
    sockets = [sock for sock in _split_freq_sockets(node) if not sock.is_linked]
    values = sorted(float(sock.default_value) for sock in sockets)
    for sock, value in zip(sockets, values):
        if sock.default_value != value:
            sock.default_value = value


def sort_all_split_freqs() -> None:
    """Order the split frequencies of every splitter in the open file. Not for use during a drag."""
    trees = [ng for ng in bpy.data.node_groups if ng.library is None and not ng.name.startswith(".ACE_")]
    trees += [m.node_tree for m in bpy.data.materials if m.library is None and m.node_tree is not None]
    for tree in trees:
        for node in list(tree.nodes):
            if node_kind(node) == ("delta", "frequency_splitter"):
                with utils.logged_errors("split_freqs.sort_all_split_freqs"):
                    sort_split_freqs(node)
