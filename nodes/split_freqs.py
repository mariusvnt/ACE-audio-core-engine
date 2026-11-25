"""Frequency splitter split frequencies: edited by the panel's sliders, kept in ascending order."""

import math
import bpy  # type: ignore
from ..core.id_management import ACEIdManager
from .. import utils
from . import driven_values


_SPLITTER_ID_PREFIX = f"{ACEIdManager.PREFIX}delta_frequency_splitter_"

# Range of the logarithmic sliders (the band strip's axis). The Hz fields accept 0 to MAX_HZ.
_SLIDER_LOW_HZ = 20.0
MAX_HZ = 20000.0


def position_to_freq(position: float) -> float:
    """Slider position (0..1) to Hz, logarithmically."""
    return _SLIDER_LOW_HZ * (MAX_HZ / _SLIDER_LOW_HZ) ** min(max(float(position), 0.0), 1.0)


def freq_to_position(hz: float) -> float:
    hz = min(max(float(hz), _SLIDER_LOW_HZ), MAX_HZ)
    return math.log(hz / _SLIDER_LOW_HZ) / math.log(MAX_HZ / _SLIDER_LOW_HZ)


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

    state: "own" (unlinked), "linked" (computed from the link), "unreadable" (the link cannot be
    computed: the socket's own value is used) or "audio" (driven by an ACE output: not known here).
    """
    states = []
    for sock in _split_freq_sockets(node):
        value, state = float(sock.default_value), "own"
        if sock.is_linked:
            driven = driven_values.linked_value(sock, scene)
            if driven is not None:
                value, state = driven, "linked"
            elif driven_values.fed_by_audio(sock):
                value, state = None, "audio"
            else:
                state = "unreadable"
        states.append((sock, value, state))
    return states


def capture_driven_splits(node, params: dict) -> None:
    """Give the engine the values of linked split frequencies, and where to read them again."""
    linked = [i for i, sock in enumerate(node.inputs) if i and sock.is_linked]
    if not linked:
        return
    for i in linked:
        value = driven_values.linked_value(node.inputs[i], bpy.context.scene)
        # An ACE output reaches the engine through its own link, which takes precedence.
        params[f"in_{i}"] = float(node.inputs[i].default_value) if value is None else value
    params["driven_inputs"] = linked
    params["driven_node"] = driven_values.node_ref(node)


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
            for i in params["driven_inputs"]:
                if i >= len(node.inputs) or not node.inputs[i].is_linked:
                    continue
                value = driven_values.linked_value(node.inputs[i], scene)
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
            if str(node.get("ace_id", "") or "").startswith(_SPLITTER_ID_PREFIX):
                with utils.logged_errors("split_freqs.sort_all_split_freqs"):
                    sort_split_freqs(node)
