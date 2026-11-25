"""Frequency splitter split frequencies: edited by the panel's sliders, kept in ascending order."""

import bpy  # type: ignore
from ..core.id_management import ACEIdManager
from .. import utils


_SPLITTER_ID_PREFIX = f"{ACEIdManager.PREFIX}delta_frequency_splitter_"


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
    value = min(max(float(value), 0.0), 20000.0)
    if abs(sockets[index].default_value - value) < 1e-3:
        return False
    sockets[index].default_value = value
    return True


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
