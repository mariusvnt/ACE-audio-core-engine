"""Frequency splitter display: band bounds and the band strip."""

import bpy  # type: ignore
from ... import utils
from ...nodes.split_freqs import split_freq_states


_SPLITTER_LOW_HZ = 20.0


_SPLITTER_HIGH_HZ = 20000.0


# Hidden ramp editor group made by earlier versions; removed when found.
_SPLITTER_RAMP_NODETREE_NAME = ".ACE_SplitterRampUI"


def _splitter_hz_to_norm(hz: float, low_hz: float = _SPLITTER_LOW_HZ, high_hz: float = _SPLITTER_HIGH_HZ) -> float:
    import math
    low = max(float(low_hz), 1e-3)
    high = max(float(high_hz), low + 1e-3)
    h = min(max(float(hz), low), high)
    return float((math.log(h) - math.log(low)) / (math.log(high) - math.log(low)))


def _splitter_display_bounds(scene) -> tuple[float, float]:
    """Return the (low_hz, high_hz) display range for the splitter ramp."""
    low_hz = float(_SPLITTER_LOW_HZ)
    high_hz = float(_SPLITTER_HIGH_HZ)
    ace = getattr(scene, "audiocore", None)
    if ace and getattr(ace, "audio_sources", None):
        idx = int(getattr(ace, "active_audio_source_index", 0) or 0)
        if 0 <= idx < len(ace.audio_sources):
            src = ace.audio_sources[idx]
            sr = float(getattr(src, "sample_rate", 0.0) or 0.0)
            if sr > 0.0:
                high_hz = min(high_hz, max(100.0, 0.49 * sr))
    if high_hz <= low_hz:
        high_hz = low_hz + 1e-3
    return low_hz, high_hz


def splitter_band_widths(node, scene) -> list[float]:
    """Width of each band on the log-frequency axis (sums to 1), from the cut frequencies.

    A cut driven by an audio value is left out: it is not known here.
    """
    low_hz, high_hz = _splitter_display_bounds(scene)
    cuts = [
        min(max(_splitter_hz_to_norm(value, low_hz=low_hz, high_hz=high_hz), 0.0), 1.0)
        for _sock, value, _state in split_freq_states(node, scene)
        if value is not None
    ]
    edges = [0.0] + sorted(cuts) + [1.0]
    return [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]


def remove_legacy_ramp_group() -> None:
    """Files from earlier versions hold a hidden ramp editor group (with leaked nodes): drop it."""
    tree = bpy.data.node_groups.get(_SPLITTER_RAMP_NODETREE_NAME)
    if tree is not None:
        bpy.data.node_groups.remove(tree)
