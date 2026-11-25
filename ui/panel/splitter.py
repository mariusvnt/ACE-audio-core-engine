"""Frequency splitter display: band bounds and the ramp preview texture."""

import bpy  # type: ignore
from ...core.engine.engine import ACEAudioEngine
from ... import utils


_SPLITTER_LOW_HZ = 20.0


_SPLITTER_HIGH_HZ = 20000.0


_SPLITTER_RAMP_NODETREE_NAME = ".ACE_SplitterRampUI"


_SPLITTER_RAMP_NODE_NAME = "ACE_SplitterRampNode"


_splitter_ramp_last_key = ""


_splitter_ramp_last_sig = ""


def _splitter_hz_to_norm(hz: float, low_hz: float = _SPLITTER_LOW_HZ, high_hz: float = _SPLITTER_HIGH_HZ) -> float:
    """Map a frequency in Hz to [0, 1] on a logarithmic scale.

    Args:
        hz: Frequency value in Hz.
        low_hz: Lower bound of the display range in Hz.
        high_hz: Upper bound of the display range in Hz.

    Returns:
        Normalised position in [0, 1].
    """
    import math
    low = max(float(low_hz), 1e-3)
    high = max(float(high_hz), low + 1e-3)
    h = min(max(float(hz), low), high)
    return float((math.log(h) - math.log(low)) / (math.log(high) - math.log(low)))


def _splitter_norm_to_hz(t: float, low_hz: float = _SPLITTER_LOW_HZ, high_hz: float = _SPLITTER_HIGH_HZ) -> float:
    """Map a normalised [0, 1] position back to a frequency in Hz.

    Inverse of _splitter_hz_to_norm.

    Args:
        t: Normalised position in [0, 1].
        low_hz: Lower bound of the display range in Hz.
        high_hz: Upper bound of the display range in Hz.

    Returns:
        Frequency in Hz.
    """
    import math
    low = max(float(low_hz), 1e-3)
    high = max(float(high_hz), low + 1e-3)
    tc = min(max(float(t), 0.0), 1.0)
    return float(math.exp(math.log(low) + tc * (math.log(high) - math.log(low))))


def _splitter_display_bounds(scene) -> tuple[float, float]:
    """Return the (low_hz, high_hz) display range for the splitter ramp.

    The upper bound is clamped to just below the Nyquist frequency of the
    active source when its sample rate is known, so cut-point sliders never
    exceed the useful analysis range.
    """
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


def _splitter_cut_sockets(node) -> list:
    """Return the unlinked cut-frequency input sockets of a FrequencySplitter node.

    Skips the first input (Audio bundle) and any linked sockets since linked
    values are not user-editable cut points.

    Args:
        node: A Blender GeometryNodeGroup node.

    Returns:
        List of NodeSocket objects representing cut-frequency inputs.
    """
    sockets = []
    for i, sock in enumerate(getattr(node, "inputs", [])):
        if i == 0:
            continue
        if bool(getattr(sock, "is_linked", False)):
            continue
        try:
            _ = float(getattr(sock, "default_value", 0.0))
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._splitter_cut_sockets", exc)
            continue
        sockets.append(sock)
    return sockets


def _splitter_node_signature(node, scene) -> str:
    """Build a cache-key string representing the current splitter cut-point configuration.

    Encodes the display bounds (Hz) and all unlinked cut-frequency socket values
    so the ramp texture is only redrawn when the configuration actually changes.
    """
    low_hz, high_hz = _splitter_display_bounds(scene)
    values = []
    for sock in _splitter_cut_sockets(node):
        with utils.logged_errors("n_panel_ui._splitter_node_signature"):
            values.append(float(getattr(sock, "default_value", 0.0)))
    values = sorted(values)
    rounded = ",".join(f"{v:.3f}" for v in values)
    return f"{low_hz:.3f}|{high_hz:.3f}|{rounded}"


def _ensure_splitter_ramp_texture():
    """
    Returns an RNA owner that exposes a `color_ramp` property for
    `layout.template_color_ramp(...)`.
    """
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    try:
        node_groups = bpy.data.node_groups
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._ensure_splitter_ramp_texture", exc)
        return None

    tree = node_groups.get(_SPLITTER_RAMP_NODETREE_NAME)
    if ACEAudioEngine._undo_recovery_active_for_scene(scene):
        if tree is None:
            return None
        ramp_node = tree.nodes.get(_SPLITTER_RAMP_NODE_NAME)
        if ramp_node is None or getattr(ramp_node, "bl_idname", "") != 'ShaderNodeValToRGB':
            return None
        return ramp_node
    if tree is None:
        try:
            tree = node_groups.new(_SPLITTER_RAMP_NODETREE_NAME, 'ShaderNodeTree')
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ensure_splitter_ramp_texture#2", exc)
            return None

    ramp_node = tree.nodes.get(_SPLITTER_RAMP_NODE_NAME)
    if ramp_node is None or getattr(ramp_node, "bl_idname", "") != 'ShaderNodeValToRGB':
        try:
            if ramp_node is not None:
                tree.nodes.remove(ramp_node)
            ramp_node = tree.nodes.new('ShaderNodeValToRGB')
            ramp_node.name = _SPLITTER_RAMP_NODE_NAME
            ramp_node.label = "ACE Splitter Ramp"
            ramp_node.location = (0, 0)
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ensure_splitter_ramp_texture#3", exc)
            return None

    return ramp_node


def _sync_splitter_ramp_texture_from_node(node, scene, force: bool = False):
    """Update the hidden splitter ramp ColorRamp from the current node cut-point values.

    Rate-limited by a signature cache: the ramp is only redrawn when the cut-point
    configuration has changed since the last call.  Pass ``force=True`` to skip the
    cache check (used when the node selection changes).

    Returns the ramp_owner node, or None if the ShaderNodeTree could not be created.
    """
    global _splitter_ramp_last_key, _splitter_ramp_last_sig
    ramp_owner = _ensure_splitter_ramp_texture()
    if ramp_owner is None:
        return None

    node_key = str(getattr(node, "as_pointer", lambda: 0)() or 0)
    sig = _splitter_node_signature(node, scene)
    if (not force) and node_key == _splitter_ramp_last_key and sig == _splitter_ramp_last_sig:
        return ramp_owner

    cut_values = []
    for sock in _splitter_cut_sockets(node):
        with utils.logged_errors("n_panel_ui._sync_splitter_ramp_texture_from_node"):
            cut_values.append(float(getattr(sock, "default_value", 0.0)))
    cut_values = sorted(cut_values)

    ramp = ramp_owner.color_ramp
    target_len = len(cut_values) + 2
    while len(ramp.elements) < target_len:
        ramp.elements.new(1.0)
    while len(ramp.elements) > target_len and len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])

    low_hz, high_hz = _splitter_display_bounds(scene)
    ramp.elements[0].position = 0.0
    ramp.elements[0].color = (0.05, 0.35, 0.95, 1.0)
    ramp.elements[-1].position = 1.0
    ramp.elements[-1].color = (0.95, 0.25, 0.05, 1.0)
    for i, hz in enumerate(cut_values, start=1):
        t = _splitter_hz_to_norm(hz, low_hz=low_hz, high_hz=high_hz)
        t = min(max(t, 0.0), 1.0)
        hue = float(i) / float(max(1, len(cut_values) + 1))
        ramp.elements[i].position = t
        ramp.elements[i].color = (0.15 + 0.65 * hue, 0.85 - 0.45 * hue, 0.30, 1.0)

    _splitter_ramp_last_key = node_key
    _splitter_ramp_last_sig = sig
    return ramp_owner


def _splitter_ramp_ascii(
    freqs: list[float],
    width: int = 28,
    low_hz: float = _SPLITTER_LOW_HZ,
    high_hz: float = _SPLITTER_HIGH_HZ,
) -> str:
    """Render splitter cut-point positions as a compact ASCII ruler string.

    Positions each frequency as a ``|`` mark on a ``-``-filled ruler of the
    given *width*, using a logarithmic frequency scale.  Used for compact
    text-based band visualisation in the N-panel header.

    Args:
        freqs: List of cut-point frequencies in Hz.
        width: Total character width of the output string.
        low_hz: Lower Hz bound for the logarithmic scale.
        high_hz: Upper Hz bound for the logarithmic scale.

    Returns:
        A string of length *width* consisting of ``-`` and ``|`` characters.
    """
    if width <= 2:
        return ""
    marks = ["-"] * width
    for f in freqs:
        t = _splitter_hz_to_norm(float(f), low_hz=low_hz, high_hz=high_hz)
        idx = max(0, min(width - 1, int(round(t * (width - 1)))))
        marks[idx] = "|"
    return "".join(marks)
