"""Selected-node settings and bake controls drawn in the N-panel."""

import textwrap
from ...core.id_management import ACEIdManager
from ...core.engine.engine import ACEAudioEngine
from ... import utils
from ...vars import ACE_NODES, node_modes, resolve_node_mode
from .splitter import _splitter_display_bounds, splitter_band_widths
from ...nodes.split_freqs import split_freq_states
from .helpers import _describe_node, _find_active_node_tree, _find_tree_node_by_ace_id
from .sync import _resolve_ui_list_selected_node, is_setting_active_row, note_editor_active, reveal_operator, is_setting_source_row, source_for_row
from ...nodes.modes import automatic_node_label


def _resolve_capture_node(context):
    node, parsed = _resolve_ui_list_selected_node(context)
    if node and parsed:
        return node, parsed

    node = getattr(context, "active_node", None)
    ace_id = str(node.get("ace_id", "") or "") if node and hasattr(node, "get") else ""
    if node and ace_id:
        parsed = ACEIdManager.parse_node_id(ace_id)
        if parsed:
            return node, parsed
    return None, None


def _source_for_node(node, scene):
    """The audio source a source placeholder node stands for."""
    source_id = str(node.get("ace_source_id", "") or "")
    for src in getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []:
        if source_id and str(getattr(src, "stable_id", "") or "") == source_id:
            return src
    return None


def _selected_node_for(active_source):
    """Node and parsed id of the operator selected in *active_source*'s list."""
    import bpy  # type: ignore

    node_id = str(getattr(active_source, "active_ui_node_id", "") or "")
    if not node_id:
        return None, None
    tree = _find_active_node_tree(bpy.context, editor_context_only=True)
    node, parsed = _find_tree_node_by_ace_id(tree, node_id)
    return (node, parsed) if node is not None and parsed else (None, None)


# Operator the rename box (ACE_PT_rename_operator) edits.
_rename_target_id = ""


def start_rename(node_id: str) -> None:
    global _rename_target_id
    _rename_target_id = str(node_id or "")


def rename_box_name(_wm) -> str:
    """Getter of ``WindowManager.ace_rename_name``."""
    import bpy  # type: ignore

    tree = _find_active_node_tree(bpy.context, editor_context_only=True)
    node, _parsed = _find_tree_node_by_ace_id(tree, _rename_target_id)
    return str(node.label or node.name or "") if node is not None else ""


def set_rename_box_name(_wm, value: str) -> None:
    """Setter of ``WindowManager.ace_rename_name``."""
    if _rename_target_id:
        rename_operator(_rename_target_id, value)


def selected_operator_name(active_source) -> str:
    """Name of the operator selected in *active_source*'s list."""
    node, _parsed = _selected_node_for(active_source)
    if node is None:
        return ""
    return str(node.label or node.name or "")


def rename_operator(node_id: str, value: str) -> str:
    """Rename operator *node_id*: the list and the node's title follow.

    An empty name restores the automatic one. A source placeholder renames its audio source,
    since the placeholder's title follows the source.
    """
    import bpy  # type: ignore

    tree = _find_active_node_tree(bpy.context, editor_context_only=True)
    node, parsed = _find_tree_node_by_ace_id(tree, node_id)
    if node is None or not parsed:
        return ""
    category, node_type = str(parsed.get("category", "")), str(parsed.get("type", ""))
    name = str(value or "").strip()
    if (category, node_type) == ("source_mngmt", "src"):
        source = _source_for_node(node, bpy.context.scene)
        if source is not None and name:
            source.name = name
            node.label = name
            node.name = name
        return str(node.label or node.name or "")
    if not name:
        name = automatic_node_label(node, category, node_type)
    node.label = name
    node.name = name
    return name


def rename_selected_operator(active_source, value: str) -> None:
    """Rename the operator selected in *active_source*'s list."""
    node_id = str(getattr(active_source, "active_ui_node_id", "") or "")
    if node_id:
        rename_operator(node_id, value)


def select_row_operator(active_source, _context) -> None:
    """Update of ``active_ui_node_index``: clicking a row's background selects its last operator."""
    if is_setting_active_row():
        return
    idx = int(active_source.active_ui_node_index)
    if 0 <= idx < len(active_source.ui_nodes):
        item = active_source.ui_nodes[idx]
        selected = str(active_source.active_ui_node_id or "")
        # A click on the selected name (drawn as a text field) lands on the row: keep it selected.
        if any(seg.kind == "node" and seg.node_id == selected for seg in item.segments):
            return
        if item.segments:
            focus_operator(bpy_context(), item.segments[-1].node_id, frame=False)


def select_source_row(project, context) -> None:
    """Update of ``active_source_row``: a source becomes active; an instance also selects its object."""
    if is_setting_source_row():
        return
    idx = int(project.active_source_row)
    if not 0 <= idx < len(project.source_rows):
        return
    item = project.source_rows[idx]
    owner = project.source_rows[item.parent_row] if item.parent_row >= 0 else item
    src_idx, _source = source_for_row(project, owner)
    if src_idx >= 0 and project.active_audio_source_index != src_idx:
        project.active_audio_source_index = src_idx
    if item.kind != "instance":
        return
    import bpy  # type: ignore

    obj = bpy.data.objects.get(item.target)
    view_layer = getattr(context, "view_layer", None)
    if obj is None or view_layer is None or view_layer.objects.get(obj.name) is None:
        return
    for other in view_layer.objects:
        if other.select_get():
            other.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj


def bpy_context():
    import bpy  # type: ignore

    return bpy.context


def _node_editor_area_for(context, tree):
    """(window, area, region) of a Node Editor showing *tree*, or None."""
    wm = getattr(context, "window_manager", None)
    for window in getattr(wm, "windows", []) or []:
        for area in getattr(getattr(window, "screen", None), "areas", []) or []:
            if area.type != "NODE_EDITOR":
                continue
            space = area.spaces.active
            if getattr(space, "edit_tree", None) != tree:
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                return window, area, region
    return None


def focus_operator(context, node_id: str, frame: bool = True) -> bool:
    """Select operator *node_id* in the list and in the Node Editor; *frame* centres the view on it."""
    import bpy  # type: ignore

    _, active_source = utils._get_active_source(context)
    if active_source is None:
        return False
    tree = _find_active_node_tree(context, editor_context_only=True)
    node, _parsed = _find_tree_node_by_ace_id(tree, node_id)
    if node is None:
        return False
    active_source.active_ui_node_id = node_id
    reveal_operator(active_source, node_id)
    for other in tree.nodes:
        if other.select:
            other.select = False
    node.select = True
    tree.nodes.active = node
    note_editor_active(tree, node_id)
    if frame:
        target = _node_editor_area_for(context, tree)
        if target is not None:
            window, area, region = target
            with context.temp_override(window=window, area=area, region=region):
                bpy.ops.node.view_selected()
    return True


def _parsed_node_has_operator_settings(parsed) -> bool:
    if not parsed:
        return False
    key = (str(parsed.get("category", "") or ""), str(parsed.get("type", "") or ""))
    if node_modes(*key):
        return True
    return key in {
        ("delta", "frequency_splitter"),
        ("utilities", "audio_math"),
        ("utilities", "preview_sink"),
        ("utilities", "util_rolling_avg"),
    }


def _selected_node_has_operator_settings(context) -> bool:
    node, parsed = _resolve_capture_node(context)
    return bool(node and _parsed_node_has_operator_settings(parsed))


def _selected_node_description(context) -> str:
    node, parsed = _resolve_ui_list_selected_node(context)
    if node is None or not parsed:
        return ""
    category, node_type = str(parsed.get("category", "")), str(parsed.get("type", ""))
    text = _describe_node(category, node_type)
    mode = resolve_node_mode(category, node_type, node.get("ace_mode"))
    for mode_id, label, desc in node_modes(category, node_type):
        if mode_id == mode:
            text = f"{text} {label}: {desc}"
    return text.strip()


def _draw_selected_operator(layout, context) -> None:
    """Settings of the selected operator (it is renamed in the list itself)."""
    node, parsed = _resolve_ui_list_selected_node(context)
    if node is not None and _parsed_node_has_operator_settings(parsed):
        _draw_selected_node_operator_settings(layout, context)


# Gap at each cut, before UI scale. Blender truncates widths to whole pixels: 1 px at scale 1.
_BAND_GAP_PX = 1.5

# Blender's colour for bundle sockets and their links.
_BUNDLE_SOCKET_COLOR = (0.30, 0.50, 0.50)

_band_fill: tuple[float, float, float] = _BUNDLE_SOCKET_COLOR


def band_fill_color() -> tuple[float, float, float]:
    return _band_fill


def _match_band_fill(context, shown: tuple) -> None:
    """Pick the swatch colour that looks like *shown* once drawn disabled, at half opacity over the box."""
    global _band_fill
    ui = context.preferences.themes[0].user_interface
    back, box = ui.panel_back, ui.wcol_box.inner
    under = [back[i] * (1.0 - box[3]) + box[i] * box[3] for i in range(3)]
    _band_fill = tuple(min(max(2.0 * shown[i] - under[i], 0.0), 1.0) for i in range(3))


def _draw_band_strip(layout, context, widths: list[float], low_hz: float, high_hz: float) -> None:
    """Bands as blocks in the audio socket colour, sized on the log-frequency axis, with a gap at each cut."""
    wm = context.window_manager
    _match_band_fill(context, _BUNDLE_SOCKET_COLOR)
    # The gap is a fixed number of pixels, so it stays visible in a narrow sidebar.
    scale_factor = context.preferences.system.ui_scale or 1.0  # 0 without a window
    strip_px = max(context.region.width - 50 * scale_factor, 100) if context.region else 300
    gap = min(_BAND_GAP_PX * scale_factor / strip_px, 0.02)
    band_share = max(1.0 - gap * max(len(widths) - 1, 0), 0.0)
    pieces: list[tuple[bool, float]] = []  # (is a band, share of the strip)
    for idx, width in enumerate(widths):
        if idx:
            pieces.append((False, gap))
        pieces.append((True, width * band_share))
    strip = layout.row(align=True)
    strip.scale_y = 0.8
    strip.enabled = False  # a colour swatch opens a picker unless disabled (drawn at half opacity)
    rest = strip
    remaining = 1.0

    def _piece(slot, is_band: bool) -> None:
        if is_band:
            slot.prop(wm, "ace_band_fill", text="")
        else:
            slot.label(text="")

    for idx, (is_band, width) in enumerate(pieces):
        if idx == len(pieces) - 1:
            _piece(rest, is_band)
            break
        # Never 0: Blender reads a zero split factor as "share equally".
        factor = min(max(width / remaining, 1e-4), 1.0) if remaining > 1e-6 else 1e-4
        split = rest.split(factor=factor, align=True)
        _piece(split, is_band)
        rest = split
        remaining -= width
    scale = layout.row()
    scale.label(text=f"{int(low_hz)} Hz")
    right = scale.row()
    right.alignment = 'RIGHT'
    right.label(text=f"{high_hz / 1000.0:g} kHz" if high_hz >= 1000.0 else f"{int(high_hz)} Hz")


def _draw_selected_node_operator_settings(layout, context) -> bool:
    node, parsed = _resolve_capture_node(context)
    if not node or not _parsed_node_has_operator_settings(parsed):
        return False

    drew_settings = False

    category = str(parsed.get("category", "") or "")
    node_type = str(parsed.get("type", "") or "")
    modes = node_modes(category, node_type)
    if modes:
        drew_settings = True
        box = layout.box()
        node_def = ACE_NODES.get(category, {}).get("nodes", {}).get(node_type, {})
        box.label(text=str(node_def.get("label", node_type)), icon="PRESET")
        current = resolve_node_mode(category, node_type, node.get("ace_mode"))
        row = box.row(align=True)
        for mode_id, label, _desc in modes:
            op = row.operator("ace.set_node_mode", text=label, depress=(current == mode_id))
            op.mode = mode_id
        for sock in getattr(node, "inputs", []):
            if sock.name == "Window (s)" and not sock.hide and not sock.is_linked:
                box.prop(sock, "default_value", text=sock.name)

    if parsed.get("category") == "delta" and parsed.get("type") == "frequency_splitter":
        drew_settings = True
        box = layout.box()
        box.label(text="Frequency Splitter", icon="MOD_WAVE")
        bands = int(node.get("ace_band_count", max(2, len(getattr(node, "outputs", []))))) if node else 2
        low_hz, high_hz = _splitter_display_bounds(context.scene)
        row = box.row(align=True)
        op = row.operator("ace.update_splitter_bands", text="", icon="REMOVE")
        op.direction = 'DOWN'
        row.label(text=f"{bands} Bands")
        op = row.operator("ace.update_splitter_bands", text="", icon="ADD")
        op.direction = 'UP'

        split_sockets = list(node.inputs)[1:] if node else []
        if split_sockets:
            _draw_band_strip(box, context, splitter_band_widths(node, context.scene), low_hz, high_hz)
            box.operator("ace.even_splitter_freqs", text="Distribute horizontally", icon="SNAP_MIDPOINT")
            for idx, (sock, value, state) in enumerate(split_freq_states(node, context.scene)):
                split = box.split(factor=0.28)
                split.label(text=sock.name)
                if state != "own":
                    text = {
                        "linked": f"Linked: {value:.0f} Hz" if value is not None else "",
                        "unreadable": f"Link not readable, uses {value:.0f} Hz" if value is not None else "",
                        "audio": "Linked to audio",
                    }[state]
                    split.label(text=text, icon="LINKED")
                    continue
                fields = split.split(factor=0.65, align=True)
                fields.prop(node, f"ace_split_pos_{idx + 1}", text="", slider=True)
                fields.prop(node, f"ace_split_freq_{idx + 1}", text="Hz")

    if parsed.get("category") == "utilities" and parsed.get("type") == "audio_math":
        drew_settings = True
        box = layout.box()
        box.label(text="Audio Math", icon="MODIFIER")
        inputs = int(node.get("ace_input_count", max(2, len(getattr(node, "inputs", []))))) if node else 2
        row = box.row(align=True)
        op = row.operator("ace.update_audio_math_inputs", text="", icon="REMOVE")
        op.direction = 'DOWN'
        row.label(text=f"{inputs} Inputs")
        op = row.operator("ace.update_audio_math_inputs", text="", icon="ADD")
        op.direction = 'UP'

        current_op = str(node.get("ace_op", "add")).lower()
        if current_op not in {"add", "subtract", "multiply"}:
            current_op = "add"
        row = box.row(align=True)
        for op_id, label in (("add", "Add"), ("subtract", "Subtract"), ("multiply", "Multiply")):
            op = row.operator("ace.set_audio_math_op", text=label, depress=(current_op == op_id))
            op.op = op_id

    if parsed.get("category") == "utilities" and parsed.get("type") == "preview_sink":
        drew_settings = True
        box = layout.box()
        box.label(text="Audio Preview", icon="PLAY_SOUND")
        enabled = bool(node.get("ace_preview_enabled", True))
        row = box.row(align=True)
        op = row.operator("ace.toggle_preview_sink", text="Enable", icon="PLAY", depress=enabled)
        op.preview_enabled = True
        op = row.operator("ace.toggle_preview_sink", text="Disable", icon="PAUSE", depress=(not enabled))
        op.preview_enabled = False
        box.label(text="Manual override for Group Output preview.", icon="INFO")
        box.label(text="Play timeline to hear this branch on the active source.", icon="INFO")
        if not ACEAudioEngine.preview_monitor_available():
            warn = box.row()
            warn.alert = True
            warn.label(text="Install SoundDevice in Preferences to enable preview audio.", icon="ERROR")
        else:
            err_count = ACEAudioEngine.preview_monitor_consecutive_errors()
            if err_count >= 5:
                warn = box.row()
                warn.alert = True
                warn.label(
                    text=f"Preview DSP error \u00d7{err_count} — check console for details.",
                    icon="ERROR",
                )

    if parsed.get("category") == "utilities" and parsed.get("type") == "util_rolling_avg":
        drew_settings = True
        box = layout.box()
        box.label(text="Rolling Average", icon="DRIVER")
        frames_sock = next(
            (sock for sock in getattr(node, "inputs", []) if getattr(sock, "name", "") == "Number of Frames"),
            None,
        )
        if frames_sock is not None:
            row = box.row(align=True)
            row.prop(frames_sock, "default_value", text="Number of Frames")
        else:
            box.label(text="Number of Frames input is unavailable.", icon="INFO")

    return drew_settings


def _draw_selected_node_description(layout, context, description: str):
    desc = str(description or "").strip()
    if not desc:
        return
    scale_factor = context.preferences.system.ui_scale or 1.0  # 0 without a window
    width_px = (context.region.width if context.region else 300) - 40 * scale_factor
    # Labels are a full widget row tall: a tight, squashed column reads as one paragraph.
    column = layout.column(align=True)
    column.scale_y = 0.7
    for line in textwrap.wrap(desc, width=max(24, int(width_px / (6.5 * scale_factor)))) or [desc]:
        column.label(text=line)


def _draw_bake_controls(layout, context):
    from ...utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return

    row = layout.row()
    row.enabled = getattr(active_source, "source_type", "") == 'FILE'
    row.operator("audiocore.bake_active_source", icon="FILE_TICK", text="Bake Active Source")
    if not row.enabled:
        return

    try:
        status = str(active_source.get("ace_bake_status", "") or "")
    except Exception as exc:
        utils.note_swallowed("selection._draw_bake_controls", exc)
        status = ""
    if not status:
        layout.label(text="Bake stream outputs over the scene frame range.", icon="INFO")
        return
    if status in {"failed", "error"}:
        err = ""
        try:
            err = str(active_source.get("ace_bake_error", "") or "")
        except Exception as exc:
            utils.note_swallowed("selection._draw_bake_controls#2", exc)
            err = ""
        warn = layout.row()
        warn.alert = True
        warn.label(text=f"Bake failed: {err or 'Unknown error'}", icon="ERROR")
    elif status in {"ready", "done"}:
        start = int(active_source.get("ace_bake_frame_start", 0))
        end = int(active_source.get("ace_bake_frame_end", -1))
        layout.label(text=f"Bake ready: frames {start} -> {end}", icon="CHECKMARK")
    elif status == "baking":
        progress = 0.0
        try:
            progress = float(active_source.get("ace_bake_progress", 0.0) or 0.0)
        except Exception as exc:
            utils.note_swallowed("selection._draw_bake_controls#3", exc)
            progress = 0.0
        pct = int(max(0.0, min(progress, 1.0)) * 100.0)
        layout.label(text=f"Baking in progress... {pct}%", icon="TIME")
