"""Selected-node settings and bake controls drawn in the N-panel."""

import textwrap
from ...nodes.identity import node_id as ace_node_id, parsed_node
from ...nodes.source_nodes import node_source_id
from ...core.engine.engine import ACEAudioEngine
from ... import utils
from ...vars import ACE_NODES, PREVIEW_KIND, SOURCE_KIND, SPLITTER_KIND, node_modes, resolve_node_mode
from .splitter import _splitter_display_bounds, splitter_band_widths
from ...nodes.split_freqs import split_freq_states
from .helpers import _describe_node, _find_active_node_tree, _find_tree_node_by_ace_id, property_column, property_split_row
from .sync import _resolve_ui_list_selected_node, is_setting_active_row, note_editor_active, reveal_operator, is_setting_source_row, source_for_row
from ...nodes.modes import automatic_node_label


def _resolve_capture_node(context):
    node, parsed = _resolve_ui_list_selected_node(context)
    if node and parsed:
        return node, parsed

    node = getattr(context, "active_node", None)
    parsed = parsed_node(node) if node is not None else None
    if parsed:
        return node, parsed
    return None, None


def _source_for_node(node, scene):
    """The audio source a source placeholder node stands for."""
    source_id = node_source_id(node)
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
    if (category, node_type) == SOURCE_KIND:
        source = _source_for_node(node, bpy.context.scene)
        if source is not None and name:
            source.name = name
            node.label = name
            node.name = name
            _follow_renamed_node(node, node_id)
        return str(node.label or node.name or "")
    if not name:
        name = automatic_node_label(node, category, node_type)
    node.label = name
    node.name = name
    _follow_renamed_node(node, node_id)
    return name


def _follow_renamed_node(node, old_id: str) -> None:
    """A node's id comes from its name: the lists that selected it select its new id."""
    import bpy  # type: ignore

    new_id = ace_node_id(node)
    for source in getattr(getattr(bpy.context.scene, "audiocore", None), "audio_sources", []) or []:
        if str(getattr(source, "active_ui_node_id", "") or "") == old_id:
            source.active_ui_node_id = new_id
    global _rename_target_id
    if _rename_target_id == old_id:
        _rename_target_id = new_id


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
        if any(seg.kind in ("node", "split") and seg.node_id == selected for seg in item.segments):
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


def _source_choices() -> list:
    project = getattr(getattr(bpy_context(), "scene", None), "audiocore", None)
    return list(getattr(project, "audio_sources", []) or [])


def _parsed_node_has_operator_settings(parsed) -> bool:
    if not parsed:
        return False
    key = (str(parsed.get("category", "") or ""), str(parsed.get("type", "") or ""))
    if key == SOURCE_KIND:
        return len(_source_choices()) > 1
    if node_modes(*key):
        return True
    return key in {SPLITTER_KIND, PREVIEW_KIND} or key[1] == "util_rolling_avg"


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
    """Settings of the selected operator (it is renamed in the list itself), and the links it ignores."""
    node, parsed = _resolve_ui_list_selected_node(context)
    if node is not None and _parsed_node_has_operator_settings(parsed):
        _draw_selected_node_operator_settings(layout, context)
    if node is not None:
        draw_ignored_links(layout, context, ignored_links_into(node))


def ignored_links_into(node) -> list:
    """The links into *node* that ACE ignores, read from its tree's definition."""
    from ...nodes.definitions import definition

    tree = getattr(node, "id_data", None)
    if tree is None:
        return []
    with utils.logged_errors("selection.ignored_links_into"):
        return [item for item in definition(tree).ignored if item.node_name == node.name]
    return []


def draw_ignored_links(layout, context, items, show_node: bool = False) -> None:
    """One box per ignored link: the input, where it comes from, and why ACE ignores it."""
    for item in items:
        box = layout.box()
        head = box.row()
        head.alert = True
        title = f"{item.node_name}: {item.input_name}" if show_node else item.input_name
        head.label(text=f"{title} ← {item.from_node} (link ignored)", icon="ERROR")
        _draw_selected_node_description(box, context, item.reason)


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
    if (category, node_type) == SOURCE_KIND:
        drew_settings = True
        box = layout.box()
        box.label(text="Audio Source", icon="SOUND")
        current = node_source_id(node)
        name = next((s.name for s in _source_choices() if s.stable_id == current), "Choose a source")
        property_split_row(box, "Input").operator_menu_enum("ace.set_node_source", "source", text=name)

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
                property_column(box).prop(sock, "default_value", text=sock.name)

    if (parsed.get("category"), parsed.get("type")) == SPLITTER_KIND:
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
                field = property_split_row(box, sock.name)
                if state == "own":
                    # The frequency unit writes "Hz" after the value, unless the scene has no units.
                    unitless = context.scene.unit_settings.system == "NONE"
                    field.prop(node, f"ace_split_freq_{idx + 1}", text="Hz" if unitless else "", slider=True)
                    continue
                field.label(text="" if value is None else f"{value:.0f} Hz")
                if state == "ignored":
                    field.label(text="Ignored", icon="ERROR")
                else:
                    field.label(text="Linked", icon="LINKED")

    if (parsed.get("category"), parsed.get("type")) == PREVIEW_KIND:
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
            warn.label(text="SoundDevice failed to load: the console shows why.", icon="ERROR")
        else:
            err_count = ACEAudioEngine.preview_monitor_consecutive_errors()
            if err_count >= 5:
                warn = box.row()
                warn.alert = True
                warn.label(
                    text=f"Preview DSP error \u00d7{err_count} — check console for details.",
                    icon="ERROR",
                )

    if parsed.get("type") == "util_rolling_avg":
        drew_settings = True
        box = layout.box()
        box.label(text="Rolling Average", icon="DRIVER")
        frames_sock = next(
            (sock for sock in getattr(node, "inputs", []) if getattr(sock, "name", "") == "Window"),
            None,
        )
        if frames_sock is not None:
            property_column(box).prop(frames_sock, "default_value", text="Window")
        else:
            box.label(text="Window input is unavailable.", icon="INFO")

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


def _draw_render_notes(layout, context):
    """What the last render could not show exactly (core/engine/handlers.py, render_init)."""
    engine = ACEAudioEngine.get_instance(context.scene)
    notes = list(getattr(engine, "render_notes", []) or []) if engine is not None else []
    if not notes:
        return
    box = layout.box()
    box.label(text="Last render:", icon="RENDER_ANIMATION")
    for note in notes:
        row = box.row()
        row.alert = True
        row.label(text=note, icon="ERROR")


def _draw_bake_state(layout, context, source):
    """How much of the source is baked, and whether its bake is read or it runs live."""
    engine = ACEAudioEngine.get_instance(context.scene)
    uid = str(getattr(source, "stable_id", "") or "")
    view = engine.bake.store.read(uid) if engine is not None and uid else None
    if view is None:
        return
    graph = engine.graphs.get(uid) if engine is not None else None
    outputs = list(getattr(graph, "output_attr_keys", []) or []) if graph is not None else []
    baked = sum(1 for key in outputs if key in view.series)
    is_take = view.origin == "take"
    use = bool(source.get("ace_bake_use", not is_take))
    box = layout.box()
    row = box.row()
    what = "Take" if is_take else "Baked"
    row.label(text=f"{what}: {baked} of {len(outputs)} outputs, frames {view.frame_start} -> {view.frame_end}",
              icon="CHECKMARK" if outputs and baked == len(outputs) else "INFO")
    if outputs and baked < len(outputs):
        box.label(text="Outputs changed since the bake are computed live. Bake again to include them.", icon="ERROR")
    row = box.row(align=True)
    label = ("Using Take" if use else "Live (Use Take)") if is_take else ("Using Bake" if use else "Live")
    row.operator("audiocore.toggle_source_bake", text=label, icon="CHECKBOX_HLT" if use else "CHECKBOX_DEHLT", depress=use)
    row.operator("audiocore.clear_source_bake", text="Clear Bake", icon="TRASH")


def _draw_downstream_bakes(layout, context):
    """The Bake nodes downstream of ACE nodes, and whether the sources they read are baked.

    ACE bakes nothing for them: a native bake records what ACE streams on each frame it steps.
    """
    engine = ACEAudioEngine.get_instance(context.scene)
    bakes = list(getattr(getattr(engine, "bake", None), "downstream_bakes", []) or []) if engine is not None else []
    if not bakes:
        return
    box = layout.box()
    box.label(text="Bake nodes downstream:", icon="NODETREE")
    for bake in bakes:
        baked = all(
            (graph := engine.graphs.get(uid)) is not None
            and not engine.bake.missing_keys(graph, bake["frame_start"], bake["frame_end"])
            for uid in bake["sources"]
        )
        state = "baked" if baked else "streamed"
        box.label(text=f"{bake['label']}: ACE input {state}", icon="CHECKMARK" if baked else "PLAY")


def _draw_bake_controls(layout, context):
    from ...utils import _get_active_source

    _draw_render_notes(layout, context)
    _draw_downstream_bakes(layout, context)
    _, active_source = _get_active_source(context)
    if not active_source:
        return
    if getattr(active_source, "source_type", "") == 'LIVE':
        _draw_bake_state(layout, context, active_source)  # its take, if one was recorded

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
    _draw_bake_state(layout, context, active_source)
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
