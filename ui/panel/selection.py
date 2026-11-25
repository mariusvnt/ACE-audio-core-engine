"""Selected-node settings and bake controls drawn in the N-panel."""

import textwrap
from ...core.id_management import ACEIdManager
from ...core.engine.engine import ACEAudioEngine
from ... import utils
from .splitter import _splitter_display_bounds, _splitter_hz_to_norm, _splitter_ramp_ascii, _sync_splitter_ramp_texture_from_node
from .sync import _resolve_ui_list_selected_node


def _resolve_capture_node(context):
    """
    Prefer the active graph list selection in the N-panel. Fallback to the GN
    editor active ACE node when there is no valid list selection.
    """
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


def _selected_ui_node_item(context):
    from ...utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return None
    idx = int(getattr(active_source, "active_ui_node_index", -1))
    if idx < 0 or idx >= len(getattr(active_source, "ui_nodes", [])):
        return None
    return active_source.ui_nodes[idx]


def _parsed_node_has_operator_settings(parsed) -> bool:
    if not parsed:
        return False
    key = (str(parsed.get("category", "") or ""), str(parsed.get("type", "") or ""))
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
    item = _selected_ui_node_item(context)
    if item is None:
        return ""
    return str(getattr(item, "description", "") or "").strip()


def _draw_selected_node_operator_settings(layout, context) -> bool:
    node, parsed = _resolve_capture_node(context)
    if not node or not _parsed_node_has_operator_settings(parsed):
        return False

    drew_settings = False

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

        split_sockets = []
        split_values = []
        for i, sock in enumerate(getattr(node, "inputs", [])):
            if i == 0:
                continue
            try:
                val = float(getattr(sock, "default_value", 0.0))
            except Exception as exc:
                utils.note_swallowed("n_panel_ui._draw_selected_node_operator_settings", exc)
                continue
            split_sockets.append(sock)
            split_values.append(val)
            with utils.logged_errors("n_panel_ui._draw_selected_node_operator_settings#2"):
                sock.min_value = 0.0
                sock.max_value = 20000.0
                sock.soft_min = 20.0
                sock.soft_max = 20000.0

        if split_values:
            bar = _splitter_ramp_ascii(sorted(split_values), low_hz=low_hz, high_hz=high_hz)
            box.label(text=f"{int(low_hz)}Hz {bar} {int(high_hz)}Hz")
            ramp_owner = _sync_splitter_ramp_texture_from_node(node, context.scene)
            if ramp_owner is not None:
                box.template_color_ramp(ramp_owner, "color_ramp", expand=True)
                ramp_ops = box.row(align=True)
                ramp_ops.operator("ace.sync_splitter_ramp", text="Sync From Node", icon="FILE_REFRESH")
                ramp_ops.operator("ace.apply_splitter_ramp", text="Apply Ramp", icon="CHECKMARK")
            tools = box.row(align=True)
            tools.operator("ace.sort_splitter_freqs", text="Sort", icon="SORTALPHA")
            tools.operator("ace.even_splitter_freqs", text="Even Log", icon="SNAP_MIDPOINT")
            for idx, sock in enumerate(split_sockets, start=1):
                sub = box.row(align=True)
                try:
                    hz = float(getattr(sock, "default_value", 0.0))
                except Exception as exc:
                    utils.note_swallowed("n_panel_ui._draw_selected_node_operator_settings#3", exc)
                    hz = 0.0
                pos = _splitter_hz_to_norm(hz, low_hz=low_hz, high_hz=high_hz)
                sub.label(text=f"Cut {idx} ({pos:.2f})")
                sub.prop(sock, "default_value", text="Hz", slider=True)

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


def _draw_selected_node_description(layout, description: str):
    desc = str(description or "").strip()
    if not desc:
        return
    wrapped = textwrap.wrap(desc, width=42) or [desc]
    for chunk in wrapped:
        layout.label(text=chunk)


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
        utils.note_swallowed("n_panel_ui._draw_bake_controls", exc)
        status = ""
    if not status:
        layout.label(text="Bake stream outputs over the scene frame range.", icon="INFO")
        return
    if status in {"failed", "error"}:
        err = ""
        try:
            err = str(active_source.get("ace_bake_error", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._draw_bake_controls#2", exc)
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
            utils.note_swallowed("n_panel_ui._draw_bake_controls#3", exc)
            progress = 0.0
        pct = int(max(0.0, min(progress, 1.0)) * 100.0)
        layout.label(text=f"Baking in progress... {pct}%", icon="TIME")
