"""N-panel lists and panels."""

import math
import blf  # type: ignore
import bpy  # type: ignore
import textwrap
from ...core.engine.engine import ACEAudioEngine
from ... import utils
from .helpers import _deal_breakers, _source_display_sample_rate, _source_list_icon, _ui_is_blocked, property_column
from .selection import _draw_bake_controls, _draw_selected_node_description, _draw_selected_operator, _selected_node_description
from . import operator_tree
from . import sync
from . import usage
from .sync import _engine_latency_metrics, _ui_click_timer, _ui_load_post, _ui_sync_timer


# Blender sizes a text button as its text plus this many UI units (an icon adds a little).
_BUTTON_PAD_UNITS = 1.5
_ICON_PAD_UNITS = 1.25
_SEPARATOR_UNITS = 1.0

_column_cache: tuple = (None, {})


def _operator_columns(context, items) -> dict[int, list[float]]:
    """Column widths for each group of rows under the same branch, keyed by their parent row."""
    global _column_cache
    scale = context.preferences.system.ui_scale or 1.0
    shape = tuple(
        (int(it.parent_row), int(it.depth), tuple((seg.label, seg.kind) for seg in it.segments)) for it in items
    )
    width_px = context.region.width if context.region else 300
    key = (shape, width_px, scale)
    if _column_cache[0] == key:
        return _column_cache[1]
    blf.size(0, context.preferences.ui_styles[0].widget.points * scale)
    unit = 20.0 * scale

    def units(label: str, kind: str) -> float:
        return blf.dimensions(0, label)[0] / unit + _BUTTON_PAD_UNITS + (_ICON_PAD_UNITS if kind == "ref" else 0.0)

    groups: dict[int, list] = {}
    for parent, depth, segments in shape:
        if segments:
            groups.setdefault(parent, []).append((depth + 1.0, [units(*seg) for seg in segments]))
    available = width_px / unit - 3.0
    widths = {parent: operator_tree.column_widths(rows, available, _SEPARATOR_UNITS) for parent, rows in groups.items()}
    _column_cache = (key, widths)
    return widths


def _draw_operator_indent(row, item) -> None:
    """Indentation and the fold arrow of a list row."""
    for _ in range(int(item.depth)):
        row.label(text="", icon='BLANK1')
    if not item.has_children:
        row.label(text="", icon='BLANK1')
        return
    folded = item.branch_key in sync.collapsed_keys()
    op = row.operator(
        "ace.toggle_operator_branch",
        text="",
        icon='DISCLOSURE_TRI_RIGHT' if folded else 'DISCLOSURE_TRI_DOWN',
        emboss=False,
    )
    op.branch_key = item.branch_key


class ACE_UL_dummy_tree(bpy.types.UIList):
    """Active Operators drawn as a folded tree.

    Each row is a chain of operators without branching; every name is a button that
    selects its node, and the selected one is drawn pressed. Double-click or Ctrl+click
    a name to rename it (ACE_PT_rename_operator). Rows are indented by depth and the rows below a branch fold away.
    """

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='NODE')
            return
        active_id = str(getattr(data, "active_ui_node_id", "") or "")
        widths = _operator_columns(context, data.ui_nodes).get(int(item.parent_row), [])
        row = layout.row(align=True)
        row.alignment = 'LEFT'
        last = len(item.segments) - 1
        for seg_idx, seg in enumerate(item.segments):
            if seg_idx:
                separator = row.row(align=True)
                separator.ui_units_x = _SEPARATOR_UNITS
                separator.label(text="\u203a")
            cell = row
            if seg_idx < last and seg_idx < len(widths):
                cell = row.row(align=True)
                cell.ui_units_x = widths[seg_idx]
                cell.alignment = 'LEFT'
            if seg_idx == 0:
                _draw_operator_indent(cell, item)
            # A splitter's bands are all the splitter: only the active row's one is pressed.
            selected = seg.node_id == active_id and (
                seg.kind == "node" or (seg.kind == "split" and index == getattr(active_data, active_propname, -1))
            )
            op = cell.operator(
                "ace.select_operator",
                text=seg.label,
                icon='LINKED' if seg.kind == "ref" else 'NONE',
                emboss=selected,
                depress=selected,
            )
            op.node_id = seg.node_id
            op.row = index

    def draw_filter(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "filter_name", text="", icon='VIEWZOOM')

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        rows = [
            operator_tree.Row(
                depth=int(it.depth),
                segments=[operator_tree.Segment(seg.node_id, seg.label, seg.kind) for seg in it.segments],
                has_children=bool(it.has_children),
                branch_key=str(it.branch_key),
                parent=int(it.parent_row),
            )
            for it in items
        ]
        shown = operator_tree.visible_rows(rows, sync.collapsed_keys(), self.filter_name)
        return [self.bitflag_filter_item if visible else 0 for visible in shown], []


class ACE_UL_source_list(bpy.types.UIList):
    """Audio sources, each folding open onto the objects it drives (closed by default).

    A source row shows a status icon (ERROR when its proxy is missing, FILE or SPEAKER) and its
    editable name; clicking it makes it the active source. Clicking an object selects it.
    """

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        if item.kind == "source":
            if item.has_children:
                opened = item.branch_key in sync.expanded_source_keys()
                op = row.operator(
                    "ace.toggle_operator_branch",
                    text="",
                    icon='DISCLOSURE_TRI_DOWN' if opened else 'DISCLOSURE_TRI_RIGHT',
                    emboss=False,
                )
                op.branch_key = item.branch_key
                op.list_id = "sources"
            else:
                row.label(text="", icon='BLANK1')
            _index, source = sync.source_for_row(data, item)
            if source is None:
                row.label(text=item.name)
                return
            row.prop(source, "name", text="", emboss=False, icon=_source_list_icon(source))
            return
        row.label(text="", icon='BLANK1')
        row.label(text="", icon='BLANK1')
        split = row.split(factor=0.62, align=True)
        split.label(text=item.name, icon=item.icon or 'NONE')
        right = split.row(align=True)
        right.alignment = 'RIGHT'
        right.label(text=item.detail, icon=item.detail_icon or 'NONE')

    def draw_filter(self, context, layout):
        layout.row(align=True).prop(self, "filter_name", text="", icon='VIEWZOOM')

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        rows = [
            operator_tree.Row(
                depth=int(it.depth),
                segments=[operator_tree.Segment(it.target, it.name)],
                has_children=bool(it.has_children),
                branch_key=str(it.branch_key),
                parent=int(it.parent_row),
            )
            for it in items
        ]
        closed = {row.branch_key for row in rows if row.has_children} - sync.expanded_source_keys()
        shown = operator_tree.visible_rows(rows, closed, self.filter_name)
        return [self.bitflag_filter_item if visible else 0 for visible in shown], []


class ACE_PT_rename_operator(bpy.types.Panel):
    """Rename box opened at the cursor by a double-click in the Active Operators list (like F2)."""
    bl_space_type = 'TOPBAR'  # dummy: only shown through wm.call_panel
    bl_region_type = 'HEADER'
    bl_label = "Rename Operator"
    bl_ui_units_x = 14

    def draw(self, context):
        layout = self.layout
        layout.label(text="Operator Name")
        row = layout.row()
        row.activate_init = True
        row.label(icon='NODE')
        row.prop(context.window_manager, "ace_rename_name", text="")


class ACE_PT_presentation_panel(bpy.types.Panel):
    """Panel displaying the add-on logo."""
    bl_label = "Audio Core Engine"
    bl_idname = "ACE_PT_presentation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_order = 0

    def draw(self, context):
        from ...utils import _ensure_logo_icon
        layout = self.layout
        icon_id = _ensure_logo_icon()
        if icon_id:
            layout.template_icon(icon_value=icon_id, scale=8.0)
        else:
            layout.label(text="Audio Core Engine", icon="SOUND")


class ACE_PT_unsatisfactory_config(bpy.types.Panel):
    """Shown when a blocking configuration issue is detected."""
    bl_label = "Configuration Issues"
    bl_idname = "ACE_PT_unsatisfactory_config"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return _ui_is_blocked(context)

    def draw(self, context):
        layout = self.layout
        issues = _deal_breakers()
        if not issues:
            layout.label(text="No blocking issues detected.", icon="CHECKMARK")
            return

        # Calculate the maximum width of the label
        panel_width = int(getattr(getattr(context, "region", None), "width", 0) or 280)
        uifontscale = 7 * context.preferences.view.ui_scale
        max_label_width = max(24, int(panel_width // max(uifontscale, 1.0)))

        col = layout.column(align=True)
        col.alert = True
        for issue in issues:
            col.label(text="REQUIREMENT", icon='CANCEL')
            # Split the text into lines and format each line
            for chunk in textwrap.wrap(issue, width=max_label_width):
                col.label(text=chunk)


def _averaged_latency(metrics, key: str) -> float:
    """The recent average of latency *key*; the current value before any was recorded, and 0 while stopped."""
    if not metrics.get("has_data"):
        return 0.0
    average = usage.average(key)
    return float(metrics[key]) if average is None else average


def _draw_meter(layout, label: str, share: float, text: str) -> None:
    """A labelled read-only bar."""
    row = layout.split(factor=0.25)
    row.label(text=label)
    row.progress(factor=min(max(share, 0.0), 1.0), text=text)


class ACE_PT_project_settings(bpy.types.Panel):
    """Top-level engine status panel: latency against the frame budget, CPU and GPU load."""

    bl_label = "Engine Status"
    bl_idname = "ACE_PT_project_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = -1

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        layout = self.layout
        latency_metrics = _engine_latency_metrics(context)
        render = getattr(context.scene, "render", None)
        raw_fps = float(getattr(render, "fps", 24.0) or 24.0)
        fps_base = float(getattr(render, "fps_base", 1.0) or 1.0)
        scene_fps = raw_fps / fps_base if fps_base > 0.0 else raw_fps
        budget_ms = 1000.0 / max(scene_fps, 1e-6)
        latency_ms = _averaged_latency(latency_metrics, 'total_latency_ms')
        _draw_meter(layout, "Latency", latency_ms / budget_ms, f"{latency_ms:.2f} ms")
        for key, label in (("cpu", "CPU"), ("gpu", "GPU")):
            value = usage.average(key)
            _draw_meter(layout, label, value or 0.0, "Not available" if value is None else f"{value * 100.0:.0f}%")


class ACE_PT_sub_latency_measures(bpy.types.Panel):
    """Sub-panel showing per-stage latency breakdown under Engine Status."""

    bl_label = "Latency Detail"
    bl_idname = "ACE_PT_sub_latency_measures"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_parent_id = "ACE_PT_project_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        layout = self.layout
        latency_metrics = _engine_latency_metrics(context)
        layout.label(text=f"Source Input: {_averaged_latency(latency_metrics, 'input_total_ms'):.2f} ms")
        layout.label(text=f"Audio Analysis: {_averaged_latency(latency_metrics, 'engine_total_ms'):.2f} ms")
        layout.label(text=f"Blender Side: {_averaged_latency(latency_metrics, 'blender_total_ms'):.2f} ms")


class ACE_PT_sub_engine_settings(bpy.types.Panel):
    """Sub-panel for engine configuration: working sample rate and manual reset."""

    bl_label = "Engine Settings"
    bl_idname = "ACE_PT_sub_engine_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_parent_id = "ACE_PT_project_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        ace = getattr(getattr(context, "scene", None), "audiocore", None)
        return bool(ace and getattr(ace, "engine_settings", None)) and not _ui_is_blocked(context)

    def draw(self, context):
        eng_settings = getattr(getattr(context.scene, "audiocore", None), "engine_settings", None)
        if eng_settings is None:
            return
        layout = self.layout
        property_column(layout).prop(eng_settings, "ace_file_working_sample_rate", text="Sample Rate", expand=True)
        layout.operator("audiocore.reset_engine", icon="FILE_REFRESH")


class ACE_PT_audio_sources(bpy.types.Panel):
    """Panel for managing audio sources: file or live, proxy status, and metadata."""

    bl_label = "Audio Inputs"
    bl_idname = "ACE_PT_audio_sources"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_order = 2

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        from ...utils import _get_active_source
        ace = context.scene.audiocore
        layout = self.layout

        row = layout.row()
        row.template_list(
            "ACE_UL_source_list", "",
            ace, "source_rows",
            ace, "active_source_row",
            rows=4,
        )

        col = row.column(align=True)
        col.operator("audiocore.add_audio_source", icon='ADD', text="")
        col.operator("audiocore.remove_audio_source", icon='REMOVE', text="")


        _, active_source = _get_active_source(context)
        if active_source:
            col = property_column(layout)
            col.prop(active_source, "source_type", text="Type")
            if active_source.source_type == 'FILE':
                col.prop(active_source, "filepath", text="File")
            else:
                col.prop(active_source, "live_transport", text="Input Method")
                if str(getattr(active_source, "live_transport", "SOUNDDEVICE") or "SOUNDDEVICE") == "UDP":
                    col.prop(active_source, "live_udp_port", text="UDP Port")
                    col.prop(active_source, "live_udp_channels", text="Channels")
                    col.prop(active_source, "live_udp_sample_rate", text="Sample Rate")
                else:
                    col.prop(active_source, "live_device_selector", text="Audio Input")
                col.prop(active_source, "ace_buffer_latency", text="Buffer Latency")
                live_status = getattr(active_source, "live_status", "")
                if live_status:
                    warn = layout.row()
                    warn.alert = True
                    warn.label(text=live_status, icon="ERROR")
        else:
            return

        proxy_name = getattr(active_source, "proxy_object_name", "")

        # disabled_sources holds source uids, or proxy names for sources without one.
        engine = ACEAudioEngine.get_instance(context.scene)
        disabled_srcs = set(getattr(getattr(engine, "parallel", None), "disabled_sources", set()) or set())
        src_uid = getattr(active_source, "stable_id", "") or ""
        if engine is not None and (
            (proxy_name and proxy_name in disabled_srcs) or (src_uid and src_uid in disabled_srcs)
        ):
            warn_parallel = layout.row()
            warn_parallel.alert = True
            warn_parallel.label(
                text="Parallel DSP disabled (restart limit reached). Reset the engine to restore.",
                icon="ERROR",
            )

        # File-format error from filepath validation or failed SourceTask._load_source().
        file_error = str(getattr(active_source, "ace_source_error", "") or "")
        if file_error and str(getattr(active_source, "source_type", "FILE")) == "FILE":
            err_row = layout.row()
            err_row.alert = True
            err_row.label(text=file_error, icon="ERROR")



def _source_is_configured(source) -> bool:
    """True once the source points at audio: a file, or a live route."""
    src_type = str(getattr(source, "source_type", "") or "").upper()
    if src_type == "FILE":
        return bool(str(getattr(source, "filepath", "") or "").strip())
    transport = str(getattr(source, "live_transport", "SOUNDDEVICE") or "SOUNDDEVICE")
    device = str(getattr(source, "live_device_selector", "") or "").strip()
    return src_type == "LIVE" and (transport == "UDP" or bool(device))


class _ACESourceSubPanel:
    """A closed-by-default sub-panel about the active audio source."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_parent_id = "ACE_PT_audio_sources"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        from ...utils import _get_active_source
        _, active_source = _get_active_source(context)
        return active_source is not None and not _ui_is_blocked(context)


class ACE_PT_sub_source_info(_ACESourceSubPanel, bpy.types.Panel):
    """Sample rate and duration of the active source."""

    bl_label = "Info"
    bl_idname = "ACE_PT_sub_source_info"
    bl_order = 0

    def draw(self, context):
        from ...utils import _get_active_source
        layout = self.layout
        _, source = _get_active_source(context)
        if source is None or not _source_is_configured(source):
            layout.label(text="No audio selected for this source.", icon="INFO")
            return
        col = layout.column(align=True)
        duration = float(getattr(source, "duration", 0.0) or 0.0)
        rate = _source_display_sample_rate(source)
        if rate > 0:
            col.label(text=f"Sample Rate: {rate}Hz")
        if duration > 0.0:
            minutes, seconds = divmod(int(round(duration)), 60)
            col.label(text=f"Duration: {minutes} min {seconds} s")
            # The audio starts on the scene's first frame and runs at the file's frame rate.
            render = context.scene.render
            frames = max(math.ceil(duration * render.fps / render.fps_base - 1e-6), 1)
            first = context.scene.frame_start
            col.label(text=f"Frame Span: {first} - {first + frames - 1} ({frames} frames)")
        if rate <= 0 and duration <= 0.0:
            col.label(text="Metadata unavailable for the selected source.", icon="INFO")


class ACE_PT_sub_tempo_meter(_ACESourceSubPanel, bpy.types.Panel):
    """Sub-panel for per-source BPM and beats-per-bar settings."""

    bl_label = "Tempo & Meter"
    bl_idname = "ACE_PT_sub_tempo_meter"
    bl_order = 1

    def draw(self, context):
        from ...utils import _get_active_source
        layout = self.layout
        _, src = _get_active_source(context)

        if not src:
            layout.label(text="No Source Selected")
            return

        trigger_row = layout.row()
        trigger_row.enabled = bool(
            str(getattr(src, "source_type", "") or "").upper() == "FILE"
            and str(getattr(src, "filepath", "") or "").strip()
        )
        trigger_row.operator(
            "audiocore.detect_source_tempo_meter",
            text="Auto Detect BPM / BPB",
            icon="FILE_REFRESH",
        )

        try:
            detect_status = str(src.get("ace_tempo_detect_status", "") or "")
            detect_message = str(src.get("ace_tempo_detect_message", "") or "")
        except Exception as exc:
            utils.note_swallowed("panels.ACE_PT_sub_tempo_meter.draw#2", exc)
            detect_status = ""
            detect_message = ""
        if detect_status == "pending":
            layout.label(text=detect_message or "Tempo detection in progress...", icon="TIME")
        elif detect_status == "ready":
            layout.label(text=detect_message or "Tempo detection updated the active source.", icon="CHECKMARK")
        elif detect_status == "failed":
            warn = layout.row()
            warn.alert = True
            warn.label(text=detect_message or "Tempo detection failed.", icon="ERROR")

        row = layout.row(align=True)
        row.prop(src, "bpm", text="BPM")
        row.prop(src, "beats_per_bar", text="Beats/Bar")


class ACE_PT_information_streams(bpy.types.Panel):
    """Panel listing all ACE graph nodes for the active source with live values."""

    bl_label = "Operators"
    bl_idname = "ACE_PT_information_streams"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        from ...utils import _get_active_source
        layout = self.layout
        _, active_source = _get_active_source(context)
        if not active_source:
            return

        list_col = layout.column()
        list_col.template_list(
            "ACE_UL_dummy_tree", "",
            active_source, "ui_nodes",
            active_source, "active_ui_node_index",
            rows=6,
        )
        _draw_selected_operator(layout, context)


class ACE_PT_information_streams_description(bpy.types.Panel):
    """Sub-panel showing the description of the currently selected graph node."""

    bl_label = "Description"
    bl_idname = "ACE_PT_information_streams_description"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_parent_id = "ACE_PT_information_streams"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (not _ui_is_blocked(context)) and bool(_selected_node_description(context))

    def draw(self, context):
        description = _selected_node_description(context)
        if description:
            _draw_selected_node_description(self.layout, context, description)


class ACE_PT_capture_influx(bpy.types.Panel):
    """Panel providing bake controls for writing stream outputs to f-curves."""

    bl_label = "Baking"
    bl_idname = "ACE_PT_capture_influx"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Core Engine"
    bl_order = 4
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        layout = self.layout
        _draw_bake_controls(layout, context)


classes = (
    ACE_UL_source_list,
    ACE_UL_dummy_tree,
    ACE_PT_rename_operator,
    ACE_PT_presentation_panel,
    ACE_PT_unsatisfactory_config,
    ACE_PT_project_settings,
    ACE_PT_sub_latency_measures,
    ACE_PT_sub_engine_settings,
    ACE_PT_audio_sources,
    ACE_PT_sub_source_info,
    ACE_PT_sub_tempo_meter,
    ACE_PT_information_streams,
    ACE_PT_information_streams_description,
    ACE_PT_capture_influx,
)


def register():
    """Register all N-panel UI classes and start the background sync timer."""
    for cls in classes:
        bpy.utils.register_class(cls)
    if _ui_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_ui_load_post)
    if not bpy.app.timers.is_registered(_ui_sync_timer):
        bpy.app.timers.register(_ui_sync_timer, first_interval=0.1, persistent=True)
    if not bpy.app.timers.is_registered(_ui_click_timer):
        bpy.app.timers.register(_ui_click_timer, first_interval=0.1, persistent=True)


def unregister():
    """Unregister all N-panel UI classes, stop the sync timer, and clear UI caches."""
    sync.reset_state()
    if _ui_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_ui_load_post)
    if bpy.app.timers.is_registered(_ui_sync_timer):
        bpy.app.timers.unregister(_ui_sync_timer)
    if bpy.app.timers.is_registered(_ui_click_timer):
        bpy.app.timers.unregister(_ui_click_timer)
    usage.close()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
