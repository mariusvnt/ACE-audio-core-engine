"""N-panel lists and panels."""

import bpy  # type: ignore
import textwrap
from ...core.engine.engine import ACEAudioEngine
from ... import utils
from .helpers import _compiled_source_instance_rows, _deal_breakers, _find_active_node_tree, _frontend_instance_type, _frontend_kind_icon, _frontend_object_name, _source_display_sample_rate, _source_list_icon, _ui_is_blocked
from .selection import _draw_bake_controls, _draw_selected_node_description, _draw_selected_node_operator_settings, _selected_node_description, _selected_node_has_operator_settings
from . import sync
from .sync import _engine_latency_metrics, _ui_has_pending_engine_changes, _ui_load_post, _ui_sync_timer


class ACE_UL_source_list(bpy.types.UIList):
    """UIList row renderer for the audio source collection.

    Shows a status icon (ERROR when proxy is missing, FILE for file sources,
    SPEAKER for live sources) followed by the editable source name.
    """

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # Status icon policy is shared with the Active ACE Instances panel.
        icon = _source_list_icon(item)
        layout.prop(item, "name", text="", emboss=False, icon=icon)


class ACE_UL_dummy_tree(bpy.types.UIList):
    """UIList row renderer for the active-source graph node list.

    Displays the node name in the left column and the category label in the
    right column.  Values are read directly from the ui_nodes collection
    property populated by _sync_ui_dummy_tree().
    """

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # Keep drawing lean; only touch attributes that commonly exist on the ui_nodes items
        name = getattr(item, "name", f"Node {index}")
        node_type = getattr(item, "node_type", "")

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            split = layout.split(factor=0.55, align=True)
            split.label(text=name)
            split.label(text=node_type or "")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='NODE')


class ACE_PT_presentation_panel(bpy.types.Panel):
    """Panel displaying the add-on logo."""
    bl_label = "AudioCore Engine"
    bl_idname = "ACE_PT_presentation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_order = 0

    def draw(self, context):
        from ...utils import _ensure_logo_icon
        layout = self.layout
        icon_id = _ensure_logo_icon()
        if icon_id:
            layout.template_icon(icon_value=icon_id, scale=8.0)
        else:
            layout.label(text="AudioCore Engine", icon="SOUND")


class ACE_PT_unsatisfactory_config(bpy.types.Panel):
    """Shown when a blocking configuration issue is detected."""
    bl_label = "Configuration Issues"
    bl_idname = "ACE_PT_unsatisfactory_config"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
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


class ACE_PT_project_settings(bpy.types.Panel):
    """Top-level engine status panel showing compile state and latency summary."""

    bl_label = "Engine Status"
    bl_idname = "ACE_PT_project_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = -1

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        ace = context.scene.audiocore
        layout = self.layout
        eng_settings = getattr(ace, "engine_settings", None)
        engine = ACEAudioEngine.get_instance(context.scene)
        tree = _find_active_node_tree(context)
        pending_changes = _ui_has_pending_engine_changes(context.scene, tree)
        auto_compile_enabled = bool(getattr(eng_settings, "auto_compile", True)) if eng_settings else True
        is_playing = bool(engine and ACEAudioEngine._is_animation_playing())
        is_compiling = bool(engine and getattr(getattr(engine, "frontends", None), "_is_compiling", False))
        if (not is_compiling) and auto_compile_enabled and pending_changes:
            is_compiling = True
        has_runtime_graph = bool(engine and getattr(engine, "graphs", None))
        if is_compiling:
            status_text = "Compiling..."
            status_icon = "FILE_REFRESH"
        elif has_runtime_graph and is_playing:
            status_text = "Active"
            status_icon = "PLAY"
        else:
            status_text = "Idle"
            status_icon = "PAUSE"

        status_row = layout.row(align=True)
        status_row.label(text=status_text, icon=status_icon)

        if not auto_compile_enabled:
            action_row = layout.row()
            action_row.enabled = tree is not None
            action_row.operator(
                "audiocore.rebuild_engine",
                icon="FILE_REFRESH",
                text="Refresh Engine",
            )

        latency_metrics = _engine_latency_metrics(context)
        render = getattr(context.scene, "render", None)
        raw_fps = float(getattr(render, "fps", 24.0) or 24.0)
        fps_base = float(getattr(render, "fps_base", 1.0) or 1.0)
        scene_fps = raw_fps / fps_base if fps_base > 0.0 else raw_fps
        frame_budget_ms = 1000.0 / max(scene_fps, 1e-6)
        performance_box = layout.box()
        performance_box.label(text="Engine Performance", icon="DRIVER")
        performance_row = performance_box.row(align=True)
        current_col = performance_row.column(align=True)
        current_col.label(text="Current Latency")
        current_col.label(text=f"{float(latency_metrics['total_latency_ms']):.2f} ms")
        max_col = performance_row.column(align=True)
        max_col.label(text="Latency Budget")
        max_col.label(text=f"{frame_budget_ms:.2f} ms")


class ACE_PT_sub_latency_measures(bpy.types.Panel):
    """Sub-panel showing per-stage latency breakdown under Engine Status."""

    bl_label = "Latency Detail"
    bl_idname = "ACE_PT_sub_latency_measures"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_parent_id = "ACE_PT_project_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        layout = self.layout
        latency_metrics = _engine_latency_metrics(context)
        layout.label(text=f"Source Input: {float(latency_metrics['input_total_ms']):.2f} ms")
        layout.label(text=f"Audio Analysis: {float(latency_metrics['engine_total_ms']):.2f} ms")
        layout.label(text=f"Blender Side: {float(latency_metrics['blender_total_ms']):.2f} ms")


class ACE_PT_sub_engine_settings(bpy.types.Panel):
    """Sub-panel for engine configuration: sample rate, preprocessing, and compile mode."""

    bl_label = "Engine Settings"
    bl_idname = "ACE_PT_sub_engine_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_parent_id = "ACE_PT_project_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        ace = getattr(getattr(context, "scene", None), "audiocore", None)
        return bool(ace and getattr(ace, "engine_settings", None)) and not _ui_is_blocked(context)

    def draw(self, context):
        from ...utils import _get_active_source

        eng_settings = getattr(getattr(context.scene, "audiocore", None), "engine_settings", None)
        if eng_settings is None:
            return
        layout = self.layout
        _, src = _get_active_source(context)
        sample_rate_row = layout.row(align=True)
        sample_rate_row.label(text="Sample Rate")
        sample_rate_buttons = sample_rate_row.row(align=True)
        for option in ("22050", "44100", "48000", "96000"):
            sample_rate_buttons.prop_enum(eng_settings, "ace_file_working_sample_rate", option)
        if src:
            layout.prop(src, "enable_preprocessing", text="Enable Source Preprocessing")
            split = layout.split(factor=0.5, align=True)
            split.enabled = bool(getattr(src, "enable_preprocessing", False))

            left = split.column(align=True)
            left.prop(src, "freq_min", text="Min Hz")
            left.prop(src, "envelope_attack", text="Attack (ms)")

            right = split.column(align=True)
            right.prop(src, "freq_max", text="Max Hz")
            right.prop(src, "envelope_release", text="Release (ms)")
        layout.prop(eng_settings, "auto_compile", text="Auto-Compile Graph Changes")


class ACE_PT_audio_sources(bpy.types.Panel):
    """Panel for managing audio sources: file or live, proxy status, and metadata."""

    bl_label = "Audio Sources"
    bl_idname = "ACE_PT_audio_sources"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
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
            ace, "audio_sources", 
            ace, "active_audio_source_index"
        )
        
        col = row.column(align=True)
        col.operator("audiocore.add_audio_source", icon='ADD', text="")
        col.operator("audiocore.remove_audio_source", icon='REMOVE', text="")


        row = layout.row()
        _, active_source = _get_active_source(context)
        if active_source:
            layout.prop(active_source, "source_type", text="Type")
            row = layout.row()
            if active_source.source_type == 'FILE':
                row.prop(active_source, "filepath", text="File")
            else:
                col = layout.column(align=True)
                col.prop(active_source, "live_transport", text="Input Method")
                if str(getattr(active_source, "live_transport", "SOUNDDEVICE") or "SOUNDDEVICE") == "UDP":
                    col.prop(active_source, "live_udp_port", text="UDP Port")
                    split_udp = col.split(factor=0.5, align=True)
                    split_udp.prop(active_source, "live_udp_channels", text="Channels")
                    split_udp.prop(active_source, "live_udp_sample_rate", text="Sample Rate")
                else:
                    col.prop(active_source, "live_device_selector", text="Audio Input")
                layout.prop(active_source, "live_latency_ms", text="Latency (ms)")
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
                text="Parallel DSP disabled (restart limit reached). Re-compile to restore.",
                icon="ERROR",
            )

        # File-format error from filepath validation or failed SourceTask._load_source().
        file_error = str(getattr(active_source, "ace_source_error", "") or "")
        if file_error and str(getattr(active_source, "source_type", "FILE")) == "FILE":
            err_row = layout.row()
            err_row.alert = True
            err_row.label(text=file_error, icon="ERROR")

        # Analysis Data — only shown when the source is actually linked to audio.
        src_type = str(getattr(active_source, "source_type", "") or "").upper()
        filepath = str(getattr(active_source, "filepath", "") or "").strip()
        transport = str(getattr(active_source, "live_transport", "SOUNDDEVICE") or "SOUNDDEVICE")
        device_sel = str(getattr(active_source, "live_device_selector", "") or "").strip()
        source_configured = (
            (src_type == "FILE" and bool(filepath))
            or (src_type == "LIVE" and (transport == "UDP" or bool(device_sel)))
        )
        if source_configured:
            col = layout.column(align=True)
            dur = float(getattr(active_source, "duration", 0.0) or 0.0)
            rate = _source_display_sample_rate(active_source)
            if rate > 0:
                col.label(text=f"Sample Rate: {rate}Hz")
            if dur > 0.0:
                col.label(text=f"Duration: {dur:.2f}s")
            if rate <= 0 and dur <= 0.0:
                col.label(text="Metadata unavailable for the selected source.", icon="INFO")


class ACE_PT_sub_tempo_meter(bpy.types.Panel):
    """Sub-panel for per-source BPM and beats-per-bar settings."""

    bl_label = "Tempo & Meter"
    bl_idname = "ACE_PT_sub_tempo_meter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_parent_id = "ACE_PT_audio_sources"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, context):
        from ...utils import _get_active_source
        _, active_source = _get_active_source(context)
        return active_source is not None and not _ui_is_blocked(context)

    def draw(self, context):
        from ...utils import _get_active_source
        layout = self.layout
        _, src = _get_active_source(context)
        
        if not src:
            layout.label(text="No Source Selected")
            return

        panel_prop = getattr(layout, "panel_prop", None)
        if callable(panel_prop):
            with utils.logged_errors("panels.ACE_PT_sub_tempo_meter.draw"):
                header, panel = panel_prop(src, "tempo_meter_panel_open")
                header.label(text=self.bl_label)
                if panel is None:
                    return
                layout = panel

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

    bl_label = "Active Operators"
    bl_idname = "ACE_PT_information_streams"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
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

        refresh_row = layout.row(align=True)
        refresh_row.operator("audiocore.refresh_stream_list", icon="FILE_REFRESH", text="Refresh Active Node List")

        list_col = layout.column()
        list_col.template_list(
            "ACE_UL_dummy_tree", "",
            active_source, "ui_nodes",
            active_source, "active_ui_node_index"
        )
        if _selected_node_has_operator_settings(context):
            _draw_selected_node_operator_settings(layout, context)


class ACE_PT_information_streams_description(bpy.types.Panel):
    """Sub-panel showing the description of the currently selected graph node."""

    bl_label = "Description"
    bl_idname = "ACE_PT_information_streams_description"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_parent_id = "ACE_PT_information_streams"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (not _ui_is_blocked(context)) and bool(_selected_node_description(context))

    def draw(self, context):
        description = _selected_node_description(context)
        if description:
            _draw_selected_node_description(self.layout, description)


class ACE_PT_capture_influx(bpy.types.Panel):
    """Panel providing bake controls for writing stream outputs to f-curves."""

    bl_label = "Baking"
    bl_idname = "ACE_PT_capture_influx"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_order = 4
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "audiocore") and not _ui_is_blocked(context)

    def draw(self, context):
        layout = self.layout
        _draw_bake_controls(layout, context)


class ACE_PT_stream_confluence(bpy.types.Panel):
    """Read-only ledger of compiled source-to-frontend routing."""

    bl_label = "Active ACE Instances"
    bl_idname = "ACE_PT_stream_confluence"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_order = 5

    @classmethod
    def poll(cls, context):
        return (
            hasattr(context.scene, "audiocore")
            and not _ui_is_blocked(context)
            and sync._has_ace_nodes_in_trees
        )

    def draw(self, context):
        layout = self.layout
        ledger_rows = _compiled_source_instance_rows(getattr(context, "scene", None))
        if ledger_rows is None or not ledger_rows:
            layout.label(text="Engine offline or no sources found.")
            return
        for entry in ledger_rows:
            source_box = layout.box()
            header = source_box.row(align=True)
            source_icon = _source_list_icon(entry.get("source_prop"))
            header.label(text=str(entry.get("display_name", "") or "Unnamed Source"), icon=source_icon)

            frontend_refs = list(entry.get("frontend_refs", []) or [])
            if not frontend_refs:
                empty_row = source_box.row(align=True)
                empty_row.separator(factor=2.0)
                empty_row.label(text="No connected instances")
                continue

            for frontend_ref in frontend_refs:
                frontend_row = source_box.row(align=True)
                frontend_row.separator(factor=2.0)
                frontend_kind = str(getattr(frontend_ref, "kind", "") or "")
                split = frontend_row.split(factor=0.62, align=True)
                object_col = split.row(align=True)
                object_col.label(text=_frontend_object_name(frontend_ref), icon='OBJECT_DATA')
                type_col = split.row(align=True)
                type_col.alignment = 'RIGHT'
                type_col.label(
                    text=_frontend_instance_type(frontend_kind),
                    icon=_frontend_kind_icon(frontend_kind),
                )


classes = (
    ACE_UL_source_list,
    ACE_UL_dummy_tree,
    ACE_PT_presentation_panel,
    ACE_PT_unsatisfactory_config,
    ACE_PT_project_settings,
    ACE_PT_sub_latency_measures,
    ACE_PT_sub_engine_settings,
    ACE_PT_audio_sources,
    ACE_PT_sub_tempo_meter,
    ACE_PT_information_streams,
    ACE_PT_information_streams_description,
    ACE_PT_capture_influx,
    ACE_PT_stream_confluence,
)


def register():
    """Register all N-panel UI classes and start the background sync timer."""
    for cls in classes:
        bpy.utils.register_class(cls)
    if _ui_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_ui_load_post)
    if not bpy.app.timers.is_registered(_ui_sync_timer):
        bpy.app.timers.register(_ui_sync_timer, first_interval=0.1, persistent=True)


def unregister():
    """Unregister all N-panel UI classes, stop the sync timer, and clear UI caches."""
    sync.reset_state()
    if _ui_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_ui_load_post)
    if bpy.app.timers.is_registered(_ui_sync_timer):
        bpy.app.timers.unregister(_ui_sync_timer)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
