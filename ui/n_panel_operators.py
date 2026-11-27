"""n_panel_operators.py includes all the bpy.types.Operator classes called in the N-Panel."""

import bpy                                                                                                                              # type: ignore
import os
from bpy_extras.io_utils import ImportHelper                                                                                            # type: ignore

from ..core.id_management import ACEIdManager
from ..nodes.dynamic_audio_operators import ACESplitterManager
from ..nodes.split_freqs import sort_all_split_freqs, sort_split_freqs
from ..nodes.source_nodes import set_node_source
from ..nodes.identity import find_node, node_id as ace_node_id, parsed_node
from ..nodes.modes import apply_node_mode
from ..vars import node_modes
from .. import utils
from ..core.engine.engine import ACEAudioEngine
from ..core.input.reception import AudioReception, build_live_flow_identifier
from ..core.proxy.manager import ACEProxyManager
def _supported_audio_filter_glob() -> str:
    with utils.logged_errors("n_panel_operators._supported_audio_filter_glob"):
        glob = AudioReception.supported_file_filter_glob()
        if glob:
            return glob
    return "*.wav;*.wave"


def _resolve_panel_selected_node(context):
    """Resolve the ACE node currently targeted by the N-panel."""
    with utils.logged_errors("n_panel_operators._resolve_panel_selected_node"):
        from .panel.selection import _resolve_capture_node
        node, parsed = _resolve_capture_node(context)
        if node and parsed:
            return node, parsed

    node = getattr(context, "active_node", None)
    parsed = parsed_node(node) if node is not None else None
    if parsed:
        return node, parsed

    try:
        from .panel.helpers import _find_active_node_tree
        tree = _find_active_node_tree(context)
    except Exception as exc:
        utils.note_swallowed("n_panel_operators._resolve_panel_selected_node#2", exc)
        tree = None
    if not tree:
        return None, None

    try:
        from ..utils import _get_active_source
        _, active_source = _get_active_source(context)
    except Exception as exc:
        utils.note_swallowed("n_panel_operators._resolve_panel_selected_node#3", exc)
        active_source = None
    if not active_source:
        return None, None

    node_id = str(getattr(active_source, "active_ui_node_id", "") or "")
    if not node_id:
        return None, None

    candidate = find_node(tree, node_id)
    parsed = parsed_node(candidate) if candidate is not None else None
    if parsed:
        return candidate, parsed
    return None, None


def _refresh_node_instance(node):
    if not node:
        return

    with utils.logged_errors("n_panel_operators._refresh_node_instance"):
        node.update()

    tree = getattr(node, "id_data", None)
    if tree is None:
        return

    with utils.logged_errors("n_panel_operators._refresh_node_instance#2"):
        tree.update_tag()


def _queue_engine_recompile(context, tree=None):
    """Queue a non-blocking compile through the engine dirty-tree path."""
    resolved_tree = tree
    if resolved_tree is None:
        try:
            from .panel.helpers import _find_active_node_tree
            resolved_tree = _find_active_node_tree(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators._queue_engine_recompile", exc)
            resolved_tree = None
    if not resolved_tree:
        return None
    engine = ACEAudioEngine.ensure_instance(context.scene)
    engine.request_recompile(resolved_tree)
    return resolved_tree


def _refresh_source_list(scene) -> None:
    from .panel.sync import _sync_source_rows
    with utils.logged_errors("n_panel_operators._refresh_source_list"):
        _sync_source_rows(scene)


class ACE_OT_reset_audio_engine(bpy.types.Operator):
    """Restart the whole engine: workers, caches and graphs are dropped and rebuilt from the scene"""
    bl_idname = "audiocore.reset_engine"
    bl_label = "Manual Engine Reset"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return getattr(getattr(context, "scene", None), "audiocore", None) is not None

    def execute(self, context):
        from ..core.engine.handlers import hard_reset_engine
        from .panel.sync import _sync_source_rows, _sync_ui_dummy_tree
        try:
            engine = hard_reset_engine(context.scene)
        except Exception as exc:  # noqa: BLE001 boundary: the failure is shown to the user
            self.report({'ERROR'}, f"Engine reset failed: {exc}")
            return {'CANCELLED'}
        with utils.logged_errors("n_panel_operators.ACE_OT_reset_audio_engine.execute"):
            _sync_ui_dummy_tree(context, force=True)
            _sync_source_rows(context.scene)
        task_count = sum(len(g.tasks) for g in engine.graphs.values())
        self.report({'INFO'}, f"Engine reset: {task_count} tasks.")
        return {'FINISHED'}


class ACE_OT_add_audio_sources_from_files(bpy.types.Operator, ImportHelper):
    """Import audio files as sources"""
    bl_idname = "audiocore.add_audio_sources_from_files"
    bl_label = "Import Audio Files"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: bpy.props.StringProperty(                                                                                              # type: ignore
        default=_supported_audio_filter_glob(),
        options={'HIDDEN'},
    )
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)                                                         # type: ignore
    directory: bpy.props.StringProperty(subtype='DIR_PATH')                                                                             # type: ignore

    def execute(self, context):
        ace = context.scene.audiocore
        existing_ids = [getattr(s, "stable_id", "") for s in ace.audio_sources if getattr(s, "stable_id", "")]

        selected_files: list[tuple[str, str]] = []
        directory = self.directory or ""
        for file_elem in self.files:
            filepath = os.path.join(directory, file_elem.name)
            selected_files.append((file_elem.name, filepath))

        # Single-selection fallback: some Blender contexts provide filepath only.
        if not selected_files:
            filepath = getattr(self, "filepath", "") or ""
            if filepath and os.path.isfile(filepath):
                selected_files.append((os.path.basename(filepath), filepath))

        if not selected_files:
            self.report({'ERROR'}, "No valid audio file selected.")
            return {'CANCELLED'}

        valid_files = []
        skipped_files = []
        for file_name, filepath in selected_files:
            if not AudioReception.is_supported_file_path(filepath):
                skipped_files.append(file_name)
                continue
            valid_files.append((file_name, filepath))

        if not valid_files:
            self.report({'ERROR'}, "No supported audio file selected.")
            return {'CANCELLED'}
        if skipped_files:
            preview = ", ".join(skipped_files[:3])
            if len(skipped_files) > 3:
                preview += f" (+{len(skipped_files) - 3} more)"
            self.report({'WARNING'}, f"Skipped unsupported files: {preview}")

        for file_name, filepath in valid_files:
            
            # Create Source
            new_source = ace.audio_sources.add()
            ACEIdManager.ensure_source_uid(new_source, existing_ids=existing_ids)
            existing_ids.append(getattr(new_source, "stable_id", ""))
            new_source.name = file_name
            new_source.filepath = filepath
            new_source.source_type = 'FILE'
            with utils.logged_errors("n_panel_operators.ACE_OT_add_audio_sources_from_files.execute"):
                engine = ACEAudioEngine.ensure_instance(context.scene)
                engine.probes._probe_source_metadata(new_source)
            
        ace.active_audio_source_index = len(ace.audio_sources) - 1
        _refresh_source_list(context.scene)
        return {'FINISHED'}


class ACE_OT_add_audio_source(bpy.types.Operator):
    """Add a blank Audio Source"""
    bl_idname = "audiocore.add_audio_source"
    bl_label = "Add Source"
    bl_options = {'REGISTER', 'UNDO'}

    source_type: bpy.props.EnumProperty(                                                                                              # type: ignore
        name="Source Type",
        items=[
            ('FILE', "File", "Create a file-based source"),
            ('LIVE', "Live", "Create a live-capture source"),
        ],
        default='FILE',
        options={'SKIP_SAVE'},
    )
    
    def execute(self, context):
        ace = context.scene.audiocore
        new_source = ace.audio_sources.add()
        ACEIdManager.ensure_source_uid(new_source, existing_ids=[getattr(s, "stable_id", "") for s in ace.audio_sources if getattr(s, "stable_id", "")])
        source_type = str(getattr(self, "source_type", "FILE") or "FILE").upper()
        if source_type not in {'FILE', 'LIVE'}:
            source_type = 'FILE'
        new_source.source_type = source_type
        new_source.name = f"Audio Source {len(ace.audio_sources)}"
        if source_type == 'LIVE':
            with utils.logged_errors("n_panel_operators.ACE_OT_add_audio_source.execute"):
                new_source.flow_identifier = build_live_flow_identifier(new_source)
        
        ace.active_audio_source_index = len(ace.audio_sources) - 1
        _refresh_source_list(context.scene)
        return {'FINISHED'}


class ACE_OT_remove_audio_source(bpy.types.Operator):
    """Remove source and clean up proxy object"""
    bl_idname = "audiocore.remove_audio_source"
    bl_label = "Remove Source"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        ace = getattr(context.scene, "audiocore", None)
        return ace and len(ace.audio_sources) > 0

    def execute(self, context):
        ace = context.scene.audiocore
        idx = ace.active_audio_source_index
        source = ace.audio_sources[idx]

        # Capture stable_id before the PropertyGroup is removed from the collection,
        # because the Python reference may become invalid after audio_sources.remove().
        src_uid = getattr(source, "stable_id", "") or ""

        ACEProxyManager.delete_proxy_for_source(source)

        ace.audio_sources.remove(idx)

        if ace.active_audio_source_index >= len(ace.audio_sources):
            ace.active_audio_source_index = max(0, len(ace.audio_sources) - 1)

        engine = ACEAudioEngine.get_instance(context.scene)
        if engine:
            # The source's bake goes with it.
            if src_uid:
                with utils.logged_errors("n_panel_operators.ACE_OT_remove_audio_source.execute#3"):
                    engine.bake.store.clear(src_uid)
            engine._shutdown_graphs()

        _refresh_source_list(context.scene)
        return {'FINISHED'}
    

class ACE_OT_update_splitter_bands(bpy.types.Operator):
    bl_idname = "ace.update_splitter_bands"
    bl_label = "Change Bands"
    bl_options = {'INTERNAL', 'UNDO'}
    direction: bpy.props.EnumProperty(items=[('UP', "Up", ""), ('DOWN', "Down", "")])                                                           # type: ignore
    
    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        is_splitter = bool(parsed and parsed.get("category") == "delta" and parsed.get("type") == "frequency_splitter")
        if not node or not is_splitter:
            return {'CANCELLED'}
        current = node.get('ace_band_count', 2)
        new_cnt = min(current + 1, 8) if self.direction == 'UP' else max(current - 1, 2)
        if new_cnt != current:
            ACESplitterManager.apply_variant_to_node(node, new_cnt)
            sort_split_freqs(node)  # the added socket's default may sit below a kept value
        return {'FINISHED'}


# Releases that end an edit of a number field: drag, cancelled drag, typed value.
_EDIT_END_EVENTS = {'LEFTMOUSE', 'RIGHTMOUSE', 'ESC', 'RET', 'NUMPAD_ENTER'}


class ACE_OT_sort_split_freqs_on_release(bpy.types.Operator):
    """Order the split frequencies once the field being edited is let go."""
    bl_idname = "ace.sort_split_freqs_on_release"
    bl_label = "Sort Split Frequencies"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.value != 'RELEASE' or event.type not in _EDIT_END_EVENTS:
            return {'PASS_THROUGH'}
        # The field applies its value on this same event, after us: sort once that is done.
        bpy.app.timers.register(sort_all_split_freqs, first_interval=0.0, persistent=True)
        return {'FINISHED', 'PASS_THROUGH'}


def sort_split_freqs_when_released() -> None:
    """Called while a split frequency field is being edited; watches for the end of the edit."""
    window = bpy.context.window
    if window is None or bpy.app.background:  # no input events: nothing is being dragged
        return
    if any(op.bl_idname == "ACE_OT_sort_split_freqs_on_release" for op in window.modal_operators):
        return
    with utils.logged_errors("n_panel_operators.sort_split_freqs_when_released"):
        bpy.ops.ace.sort_split_freqs_on_release('INVOKE_DEFAULT')


class ACE_OT_even_splitter_freqs(bpy.types.Operator):
    """Space the split frequencies evenly across the strip (logarithmic frequency scale)."""
    bl_idname = "ace.even_splitter_freqs"
    bl_label = "Distribute horizontally"
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        if not parsed or parsed.get("category") != "delta" or parsed.get("type") != "frequency_splitter":
            return {'CANCELLED'}

        split_sockets = []
        for idx, sock in enumerate(getattr(node, "inputs", [])):
            if idx == 0 or getattr(sock, "is_linked", False):
                continue
            try:
                _ = float(getattr(sock, "default_value", 0.0))
            except Exception as exc:
                utils.note_swallowed("n_panel_operators.ACE_OT_even_splitter_freqs.execute", exc)
                continue
            split_sockets.append(sock)
        if not split_sockets:
            return {'CANCELLED'}

        low_hz = 20.0
        high_hz = 20000.0
        ace = getattr(context.scene, "audiocore", None)
        if ace and getattr(ace, "audio_sources", None):
            idx = int(getattr(ace, "active_audio_source_index", 0) or 0)
            if 0 <= idx < len(ace.audio_sources):
                src = ace.audio_sources[idx]
                sr = float(getattr(src, "sample_rate", 0) or 0.0)
                if sr > 0:
                    high_hz = min(high_hz, max(100.0, 0.49 * sr))
        ratio = max(high_hz / low_hz, 1.0001)
        count = len(split_sockets)
        for i, sock in enumerate(split_sockets):
            t = float(i + 1) / float(count + 1)
            freq = low_hz * (ratio ** t)
            with utils.logged_errors("n_panel_operators.ACE_OT_even_splitter_freqs.execute#2"):
                sock.default_value = float(freq)
        _refresh_node_instance(node)

        try:
            from .panel.helpers import _find_active_node_tree
            from .panel.sync import _sync_ui_dummy_tree
            tree = _find_active_node_tree(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_even_splitter_freqs.execute#3", exc)
            tree = None
            _sync_ui_dummy_tree = None  # type: ignore
        _queue_engine_recompile(context, tree=tree)
        if _sync_ui_dummy_tree:
            with utils.logged_errors("n_panel_operators.ACE_OT_even_splitter_freqs.execute#4"):
                _sync_ui_dummy_tree(context, force=False)
        return {'FINISHED'}


class ACE_OT_set_node_mode(bpy.types.Operator):
    """Switch the method of the selected Loudness or Stereo operator."""
    bl_idname = "ace.set_node_mode"
    bl_label = "Set Operator Method"
    bl_options = {'INTERNAL', 'UNDO'}
    mode: bpy.props.StringProperty()                                                                                                            # type: ignore

    @classmethod
    def description(cls, context, properties):
        node, parsed = _resolve_panel_selected_node(context)
        if not node or not parsed:
            return cls.__doc__
        for mode_id, _label, desc in node_modes(parsed.get("category", ""), parsed.get("type", "")):
            if mode_id == properties.mode:
                return desc
        return cls.__doc__

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        if not node or not parsed or not node_modes(parsed.get("category", ""), parsed.get("type", "")):
            return {'CANCELLED'}
        apply_node_mode(node, parsed["category"], parsed["type"], self.mode)
        _refresh_node_instance(node)
        # A node's custom property sends no depsgraph update: ask for the compile.
        _queue_engine_recompile(context, tree=node.id_data)
        return {'FINISHED'}


class ACE_OT_select_operator(bpy.types.Operator):
    """Select this operator and show it in the Node Editor. Double-click or Ctrl+click to rename it"""
    bl_idname = "ace.select_operator"
    bl_label = "Select Operator"
    bl_options = {'INTERNAL', 'UNDO'}
    node_id: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})                                                          # type: ignore
    row: bpy.props.IntProperty(default=-1, options={'HIDDEN', 'SKIP_SAVE'})                                                      # type: ignore

    # (node_id, time) of the last click, to tell a double-click on the same name.
    _last_click: tuple[str, float] = ("", 0.0)

    def invoke(self, context, event):
        import time
        from .panel.selection import focus_operator, start_rename

        now = time.monotonic()
        double_s = float(context.preferences.inputs.mouse_double_click_time) / 1000.0
        last_id, last_ts = ACE_OT_select_operator._last_click
        is_double = last_id == self.node_id and (now - last_ts) <= double_s
        ACE_OT_select_operator._last_click = ("", 0.0) if is_double else (self.node_id, now)
        rename = bool(event.ctrl or is_double)
        if not focus_operator(context, self.node_id, frame=not rename):
            return {'CANCELLED'}
        self._keep_row(context)
        if rename:
            start_rename(self.node_id)
            bpy.ops.wm.call_panel(name="ACE_PT_rename_operator", keep_open=False)
        return {'FINISHED'}

    def execute(self, context):
        from .panel.selection import focus_operator
        if not focus_operator(context, self.node_id):
            return {'CANCELLED'}
        self._keep_row(context)
        return {'FINISHED'}

    def _keep_row(self, context) -> None:
        """A splitter shows on one row per band: highlight the clicked one."""
        from .panel.sync import set_active_row
        _, source = utils._get_active_source(context)
        if source is not None and 0 <= self.row < len(source.ui_nodes):
            set_active_row(source, self.row)


class ACE_OT_toggle_operator_branch(bpy.types.Operator):
    """Fold or unfold the operators below this one"""
    bl_idname = "ace.toggle_operator_branch"
    bl_label = "Fold Operators"
    bl_options = {'INTERNAL'}
    branch_key: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})                                                       # type: ignore
    list_id: bpy.props.StringProperty(default="operators", options={'HIDDEN', 'SKIP_SAVE'})                                      # type: ignore

    def execute(self, context):
        from .panel.sync import toggle_branch, toggle_source_branch
        if self.list_id == "sources":
            toggle_source_branch(self.branch_key)
        else:
            toggle_branch(self.branch_key)
        area = getattr(context, "area", None)
        if area is not None:
            area.tag_redraw()
        return {'FINISHED'}


class ACE_OT_detect_source_tempo_meter(bpy.types.Operator):
    """Queue non-blocking BPM / beats-per-bar detection for the active file source."""
    bl_idname = "audiocore.detect_source_tempo_meter"
    bl_label = "Auto Detect BPM / BPB"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        try:
            _, active_source = utils._get_active_source(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_detect_source_tempo_meter.poll", exc)
            active_source = None
        if not active_source:
            return False
        if str(getattr(active_source, "source_type", "") or "").upper() != "FILE":
            return False
        return bool(str(getattr(active_source, "filepath", "") or "").strip())

    def execute(self, context):
        try:
            _, active_source = utils._get_active_source(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_detect_source_tempo_meter.execute", exc)
            active_source = None
        if not active_source:
            self.report({'ERROR'}, "No active source selected.")
            return {'CANCELLED'}

        engine = ACEAudioEngine.ensure_instance(context.scene)
        ok, message = engine.request_source_tempo_detection(active_source)
        with utils.logged_errors("n_panel_operators.ACE_OT_detect_source_tempo_meter.execute#2"):
            engine.flush_source_tempo_detection_updates([active_source])

        if not ok:
            self.report({'ERROR'}, message or "Tempo detection could not start.")
            return {'CANCELLED'}

        self.report({'INFO'}, message or "Tempo detection queued.")
        return {'FINISHED'}


_source_items: list = []  # Blender needs the strings of a dynamic enum kept alive


def _node_source_items(_self, context):
    global _source_items
    project = getattr(getattr(context, "scene", None), "audiocore", None)
    _source_items = [(s.stable_id, s.name, "") for s in getattr(project, "audio_sources", []) if s.stable_id]
    return _source_items


class ACE_OT_set_node_source(bpy.types.Operator):
    """Choose the audio source this Source node reads"""
    bl_idname = "ace.set_node_source"
    bl_label = "Audio Source"
    bl_options = {'INTERNAL', 'UNDO'}
    source: bpy.props.EnumProperty(name="Source", items=_node_source_items)                                                          # type: ignore

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        if not parsed or parsed.get("category") != "source_mngmt" or parsed.get("type") != "src":
            return {'CANCELLED'}
        project = context.scene.audiocore
        for index, source in enumerate(project.audio_sources):
            if source.stable_id == self.source:
                break
        else:
            return {'CANCELLED'}
        node_id = ace_node_id(node)
        tree = node.id_data
        set_node_source(node, source, context.scene)
        # The list shows one source's graph: follow the node to its new source.
        project.active_audio_source_index = index
        from .panel.sync import _sync_ui_dummy_tree
        _sync_ui_dummy_tree(context, force=True)
        project.audio_sources[index].active_ui_node_id = node_id
        _sync_ui_dummy_tree(context, force=True)
        _queue_engine_recompile(context, tree=tree)
        return {'FINISHED'}


class ACE_OT_toggle_preview_sink(bpy.types.Operator):
    """Enable/disable branch audio preview on the selected Audio Preview node."""
    bl_idname = "ace.toggle_preview_sink"
    bl_label = "Toggle Audio Preview"
    bl_options = {'INTERNAL', 'UNDO'}

    preview_enabled: bpy.props.BoolProperty(name="Enabled", default=True)  # type: ignore

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        is_preview_sink = bool(parsed and parsed.get("category") == "utilities" and parsed.get("type") == "preview_sink")
        if not is_preview_sink:
            self.report({'ERROR'}, "Selected node is not an ACE Audio Preview.")
            return {'CANCELLED'}

        try:
            node["ace_preview_enabled"] = bool(self.preview_enabled)
            _refresh_node_instance(node)
        except Exception:
            self.report({'ERROR'}, "Failed to toggle audio preview.")
            return {'CANCELLED'}

        try:
            from .panel.helpers import _find_active_node_tree
            from .panel.sync import _sync_ui_dummy_tree
            tree = _find_active_node_tree(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_toggle_preview_sink.execute", exc)
            tree = None
            _sync_ui_dummy_tree = None  # type: ignore

        if tree:
            _queue_engine_recompile(context, tree=tree)

        if _sync_ui_dummy_tree:
            with utils.logged_errors("n_panel_operators.ACE_OT_toggle_preview_sink.execute#2"):
                _sync_ui_dummy_tree(context, force=False)

        return {'FINISHED'}


class ACE_OT_bake_active_source(bpy.types.Operator):
    """Bake the active file source outputs over the scene frame range."""
    bl_idname = "audiocore.bake_active_source"
    bl_label = "Bake Active Source"
    bl_options = {'REGISTER'}

    _timer = None
    _source_uid: str = ""
    _source_name: str = ""

    @classmethod
    def poll(cls, context):
        ace = getattr(getattr(context, "scene", None), "audiocore", None)
        if not ace:
            return False
        idx = utils.as_int(getattr(ace, "active_audio_source_index", None), -1)
        sources = getattr(ace, "audio_sources", None)
        if sources is None or idx < 0 or idx >= len(sources):
            return False
        source = sources[idx]
        return bool(getattr(source, "source_type", "") == 'FILE')

    @staticmethod
    def _find_source_by_uid(context, source_uid: str):
        ace = getattr(getattr(context, "scene", None), "audiocore", None)
        if not ace:
            return None
        for src in getattr(ace, "audio_sources", []) or []:
            if str(getattr(src, "stable_id", "") or "") == str(source_uid or ""):
                return src
        return None

    def _finish_modal(self, context):
        if self._timer is not None:
            with utils.logged_errors("n_panel_operators.ACE_OT_bake_active_source._finish_modal"):
                context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        engine = ACEAudioEngine.ensure_instance(context.scene)
        with utils.logged_errors("n_panel_operators.ACE_OT_bake_active_source.modal"):
            engine.flush_bake_results()

        source = self._find_source_by_uid(context, self._source_uid)
        if source is None:
            self._finish_modal(context)
            self.report({'WARNING'}, "Bake source is no longer available.")
            return {'CANCELLED'}

        try:
            status = str(source.get("ace_bake_status", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_bake_active_source.modal#2", exc)
            status = ""
        if status in {"done", "ready"}:
            self._finish_modal(context)
            self.report({'INFO'}, f"Bake completed for '{self._source_name or getattr(source, 'name', 'source')}'.")
            return {'FINISHED'}
        if status in {"error", "failed"}:
            err = ""
            try:
                err = str(source.get("ace_bake_error", "") or "")
            except Exception as exc:
                utils.note_swallowed("n_panel_operators.ACE_OT_bake_active_source.modal#3", exc)
                err = ""
            self._finish_modal(context)
            self.report({'ERROR'}, f"Bake failed: {err or 'Unknown error'}")
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ace = getattr(context.scene, "audiocore", None)
        if not ace:
            self.report({'ERROR'}, "Audio Core settings unavailable.")
            return {'CANCELLED'}

        from ..utils import _get_active_source
        _, active_source = _get_active_source(context)
        if not active_source:
            self.report({'ERROR'}, "No active source.")
            return {'CANCELLED'}
        if getattr(active_source, "source_type", "") != 'FILE':
            self.report({'ERROR'}, "Baking is only available for file sources.")
            return {'CANCELLED'}
        if not getattr(active_source, "filepath", ""):
            self.report({'ERROR'}, "Select a file path before baking.")
            return {'CANCELLED'}

        engine = ACEAudioEngine.ensure_instance(context.scene)
        with utils.logged_errors("n_panel_operators.ACE_OT_bake_active_source.execute"):
            engine.compile_scene_frontends(context.scene)
        ok = engine.bake_source_to_cache(active_source, None, scene=context.scene)
        if not ok:
            self.report({'ERROR'}, "Bake request failed.")
            return {'CANCELLED'}
        self._source_uid = str(getattr(active_source, "stable_id", "") or "")
        self._source_name = str(getattr(active_source, "name", "") or "")
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self._finish_modal(context)

def _active_file_source(context):
    from ..utils import _get_active_source

    _, source = _get_active_source(context)
    return source if source is not None and getattr(source, "source_type", "") == 'FILE' else None


class ACE_OT_clear_source_bake(bpy.types.Operator):
    """Delete the active source's bake: its outputs are computed live again."""
    bl_idname = "audiocore.clear_source_bake"
    bl_label = "Clear Bake"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        source = _active_file_source(context)
        from ..core.bake_store import store_name

        return source is not None and bpy.data.meshes.get(store_name(str(source.stable_id))) is not None

    def execute(self, context):
        source = _active_file_source(context)
        engine = ACEAudioEngine.ensure_instance(context.scene)
        engine.bake.clear_source(str(source.stable_id))
        source["ace_bake_status"] = ""
        return {'FINISHED'}


class ACE_OT_toggle_source_bake(bpy.types.Operator):
    """Read the active source's bake, or compute its outputs live while keeping the bake."""
    bl_idname = "audiocore.toggle_source_bake"
    bl_label = "Use Bake"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_file_source(context) is not None

    def execute(self, context):
        source = _active_file_source(context)
        source["ace_bake_use"] = not bool(source.get("ace_bake_use", True))
        engine = ACEAudioEngine.ensure_instance(context.scene)
        engine.bake.layout()
        return {'FINISHED'}


classes = (
    ACE_OT_reset_audio_engine,
    ACE_OT_add_audio_sources_from_files,
    ACE_OT_add_audio_source,
    ACE_OT_remove_audio_source,
    ACE_OT_update_splitter_bands,
    ACE_OT_even_splitter_freqs,
    ACE_OT_sort_split_freqs_on_release,
    ACE_OT_set_node_mode,
    ACE_OT_select_operator,
    ACE_OT_toggle_operator_branch,
    ACE_OT_detect_source_tempo_meter,
    ACE_OT_set_node_source,
    ACE_OT_toggle_preview_sink,
    ACE_OT_bake_active_source,
    ACE_OT_clear_source_bake,
    ACE_OT_toggle_source_bake,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    
