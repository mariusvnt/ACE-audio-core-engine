"""
n_panel_operators.py includes all the bpy.types.Operator classes called in the N-Panel.
"""

import bpy                                                                                                                              # type: ignore
import math
import os
from bpy_extras.io_utils import ImportHelper                                                                                            # type: ignore

from ..core.id_management import ACEIdManager
from ..nodes.dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager
from .. import utils
from ..core.audio_engine import ACEAudioEngine
from ..core.audio_influx import AudioReception, build_live_flow_identifier
from ..core.proxy_management import ACEProxyManager
def _supported_audio_filter_glob() -> str:
    with utils.logged_errors("n_panel_operators._supported_audio_filter_glob"):
        glob = AudioReception.supported_file_filter_glob()
        if glob:
            return glob
    return "*.wav;*.wave"


def _resolve_panel_selected_node(context):
    """
    Resolve the ACE node currently targeted by the N-panel.
    Prefer the active graph list selection used by the N-panel UI itself.
    """
    with utils.logged_errors("n_panel_operators._resolve_panel_selected_node"):
        from .n_panel_ui import _resolve_capture_node
        node, parsed = _resolve_capture_node(context)
        if node and parsed:
            return node, parsed

    node = getattr(context, "active_node", None)
    if node and getattr(node, "type", "") == 'GROUP':
        ace_id = str(node.get("ace_id", "") or "")
        parsed = ACEIdManager.parse_node_id(ace_id) if ace_id else None
        if parsed:
            return node, parsed

    try:
        from .n_panel_ui import _find_active_node_tree
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

    idx = int(getattr(active_source, "active_ui_node_index", -1))
    if idx < 0 or idx >= len(getattr(active_source, "ui_nodes", [])):
        return None, None
    ui_node = active_source.ui_nodes[idx]
    node_id = str(getattr(ui_node, "node_id", "") or "")
    if not node_id:
        return None, None

    for candidate in getattr(tree, "nodes", []):
        try:
            ace_id = str(candidate.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_operators._resolve_panel_selected_node#4", exc)
            ace_id = ""
        if ace_id == node_id:
            parsed = ACEIdManager.parse_node_id(ace_id)
            if parsed:
                return candidate, parsed
            break
    return None, None


def _refresh_node_instance(node):
    """
    Refresh a GN node instance safely across Blender versions.
    GeometryNodeGroup does not expose `update_tag()` in Blender 5.0.
    """
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
    """
    Queue a non-blocking compile through the engine dirty-tree path.
    Heavy compile work is handled by the engine timer/handler pipeline.
    """
    resolved_tree = tree
    if resolved_tree is None:
        try:
            from .n_panel_ui import _find_active_node_tree
            resolved_tree = _find_active_node_tree(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators._queue_engine_recompile", exc)
            resolved_tree = None
    if not resolved_tree:
        return None
    engine = ACEAudioEngine.ensure_instance(context.scene)
    engine.request_recompile(resolved_tree)
    return resolved_tree


class ACE_OT_rebuild_audio_engine(bpy.types.Operator):
    """Recompiles the Audio Engine from the active object's Node Tree"""
    bl_idname = "audiocore.rebuild_engine"
    bl_label = "Rebuild Engine"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            scene = getattr(context, "scene", None)
            if scene is None:
                return False
            engine = ACEAudioEngine.ensure_instance(scene)
            return bool(engine.discover_scene_frontends(scene))
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_rebuild_audio_engine.poll", exc)
            return False
    
    def execute(self, context):
        scene = getattr(context, "scene", None)
        if scene is None:
            self.report({'ERROR'}, "Scene unavailable.")
            return {'CANCELLED'}

        try:
            engine = ACEAudioEngine.ensure_instance(scene)
            if not engine.discover_scene_frontends(scene):
                self.report({'ERROR'}, "No ACE-managed scene frontends found to compile.")
                return {'CANCELLED'}
            engine.compile_scene_frontends(scene)
            engine.execute_next_frame(scene, force=True)
            
            task_count = sum(len(g.tasks) for g in engine.graphs.values()) if engine.graphs else 0
            self.report({'INFO'}, f"Engine compiled: {task_count} tasks.")
            
        except Exception as e:
             self.report({'ERROR'}, f"Compilation failed: {str(e)}")
             return {'CANCELLED'}
             
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
            
            # Create Proxy
            self._create_proxy(context, new_source)
            
        ace.active_audio_source_index = len(ace.audio_sources) - 1
        sonar_obj = ACEProxyManager.ensure_sonar_setup(context.scene, ace.audio_sources)
        if sonar_obj and not getattr(ace, "gn_target_object", None):
            with utils.logged_errors("n_panel_operators.ACE_OT_add_audio_sources_from_files.execute#2"):
                ace.gn_target_object = sonar_obj
        return {'FINISHED'}

    def _create_proxy(self, context, source):
        # Delegate to proxy manager so proxies land in the hidden/internal collection
        ACEProxyManager.ensure_proxy_for_source(source, scene=context.scene)


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
        
        # Create Proxy (hidden + locked transforms)
        ACEProxyManager.ensure_proxy_for_source(new_source, scene=context.scene)
        sonar_obj = ACEProxyManager.ensure_sonar_setup(context.scene, ace.audio_sources)
        if sonar_obj and not getattr(ace, "gn_target_object", None):
            with utils.logged_errors("n_panel_operators.ACE_OT_add_audio_source.execute#2"):
                ace.gn_target_object = sonar_obj
            
        ace.active_audio_source_index = len(ace.audio_sources) - 1
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

        # --- PROXY CLEANUP ---
        ACEProxyManager.delete_proxy_for_source(source)

        ace.audio_sources.remove(idx)

        if ace.active_audio_source_index >= len(ace.audio_sources):
            ace.active_audio_source_index = max(0, len(ace.audio_sources) - 1)

        sonar_obj = ACEProxyManager.ensure_sonar_setup(context.scene, ace.audio_sources)
        if not ace.audio_sources and getattr(ace, "gn_target_object", None) and getattr(ace.gn_target_object, "name", "") == ACEProxyManager.SONAR_OBJECT_NAME:
            with utils.logged_errors("n_panel_operators.ACE_OT_remove_audio_source.execute"):
                ace.gn_target_object = None
        elif sonar_obj and not getattr(ace, "gn_target_object", None):
            with utils.logged_errors("n_panel_operators.ACE_OT_remove_audio_source.execute#2"):
                ace.gn_target_object = sonar_obj
        engine = ACEAudioEngine.get_instance(context.scene)
        if engine:
            # Eagerly release the bake cache for this source so its SHM/disk
            # entries are freed immediately rather than waiting for the next
            # compile's _prune_bake_cache() sweep.
            if src_uid:
                entry = engine._bake_cache.pop(src_uid, None)
                if isinstance(entry, dict):
                    with utils.logged_errors("n_panel_operators.ACE_OT_remove_audio_source.execute#3"):
                        engine._cleanup_bake_entry(entry)
            engine._shutdown_graphs()

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
        return {'FINISHED'}


class ACE_OT_sort_splitter_freqs(bpy.types.Operator):
    """Sort splitter cut frequencies in ascending order."""
    bl_idname = "ace.sort_splitter_freqs"
    bl_label = "Sort Split Frequencies"
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
                val = float(getattr(sock, "default_value", 0.0))
            except Exception as exc:
                utils.note_swallowed("n_panel_operators.ACE_OT_sort_splitter_freqs.execute", exc)
                continue
            split_sockets.append((sock, val))
        if not split_sockets:
            return {'CANCELLED'}

        sorted_vals = sorted(v for _, v in split_sockets)
        for (sock, _), val in zip(split_sockets, sorted_vals):
            with utils.logged_errors("n_panel_operators.ACE_OT_sort_splitter_freqs.execute#2"):
                sock.default_value = float(val)
        _refresh_node_instance(node)

        try:
            from .n_panel_ui import _find_active_node_tree, _sync_ui_dummy_tree
            tree = _find_active_node_tree(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_sort_splitter_freqs.execute#3", exc)
            tree = None
            _sync_ui_dummy_tree = None  # type: ignore
        _queue_engine_recompile(context, tree=tree)
        if _sync_ui_dummy_tree:
            with utils.logged_errors("n_panel_operators.ACE_OT_sort_splitter_freqs.execute#4"):
                _sync_ui_dummy_tree(context, force=False)
        return {'FINISHED'}


class ACE_OT_even_splitter_freqs(bpy.types.Operator):
    """Distribute splitter cuts evenly on a logarithmic frequency scale."""
    bl_idname = "ace.even_splitter_freqs"
    bl_label = "Even Log Distribution"
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
            from .n_panel_ui import _find_active_node_tree, _sync_ui_dummy_tree
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


class ACE_OT_sync_splitter_ramp(bpy.types.Operator):
    """Synchronize the N-panel splitter ramp editor from the selected splitter node."""
    bl_idname = "ace.sync_splitter_ramp"
    bl_label = "Sync Splitter Ramp"
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        if not parsed or parsed.get("category") != "delta" or parsed.get("type") != "frequency_splitter":
            return {'CANCELLED'}

        try:
            from .n_panel_ui import _sync_splitter_ramp_texture_from_node
            tex = _sync_splitter_ramp_texture_from_node(node, context.scene, force=True)
            if tex is None:
                self.report({'ERROR'}, "Could not create ramp editor texture.")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Ramp sync failed: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class ACE_OT_apply_splitter_ramp(bpy.types.Operator):
    """Apply current N-panel splitter ramp handle positions to node split frequencies."""
    bl_idname = "ace.apply_splitter_ramp"
    bl_label = "Apply Splitter Ramp"
    bl_options = {'INTERNAL', 'UNDO'}

    @staticmethod
    def _resample_positions(positions, target_count: int) -> list[float]:
        if target_count <= 0:
            return []
        src = sorted(float(p) for p in positions if p is not None)
        if not src:
            return [float(i + 1) / float(target_count + 1) for i in range(target_count)]
        if len(src) == target_count:
            return src
        if len(src) == 1:
            return [src[0] for _ in range(target_count)]

        out: list[float] = []
        span = float(len(src) - 1)
        for i in range(target_count):
            t = float(i + 1) / float(target_count + 1)
            src_pos = t * span
            lo = int(math.floor(src_pos))
            hi = min(lo + 1, len(src) - 1)
            a = src_pos - float(lo)
            out.append((1.0 - a) * src[lo] + a * src[hi])
        return out

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        if not parsed or parsed.get("category") != "delta" or parsed.get("type") != "frequency_splitter":
            return {'CANCELLED'}

        cut_sockets = []
        for idx, sock in enumerate(getattr(node, "inputs", [])):
            if idx == 0 or getattr(sock, "is_linked", False):
                continue
            try:
                _ = float(getattr(sock, "default_value", 0.0))
            except Exception as exc:
                utils.note_swallowed("n_panel_operators.ACE_OT_apply_splitter_ramp.execute", exc)
                continue
            cut_sockets.append(sock)
        if not cut_sockets:
            return {'CANCELLED'}

        try:
            from .n_panel_ui import (
                _ensure_splitter_ramp_texture,
                _splitter_display_bounds,
                _splitter_norm_to_hz,
                _sync_splitter_ramp_texture_from_node,
            )
            tex = _ensure_splitter_ramp_texture()
            if tex is None:
                self.report({'ERROR'}, "Ramp editor is unavailable.")
                return {'CANCELLED'}
            ramp = tex.color_ramp
            inner_positions = [float(el.position) for el in ramp.elements[1:-1]]
            normalized = self._resample_positions(inner_positions, len(cut_sockets))
            low_hz, high_hz = _splitter_display_bounds(context.scene)
            hz_values = sorted(_splitter_norm_to_hz(p, low_hz=low_hz, high_hz=high_hz) for p in normalized)
        except Exception as e:
            self.report({'ERROR'}, f"Could not read ramp editor: {e}")
            return {'CANCELLED'}

        for sock, hz in zip(cut_sockets, hz_values):
            with utils.logged_errors("n_panel_operators.ACE_OT_apply_splitter_ramp.execute#2"):
                sock.default_value = float(hz)
        _refresh_node_instance(node)

        try:
            from .n_panel_ui import _find_active_node_tree, _sync_ui_dummy_tree
            tree = _find_active_node_tree(context)
        except Exception as exc:
            utils.note_swallowed("n_panel_operators.ACE_OT_apply_splitter_ramp.execute#3", exc)
            tree = None
            _sync_ui_dummy_tree = None  # type: ignore
        _queue_engine_recompile(context, tree=tree)
        if _sync_ui_dummy_tree:
            with utils.logged_errors("n_panel_operators.ACE_OT_apply_splitter_ramp.execute#4"):
                _sync_ui_dummy_tree(context, force=False)

        with utils.logged_errors("n_panel_operators.ACE_OT_apply_splitter_ramp.execute#5"):
            _sync_splitter_ramp_texture_from_node(node, context.scene, force=True)
        return {'FINISHED'}


class ACE_OT_update_audio_math_inputs(bpy.types.Operator):
    bl_idname = "ace.update_audio_math_inputs"
    bl_label = "Change Audio Math Inputs"
    bl_options = {'INTERNAL', 'UNDO'}
    direction: bpy.props.EnumProperty(items=[('UP', "Up", ""), ('DOWN', "Down", "")])  # type: ignore

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        is_audio_math = bool(parsed and parsed.get("category") == "utilities" and parsed.get("type") == "audio_math")
        if not node or not is_audio_math:
            return {'CANCELLED'}
        current = int(node.get('ace_input_count', max(2, len(getattr(node, "inputs", [])))))
        new_cnt = min(current + 1, 8) if self.direction == 'UP' else max(current - 1, 2)
        if new_cnt != current:
            ACEAudioMathManager.apply_variant_to_node(node, new_cnt)
        return {'FINISHED'}


class ACE_OT_set_audio_math_op(bpy.types.Operator):
    bl_idname = "ace.set_audio_math_op"
    bl_label = "Set Audio Math Operation"
    bl_options = {'INTERNAL', 'UNDO'}
    op: bpy.props.StringProperty()                                                                                                              # type: ignore

    def execute(self, context):
        node, parsed = _resolve_panel_selected_node(context)
        is_audio_math = bool(parsed and parsed.get("category") == "utilities" and parsed.get("type") == "audio_math")
        if not node or not is_audio_math:
            return {'CANCELLED'}
        op = str(self.op or "add").strip().lower()
        if op not in {"add", "subtract", "multiply"}:
            op = "add"
        node["ace_op"] = op
        _refresh_node_instance(node)
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


class ACE_OT_refresh_stream_list(bpy.types.Operator):
    """Refresh the UI dummy tree list for the active source."""
    bl_idname = "audiocore.refresh_stream_list"
    bl_label = "Refresh Streams"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        try:
            from .n_panel_ui import _sync_ui_dummy_tree, _find_active_node_tree
            tree = _find_active_node_tree(context)
            _queue_engine_recompile(context, tree=tree)
            _sync_ui_dummy_tree(context, force=False)
            utils.log("UI stream list refreshed.", "DEBUG")
        except Exception as e:
            utils.log(f"Failed to refresh stream list: {e}", "ERROR")
            self.report({'ERROR'}, "Could not refresh streams.")
            return {'CANCELLED'}
        return {'FINISHED'}


class ACE_OT_set_target_object(bpy.types.Operator):
    """Assign the GN target object used for ACE processing."""
    bl_idname = "audiocore.set_target_object"
    bl_label = "Set Target Object"
    bl_options = {'REGISTER', 'UNDO'}

    target: bpy.props.EnumProperty(  # type: ignore
        name="Target",
        items=[
            ('ACTIVE', "Active Object", "Use the current active object"),
            ('SONAR', "ACE Sonar", "Use the ACE_Sonar object"),
        ],
        default='ACTIVE',
    )

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        ace = getattr(scene, "audiocore", None)
        if not ace:
            return False
        if getattr(context, "active_object", None) is not None:
            return True
        if bpy.data.objects.get(ACEProxyManager.SONAR_OBJECT_NAME) is not None:
            return True
        return bool(getattr(ace, "audio_sources", None))

    def execute(self, context):
        ace = getattr(context.scene, "audiocore", None)
        if not ace:
            self.report({'ERROR'}, "AudioCore settings unavailable.")
            return {'CANCELLED'}

        target_obj = None
        if self.target == 'ACTIVE':
            target_obj = getattr(context, "active_object", None)
        else:
            target_obj = ACEProxyManager.ensure_sonar_setup(context.scene, getattr(ace, "audio_sources", []))

        if not target_obj:
            self.report({'ERROR'}, "No suitable target object.")
            return {'CANCELLED'}

        try:
            ace.gn_target_object = target_obj
        except Exception:
            self.report({'ERROR'}, "Could not assign target object.")
            return {'CANCELLED'}
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
            from .n_panel_ui import _find_active_node_tree, _sync_ui_dummy_tree
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
            self.report({'ERROR'}, "AudioCore settings unavailable.")
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

# --- REGISTRATION ---
classes = (
    ACE_OT_rebuild_audio_engine,
    ACE_OT_add_audio_sources_from_files,
    ACE_OT_add_audio_source,
    ACE_OT_remove_audio_source,
    ACE_OT_update_splitter_bands,
    ACE_OT_sort_splitter_freqs,
    ACE_OT_even_splitter_freqs,
    ACE_OT_sync_splitter_ramp,
    ACE_OT_apply_splitter_ramp,
    ACE_OT_update_audio_math_inputs,
    ACE_OT_set_audio_math_op,
    ACE_OT_detect_source_tempo_meter,
    ACE_OT_refresh_stream_list,
    ACE_OT_set_target_object,
    ACE_OT_toggle_preview_sink,
    ACE_OT_bake_active_source,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    
