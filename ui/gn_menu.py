import bpy # type: ignore
from ..vars import ACE_NODES
from ..nodes.dummy_tree import (
    ACEDummyNodeFactory,
    ensure_frontend_uid,
    FRONTEND_KIND_SHADER,
)
from ..nodes.shader_tree import ACEShaderNodeFactory
from ..core.abstract_tree import ace_node_type_is_writeable_output
from ..core.id_management import ACEIdManager
from ..core.proxy_management import ACEProxyManager
from ..nodes.dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager
from .. import utils

_ALLOWED_CATEGORIES_BY_TREE_TYPE = {
    "GeometryNodeTree": ("source_mngmt", "delta", "stream", "utilities"),
    "ShaderNodeTree": ("source_mngmt", "delta", "stream", "utilities"),
}

_INPUT_MENU_ITEMS = (
    ("source_mngmt", "src", None),
    ("utilities", "src_specs", "Input Info"),
)

_OUTPUT_MENU_ITEMS = (
    ("utilities", "preview_sink", None),
)

_STREAM_MENU_GROUPS = (
    ("peak", "rms", "lufs"),
    ("spectral_centroid", "spectral_flux"),
    ("pitch", "onset", "zero_crossing_rate"),
    ("stereo_width", "stereo_balance"),
)

_UTILITIES_MENU_ORDER = (
    "audio_math",
    "logic",
    "util_rolling_avg",
    "stream_accumulator",
)


def _menu_tree_type(context) -> str:
    space = getattr(context, "space_data", None)
    tree = getattr(space, "node_tree", None) if space is not None else None
    return str(getattr(tree, "bl_idname", "") or "")


def _allowed_categories_for_context(context) -> tuple[str, ...]:
    return tuple(_ALLOWED_CATEGORIES_BY_TREE_TYPE.get(_menu_tree_type(context), ()))


# ==============================================================================
# 1. THE OPERATOR (Logic)
# ==============================================================================

class ACE_OT_add_node_group(bpy.types.Operator):
    """
    Add an ACE Node Group to the active node tree
    """
    bl_idname = "ace.add_node_group"
    bl_label = "Add ACE Node"
    bl_options = {'REGISTER', 'UNDO'}
    
    # We need both keys to find the definition in ACE_NODES
    category_key: bpy.props.StringProperty()                    # type: ignore
    node_type_key: bpy.props.StringProperty()                   # type: ignore

    @staticmethod
    def _active_tree_type(context) -> str:
        return _menu_tree_type(context)

    @classmethod
    def _category_allowed_for_context(cls, category_key: str, context) -> bool:
        return str(category_key or "") in _allowed_categories_for_context(context)

    @classmethod
    def _factory_for_context(cls, context):
        if cls._active_tree_type(context) == "ShaderNodeTree":
            return ACEShaderNodeFactory
        return ACEDummyNodeFactory

    @classmethod
    def _group_node_type_for_context(cls, context) -> str:
        if cls._active_tree_type(context) == "ShaderNodeTree":
            return "ShaderNodeGroup"
        return "GeometryNodeGroup"

    @staticmethod
    def _resolve_active_proxy_name(context) -> str:
        ace = getattr(getattr(context, "scene", None), "audiocore", None)
        if not ace or not getattr(ace, "audio_sources", None):
            return ""
        idx = getattr(ace, "active_audio_source_index", 0)
        if idx < 0 or idx >= len(ace.audio_sources):
            return ""
        source = ace.audio_sources[idx]
        try:
            ACEProxyManager.ensure_proxy_for_source(source, scene=getattr(context, "scene", None))
        except Exception:
            pass
        return getattr(source, "proxy_object_name", "") or ""

    @staticmethod
    def _default_attr_name(proxy_name: str, node_id: str, output_name: str, socket_idx: int = 0) -> str:
        if proxy_name:
            return ACEIdManager.generate_compact_attr_name(proxy_name, node_id, output_name, socket_idx)
        metric = ACEIdManager._clean_string(output_name or "out").lower()[:6] or "val"
        sig = ACEIdManager._clean_string(f"{node_id}{socket_idx}").lower()[-3:] or "000"
        return f"{ACEIdManager.PREFIX}{metric}_src{sig}"[:63]

    @staticmethod
    def _initialize_source_binding(node, context):
        _, active_source = utils._get_active_source(context)
        if active_source is None:
            return
        existing_ids = [
            getattr(src, "stable_id", "")
            for src in getattr(getattr(context.scene, "audiocore", None), "audio_sources", [])
            if getattr(src, "stable_id", "")
        ]
        source_uid = ACEIdManager.ensure_source_uid(active_source, existing_ids=existing_ids)
        source_name = str(getattr(active_source, "name", "") or "Source")
        try:
            node["ace_source_id"] = str(source_uid or "")
        except Exception:
            pass
        try:
            node["ace_source_name"] = source_name
        except Exception:
            pass
        node.label = source_name
        node.name = source_name

        try:
            ACEProxyManager.ensure_proxy_for_source(active_source, scene=getattr(context, "scene", None))
        except Exception:
            pass

        proxy_name = str(getattr(active_source, "proxy_object_name", "") or "")
        proxy_sock = next((s for s in getattr(node, "inputs", []) if getattr(s, "name", "") == "Proxy Object"), None)
        if proxy_sock is not None:
            proxy_obj = None
            if proxy_name:
                try:
                    proxy_obj = bpy.data.objects.get(proxy_name)
                except Exception:
                    proxy_obj = None
            try:
                if proxy_obj is not None:
                    proxy_sock.default_value = proxy_obj
                elif proxy_name:
                    ACE_OT_add_node_group._set_socket_default_value(proxy_sock, proxy_name)
            except Exception:
                if proxy_name:
                    ACE_OT_add_node_group._set_socket_default_value(proxy_sock, proxy_name)

    @staticmethod
    def _collect_scene_frontend_uids(scene) -> list[str]:
        if scene is None:
            return []
        ordered: list[str] = []

        def _push_uid(tree):
            if tree is None:
                return
            uid = ""
            try:
                uid = str(tree.get("ace_frontend_uid", "") or "")
            except Exception:
                uid = ""
            if uid and uid not in ordered:
                ordered.append(uid)

        for obj in getattr(scene, "objects", []) or []:
            for mod in getattr(obj, "modifiers", []) or []:
                if getattr(mod, "type", None) != "NODES":
                    continue
                _push_uid(getattr(mod, "node_group", None))
            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
                    continue
                _push_uid(tree)
        return ordered

    @classmethod
    def _ensure_shader_frontend_uid(cls, tree, context):
        if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
            return ""
        existing_uids = cls._collect_scene_frontend_uids(getattr(context, "scene", None))
        return ensure_frontend_uid(tree, FRONTEND_KIND_SHADER, existing_uids)

    @staticmethod
    def _initialize_auto_attr_ids(node, context):
        parsed = ACEIdManager.parse_node_id(node.get("ace_id", ""))
        if not parsed:
            return
        if not ace_node_type_is_writeable_output(parsed.get("category", ""), parsed.get("type", "")):
            return
        is_stream = parsed.get("category") == "stream"
        is_specs = parsed.get("category") == "utilities" and parsed.get("type") == "src_specs"
        is_rolling_avg = parsed.get("category") == "utilities" and parsed.get("type") == "util_rolling_avg"
        is_stream_accumulator = (
            parsed.get("category") == "utilities" and parsed.get("type") == "stream_accumulator"
        )
        is_logic = parsed.get("category") == "utilities" and parsed.get("type") == "logic"
        if not (is_stream or is_specs or is_rolling_avg or is_stream_accumulator or is_logic):
            return

        proxy_name = ACE_OT_add_node_group._resolve_active_proxy_name(context)
        proxy_obj = None
        if proxy_name:
            try:
                proxy_obj = bpy.data.objects.get(proxy_name)
            except Exception:
                proxy_obj = None

        if is_stream and len(node.outputs) > 0:
            for out_idx, out_sock in enumerate(node.outputs):
                auto_name = ACE_OT_add_node_group._default_attr_name(
                    proxy_name,
                    node["ace_id"],
                    out_sock.name,
                    out_idx,
                )
                socket_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                attr_sock = next((s for s in node.inputs if getattr(s, "name", "") == socket_name), None)
                if attr_sock:
                    ACE_OT_add_node_group._set_socket_default_value(attr_sock, auto_name)

        if is_specs:
            socket_map = (
                ("Frames per Beat Attribute ID", 0),
                ("Beat Phase Attribute ID", 1),
            )
            for socket_name, out_idx in socket_map:
                if out_idx >= len(node.outputs):
                    continue
                auto_name = ACE_OT_add_node_group._default_attr_name(
                    proxy_name,
                    node["ace_id"],
                    node.outputs[out_idx].name,
                    out_idx,
                )
                attr_sock = next((s for s in node.inputs if getattr(s, "name", "") == socket_name), None)
                if attr_sock:
                    ACE_OT_add_node_group._set_socket_default_value(attr_sock, auto_name)

        if (is_rolling_avg or is_logic) and len(node.outputs) > 0:
            auto_name = ACE_OT_add_node_group._default_attr_name(
                proxy_name,
                node["ace_id"],
                node.outputs[0].name,
                0,
            )
            attr_sock = next((s for s in node.inputs if getattr(s, "name", "") == "Attribute ID"), None)
            if attr_sock:
                ACE_OT_add_node_group._set_socket_default_value(attr_sock, auto_name)

        if is_stream_accumulator and len(node.outputs) > 0:
            auto_name = ACE_OT_add_node_group._default_attr_name(
                proxy_name,
                node["ace_id"],
                node.outputs[0].name,
                0,
            )
            attr_sock = next((s for s in node.inputs if getattr(s, "name", "") == "Attribute ID"), None)
            if attr_sock:
                ACE_OT_add_node_group._set_socket_default_value(attr_sock, auto_name)

        if (is_specs or is_rolling_avg or is_logic) and proxy_obj:
            proxy_sock = next((s for s in node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
            if proxy_sock:
                try:
                    proxy_sock.default_value = proxy_obj
                except Exception:
                    if proxy_name:
                        ACE_OT_add_node_group._set_socket_default_value(proxy_sock, proxy_name)
        if is_stream_accumulator and proxy_obj:
            proxy_sock = next((s for s in node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
            if proxy_sock:
                try:
                    proxy_sock.default_value = proxy_obj
                except Exception:
                    if proxy_name:
                        ACE_OT_add_node_group._set_socket_default_value(proxy_sock, proxy_name)

    @staticmethod
    def _set_socket_default_value(sock, value) -> bool:
        """Safely assign a socket default, skipping incompatible socket types."""
        if sock is None:
            return False
        try:
            current = getattr(sock, "default_value", None)
        except Exception:
            current = None
        if hasattr(value, "name"):
            try:
                value_name = str(getattr(value, "name", "") or "")
            except Exception:
                value_name = ""
            if isinstance(current, str):
                value = value_name
        if isinstance(value, str) and current is not None and not isinstance(current, str):
            # Some Shader sockets are float fallbacks when String sockets are unavailable.
            return False
        try:
            sock.default_value = value
            return True
        except Exception:
            return False

    @staticmethod
    def _show_attr_id_socket_values(node):
        attr_socket_names = {
            "Attribute ID",
            "Frames per Beat Attribute ID",
            "Beat Phase Attribute ID",
        }
        try:
            node_tree = getattr(node, "node_tree", None)
            if node_tree is not None:
                for item in getattr(node_tree.interface, "items_tree", []):
                    if (
                        getattr(item, "in_out", None) == 'INPUT'
                        and getattr(item, "name", "") in attr_socket_names
                        and hasattr(item, "hide_value")
                    ):
                        item.hide_value = False
        except Exception:
            pass
        for sock in getattr(node, "inputs", []):
            if getattr(sock, "name", "") in attr_socket_names:
                try:
                    sock.hide_value = False
                except Exception:
                    pass

    @classmethod
    def poll(cls, context):
        if not context or not getattr(context, "space_data", None):
            return False
        space = context.space_data
        if getattr(space, "type", "") != "NODE_EDITOR":
            return False
        return cls._active_tree_type(context) in _ALLOWED_CATEGORIES_BY_TREE_TYPE

    def execute(self, context):
        space = context.space_data
        if not space or space.type != 'NODE_EDITOR' or not space.node_tree:
            self.report({'ERROR'}, "No active node tree.")
            return {'CANCELLED'}

        tree = space.node_tree
        tree_type = str(getattr(tree, "bl_idname", "") or "")
        if tree_type not in {"GeometryNodeTree", "ShaderNodeTree"}:
            self.report({'ERROR'}, f"Unsupported node tree type: {tree_type or 'Unknown'}")
            return {'CANCELLED'}
        if not self._category_allowed_for_context(self.category_key, context):
            self.report({'ERROR'}, f"Category '{self.category_key}' is unavailable in {tree_type}.")
            return {'CANCELLED'}
        factory_cls = self._factory_for_context(context)
        group_node_type = self._group_node_type_for_context(context)
        splitter_bands = 2
        audio_math_inputs = 2
        
        # 1. Resolve the Group data-block
        if self.category_key == "delta" and self.node_type_key == "frequency_splitter":
            try:
                ace = getattr(getattr(context, "scene", None), "audiocore", None)
                settings = getattr(ace, "engine_settings", None) if ace else None
                splitter_bands = int(getattr(settings, "splitter_default_bands", 2) or 2)
            except Exception:
                splitter_bands = 2
            splitter_bands = max(2, min(splitter_bands, 8))
            group = ACESplitterManager.get_or_create_variant(splitter_bands, tree)
        elif self.category_key == "utilities" and self.node_type_key == "audio_math":
            try:
                ace = getattr(getattr(context, "scene", None), "audiocore", None)
                settings = getattr(ace, "engine_settings", None) if ace else None
                audio_math_inputs = int(getattr(settings, "audio_math_default_inputs", 2) or 2)
            except Exception:
                audio_math_inputs = 2
            audio_math_inputs = max(2, min(audio_math_inputs, 8))
            group = ACEAudioMathManager.get_or_create_variant(audio_math_inputs, tree)
        else:
            group_name = factory_cls.get_group_name(self.category_key, self.node_type_key)
            existing_group = bpy.data.node_groups.get(group_name)
            if existing_group is None or str(getattr(existing_group, "bl_idname", "") or "") != tree_type:
                try:
                    factory_cls.rebuild_node_group(self.category_key, self.node_type_key)
                except Exception as e:
                    self.report({'ERROR'}, f"Could not build node group: {str(e)}")
                    return {'CANCELLED'}
            group = bpy.data.node_groups.get(group_name)
        if group is None:
            self.report({'ERROR'}, "Could not resolve node group.")
            return {'CANCELLED'}

        # 2. Create the Node instance
        node = tree.nodes.new(group_node_type)
        node.node_tree = group
        node.location = context.space_data.cursor_location
        
        # 3. CRITICAL: Assign the Unique ID
        # This ties the visual node to the backend logic
        try:
            # We pass existing IDs to ensure absolute uniqueness
            existing_ids = [n.get('ace_id') for n in tree.nodes if n.get('ace_id')]
            
            new_id = ACEIdManager.generate_node_id(
                self.category_key, 
                self.node_type_key, 
                existing_ids=existing_ids
            )
            
            node['ace_id'] = new_id
            node.label = ACE_NODES[self.category_key]['nodes'][self.node_type_key]['label']
            node.name = node.label
            
        except ValueError as e:
            self.report({'ERROR'}, f"ID Generation failed: {str(e)}")
            tree.nodes.remove(node)
            return {'CANCELLED'}

        if self.node_type_key == 'frequency_splitter':
            node['ace_band_count'] = max(2, min(int(node.get('ace_band_count', splitter_bands)), 8))
            node.label = f"{node.label} ({int(node['ace_band_count'])} Bands)"
            try:
                ACESplitterManager._ensure_node_socket_limits(node)
            except Exception:
                pass
        if self.node_type_key == 'audio_math':
            node["ace_input_count"] = max(2, min(int(node.get("ace_input_count", audio_math_inputs)), 8))
            node["ace_op"] = node.get("ace_op", "add")
            node.label = f"{node.label} ({int(node['ace_input_count'])} Inputs)"
        if self.node_type_key == 'preview_sink':
            node["ace_preview_enabled"] = bool(node.get("ace_preview_enabled", True))
        if self.category_key == "source_mngmt" and self.node_type_key == "src":
            if tree_type == "ShaderNodeTree":
                self._ensure_shader_frontend_uid(tree, context)
            self._initialize_source_binding(node, context)

        # 4. Initialize deterministic attribute IDs for stream/spec nodes
        self._initialize_auto_attr_ids(node, context)
        self._show_attr_id_socket_values(node)
        if tree_type == "ShaderNodeTree":
            try:
                from ..core.audio_engine import ACEAudioEngine

                scene = getattr(context, "scene", None)
                if not ACEAudioEngine._structural_tree_writes_allowed(
                    scene,
                    tree=tree,
                    frontend_kind="SHADER_NODES",
                ):
                    engine = ACEAudioEngine.get_instance(scene)
                    if engine is None and scene is not None:
                        engine = ACEAudioEngine.ensure_instance(scene)
                    if engine is not None:
                        engine.request_recompile(tree)
                else:
                    proxy_name = self._resolve_active_proxy_name(context)
                    ACEShaderNodeFactory.sync_node_instance_group_bindings(
                        node,
                        proxy_name=proxy_name,
                        scene=scene,
                    )
            except Exception:
                pass

        # 5. UI Cleanup (Selection)
        for n in tree.nodes:
            n.select = False
        node.select = True
        tree.nodes.active = node
        
        return {'FINISHED'}

# ==============================================================================
# 2. THE MENUS (View)
# ==============================================================================

class ACE_MT_add_node_menu_base(bpy.types.Menu):
    """Generic Helper to draw items for a specific category"""

    def _draw_node_item(self, layout, category_key: str, type_key: str, label_override: str | None = None):
        if category_key not in ACE_NODES:
            return
        node_def = ACE_NODES[category_key]['nodes'].get(type_key)
        if not node_def:
            return
        op = layout.operator("ace.add_node_group", text=label_override or node_def['label'], icon=node_def['icon'] or 'NONE')
        op.category_key = category_key
        op.node_type_key = type_key

    def draw_category_items(self, context, category_key):
        layout = self.layout
        if category_key not in ACE_NODES:
            return
        if str(category_key or "") not in _allowed_categories_for_context(context):
            return
            
        # Sort nodes by label for easier reading
        nodes_dict = ACE_NODES[category_key]['nodes']
        sorted_keys = sorted(nodes_dict.keys(), key=lambda k: nodes_dict[k]['label'])
        
        for type_key in sorted_keys:
            self._draw_node_item(layout, category_key, type_key)

# --- Submenus for each Top-Level Category ---

class ACE_MT_delta(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_delta"
    bl_label = "Delta"

    @classmethod
    def poll(cls, context):
        return "delta" in _allowed_categories_for_context(context)

    def draw(self, context):
        self.draw_category_items(context, "delta")

class ACE_MT_source_mngmt(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_source_mngmt"
    bl_label = "Input"

    @classmethod
    def poll(cls, context):
        return "source_mngmt" in _allowed_categories_for_context(context)

    def draw(self, context):
        allowed = set(_allowed_categories_for_context(context))
        layout = self.layout
        for category_key, type_key, label_override in _INPUT_MENU_ITEMS:
            if category_key not in allowed:
                continue
            self._draw_node_item(layout, category_key, type_key, label_override=label_override)

class ACE_MT_stream(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_stream"
    bl_label = "Stream"

    @classmethod
    def poll(cls, context):
        return "stream" in _allowed_categories_for_context(context)

    def draw(self, context):
        layout = self.layout
        for group_idx, group in enumerate(_STREAM_MENU_GROUPS):
            drew_any = False
            for type_key in group:
                if type_key not in ACE_NODES.get("stream", {}).get("nodes", {}):
                    continue
                self._draw_node_item(layout, "stream", type_key)
                drew_any = True
            if drew_any and group_idx < (len(_STREAM_MENU_GROUPS) - 1):
                layout.separator()

class ACE_MT_utilities(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_utilities"
    bl_label = "Utilities"

    @classmethod
    def poll(cls, context):
        return "utilities" in _allowed_categories_for_context(context)

    def draw(self, context):
        layout = self.layout
        for type_key in _UTILITIES_MENU_ORDER:
            self._draw_node_item(layout, "utilities", type_key)

class ACE_MT_output(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_output"
    bl_label = "Output"

    @classmethod
    def poll(cls, context):
        return "utilities" in _allowed_categories_for_context(context)

    def draw(self, context):
        layout = self.layout
        for category_key, type_key, label_override in _OUTPUT_MENU_ITEMS:
            self._draw_node_item(layout, category_key, type_key, label_override=label_override)

# --- Main Entry Point ---

class ACE_MT_add_node_menu(bpy.types.Menu):
    bl_idname = "ACE_MT_add_node_menu"
    bl_label = "AudioCore Engine"

    def draw(self, context):
        layout = self.layout
        allowed = set(_allowed_categories_for_context(context))
        if "source_mngmt" in allowed:
            layout.menu("ACE_MT_source_mngmt", icon='SOUND')
        if "delta" in allowed:
            layout.menu("ACE_MT_delta", icon='MOD_WAVE')
        if "stream" in allowed:
            layout.menu("ACE_MT_stream", icon='GRAPH')
        if "utilities" in allowed:
            layout.menu("ACE_MT_utilities", icon='NODE')
            layout.menu("ACE_MT_output", icon='PLAY_SOUND')

def add_node_menu_func(self, context):
    space = getattr(context, "space_data", None)
    tree = getattr(space, "node_tree", None) if space is not None else None
    tree_type = str(getattr(tree, "bl_idname", "") or "")
    if tree_type not in {"GeometryNodeTree", "ShaderNodeTree"}:
        return
    self.layout.separator()
    self.layout.menu("ACE_MT_add_node_menu", icon='SOUND')


def add_shader_category_menu_func(self, context):
    space = getattr(context, "space_data", None)
    tree = getattr(space, "node_tree", None) if space is not None else None
    if str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
        return
    self.layout.separator()
    self.layout.menu("ACE_MT_add_node_menu", icon='SOUND')

# ==============================================================================
# 3. REGISTRATION
# ==============================================================================

classes = (
    ACE_OT_add_node_group,
    ACE_MT_source_mngmt,
    ACE_MT_delta,
    ACE_MT_stream,
    ACE_MT_utilities,
    ACE_MT_output,
    ACE_MT_add_node_menu,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_add.append(add_node_menu_func)
    if hasattr(bpy.types, "NODE_MT_category_shader_add"):
        bpy.types.NODE_MT_category_shader_add.append(add_shader_category_menu_func)

def unregister():
    bpy.types.NODE_MT_add.remove(add_node_menu_func)
    if hasattr(bpy.types, "NODE_MT_category_shader_add"):
        try:
            bpy.types.NODE_MT_category_shader_add.remove(add_shader_category_menu_func)
        except Exception:
            pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
