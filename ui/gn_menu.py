import bpy # type: ignore
from ..vars import ACE_NODES
from ..nodes.dummy_tree import ACEDummyNodeFactory
from ..core.id_management import ACEIdManager
from ..nodes.dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager

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
    def _resolve_active_proxy_name(context) -> str:
        ace = getattr(getattr(context, "scene", None), "audiocore", None)
        if not ace or not getattr(ace, "audio_sources", None):
            return ""
        idx = getattr(ace, "active_audio_source_index", 0)
        if idx < 0 or idx >= len(ace.audio_sources):
            return ""
        source = ace.audio_sources[idx]
        return getattr(source, "proxy_object_name", "") or ""

    @staticmethod
    def _default_attr_name(proxy_name: str, node_id: str, output_name: str, socket_idx: int = 0) -> str:
        if proxy_name:
            return ACEIdManager.generate_compact_attr_name(proxy_name, node_id, output_name, socket_idx)
        metric = ACEIdManager._clean_string(output_name or "out").lower()[:6] or "val"
        sig = ACEIdManager._clean_string(f"{node_id}{socket_idx}").lower()[-3:] or "000"
        return f"{ACEIdManager.PREFIX}{metric}_src{sig}"[:63]

    @staticmethod
    def _initialize_auto_attr_ids(node, context):
        parsed = ACEIdManager.parse_node_id(node.get("ace_id", ""))
        if not parsed:
            return
        is_stream = parsed.get("category") == "stream"
        is_specs = parsed.get("category") == "utilities" and parsed.get("type") == "src_specs"
        is_scanner = parsed.get("category") == "utilities" and parsed.get("type") == "scanner"
        if not (is_stream or is_specs or is_scanner):
            return

        proxy_name = ACE_OT_add_node_group._resolve_active_proxy_name(context)

        if is_stream and len(node.outputs) > 0:
            auto_name = ACE_OT_add_node_group._default_attr_name(
                proxy_name,
                node["ace_id"],
                node.outputs[0].name,
                0,
            )
            attr_sock = next((s for s in node.inputs if getattr(s, "name", "") == "Attribute ID"), None)
            if attr_sock:
                attr_sock.default_value = auto_name

        if is_specs:
            socket_map = (
                ("BPF Attribute ID", 0),
                ("Time Signature Attribute ID", 1),
                ("Beat Phase Attribute ID", 2),
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
                    attr_sock.default_value = auto_name

        if is_scanner and len(node.outputs) > 0:
            auto_name = ACE_OT_add_node_group._default_attr_name(
                proxy_name,
                node["ace_id"],
                node.outputs[0].name,
                0,
            )
            attr_sock = next((s for s in node.inputs if getattr(s, "name", "") == "Latency Attribute ID"), None)
            if attr_sock:
                attr_sock.default_value = auto_name

    @staticmethod
    def _show_attr_id_socket_values(node):
        attr_socket_names = {
            "Attribute ID",
            "BPF Attribute ID",
            "Time Signature Attribute ID",
            "Beat Phase Attribute ID",
            "Latency Attribute ID",
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

    def execute(self, context):
        space = context.space_data
        if not space or space.type != 'NODE_EDITOR' or not space.node_tree:
            self.report({'ERROR'}, "No active node tree.")
            return {'CANCELLED'}

        if self.category_key == "source_mngmt":
            self.report({'ERROR'}, "Source Management nodes are managed automatically per audio source.")
            return {'CANCELLED'}
        
        tree = space.node_tree
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
            group = ACESplitterManager.get_or_create_variant(splitter_bands)
        elif self.category_key == "utilities" and self.node_type_key == "audio_math":
            try:
                ace = getattr(getattr(context, "scene", None), "audiocore", None)
                settings = getattr(ace, "engine_settings", None) if ace else None
                audio_math_inputs = int(getattr(settings, "audio_math_default_inputs", 2) or 2)
            except Exception:
                audio_math_inputs = 2
            audio_math_inputs = max(2, min(audio_math_inputs, 8))
            group = ACEAudioMathManager.get_or_create_variant(audio_math_inputs)
        else:
            group_name = ACEDummyNodeFactory.get_group_name(self.category_key, self.node_type_key)
            if group_name not in bpy.data.node_groups:
                try:
                    ACEDummyNodeFactory.rebuild_node_group(self.category_key, self.node_type_key)
                except Exception as e:
                    self.report({'ERROR'}, f"Could not build node group: {str(e)}")
                    return {'CANCELLED'}
            group = bpy.data.node_groups.get(group_name)

        # 2. Create the Node instance
        node = tree.nodes.new('GeometryNodeGroup')
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

        # 4. Initialize deterministic attribute IDs for stream/spec nodes
        self._initialize_auto_attr_ids(node, context)
        self._show_attr_id_socket_values(node)

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
    
    def draw_category_items(self, context, category_key):
        layout = self.layout
        if category_key not in ACE_NODES:
            return
            
        # Sort nodes by label for easier reading
        nodes_dict = ACE_NODES[category_key]['nodes']
        sorted_keys = sorted(nodes_dict.keys(), key=lambda k: nodes_dict[k]['label'])
        
        for type_key in sorted_keys:
            # Source placeholders are managed by ACE_Sonar sync and should not be user-added.
            if category_key == "source_mngmt" and type_key == "src":
                continue
            node_def = nodes_dict[type_key]
            
            op = layout.operator("ace.add_node_group", text=node_def['label'], icon=node_def['icon'] or 'NONE')
            op.category_key = category_key
            op.node_type_key = type_key

# --- Submenus for each Top-Level Category ---

class ACE_MT_delta(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_delta"
    bl_label = "Delta"
    def draw(self, context):
        self.draw_category_items(context, "delta")

class ACE_MT_stream(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_stream"
    bl_label = "Stream"
    def draw(self, context):
        self.draw_category_items(context, "stream")

class ACE_MT_utilities(ACE_MT_add_node_menu_base):
    bl_idname = "ACE_MT_utilities"
    bl_label = "Utilities"
    def draw(self, context):
        self.draw_category_items(context, "utilities")

# --- Main Entry Point ---

class ACE_MT_add_node_menu(bpy.types.Menu):
    bl_idname = "ACE_MT_add_node_menu"
    bl_label = "AudioCore Engine"

    def draw(self, context):
        layout = self.layout
        
        # Source management is internal and never user-added.
        layout.menu("ACE_MT_delta", icon='MOD_WAVE')
        layout.menu("ACE_MT_stream", icon='GRAPH')
        layout.menu("ACE_MT_utilities", icon='MODIFIER')

def add_node_menu_func(self, context):
    self.layout.separator()
    self.layout.menu("ACE_MT_add_node_menu", icon='SOUND')

# ==============================================================================
# 3. REGISTRATION
# ==============================================================================

classes = (
    ACE_OT_add_node_group,
    ACE_MT_delta,
    ACE_MT_stream,
    ACE_MT_utilities,
    ACE_MT_add_node_menu,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_add.append(add_node_menu_func)

def unregister():
    bpy.types.NODE_MT_add.remove(add_node_menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
