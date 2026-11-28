import bpy # type: ignore
from ..vars import ACE_NODES, MENU_SEPARATOR_AFTER, SPLITTER_KIND, node_is_native
from ..nodes.placement import place_node
from ..nodes.geometry.factory import ACEDummyNodeFactory
from ..nodes.shader.factory import ACEShaderNodeFactory
from ..nodes.identity import new_sink_name, node_value_outputs
from ..core.engine.sinks import write_sink_names
from ..core.proxy.manager import ACEProxyManager
from ..nodes.dynamic_audio_operators import ACESplitterManager
from .. import utils

def _node_available(category: str, node_type: str, tree_type: str) -> bool:
    """Every node is offered in both editors: a controller in a shader is computed by a Geometry Nodes copy."""
    return tree_type in ("GeometryNodeTree", "ShaderNodeTree") or not node_is_native(category, node_type)


# A category is offered where at least one of its nodes is: an empty submenu is hidden.
_ALLOWED_CATEGORIES_BY_TREE_TYPE = {
    tree_type: tuple(
        category for category, cat_def in ACE_NODES.items()
        if any(_node_available(category, node_type, tree_type) for node_type in cat_def["nodes"])
    )
    for tree_type in ("GeometryNodeTree", "ShaderNodeTree")
}


def _menu_tree_type(context) -> str:
    space = getattr(context, "space_data", None)
    tree = getattr(space, "node_tree", None) if space is not None else None
    return str(getattr(tree, "bl_idname", "") or "")


def _allowed_categories_for_context(context) -> tuple[str, ...]:
    return tuple(_ALLOWED_CATEGORIES_BY_TREE_TYPE.get(_menu_tree_type(context), ()))



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
        with utils.logged_errors("gn_menu.ACE_OT_add_node_group._resolve_active_proxy_name"):
            ACEProxyManager.ensure_proxy_for_source(source, scene=getattr(context, "scene", None))
        return getattr(source, "proxy_object_name", "") or ""

    @classmethod
    def _initialize_sink_names(cls, node, category: str, node_type: str, context):
        """Name the values a new output node writes; the names stay as they are from then on."""
        outputs = node_value_outputs(category, node_type, str(getattr(getattr(node, "id_data", None), "bl_idname", "") or ""))
        if not outputs:
            return
        names: dict[int, str] = {}
        for out_idx, output_name in outputs:
            names[out_idx] = new_sink_name(output_name, set(names.values()))
        write_sink_names(node, category, node_type, names)

    @staticmethod
    def _set_socket_default_value(sock, value) -> bool:
        """Safely assign a socket default, skipping incompatible socket types."""
        if sock is None:
            return False
        try:
            current = getattr(sock, "default_value", None)
        except Exception as exc:
            utils.note_swallowed("gn_menu.ACE_OT_add_node_group._set_socket_default_value", exc)
            current = None
        if hasattr(value, "name"):
            try:
                value_name = str(getattr(value, "name", "") or "")
            except Exception as exc:
                utils.note_swallowed("gn_menu.ACE_OT_add_node_group._set_socket_default_value#2", exc)
                value_name = ""
            if isinstance(current, str):
                value = value_name
        if isinstance(value, str) and current is not None and not isinstance(current, str):
            # Some Shader sockets are float fallbacks when String sockets are unavailable.
            return False
        try:
            sock.default_value = value
            return True
        except Exception as exc:
            utils.note_swallowed("gn_menu.ACE_OT_add_node_group._set_socket_default_value#3", exc)
            return False

    @staticmethod
    def _show_attr_id_socket_values(node):
        attr_socket_names = {"Attribute ID"}
        with utils.logged_errors("gn_menu.ACE_OT_add_node_group._show_attr_id_socket_values"):
            node_tree = getattr(node, "node_tree", None)
            if node_tree is not None:
                for item in getattr(node_tree.interface, "items_tree", []):
                    if (
                        getattr(item, "in_out", None) == 'INPUT'
                        and getattr(item, "name", "") in attr_socket_names
                        and hasattr(item, "hide_value")
                    ):
                        item.hide_value = False
        for sock in getattr(node, "inputs", []):
            if getattr(sock, "name", "") in attr_socket_names:
                with utils.logged_errors("gn_menu.ACE_OT_add_node_group._show_attr_id_socket_values#2"):
                    sock.hide_value = False

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
        
        if (self.category_key, self.node_type_key) == SPLITTER_KIND:
            try:
                ace = getattr(getattr(context, "scene", None), "audiocore", None)
                settings = getattr(ace, "engine_settings", None) if ace else None
                splitter_bands = int(getattr(settings, "splitter_default_bands", 2) or 2)
            except Exception as exc:
                utils.note_swallowed("gn_menu.ACE_OT_add_node_group.execute", exc)
                splitter_bands = 2
            splitter_bands = max(2, min(splitter_bands, 8))
            group = ACESplitterManager.get_or_create_variant(splitter_bands, tree)
        else:
            try:
                group = factory_cls.template_group(self.category_key, self.node_type_key)
            except Exception as e:
                self.report({'ERROR'}, f"Could not build node group: {str(e)}")
                return {'CANCELLED'}
        if group is None:
            self.report({'ERROR'}, "Could not resolve node group.")
            return {'CANCELLED'}

        node = tree.nodes.new(group_node_type)
        node.node_tree = group
        node.location = context.space_data.cursor_location
        # The node's kind comes from its group and its identity from its name: nothing else is stored.
        node.label = ACE_NODES[self.category_key]['nodes'][self.node_type_key]['label']
        node.name = node.label

        if (self.category_key, self.node_type_key) == SPLITTER_KIND:
            node['ace_band_count'] = splitter_bands
            node.label = f"{node.label} ({splitter_bands} Bands)"
        _, active_source = utils._get_active_source(context)
        place_node(node, self.category_key, self.node_type_key, getattr(context, "scene", None), source=active_source)

        self._initialize_sink_names(node, self.category_key, self.node_type_key, context)
        self._show_attr_id_socket_values(node)
        with utils.logged_errors("gn_menu.ACE_OT_add_node_group.execute#3"):
            from ..core.engine.engine import ACEAudioEngine

            scene = getattr(context, "scene", None)
            engine = ACEAudioEngine.ensure_instance(scene) if scene is not None else None
            if engine is not None:
                engine.request_recompile(tree)

        for n in tree.nodes:
            n.select = False
        node.select = True
        tree.nodes.active = node
        
        return {'FINISHED'}


class ACE_MT_add_node_menu_base(bpy.types.Menu):
    """Generic Helper to draw items for a specific category"""
    
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
            node_def = nodes_dict[type_key]
            if not _node_available(category_key, type_key, _menu_tree_type(context)):
                continue
            
            op = layout.operator("ace.add_node_group", text=node_def['label'], icon=node_def['icon'] or 'NONE')
            op.category_key = category_key
            op.node_type_key = type_key


def _category_menu(category_key: str):
    """The submenu of one ACE_NODES category."""

    def poll(cls, context):
        return category_key in _allowed_categories_for_context(context)

    def draw(self, context):
        self.draw_category_items(context, category_key)

    return type(
        f"ACE_MT_{category_key}",
        (ACE_MT_add_node_menu_base,),
        {
            "bl_idname": f"ACE_MT_{category_key}",
            "bl_label": ACE_NODES[category_key]["meta"]["label"],
            "poll": classmethod(poll),
            "draw": draw,
        },
    )


_CATEGORY_MENUS = tuple(_category_menu(category_key) for category_key in ACE_NODES)


class ACE_MT_add_node_menu(bpy.types.Menu):
    bl_idname = "ACE_MT_add_node_menu"
    bl_label = "Audio Core Engine"

    def draw(self, context):
        layout = self.layout
        allowed = set(_allowed_categories_for_context(context))
        for category_key in ACE_NODES:
            if category_key in allowed:
                icon = ACE_NODES[category_key]["meta"].get("icon") or "NONE"
                layout.menu(f"ACE_MT_{category_key}", icon=icon)
            if category_key == MENU_SEPARATOR_AFTER:
                layout.separator()

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


classes = (
    ACE_OT_add_node_group,
    *_CATEGORY_MENUS,
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
        with utils.logged_errors("gn_menu.unregister"):
            bpy.types.NODE_MT_category_shader_add.remove(add_shader_category_menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
