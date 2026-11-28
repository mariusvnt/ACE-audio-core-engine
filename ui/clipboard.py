"""Copy and paste of ACE nodes between Geometry Nodes and shader trees.

Blender refuses to paste a group node into a tree of another type ("Not a geometry node
tree"), and ACE nodes are group nodes. So Ctrl+C and Ctrl+V in the Node Editor go through ACE:
the copy also notes the selected ACE nodes (kind, label, settings, input values, links), and a
paste into a tree of the other type adds them back with that type's groups. Everything else is
Blender's own copy and paste; a paste into a tree of the same type is only Blender's.
"""

import sys

import bpy  # type: ignore

from .. import utils
from ..vars import SOURCE_KIND, SPLITTER_KIND, node_def
from ..nodes.identity import node_kind

_clipboard: dict = {"tree_type": "", "nodes": [], "links": []}

_COPIED_PROPS = ("ace_mode", "ace_band_count", "ace_preview_enabled") + tuple(f"ace_split_freq_{i}" for i in range(1, 8))


def _plain(value):
    if hasattr(value, "__len__") and not isinstance(value, str):
        return tuple(value)
    return value


def record(tree) -> None:
    """Note the selected ACE nodes of *tree*, and the links between selected nodes."""
    from ..nodes.source_nodes import node_source_id

    nodes, names = [], set()
    for node in tree.nodes:
        if not node.select:
            continue
        names.add(node.name)
        category, node_type = node_kind(node)
        if not category:
            continue
        values = {}
        for sock in node.inputs:
            if sock.is_linked or not hasattr(sock, "default_value") or sock.type in {"OBJECT", "STRING"}:
                continue
            with utils.logged_errors("clipboard.record#value"):
                values[sock.name] = _plain(sock.default_value)
        nodes.append({
            "name": node.name, "label": node.label, "kind": (category, node_type),
            "location": tuple(node.location), "width": float(node.width),
            "props": {k: _plain(node[k]) for k in _COPIED_PROPS if k in node},
            "values": values, "hidden": [s.name for s in node.inputs if s.hide],
            "source": node_source_id(node) if (category, node_type) == SOURCE_KIND else "",
        })
    links = [
        (link.from_node.name, link.from_socket.identifier, link.from_socket.name,
         link.to_node.name, link.to_socket.identifier, link.to_socket.name)
        for link in tree.links
        if link.from_node.name in names and link.to_node.name in names
    ]
    _clipboard.update(tree_type=str(tree.bl_idname), nodes=nodes, links=links,
                      names={n["name"] for n in nodes}, selected=names)


def _group_for(kind, tree_type, props):
    from ..nodes.dynamic_audio_operators import ACESplitterManager
    from ..nodes.geometry.factory import ACEDummyNodeFactory
    from ..nodes.shader.factory import ACEShaderNodeFactory

    if kind == SPLITTER_KIND:
        return ACESplitterManager.get_or_create_variant(int(props.get("ace_band_count", 2) or 2), tree_type)
    factory = ACEShaderNodeFactory if tree_type == "ShaderNodeTree" else ACEDummyNodeFactory
    return factory.template_group(*kind)


def _socket(sockets, identifier, name):
    return next((s for s in sockets if s.identifier == identifier and s.enabled), None) or next(
        (s for s in sockets if s.name == name and s.enabled), None)


def translate(tree, scene, pasted_natives) -> list:
    """Add the noted ACE nodes to *tree*, a tree of the other type; the new nodes.

    *pasted_natives* are the nodes Blender just pasted ({copied name: pasted node}): links
    between them and the ACE nodes are made again, and they set where the ACE nodes go.
    """
    from ..nodes.placement import place_node

    tree_type = str(tree.bl_idname)
    offset = None
    for entry in _clipboard.get("natives_at", {}).items():
        name, location = entry
        node = pasted_natives.get(name)
        if node is not None:
            offset = (node.location[0] - location[0], node.location[1] - location[1])
            break
    if offset is None:
        cursor = getattr(getattr(bpy.context, "space_data", None), "cursor_location", (0.0, 0.0))
        xs = [n["location"][0] for n in _clipboard["nodes"]] or [0.0]
        ys = [n["location"][1] for n in _clipboard["nodes"]] or [0.0]
        offset = (cursor[0] - min(xs), cursor[1] - max(ys))
    sources = {s.stable_id: s for s in getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []}
    made = {}
    for entry in _clipboard["nodes"]:
        kind = tuple(entry["kind"])
        group = _group_for(kind, tree_type, entry["props"])
        if group is None:
            continue
        node = tree.nodes.new("ShaderNodeGroup" if tree_type == "ShaderNodeTree" else "GeometryNodeGroup")
        node.node_tree = group
        node.name = entry["name"]
        node.label = entry["label"]
        node.width = entry["width"]
        node.location = (entry["location"][0] + offset[0], entry["location"][1] + offset[1])
        for key, value in entry["props"].items():
            node[key] = value
        for sock_name, value in entry["values"].items():
            sock = node.inputs.get(sock_name)
            if sock is not None and hasattr(sock, "default_value"):
                with utils.logged_errors("clipboard.translate#value"):
                    sock.default_value = value
        source = sources.get(entry["source"]) if entry["source"] else None
        # Its sink names are new: the compile's repair names a node that has none.
        place_node(node, *kind, scene, source=source)
        for sock_name in entry["hidden"]:
            sock = node.inputs.get(sock_name)
            if sock is not None and not sock.is_linked:
                sock.hide = True
        made[entry["name"]] = node
    ends = dict(pasted_natives)
    ends.update(made)
    for from_name, from_id, from_sock, to_name, to_id, to_sock in _clipboard["links"]:
        if from_name not in made and to_name not in made:
            continue  # Blender pasted that link itself
        src, dst = ends.get(from_name), ends.get(to_name)
        if src is None or dst is None:
            continue
        out, inp = _socket(src.outputs, from_id, from_sock), _socket(dst.inputs, to_id, to_sock)
        if out is not None and inp is not None:
            with utils.logged_errors("clipboard.translate#link"):
                tree.links.new(out, inp)
    for node in tree.nodes:
        node.select = False
    for node in list(made.values()) + list(pasted_natives.values()):
        node.select = True
    return list(made.values())


def _edit_tree(context):
    space = getattr(context, "space_data", None)
    return getattr(space, "edit_tree", None) if getattr(space, "type", "") == "NODE_EDITOR" else None


class ACE_OT_clipboard_copy(bpy.types.Operator):
    """Copy the selected nodes to the clipboard; ACE nodes can then be pasted into shader and Geometry Nodes trees alike"""

    bl_idname = "ace.clipboard_copy"
    bl_label = "Copy"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def execute(self, context):
        tree = _edit_tree(context)
        with utils.logged_errors("clipboard.ACE_OT_clipboard_copy"):
            record(tree)
            _clipboard["natives_at"] = {
                n.name: tuple(n.location) for n in tree.nodes if n.select and not node_kind(n)[0]
            }
        return bpy.ops.node.clipboard_copy()


class ACE_OT_clipboard_paste(bpy.types.Operator):
    """Paste nodes from the clipboard; ACE nodes copied from a tree of the other type are added with this type's groups"""

    bl_idname = "ace.clipboard_paste"
    bl_label = "Paste"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def invoke(self, context, event):
        """Ctrl+V: Blender's paste puts the nodes at the mouse, and so does ACE's translation."""
        region = getattr(context, "region", None)
        if region is not None and region.type == 'WINDOW':
            context.space_data.cursor_location_from_region(event.mouse_region_x, event.mouse_region_y)
        return self._paste(context, 'INVOKE_DEFAULT')

    def execute(self, context):
        return self._paste(context, 'EXEC_DEFAULT')

    def _paste(self, context, call):
        tree = _edit_tree(context)
        before = {n.as_pointer() for n in tree.nodes}
        cross = bool(_clipboard["nodes"]) and _clipboard["tree_type"] not in ("", str(tree.bl_idname))
        try:
            result = bpy.ops.node.clipboard_paste(call)
        except RuntimeError as exc:
            # Blender pasted what it could and refused the group nodes of the other type.
            if not cross:
                self.report({'ERROR'}, str(exc).removeprefix("Error: ").strip())
                return {'CANCELLED'}
            result = {'FINISHED'}
        if not cross:
            return result
        pasted = [n for n in tree.nodes if n.as_pointer() not in before]
        natives = {}
        copied_names = set(_clipboard.get("selected", ())) - set(_clipboard["names"])
        for node in pasted:
            base = str(node.name)
            copied = base if base in copied_names else None  # "Math.001" copied as itself
            if copied is None:
                # Renamed by the paste: "Math" pasted where a "Math" was already is "Math.001".
                stem, _dot, suffix = base.rpartition(".")
                copied = stem if suffix.isdigit() and stem in copied_names else None
            if copied is not None and copied not in natives:
                natives[copied] = node
        with utils.logged_errors("clipboard.ACE_OT_clipboard_paste"):
            made = translate(tree, context.scene, natives)
            if made:
                what = "Shader" if tree.bl_idname == "ShaderNodeTree" else "Geometry Nodes"
                kinds = sorted({node_def(*node_kind(n)).get("label", "") for n in made})
                self.report({'INFO'}, f"ACE nodes pasted with their {what} groups: {', '.join(kinds)}")
        return {'FINISHED'}


classes = (ACE_OT_clipboard_copy, ACE_OT_clipboard_paste)
_keymaps: list = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is None:  # background mode
        return
    keymap = keyconfig.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
    modifier = "oskey" if sys.platform == "darwin" else "ctrl"  # Blender's own copy and paste keys
    for idname, key in (("ace.clipboard_copy", "C"), ("ace.clipboard_paste", "V")):
        item = keymap.keymap_items.new(idname, key, "PRESS", **{modifier: True})
        _keymaps.append((keymap, item))


def unregister():
    for keymap, item in _keymaps:
        with utils.logged_errors("clipboard.unregister"):
            keymap.keymap_items.remove(item)
    _keymaps.clear()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
