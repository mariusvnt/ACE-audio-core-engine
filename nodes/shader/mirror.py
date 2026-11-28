"""Controllers in shaders: a hidden Geometry Nodes copy computes them, and ACE reads the result back.

Shader trees have no Simulation Zone, so a controller (Envelope, Running Total, Gate) in a
material, the World or a light runs in a Geometry Nodes tree ACE generates on a hidden object,
``.ACE_Controllers``:

- The chain that feeds each controller input in the shader tree is copied: an ACE measurement
  becomes a read of its value on its source's proxy (the engine writes a shader measurement there
  too, under its sink name, when a controller reads it); a Source node's BPM, Beats per Bar and
  Downbeat are read the same way; a node Geometry Nodes also has (Math, Map Range, Clamp, Mix,
  Float Curve, Value...) is copied with its settings; another controller is its own copy.
- Anything else (a texture, Light Path...) can't be copied: the link it feeds into the controller
  is ignored, and drawn invalid (nodes/link_rules.py). The input keeps its own value.
- Each controller output is stored as a named attribute, named by the shader node's sink name.
  After every frame (``frame_change_post``, also in a render's own depsgraph) and every update,
  ``read_back`` writes it to the scene property of that name, which the shader node's group
  reads with a View Layer Attribute node, as measurements are read.

The copy is rebuilt only when one of the shader trees changes. Controllers inside shader groups
aren't copied yet: only those directly in a material's, the World's or a light's tree.
"""

import hashlib

import bpy  # type: ignore

from ... import utils
from ...vars import ROLE_MEASURE, ROLE_SOURCE, node_is_native, node_role
from ..identity import node_kind, node_value_outputs, sink_names, tree_key
from ..link_rules import IgnoredLink

MIRROR_OBJECT = ".ACE_Controllers"
MIRROR_TREE = ".ACE_Controllers_Mirror"
SIGNATURE_KEY = "ace_mirror_signature"
NAMES_KEY = "ace_mirror_names"

# Nodes that compute the same in Geometry Nodes and in shaders, so a copy gives the same value.
COPYABLE = frozenset({
    "NodeReroute", "ShaderNodeMath", "ShaderNodeMapRange", "ShaderNodeClamp", "ShaderNodeMix",
    "ShaderNodeValue", "ShaderNodeFloatCurve", "ShaderNodeCombineXYZ", "ShaderNodeSeparateXYZ",
    "ShaderNodeVectorMath", "GeometryNodeMenuSwitch",
})
_UNCOPIED_PROPS = frozenset({
    "name", "label", "location", "location_absolute", "width", "height", "select", "hide", "mute",
    "show_options", "show_preview", "show_texture", "use_custom_color", "color", "parent", "bl_idname",
    "bl_label", "bl_description", "bl_icon", "bl_static_type", "bl_width_default", "bl_width_min",
    "bl_width_max", "bl_height_default", "bl_height_min", "bl_height_max", "type", "warning_propagation",
    "color_tag",
})

_cache: dict = {"signature": None, "ignored": {}}


class _Ignore(Exception):
    """The chain holds something the copy can't hold. The argument names it."""


# Where the controllers are


def _owner_trees(scene):
    """The shader trees a scene renders with: its objects' materials, its World, its lights."""
    seen = set()
    for obj in getattr(scene, "objects", []) or []:
        for slot in getattr(obj, "material_slots", []) or []:
            material = getattr(slot, "material", None)
            tree = getattr(material, "node_tree", None) if material is not None else None
            if tree is not None and id(tree) not in seen:
                seen.add(id(tree))
                yield tree
        if getattr(obj, "type", "") == "LIGHT":
            tree = getattr(obj.data, "node_tree", None)
            if tree is not None and id(tree) not in seen:
                seen.add(id(tree))
                yield tree
    world = getattr(scene, "world", None)
    tree = getattr(world, "node_tree", None) if world is not None else None
    if tree is not None and id(tree) not in seen:
        yield tree


def controller_uses(scene):
    """(shader tree, controller node) for each controller directly in a tree the scene renders with."""
    uses = []
    for tree in _owner_trees(scene):
        for node in getattr(tree, "nodes", []) or []:
            kind = node_kind(node)
            if kind[0] and node_is_native(*kind):
                uses.append((tree, node))
    return uses


def controller_sink_names(scene) -> set:
    """The scene properties the shader controllers read: read_back writes them, a compile keeps them."""
    names = set()
    for _tree, node in controller_uses(scene):
        names.update(sink_names(node).values())
    return names


# The copy


def _signature(uses, scene) -> str:
    from ..definitions import tree_content_signature

    parts = []
    for tree in sorted({t for t, _n in uses}, key=lambda t: t.name):
        parts.append(f"{tree.name}:{tree_content_signature(tree)}")
    for source in getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []:
        parts.append(f"{getattr(source, 'stable_id', '')}:{getattr(source, 'proxy_object_name', '')}")
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).hexdigest()


def _feeding_link(sock):
    return next((lk for lk in getattr(sock, "links", []) or [] if not getattr(lk, "is_muted", False)), None)


def _label(node) -> str:
    return str(getattr(node, "label", "") or getattr(node, "name", "") or "a node")


def _source_of(node, scene):
    """The source an ACE node in a shader tree reads: the Source node up its Audio inputs."""
    from ..source_nodes import node_source_id

    for _ in range(64):
        kind = node_kind(node)
        if node_role(*kind) == ROLE_SOURCE:
            uid = node_source_id(node)
            return next((s for s in scene.audiocore.audio_sources if s.stable_id == uid), None)
        audio = next((s for s in node.inputs if str(getattr(s, "type", "")) == "BUNDLE" or "Audio" in s.name), None)
        link = _feeding_link(audio) if audio is not None else None
        if link is None:
            return None
        node = link.from_node
        while str(getattr(node, "bl_idname", "")) == "NodeReroute":
            link = _feeding_link(node.inputs[0])
            if link is None:
                return None
            node = link.from_node
    return None


class _Copy:
    """Builds the mirror tree: one Geometry Nodes group node per controller, fed by copies of its chains."""

    def __init__(self, ng, scene):
        self.ng = ng
        self.scene = scene
        self.done: dict = {}  # (node pointer, output identifier) -> Geometry Nodes output socket
        self.controllers: dict = {}  # shader controller node pointer -> its Geometry Nodes group node
        self.reads: set = set()  # sink names of shader measurements read from proxies
        self.ignored: dict = {}  # tree key -> IgnoredLinks into shader controllers
        self.column = 0

    def new(self, bl_idname: str):
        node = self.ng.nodes.new(bl_idname)
        node.location = (-300 * (self.column // 20), -150 * (self.column % 20))
        self.column += 1
        return node

    def proxy_value(self, proxy, attr_name: str, data_type: str = "FLOAT", frame_indexed: bool = True):
        """The value of *attr_name* on *proxy* at the frame (a baked proxy has one point per frame)."""
        from ..geometry.factory import _frame_index_socket

        info = self.new("GeometryNodeObjectInfo")
        info.transform_space = "ORIGINAL"
        info.inputs["Object"].default_value = proxy
        attr = self.new("GeometryNodeInputNamedAttribute")
        attr.data_type = data_type
        attr.inputs["Name"].default_value = attr_name
        sample = self.new("GeometryNodeSampleIndex")
        sample.data_type, sample.domain, sample.clamp = data_type, "POINT", True
        self.ng.links.new(info.outputs["Geometry"], sample.inputs["Geometry"])
        self.ng.links.new(attr.outputs[0], sample.inputs["Value"])
        if frame_indexed:
            self.ng.links.new(_frame_index_socket(self.ng, info.outputs["Geometry"], None, (0, 0)), sample.inputs["Index"])
        return sample.outputs["Value"]

    def output(self, sock):
        """The Geometry Nodes socket computing what shader output *sock* carries."""
        node = sock.node
        key = (node.as_pointer(), str(sock.identifier))
        if key in self.done:
            return self.done[key]
        kind = node_kind(node)
        if kind[0]:
            result = self.ace_output(node, sock, kind)
        elif str(getattr(node, "bl_idname", "")) in COPYABLE and not getattr(node, "mute", False):
            result = self.copied_output(node, sock)
        else:
            raise _Ignore(_label(node))
        self.done[key] = result
        return result

    def ace_output(self, node, sock, kind):
        index = next(i for i, s in enumerate(node.outputs) if s == sock)
        if node_is_native(*kind):
            return self.controller(node).outputs[index]
        role = node_role(*kind)
        source = _source_of(node, self.scene)
        proxy = bpy.data.objects.get(getattr(source, "proxy_object_name", "") or "") if source is not None else None
        if proxy is None:
            raise _Ignore(f"{_label(node)} (its source has no proxy yet)")
        if role == ROLE_SOURCE:
            from ...core.proxy.attributes import SOURCE_TEMPO_ATTRS

            items = list(SOURCE_TEMPO_ATTRS.items())
            if not 1 <= index <= len(items):
                raise _Ignore(_label(node))
            name, data_type = items[index - 1]
            return self.proxy_value(proxy, name, data_type, frame_indexed=False)
        if role == ROLE_MEASURE:
            name = sink_names(node).get(index)
            if not name:
                raise _Ignore(f"{_label(node)} (it has no sink name)")
            self.reads.add(name)
            return self.proxy_value(proxy, name)
        raise _Ignore(_label(node))

    def copied_output(self, node, sock):
        if str(node.bl_idname) == "NodeReroute":
            link = _feeding_link(node.inputs[0])
            if link is None:
                raise _Ignore(_label(node))
            return self.output(link.from_socket)
        copy = self.new(node.bl_idname)
        for prop in node.bl_rna.properties:
            ident = prop.identifier
            if prop.is_readonly or ident in _UNCOPIED_PROPS or prop.type not in {"ENUM", "BOOLEAN", "INT", "FLOAT", "STRING"}:
                continue
            with utils.logged_errors("mirror._Copy.copied_output#prop"):
                setattr(copy, ident, getattr(node, ident))
        if str(node.bl_idname) == "ShaderNodeFloatCurve":
            _copy_curve(node.mapping, copy.mapping)
        if str(node.bl_idname) == "GeometryNodeMenuSwitch":
            for item, source_item in zip(copy.enum_items, node.enum_items):
                item.name = source_item.name
        for source_input in node.inputs:
            target = next((s for s in copy.inputs if s.identifier == source_input.identifier), None)
            if target is None:
                continue
            link = _feeding_link(source_input)
            if link is not None:
                self.ng.links.new(self.output(link.from_socket), target)
            elif hasattr(source_input, "default_value") and hasattr(target, "default_value"):
                with utils.logged_errors("mirror._Copy.copied_output#default"):
                    target.default_value = source_input.default_value
        out = next((s for s in copy.outputs if s.identifier == sock.identifier), None)
        if out is None:
            raise _Ignore(_label(node))
        return out

    def controller(self, node):
        """The Geometry Nodes copy of a shader controller node, with its inputs fed."""
        from ..geometry.factory import ACEDummyNodeFactory

        key = node.as_pointer()
        if key in self.controllers:
            return self.controllers[key]
        category, node_type = node_kind(node)
        group_name = ACEDummyNodeFactory.get_group_name(category, node_type)
        if bpy.data.node_groups.get(group_name) is None:
            ACEDummyNodeFactory.rebuild_node_group(category, node_type)
        copy = self.new("GeometryNodeGroup")
        copy.node_tree = bpy.data.node_groups[group_name]
        copy.label = node.name
        self.controllers[key] = copy
        for index, source_input in enumerate(node.inputs):
            target = copy.inputs.get(source_input.name)
            if target is None:
                continue
            if hasattr(source_input, "default_value"):
                with utils.logged_errors("mirror._Copy.controller#default"):
                    target.default_value = source_input.default_value
            link = _feeding_link(source_input)
            if link is None:
                continue
            try:
                self.ng.links.new(self.output(link.from_socket), target)
            except _Ignore as exc:
                self.ignored.setdefault(tree_key(node.id_data), []).append(IgnoredLink(
                    str(node.name), index, str(source_input.identifier), str(source_input.name), _label(link.from_node),
                    f"{source_input.name} comes through {exc.args[0]}, which the Geometry Nodes copy that computes "
                    "this controller can't hold: its own value is used.",
                ))
        return copy


def _copy_curve(source, target) -> None:
    for curve_source, curve_target in zip(source.curves, target.curves):
        points = curve_source.points
        while len(curve_target.points) < len(points):
            curve_target.points.new(0.0, 0.0)
        for point_target, point_source in zip(curve_target.points, points):
            point_target.location = point_source.location
            point_target.handle_type = point_source.handle_type
    target.use_clip = source.use_clip
    target.update()


def _mirror_object(scene, tree):
    """The object running the mirror. It must be evaluated, in the viewport and in renders: unlike
    the proxies, it can't sit in the hidden ACE_Internals collection. Its one loose vertex draws
    nothing in Object Mode and never renders."""
    obj = bpy.data.objects.get(MIRROR_OBJECT)
    if obj is None or obj.type != "MESH":
        mesh = bpy.data.meshes.new(f"{MIRROR_OBJECT}_Mesh")
        obj = bpy.data.objects.new(MIRROR_OBJECT, mesh)
    if scene.collection.objects.get(obj.name) is None:
        scene.collection.objects.link(obj)
    for collection in list(obj.users_collection):
        if collection != scene.collection:
            collection.objects.unlink(obj)
    obj.hide_select = True
    modifier = next((m for m in obj.modifiers if m.type == "NODES"), None) or obj.modifiers.new("ACE Controllers", "NODES")
    if modifier.node_group != tree:
        modifier.node_group = tree
    return obj


def remove_mirror() -> None:
    obj = bpy.data.objects.get(MIRROR_OBJECT)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    tree = bpy.data.node_groups.get(MIRROR_TREE)
    if tree is not None:
        bpy.data.node_groups.remove(tree)
    _cache["signature"], _cache["ignored"] = None, {}


def sync(scene, engine=None) -> dict:
    """Build (or keep) the mirror for the scene's shader controllers; the links it ignores, by tree key.

    Also tells the engine which shader measurements the copy reads from proxies
    (``engine.mirror_reads``), so the next compile writes them there.
    """
    from ...core.proxy.manager import ACEProxyManager

    uses = controller_uses(scene)
    if not uses:
        if bpy.data.objects.get(MIRROR_OBJECT) is not None or bpy.data.node_groups.get(MIRROR_TREE) is not None:
            remove_mirror()
        if engine is not None:
            engine.mirror_reads = set()
        return {}
    for source in getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []:
        if any(_source_of(node, scene) is source for _tree, node in uses):
            with utils.logged_errors("mirror.sync#proxy"):
                ACEProxyManager.ensure_proxy_for_source(source, scene=scene)
    signature = _signature(uses, scene)
    tree = bpy.data.node_groups.get(MIRROR_TREE)
    if tree is not None and tree.get(SIGNATURE_KEY) == signature and _cache["signature"] == signature:
        _mirror_object(scene, tree)
        if engine is not None:
            engine.mirror_reads = set(_cache.get("reads", set()))
        return dict(_cache["ignored"])
    if tree is None:
        tree = bpy.data.node_groups.new(MIRROR_TREE, "GeometryNodeTree")
        tree.is_modifier = True
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    tree.nodes.clear()
    copy = _Copy(tree, scene)
    out = copy.new("NodeGroupOutput")
    point = copy.new("GeometryNodeMeshLine")
    point.inputs["Count"].default_value = 1
    geometry = point.outputs[0]
    names = []
    for _tree, node in uses:
        gn_node = copy.controller(node)
        category, node_type = node_kind(node)
        names_of = sink_names(node)
        for out_idx, _name in node_value_outputs(category, node_type, "ShaderNodeTree"):
            sink = names_of.get(out_idx)
            if not sink:
                continue
            store = copy.new("GeometryNodeStoreNamedAttribute")
            store.data_type, store.domain = "FLOAT", "POINT"
            store.inputs["Name"].default_value = sink
            tree.links.new(geometry, store.inputs["Geometry"])
            tree.links.new(gn_node.outputs[out_idx], store.inputs["Value"])
            geometry = store.outputs[0]
            names.append(sink)
    tree.links.new(geometry, out.inputs[0])
    tree[SIGNATURE_KEY] = signature
    tree[NAMES_KEY] = names
    _mirror_object(scene, tree)
    _cache.update(signature=signature, ignored=dict(copy.ignored), reads=set(copy.reads))
    if engine is not None:
        engine.mirror_reads = set(copy.reads)
    return dict(copy.ignored)


def read_back(scene, depsgraph=None) -> int:
    """Write each shader controller's value, computed by the mirror, to its scene property; how many changed."""
    obj = bpy.data.objects.get(MIRROR_OBJECT)
    tree = bpy.data.node_groups.get(MIRROR_TREE)
    if obj is None or tree is None or scene is None:
        return 0
    names = [str(n) for n in tree.get(NAMES_KEY, [])]
    if not names:
        return 0
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    attributes = getattr(getattr(evaluated, "data", None), "attributes", None)
    if attributes is None:
        return 0
    changed = 0
    for name in names:
        attr = attributes.get(name)
        if attr is None or not len(attr.data):
            continue
        value = float(attr.data[0].value)
        if scene.get(name) != value:
            scene[name] = value
            changed += 1
    if changed:
        scene.update_tag()
    return changed
