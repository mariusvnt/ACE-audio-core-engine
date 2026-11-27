"""N-panel helpers: active tree lookup, source rows and display names."""

import bpy  # type: ignore
from ...core.id_management import ACEIdManager
from ...core.engine.engine import ACEAudioEngine
from ...nodes.identity import find_node, is_ace_node
from ...vars import ACE_NODES
from ... import utils


_ui_label_icon_ids: set[str] | None = None


def _tree_has_ace_nodes(tree) -> bool:
    if tree is None:
        return False
    return any(is_ace_node(node) for node in getattr(tree, "nodes", []) or [])


def _deal_breakers():
    issues = []
    try:
        if tuple(getattr(bpy.app, "version", (0, 0, 0))) < (5, 2, 0):
            issues.append("Blender 5.2+ is required.")
    except Exception as exc:
        utils.note_swallowed("helpers._deal_breakers", exc)
        issues.append("Unable to read Blender version.")
    return issues


def _ui_is_blocked(context):
    return len(_deal_breakers()) > 0


# Blender's property split: the label right-aligned in the left 40%, the control filling the rest.
_PROPERTY_SPLIT_FACTOR = 0.4


def property_column(layout):
    """A column whose props each take one row in Blender's property-split layout."""
    col = layout.column()
    col.use_property_split = True
    col.use_property_decorate = False
    return col


def property_split_row(layout, label: str):
    """A row shaped like a property-split prop for a control that is not a prop (a menu, a value
    label); returns the right-hand part to draw it in."""
    split = layout.split(factor=_PROPERTY_SPLIT_FACTOR, align=True)
    left = split.row()
    left.alignment = 'RIGHT'
    left.label(text=label)
    return split.row(align=True)


def _iter_visible_node_editor_trees(context):
    """Yield unique supported trees from visible Node Editor areas."""
    seen_tree_ptrs: set[int] = set()
    seen_area_ptrs: set[int] = set()

    def _tree_from_space(space):
        if getattr(space, "type", "") != "NODE_EDITOR":
            return
        for tree in (getattr(space, "edit_tree", None), getattr(space, "node_tree", None)):
            tree_type = str(getattr(tree, "bl_idname", "") or "")
            if tree_type not in {'GeometryNodeTree', 'ShaderNodeTree'}:
                continue
            tree_ptr = _tree_pointer(tree)
            if tree_ptr in seen_tree_ptrs:
                continue
            seen_tree_ptrs.add(tree_ptr)
            yield tree

    # Current context space first when already in a node editor.
    for tree in _tree_from_space(getattr(context, "space_data", None)):
        yield tree

    wm = getattr(context, "window_manager", None)
    windows = list(getattr(wm, "windows", []) or [])
    current_window = getattr(context, "window", None)
    if current_window is not None:
        windows = [current_window] + [win for win in windows if win is not current_window]

    for window in windows:
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", []) or []:
            if getattr(area, "type", "") != "NODE_EDITOR":
                continue
            try:
                area_ptr = int(area.as_pointer())
            except Exception as exc:
                utils.note_swallowed("helpers._iter_visible_node_editor_trees", exc)
                area_ptr = id(area)
            if area_ptr in seen_area_ptrs:
                continue
            seen_area_ptrs.add(area_ptr)
            spaces = getattr(area, "spaces", None)
            active_space = getattr(spaces, "active", None) if spaces is not None else None
            if active_space is None:
                continue
            for tree in _tree_from_space(active_space):
                yield tree


def _find_active_node_tree(context, editor_context_only: bool = False):
    """Return the most relevant active ACE authoring tree for the current context."""
    scene = getattr(context, "scene", None)
    ace = getattr(scene, "audiocore", None)
    target_obj = getattr(ace, "gn_target_object", None) if ace else None

    def _is_supported_tree(tree):
        return str(getattr(tree, "bl_idname", "") or "") in {'GeometryNodeTree', 'ShaderNodeTree'}

    def _tree_from_object_modifiers(obj):
        if not obj:
            return (None, None)
        fallback = None
        for mod in getattr(obj, "modifiers", []) or []:
            if getattr(mod, "type", None) == 'NODES' and getattr(mod, "node_group", None):
                tree = mod.node_group
                if not _is_supported_tree(tree):
                    continue
                if fallback is None:
                    fallback = tree
                if ACEAudioEngine._tree_has_ace_nodes(tree):
                    return (tree, fallback)
        return (None, fallback)

    def _tree_from_object_materials(obj):
        if not obj:
            return (None, None)
        fallback = None
        seen_ptrs: set[int] = set()
        materials = []
        active_material = getattr(obj, "active_material", None)
        if active_material is not None:
            materials.append(active_material)
        for slot in getattr(obj, "material_slots", []) or []:
            material = getattr(slot, "material", None)
            if material is not None:
                materials.append(material)
        for material in materials:
            tree = getattr(material, "node_tree", None)
            if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
                continue
            tree_ptr = _tree_pointer(tree)
            if tree_ptr in seen_ptrs:
                continue
            seen_ptrs.add(tree_ptr)
            if fallback is None:
                fallback = tree
            if ACEAudioEngine._tree_has_ace_nodes(tree):
                return (tree, fallback)
        return (None, fallback)

    fallback = None

    def _consider_tree(tree):
        nonlocal fallback
        if tree is None or not _is_supported_tree(tree):
            return None
        if ACEAudioEngine._tree_has_ace_nodes(tree):
            return tree
        if fallback is None:
            fallback = tree
        return None

    # Respect Blender's context resolver only when currently in the Node Editor.
    space = getattr(context, "space_data", None)
    if getattr(space, "type", "") == 'NODE_EDITOR':
        with utils.logged_errors("helpers._find_active_node_tree"):
            resolver = getattr(getattr(bpy, "types", None), "NodeTree", None)
            if resolver is not None and hasattr(resolver, "get_from_context"):
                resolved = resolver.get_from_context(context)
                if isinstance(resolved, (list, tuple)):
                    tree = resolved[0] if resolved else None
                else:
                    tree = resolved
                chosen = _consider_tree(tree)
                if chosen is not None:
                    return chosen
        for tree in (getattr(space, "edit_tree", None), getattr(space, "node_tree", None)):
            chosen = _consider_tree(tree)
            if chosen is not None:
                return chosen

    for tree in _iter_visible_node_editor_trees(context):
        chosen = _consider_tree(tree)
        if chosen is not None:
            return chosen

    if editor_context_only:
        return fallback

    obj = getattr(context, "active_object", None)
    if obj:
        tree, tree_fallback = _tree_from_object_materials(obj)
        if tree is not None:
            return tree
        if fallback is None and tree_fallback is not None:
            fallback = tree_fallback

    if target_obj:
        tree, tree_fallback = _tree_from_object_materials(target_obj)
        if tree is not None:
            return tree
        if fallback is None and tree_fallback is not None:
            fallback = tree_fallback
        tree, tree_fallback = _tree_from_object_modifiers(target_obj)
        if tree is not None:
            return tree
        if fallback is None and tree_fallback is not None:
            fallback = tree_fallback

    if obj:
        tree, tree_fallback = _tree_from_object_modifiers(obj)
        if tree is not None:
            return tree
        if fallback is None and tree_fallback is not None:
            fallback = tree_fallback

    # Fallback: try ACE_Sonar object if present
    sonar = bpy.data.objects.get("ACE_Sonar")
    if sonar:
        for mod in getattr(sonar, "modifiers", []) or []:
            if getattr(mod, "type", None) == 'NODES' and getattr(mod, "node_group", None):
                tree = mod.node_group
                chosen = _consider_tree(tree)
                if chosen is not None:
                    return chosen
    if fallback is None and scene is not None:
        for scene_obj in getattr(scene, "objects", []) or []:
            tree, tree_fallback = _tree_from_object_materials(scene_obj)
            if tree is not None:
                return tree
            if fallback is None and tree_fallback is not None:
                fallback = tree_fallback
    return fallback


def _describe_node(cat: str, typ: str) -> str:
    try:
        return ACE_NODES[cat]["nodes"][typ]["description"]
    except Exception as exc:
        utils.note_swallowed("helpers._describe_node", exc)
        return ""


def _source_list_icon(source_prop) -> str:
    if source_prop is None:
        return "SPEAKER"
    if str(getattr(source_prop, "ace_source_error", "") or ""):
        return "ERROR"
    source_type = str(getattr(source_prop, "source_type", "") or "").strip().upper()
    if source_type == "FILE":
        return "FILE"
    return "SPEAKER"


def _frontend_object_name(frontend_ref) -> str:
    owner_keys = tuple(getattr(frontend_ref, "owner_keys", ()) or ())
    if not owner_keys:
        return "Unknown Object"
    owner_key = str(owner_keys[0] or "").strip()
    if not owner_key:
        return "Unknown Object"
    object_name = owner_key.rsplit(":", 1)[0].strip()
    return object_name or owner_key


def _frontend_instance_type(frontend_kind: str) -> str:
    kind = str(frontend_kind or "").strip().upper()
    if kind == "GEOMETRY_NODES":
        return "Geometry"
    if kind == "SHADER_NODES":
        return "Shader"
    return "Unknown"


def _label_icon_supported(icon_name: str) -> bool:
    """Return True when ``UILayout.label`` supports *icon_name*."""
    global _ui_label_icon_ids
    icon_name = str(icon_name or "").strip()
    if not icon_name:
        return False
    if _ui_label_icon_ids is None:
        try:
            label_fn = bpy.types.UILayout.bl_rna.functions.get("label")
            icon_param = label_fn.parameters.get("icon") if label_fn else None
            enum_items = getattr(icon_param, "enum_items", None) if icon_param else None
            _ui_label_icon_ids = {
                str(getattr(item, "identifier", "") or "")
                for item in (enum_items or [])
                if str(getattr(item, "identifier", "") or "")
            }
        except Exception as exc:
            utils.note_swallowed("helpers._label_icon_supported", exc)
            _ui_label_icon_ids = set()
    return icon_name in (_ui_label_icon_ids or set())


def _frontend_kind_icon(frontend_kind: str) -> str:
    kind = str(frontend_kind or "").strip().upper()
    if kind == "GEOMETRY_NODES":
        if _label_icon_supported("GEOMETRY_NODES"):
            return "GEOMETRY_NODES"
        if _label_icon_supported("MOD_NODES"):
            return "MOD_NODES"
        return "NODETREE"
    return "SHADING_RENDERED"


def _source_display_sample_rate(source) -> int:
    """Return the sample rate to display in the UI for *source*."""
    source_type = str(getattr(source, "source_type", "") or "").strip().upper()
    if source_type == "FILE":
        try:
            sr = int(getattr(source, "native_sample_rate", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("helpers._source_display_sample_rate", exc)
            sr = 0
        if sr > 0:
            return sr
    try:
        return int(getattr(source, "sample_rate", 0) or 0)
    except Exception as exc:
        utils.note_swallowed("helpers._source_display_sample_rate#2", exc)
        return 0


def _node_type_display(cat: str, typ: str) -> str:
    try:
        cat_label = str(ACE_NODES[cat]["meta"]["label"] or "").strip()
    except Exception as exc:
        utils.note_swallowed("helpers._node_type_display", exc)
        cat_label = str(cat or "").replace("_", " ").strip()
    return cat_label


def _find_tree_node_by_ace_id(tree, wanted_id: str):
    """The node of *tree* whose (computed) id is *wanted_id*, and its parsed kind."""
    parsed = ACEIdManager.parse_node_id(str(wanted_id or ""))
    if not tree or not parsed:
        return None, None
    return find_node(tree, wanted_id), parsed


def _tree_pointer(tree) -> int:
    if tree is None:
        return 0
    try:
        return int(tree.as_pointer())
    except Exception as exc:
        utils.note_swallowed("helpers._tree_pointer", exc)
        return id(tree)
