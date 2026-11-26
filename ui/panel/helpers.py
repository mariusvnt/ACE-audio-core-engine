"""N-panel helpers: active tree lookup, source rows and display names."""

import bpy  # type: ignore
from ...core.id_management import ACEIdManager
from ...core.engine.engine import ACEAudioEngine
from ...nodes.geometry.reader import ACEDummyTreeReader
from ...vars import ACE_NODES
from ... import deps, utils


_ui_label_icon_ids: set[str] | None = None


def _tree_has_ace_nodes(tree) -> bool:
    if tree is None:
        return False
    for node in getattr(tree, "nodes", []) or []:
        if node.get("ace_id"):
            return True
    return False


def _deal_breakers():
    issues = []
    try:
        if tuple(getattr(bpy.app, "version", (0, 0, 0))) < (5, 2, 0):
            issues.append("Blender 5.2+ is required.")
    except Exception as exc:
        utils.note_swallowed("helpers._deal_breakers", exc)
        issues.append("Unable to read Blender version.")
    if not deps.available("audioflux"):
        issues.append("AudioFlux, which ships with the extension, failed to load; the add-on preferences show why.")
    return issues


def _ui_is_blocked(context):
    return len(_deal_breakers()) > 0


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


def _compiled_abstract_tree(scene):
    if scene is None:
        return None
    engine = ACEAudioEngine.get_instance(scene)
    if engine is None:
        return None
    return getattr(engine, "abstract_tree_ref", None)


def _scene_audio_sources_by_id(scene) -> dict[str, object]:
    ace = getattr(scene, "audiocore", None) if scene is not None else None
    sources = getattr(ace, "audio_sources", []) if ace is not None else []
    return {
        str(getattr(source, "stable_id", "") or ""): source
        for source in (sources or [])
        if str(getattr(source, "stable_id", "") or "")
    }


def _source_display_name(source_prop, source_node) -> str:
    if source_prop is not None:
        name = str(getattr(source_prop, "name", "") or "").strip()
        if name:
            return name
    if source_node is not None:
        name = str(getattr(source_node, "name", "") or "").strip()
        if name:
            return name
    return "Unnamed Source"


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


def _compiled_source_instance_rows(scene):
    """Build the read-only source-to-frontend ledger from compiled IR."""
    abstract_tree = _compiled_abstract_tree(scene)
    if abstract_tree is None:
        return None

    nodes = dict(getattr(abstract_tree, "nodes", {}) or {})
    frontends = dict(getattr(abstract_tree, "frontends", {}) or {})
    consumer_intents = list(getattr(abstract_tree, "consumer_intents", []) or [])
    source_props_by_id = _scene_audio_sources_by_id(scene)

    source_nodes_by_token: dict[str, object] = {}
    for node in nodes.values():
        if str(getattr(node, "category", "") or "") != "source_mngmt":
            continue
        if str(getattr(node, "type", "") or "") != "src":
            continue
        source_token = str(getattr(node, "source_binding_id", "") or "").strip()
        if not source_token:
            roots = sorted({str(v) for v in (getattr(node, "root_sources", set()) or set()) if str(v)})
            source_token = roots[0] if roots else ""
        if not source_token:
            continue
        if source_token not in source_nodes_by_token:
            source_nodes_by_token[source_token] = node

    if not source_nodes_by_token:
        return []

    frontend_uids_by_source: dict[str, set[str]] = {
        token: set() for token in source_nodes_by_token.keys()
    }
    for intent in consumer_intents:
        source_token = str(getattr(intent, "source_token", "") or "").strip()
        frontend_uid = str(getattr(intent, "frontend_uid", "") or "").strip()
        if not source_token or not frontend_uid:
            continue
        if source_token in frontend_uids_by_source:
            frontend_uids_by_source[source_token].add(frontend_uid)

    for node in nodes.values():
        frontend_uid = str(getattr(node, "frontend_uid", "") or "").strip()
        if not frontend_uid or str(getattr(node, "category", "") or "") == "source_mngmt":
            continue
        for source_token in sorted({str(v) for v in (getattr(node, "root_sources", set()) or set()) if str(v)}):
            if source_token in frontend_uids_by_source and not frontend_uids_by_source[source_token]:
                frontend_uids_by_source[source_token].add(frontend_uid)

    ordered_tokens = sorted(
        source_nodes_by_token.keys(),
        key=lambda token: (
            next(
                (
                    idx for idx, src in enumerate(
                        getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []
                    )
                    if str(getattr(src, "stable_id", "") or "") == token
                ),
                999999,
            ),
            _source_display_name(source_props_by_id.get(token), source_nodes_by_token.get(token)).lower(),
            token,
        ),
    )

    ledger_rows = []
    for source_token in ordered_tokens:
        source_node = source_nodes_by_token.get(source_token)
        source_prop = source_props_by_id.get(source_token)
        frontend_uids = sorted(
            frontend_uids_by_source.get(source_token, set()) or set(),
            key=lambda uid: (
                int(getattr(frontends.get(uid), "discovery_index", 0) or 0),
                str(getattr(frontends.get(uid), "tree_name", "") or "").lower(),
                str(uid),
            ),
        )
        ledger_rows.append(
            {
                "source_token": source_token,
                "source_node": source_node,
                "source_prop": source_prop,
                "display_name": _source_display_name(source_prop, source_node),
                "frontend_refs": [frontends[uid] for uid in frontend_uids if uid in frontends],
            }
        )
    return ledger_rows


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


def _find_tree_node_by_ace_id(tree, node_id: str):
    """Find a Blender GN node in *tree* matching *node_id* via its ace_id property."""
    parsed = ACEIdManager.parse_node_id(str(node_id or ""))
    if not tree or not parsed:
        return None, None

    for candidate in getattr(tree, "nodes", []):
        ace_id = ""
        try:
            ace_id = str(candidate.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("helpers._find_tree_node_by_ace_id", exc)
            ace_id = ""
        if ace_id == node_id:
            return candidate, parsed
    return None, parsed


def _build_ui_node_lookup_ids(tree, reader=None) -> dict[str, str]:
    """Map abstract tree node ids back to the real Blender node `ace_id` values."""
    if not tree:
        return {}

    if reader is None:
        try:
            reader = ACEDummyTreeReader(tree)
            reader.build()
        except Exception as exc:
            utils.note_swallowed("helpers._build_ui_node_lookup_ids", exc)
            return {}

    lookup_ids: dict[str, str] = {}
    node_id_by_ptr = getattr(reader, "_node_id_by_ptr", {}) or {}
    for candidate in getattr(tree, "nodes", []):
        try:
            node_ptr = int(candidate.as_pointer())
        except Exception as exc:
            utils.note_swallowed("helpers._build_ui_node_lookup_ids#2", exc)
            node_ptr = id(candidate)
        abstract_id = str(node_id_by_ptr.get(node_ptr, "") or "")
        if not abstract_id:
            continue
        try:
            blender_id = str(candidate.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("helpers._build_ui_node_lookup_ids#3", exc)
            blender_id = ""
        if blender_id and ACEIdManager.parse_node_id(blender_id):
            lookup_ids[abstract_id] = blender_id
    return lookup_ids


def _ui_node_lookup_id(node_obj, lookup_ids: dict[str, str]) -> str:
    abstract_id = str(getattr(node_obj, "id", "") or "")
    resolved_id = str((lookup_ids or {}).get(abstract_id, "") or "")
    return resolved_id or abstract_id


def _tree_pointer(tree) -> int:
    if tree is None:
        return 0
    try:
        return int(tree.as_pointer())
    except Exception as exc:
        utils.note_swallowed("helpers._tree_pointer", exc)
        return id(tree)
