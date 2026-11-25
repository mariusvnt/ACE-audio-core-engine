"""
N-panel UI layout for the AudioCore Engine Blender add-on.

Draws the ACE panels in the 3D View N-panel (category "AudioCore Engine"),
manages the periodic UI sync timer that rebuilds the active-source node list
from the GN tree, and exposes helpers for splitter ramp visualisation and
per-node operator settings.
"""

import bpy  # type: ignore
import textwrap
import time
import hashlib
try:
    from bpy.app.handlers import persistent  # type: ignore
except ImportError:  # pragma: no cover - Blender provides this at runtime
    def persistent(fn):
        return fn

from ..core.id_management import ACEIdManager
from ..core.audio_engine import ACEAudioEngine
from ..nodes.dummy_tree import ACEDummyTreeReader
from ..nodes.shader_tree import ACEShaderTreeReader
from ..vars import ACE_NODES
from .. import utils


_UI_SYNC_MIN_INTERVAL = 0.25
_UI_SYNC_TIMER_INTERVAL_ACTIVE = 0.25
_UI_SYNC_TIMER_INTERVAL_IDLE = 1.0
_UI_DIRTY_REBUILD_MIN_INTERVAL = 0.6
_UI_METADATA_PROBE_INTERVAL = 5.0
_UI_LATENCY_UPDATE_INTERVAL_S = 1.0
_ui_sync_last_ts = 0.0
_ui_sync_last_key = ""
_ui_timer_last_quick_key = ""
_ui_timer_last_sync_ts = 0.0
_ui_timer_last_dirty_rebuild_ts = 0.0
_ui_source_probe_last: dict[str, float] = {}
_ui_latency_cache = {
    "scene_key": 0,
    "source_uid": "",
    "ts": 0.0,
    "metrics": None,
}
_ui_tree_cache = {
    "tree_ptr": 0,
    "topology_sig": "",
    "sig_ts": 0.0,
    "source_uid": "",
    "source_proxy": "",
    "nodes": (),
    "lookup_ids": {},
}
_ui_label_icon_ids: set[str] | None = None
_has_ace_nodes_in_trees: bool = False


# --- ACE node presence detection (for Active ACE Instances panel) ---

def _tree_has_ace_nodes(tree) -> bool:
    """Return True if *tree* contains at least one node with an ``ace_id`` custom property."""
    if tree is None:
        return False
    for node in getattr(tree, "nodes", []) or []:
        if node.get("ace_id"):
            return True
    return False


def _refresh_ace_nodes_in_trees_flag(scene) -> None:
    """Update the module-level ``_has_ace_nodes_in_trees`` flag.

    Scans all GN modifiers and shader material trees attached to objects in
    *scene* for the presence of any ACE node (identified by the ``ace_id``
    custom property).  Designed to be called from the UI sync timer so the
    expensive scene walk is not repeated every redraw.
    """
    global _has_ace_nodes_in_trees
    if scene is None:
        _has_ace_nodes_in_trees = False
        return
    for obj in getattr(scene, "objects", []) or []:
        for mod in getattr(obj, "modifiers", []) or []:
            if getattr(mod, "type", None) != "NODES":
                continue
            if _tree_has_ace_nodes(getattr(mod, "node_group", None)):
                _has_ace_nodes_in_trees = True
                return
        for slot in getattr(obj, "material_slots", []) or []:
            mat = getattr(slot, "material", None)
            tree = getattr(mat, "node_tree", None) if mat is not None else None
            if _tree_has_ace_nodes(tree):
                _has_ace_nodes_in_trees = True
                return
    _has_ace_nodes_in_trees = False


# --- Deal-breaker handling ---
def _deal_breakers():
    """Return a list of blocking configuration issue strings.

    These issues prevent any ACE panel from drawing meaningful UI.  Currently
    checks Blender version (>=5.0 required) and AudioFlux availability.

    Returns:
        A list of human-readable issue description strings.  Empty when the
        configuration is valid.
    """
    issues = []
    try:
        if tuple(getattr(bpy.app, "version", (0, 0, 0))) < (5, 0, 0):
            issues.append("Blender 5.0+ is required.")
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._deal_breakers", exc)
        issues.append("Unable to read Blender version.")
    if not utils.is_module_installed("audioflux"):
        issues.append("AudioFlux is not installed although it is one of the core elements of the engine; see add-on preferences to install it.")
    return issues


def _ui_is_blocked(context):
    """Return True when one or more deal-breaker issues prevent drawing the UI."""
    return len(_deal_breakers()) > 0


# --- Active UI Dummy Tree Helpers ---
def _iter_visible_node_editor_trees(context):
    """Yield unique supported trees from visible Node Editor areas.

    The 3D View N-panel runs outside a node editor context, so this helper
    inspects open windows/screens for visible ``NODE_EDITOR`` areas and yields
    their active ``edit_tree``/``node_tree`` values.
    """
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
                utils.note_swallowed("n_panel_ui._iter_visible_node_editor_trees", exc)
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
    """Return the most relevant active ACE authoring tree for the current context.

    Priority order:
    1. Active node-editor context tree (when currently in a Node Editor).
    2. Trees visible in open Node Editor areas (current window first).
    3. Broad scene/object fallback (unless ``editor_context_only=True``).

    Returns the first found ACE-node tree, or a non-ACE fallback if nothing
    else is available.

    Args:
        context: Blender context.
        editor_context_only: When True, only resolve from visible/editor
            context trees and skip object/material fallbacks. This is used by
            the N-panel active-node list so it tracks editor context.
    """
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
        with utils.logged_errors("n_panel_ui._find_active_node_tree"):
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
    """Return the human-readable description string for an ACE node type, or ''."""
    try:
        return ACE_NODES[cat]["nodes"][typ]["description"]
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._describe_node", exc)
        return ""


def _compiled_abstract_tree(scene):
    """Return the last compiled ACEAbstractTree for *scene*, or None."""
    if scene is None:
        return None
    engine = ACEAudioEngine.get_instance(scene)
    if engine is None:
        return None
    return getattr(engine, "abstract_tree_ref", None)


def _scene_audio_sources_by_id(scene) -> dict[str, object]:
    """Return a stable-id keyed lookup of scene audio source properties."""
    ace = getattr(scene, "audiocore", None) if scene is not None else None
    sources = getattr(ace, "audio_sources", []) if ace is not None else []
    return {
        str(getattr(source, "stable_id", "") or ""): source
        for source in (sources or [])
        if str(getattr(source, "stable_id", "") or "")
    }


def _source_display_name(source_prop, source_node) -> str:
    """Return the preferred display name for a source ledger row."""
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
    """Return the same source icon policy used by ACE_UL_source_list rows."""
    if source_prop is None:
        return "SPEAKER"
    proxy_name = str(getattr(source_prop, "proxy_object_name", "") or "")
    proxy = bpy.data.objects.get(proxy_name) if proxy_name else None
    if not proxy:
        return "ERROR"
    source_type = str(getattr(source_prop, "source_type", "") or "").strip().upper()
    if source_type == "FILE":
        return "FILE"
    return "SPEAKER"


def _frontend_object_name(frontend_ref) -> str:
    """Return the object name hosting *frontend_ref* (without modifier suffix)."""
    owner_keys = tuple(getattr(frontend_ref, "owner_keys", ()) or ())
    if not owner_keys:
        return "Unknown Object"
    owner_key = str(owner_keys[0] or "").strip()
    if not owner_key:
        return "Unknown Object"
    object_name = owner_key.rsplit(":", 1)[0].strip()
    return object_name or owner_key


def _frontend_instance_type(frontend_kind: str) -> str:
    """Return a compact instance-type label for Active ACE Instances rows."""
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
            utils.note_swallowed("n_panel_ui._label_icon_supported", exc)
            _ui_label_icon_ids = set()
    return icon_name in (_ui_label_icon_ids or set())


def _frontend_kind_icon(frontend_kind: str) -> str:
    """Return a robust icon id for frontend rows in N-panel ledgers."""
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
    """Return the sample rate to display in the UI for *source*.

    For FILE sources, prefers native_sample_rate (the rate before any working
    rate cap is applied) so the user sees the file's actual rate.  Falls back
    to the engine working sample_rate for live and unchecked sources.

    Args:
        source: An ACEAudioSource property-group instance.

    Returns:
        Sample rate in Hz, or 0 if not yet known.
    """
    source_type = str(getattr(source, "source_type", "") or "").strip().upper()
    if source_type == "FILE":
        try:
            sr = int(getattr(source, "native_sample_rate", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._source_display_sample_rate", exc)
            sr = 0
        if sr > 0:
            return sr
    try:
        return int(getattr(source, "sample_rate", 0) or 0)
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._source_display_sample_rate#2", exc)
        return 0


def _node_type_display(cat: str, typ: str) -> str:
    """Return the human-readable category label for an ACE node, falling back to the raw key."""
    try:
        cat_label = str(ACE_NODES[cat]["meta"]["label"] or "").strip()
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._node_type_display", exc)
        cat_label = str(cat or "").replace("_", " ").strip()
    return cat_label


_SPLITTER_LOW_HZ = 20.0
_SPLITTER_HIGH_HZ = 20000.0
_SPLITTER_RAMP_NODETREE_NAME = ".ACE_SplitterRampUI"
_SPLITTER_RAMP_NODE_NAME = "ACE_SplitterRampNode"
_splitter_ramp_last_key = ""
_splitter_ramp_last_sig = ""


def _splitter_hz_to_norm(hz: float, low_hz: float = _SPLITTER_LOW_HZ, high_hz: float = _SPLITTER_HIGH_HZ) -> float:
    """Map a frequency in Hz to [0, 1] on a logarithmic scale.

    Args:
        hz: Frequency value in Hz.
        low_hz: Lower bound of the display range in Hz.
        high_hz: Upper bound of the display range in Hz.

    Returns:
        Normalised position in [0, 1].
    """
    import math
    low = max(float(low_hz), 1e-3)
    high = max(float(high_hz), low + 1e-3)
    h = min(max(float(hz), low), high)
    return float((math.log(h) - math.log(low)) / (math.log(high) - math.log(low)))


def _splitter_norm_to_hz(t: float, low_hz: float = _SPLITTER_LOW_HZ, high_hz: float = _SPLITTER_HIGH_HZ) -> float:
    """Map a normalised [0, 1] position back to a frequency in Hz.

    Inverse of _splitter_hz_to_norm.

    Args:
        t: Normalised position in [0, 1].
        low_hz: Lower bound of the display range in Hz.
        high_hz: Upper bound of the display range in Hz.

    Returns:
        Frequency in Hz.
    """
    import math
    low = max(float(low_hz), 1e-3)
    high = max(float(high_hz), low + 1e-3)
    tc = min(max(float(t), 0.0), 1.0)
    return float(math.exp(math.log(low) + tc * (math.log(high) - math.log(low))))


def _splitter_display_bounds(scene) -> tuple[float, float]:
    """Return the (low_hz, high_hz) display range for the splitter ramp.

    The upper bound is clamped to just below the Nyquist frequency of the
    active source when its sample rate is known, so cut-point sliders never
    exceed the useful analysis range.
    """
    low_hz = float(_SPLITTER_LOW_HZ)
    high_hz = float(_SPLITTER_HIGH_HZ)
    ace = getattr(scene, "audiocore", None)
    if ace and getattr(ace, "audio_sources", None):
        idx = int(getattr(ace, "active_audio_source_index", 0) or 0)
        if 0 <= idx < len(ace.audio_sources):
            src = ace.audio_sources[idx]
            sr = float(getattr(src, "sample_rate", 0.0) or 0.0)
            if sr > 0.0:
                high_hz = min(high_hz, max(100.0, 0.49 * sr))
    if high_hz <= low_hz:
        high_hz = low_hz + 1e-3
    return low_hz, high_hz


def _splitter_cut_sockets(node) -> list:
    """Return the unlinked cut-frequency input sockets of a FrequencySplitter node.

    Skips the first input (Audio bundle) and any linked sockets since linked
    values are not user-editable cut points.

    Args:
        node: A Blender GeometryNodeGroup node.

    Returns:
        List of NodeSocket objects representing cut-frequency inputs.
    """
    sockets = []
    for i, sock in enumerate(getattr(node, "inputs", [])):
        if i == 0:
            continue
        if bool(getattr(sock, "is_linked", False)):
            continue
        try:
            _ = float(getattr(sock, "default_value", 0.0))
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._splitter_cut_sockets", exc)
            continue
        sockets.append(sock)
    return sockets


def _find_tree_node_by_ace_id(tree, node_id: str):
    """Find a Blender GN node in *tree* matching *node_id* via its ace_id property.

    Args:
        tree: A bpy.types.NodeTree (GeometryNodeTree).
        node_id: An ACE node ID string to search for.

    Returns:
        A (node, parsed) tuple where *node* is the matching bpy.types.Node and
        *parsed* is the result of ACEIdManager.parse_node_id().  Both elements
        are None when no match is found.
    """
    parsed = ACEIdManager.parse_node_id(str(node_id or ""))
    if not tree or not parsed:
        return None, None

    for candidate in getattr(tree, "nodes", []):
        ace_id = ""
        try:
            ace_id = str(candidate.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._find_tree_node_by_ace_id", exc)
            ace_id = ""
        if ace_id == node_id:
            return candidate, parsed
    return None, parsed


def _build_ui_node_lookup_ids(tree, reader=None) -> dict[str, str]:
    """
    Map abstract tree node ids back to the real Blender node `ace_id` values.
    The active graph list is sourced from abstract nodes, while Capture Influx
    needs the live GN node instance for editable properties.
    """
    if not tree:
        return {}

    if reader is None:
        try:
            reader = ACEDummyTreeReader(tree)
            reader.build()
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._build_ui_node_lookup_ids", exc)
            return {}

    lookup_ids: dict[str, str] = {}
    node_id_by_ptr = getattr(reader, "_node_id_by_ptr", {}) or {}
    for candidate in getattr(tree, "nodes", []):
        try:
            node_ptr = int(candidate.as_pointer())
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._build_ui_node_lookup_ids#2", exc)
            node_ptr = id(candidate)
        abstract_id = str(node_id_by_ptr.get(node_ptr, "") or "")
        if not abstract_id:
            continue
        try:
            blender_id = str(candidate.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._build_ui_node_lookup_ids#3", exc)
            blender_id = ""
        if blender_id and ACEIdManager.parse_node_id(blender_id):
            lookup_ids[abstract_id] = blender_id
    return lookup_ids


def _ui_node_lookup_id(node_obj, lookup_ids: dict[str, str]) -> str:
    abstract_id = str(getattr(node_obj, "id", "") or "")
    resolved_id = str((lookup_ids or {}).get(abstract_id, "") or "")
    return resolved_id or abstract_id


def _resolve_ui_list_selected_node(context):
    """
    Resolve the node selected in the active graph list shown in the N-panel.
    In the 3D View N-panel this should take precedence over a stale GN editor
    active node so Capture Influx follows the user's list selection.
    """
    from ..utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return None, None
    idx = int(getattr(active_source, "active_ui_node_index", -1))
    if idx < 0 or idx >= len(getattr(active_source, "ui_nodes", [])):
        return None, None

    ui_item = active_source.ui_nodes[idx]
    node_id = str(getattr(ui_item, "node_id", "") or "")
    if not node_id:
        return None, None

    tree = _find_active_node_tree(context, editor_context_only=True)
    node, parsed = _find_tree_node_by_ace_id(tree, node_id)
    if node and parsed:
        return node, parsed

    lookup_ids = {}
    tree_ptr = _tree_pointer(tree)
    if tree and int(_ui_tree_cache.get("tree_ptr", 0) or 0) == int(tree_ptr):
        lookup_ids = dict(_ui_tree_cache.get("lookup_ids", {}) or {})
    if not lookup_ids and tree:
        lookup_ids = _build_ui_node_lookup_ids(tree)
    mapped_id = str(lookup_ids.get(node_id, "") or "")
    if mapped_id:
        node, parsed = _find_tree_node_by_ace_id(tree, mapped_id)
        if node and parsed:
            return node, parsed

    # If Blender is still holding the same ACE node as active in another area,
    # allow that direct reference to satisfy the selection without overriding it
    # with some unrelated active node.
    active_node = getattr(context, "active_node", None)
    active_id = str(active_node.get("ace_id", "") or "") if active_node and hasattr(active_node, "get") else ""
    if active_node and active_id == node_id:
        return active_node, parsed or ACEIdManager.parse_node_id(node_id)
    return None, parsed


def _resolve_capture_node(context):
    """
    Prefer the active graph list selection in the N-panel. Fallback to the GN
    editor active ACE node when there is no valid list selection.
    """
    node, parsed = _resolve_ui_list_selected_node(context)
    if node and parsed:
        return node, parsed

    node = getattr(context, "active_node", None)
    ace_id = str(node.get("ace_id", "") or "") if node and hasattr(node, "get") else ""
    if node and ace_id:
        parsed = ACEIdManager.parse_node_id(ace_id)
        if parsed:
            return node, parsed
    return None, None


def _selected_ui_node_item(context):
    from ..utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return None
    idx = int(getattr(active_source, "active_ui_node_index", -1))
    if idx < 0 or idx >= len(getattr(active_source, "ui_nodes", [])):
        return None
    return active_source.ui_nodes[idx]


def _parsed_node_has_operator_settings(parsed) -> bool:
    if not parsed:
        return False
    key = (str(parsed.get("category", "") or ""), str(parsed.get("type", "") or ""))
    return key in {
        ("delta", "frequency_splitter"),
        ("utilities", "audio_math"),
        ("utilities", "preview_sink"),
        ("utilities", "util_rolling_avg"),
    }


def _selected_node_has_operator_settings(context) -> bool:
    node, parsed = _resolve_capture_node(context)
    return bool(node and _parsed_node_has_operator_settings(parsed))


def _selected_node_description(context) -> str:
    item = _selected_ui_node_item(context)
    if item is None:
        return ""
    return str(getattr(item, "description", "") or "").strip()


def _draw_selected_node_operator_settings(layout, context) -> bool:
    node, parsed = _resolve_capture_node(context)
    if not node or not _parsed_node_has_operator_settings(parsed):
        return False

    drew_settings = False

    if parsed.get("category") == "delta" and parsed.get("type") == "frequency_splitter":
        drew_settings = True
        box = layout.box()
        box.label(text="Frequency Splitter", icon="MOD_WAVE")
        bands = int(node.get("ace_band_count", max(2, len(getattr(node, "outputs", []))))) if node else 2
        low_hz, high_hz = _splitter_display_bounds(context.scene)
        row = box.row(align=True)
        op = row.operator("ace.update_splitter_bands", text="", icon="REMOVE")
        op.direction = 'DOWN'
        row.label(text=f"{bands} Bands")
        op = row.operator("ace.update_splitter_bands", text="", icon="ADD")
        op.direction = 'UP'

        split_sockets = []
        split_values = []
        for i, sock in enumerate(getattr(node, "inputs", [])):
            if i == 0:
                continue
            try:
                val = float(getattr(sock, "default_value", 0.0))
            except Exception as exc:
                utils.note_swallowed("n_panel_ui._draw_selected_node_operator_settings", exc)
                continue
            split_sockets.append(sock)
            split_values.append(val)
            with utils.logged_errors("n_panel_ui._draw_selected_node_operator_settings#2"):
                sock.min_value = 0.0
                sock.max_value = 20000.0
                sock.soft_min = 20.0
                sock.soft_max = 20000.0

        if split_values:
            bar = _splitter_ramp_ascii(sorted(split_values), low_hz=low_hz, high_hz=high_hz)
            box.label(text=f"{int(low_hz)}Hz {bar} {int(high_hz)}Hz")
            ramp_owner = _sync_splitter_ramp_texture_from_node(node, context.scene)
            if ramp_owner is not None:
                box.template_color_ramp(ramp_owner, "color_ramp", expand=True)
                ramp_ops = box.row(align=True)
                ramp_ops.operator("ace.sync_splitter_ramp", text="Sync From Node", icon="FILE_REFRESH")
                ramp_ops.operator("ace.apply_splitter_ramp", text="Apply Ramp", icon="CHECKMARK")
            tools = box.row(align=True)
            tools.operator("ace.sort_splitter_freqs", text="Sort", icon="SORTALPHA")
            tools.operator("ace.even_splitter_freqs", text="Even Log", icon="SNAP_MIDPOINT")
            for idx, sock in enumerate(split_sockets, start=1):
                sub = box.row(align=True)
                try:
                    hz = float(getattr(sock, "default_value", 0.0))
                except Exception as exc:
                    utils.note_swallowed("n_panel_ui._draw_selected_node_operator_settings#3", exc)
                    hz = 0.0
                pos = _splitter_hz_to_norm(hz, low_hz=low_hz, high_hz=high_hz)
                sub.label(text=f"Cut {idx} ({pos:.2f})")
                sub.prop(sock, "default_value", text="Hz", slider=True)

    if parsed.get("category") == "utilities" and parsed.get("type") == "audio_math":
        drew_settings = True
        box = layout.box()
        box.label(text="Audio Math", icon="MODIFIER")
        inputs = int(node.get("ace_input_count", max(2, len(getattr(node, "inputs", []))))) if node else 2
        row = box.row(align=True)
        op = row.operator("ace.update_audio_math_inputs", text="", icon="REMOVE")
        op.direction = 'DOWN'
        row.label(text=f"{inputs} Inputs")
        op = row.operator("ace.update_audio_math_inputs", text="", icon="ADD")
        op.direction = 'UP'

        current_op = str(node.get("ace_op", "add")).lower()
        if current_op not in {"add", "subtract", "multiply"}:
            current_op = "add"
        row = box.row(align=True)
        for op_id, label in (("add", "Add"), ("subtract", "Subtract"), ("multiply", "Multiply")):
            op = row.operator("ace.set_audio_math_op", text=label, depress=(current_op == op_id))
            op.op = op_id

    if parsed.get("category") == "utilities" and parsed.get("type") == "preview_sink":
        drew_settings = True
        box = layout.box()
        box.label(text="Audio Preview", icon="PLAY_SOUND")
        enabled = bool(node.get("ace_preview_enabled", True))
        row = box.row(align=True)
        op = row.operator("ace.toggle_preview_sink", text="Enable", icon="PLAY", depress=enabled)
        op.preview_enabled = True
        op = row.operator("ace.toggle_preview_sink", text="Disable", icon="PAUSE", depress=(not enabled))
        op.preview_enabled = False
        box.label(text="Manual override for Group Output preview.", icon="INFO")
        box.label(text="Play timeline to hear this branch on the active source.", icon="INFO")
        if not ACEAudioEngine.preview_monitor_available():
            warn = box.row()
            warn.alert = True
            warn.label(text="Install SoundDevice in Preferences to enable preview audio.", icon="ERROR")
        else:
            err_count = ACEAudioEngine.preview_monitor_consecutive_errors()
            if err_count >= 5:
                warn = box.row()
                warn.alert = True
                warn.label(
                    text=f"Preview DSP error \u00d7{err_count} — check console for details.",
                    icon="ERROR",
                )

    if parsed.get("category") == "utilities" and parsed.get("type") == "util_rolling_avg":
        drew_settings = True
        box = layout.box()
        box.label(text="Rolling Average", icon="DRIVER")
        frames_sock = next(
            (sock for sock in getattr(node, "inputs", []) if getattr(sock, "name", "") == "Number of Frames"),
            None,
        )
        if frames_sock is not None:
            row = box.row(align=True)
            row.prop(frames_sock, "default_value", text="Number of Frames")
        else:
            box.label(text="Number of Frames input is unavailable.", icon="INFO")

    return drew_settings


def _draw_selected_node_description(layout, description: str):
    desc = str(description or "").strip()
    if not desc:
        return
    wrapped = textwrap.wrap(desc, width=42) or [desc]
    for chunk in wrapped:
        layout.label(text=chunk)


def _draw_bake_controls(layout, context):
    from ..utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return

    row = layout.row()
    row.enabled = getattr(active_source, "source_type", "") == 'FILE'
    row.operator("audiocore.bake_active_source", icon="FILE_TICK", text="Bake Active Source")
    if not row.enabled:
        return

    try:
        status = str(active_source.get("ace_bake_status", "") or "")
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._draw_bake_controls", exc)
        status = ""
    if not status:
        layout.label(text="Bake stream outputs over the scene frame range.", icon="INFO")
        return
    if status in {"failed", "error"}:
        err = ""
        try:
            err = str(active_source.get("ace_bake_error", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._draw_bake_controls#2", exc)
            err = ""
        warn = layout.row()
        warn.alert = True
        warn.label(text=f"Bake failed: {err or 'Unknown error'}", icon="ERROR")
    elif status in {"ready", "done"}:
        start = int(active_source.get("ace_bake_frame_start", 0))
        end = int(active_source.get("ace_bake_frame_end", -1))
        layout.label(text=f"Bake ready: frames {start} -> {end}", icon="CHECKMARK")
    elif status == "baking":
        progress = 0.0
        try:
            progress = float(active_source.get("ace_bake_progress", 0.0) or 0.0)
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._draw_bake_controls#3", exc)
            progress = 0.0
        pct = int(max(0.0, min(progress, 1.0)) * 100.0)
        layout.label(text=f"Baking in progress... {pct}%", icon="TIME")


def _splitter_node_signature(node, scene) -> str:
    """Build a cache-key string representing the current splitter cut-point configuration.

    Encodes the display bounds (Hz) and all unlinked cut-frequency socket values
    so the ramp texture is only redrawn when the configuration actually changes.
    """
    low_hz, high_hz = _splitter_display_bounds(scene)
    values = []
    for sock in _splitter_cut_sockets(node):
        with utils.logged_errors("n_panel_ui._splitter_node_signature"):
            values.append(float(getattr(sock, "default_value", 0.0)))
    values = sorted(values)
    rounded = ",".join(f"{v:.3f}" for v in values)
    return f"{low_hz:.3f}|{high_hz:.3f}|{rounded}"


def _ensure_splitter_ramp_texture():
    """
    Returns an RNA owner that exposes a `color_ramp` property for
    `layout.template_color_ramp(...)`.
    """
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    try:
        node_groups = bpy.data.node_groups
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._ensure_splitter_ramp_texture", exc)
        return None

    tree = node_groups.get(_SPLITTER_RAMP_NODETREE_NAME)
    if ACEAudioEngine._undo_recovery_active_for_scene(scene):
        if tree is None:
            return None
        ramp_node = tree.nodes.get(_SPLITTER_RAMP_NODE_NAME)
        if ramp_node is None or getattr(ramp_node, "bl_idname", "") != 'ShaderNodeValToRGB':
            return None
        return ramp_node
    if tree is None:
        try:
            tree = node_groups.new(_SPLITTER_RAMP_NODETREE_NAME, 'ShaderNodeTree')
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ensure_splitter_ramp_texture#2", exc)
            return None

    ramp_node = tree.nodes.get(_SPLITTER_RAMP_NODE_NAME)
    if ramp_node is None or getattr(ramp_node, "bl_idname", "") != 'ShaderNodeValToRGB':
        try:
            if ramp_node is not None:
                tree.nodes.remove(ramp_node)
            ramp_node = tree.nodes.new('ShaderNodeValToRGB')
            ramp_node.name = _SPLITTER_RAMP_NODE_NAME
            ramp_node.label = "ACE Splitter Ramp"
            ramp_node.location = (0, 0)
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ensure_splitter_ramp_texture#3", exc)
            return None

    return ramp_node


def _sync_splitter_ramp_texture_from_node(node, scene, force: bool = False):
    """Update the hidden splitter ramp ColorRamp from the current node cut-point values.

    Rate-limited by a signature cache: the ramp is only redrawn when the cut-point
    configuration has changed since the last call.  Pass ``force=True`` to skip the
    cache check (used when the node selection changes).

    Returns the ramp_owner node, or None if the ShaderNodeTree could not be created.
    """
    global _splitter_ramp_last_key, _splitter_ramp_last_sig
    ramp_owner = _ensure_splitter_ramp_texture()
    if ramp_owner is None:
        return None

    node_key = str(getattr(node, "as_pointer", lambda: 0)() or 0)
    sig = _splitter_node_signature(node, scene)
    if (not force) and node_key == _splitter_ramp_last_key and sig == _splitter_ramp_last_sig:
        return ramp_owner

    cut_values = []
    for sock in _splitter_cut_sockets(node):
        with utils.logged_errors("n_panel_ui._sync_splitter_ramp_texture_from_node"):
            cut_values.append(float(getattr(sock, "default_value", 0.0)))
    cut_values = sorted(cut_values)

    ramp = ramp_owner.color_ramp
    target_len = len(cut_values) + 2
    while len(ramp.elements) < target_len:
        ramp.elements.new(1.0)
    while len(ramp.elements) > target_len and len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])

    low_hz, high_hz = _splitter_display_bounds(scene)
    ramp.elements[0].position = 0.0
    ramp.elements[0].color = (0.05, 0.35, 0.95, 1.0)
    ramp.elements[-1].position = 1.0
    ramp.elements[-1].color = (0.95, 0.25, 0.05, 1.0)
    for i, hz in enumerate(cut_values, start=1):
        t = _splitter_hz_to_norm(hz, low_hz=low_hz, high_hz=high_hz)
        t = min(max(t, 0.0), 1.0)
        hue = float(i) / float(max(1, len(cut_values) + 1))
        ramp.elements[i].position = t
        ramp.elements[i].color = (0.15 + 0.65 * hue, 0.85 - 0.45 * hue, 0.30, 1.0)

    _splitter_ramp_last_key = node_key
    _splitter_ramp_last_sig = sig
    return ramp_owner


def _splitter_ramp_ascii(
    freqs: list[float],
    width: int = 28,
    low_hz: float = _SPLITTER_LOW_HZ,
    high_hz: float = _SPLITTER_HIGH_HZ,
) -> str:
    """Render splitter cut-point positions as a compact ASCII ruler string.

    Positions each frequency as a ``|`` mark on a ``-``-filled ruler of the
    given *width*, using a logarithmic frequency scale.  Used for compact
    text-based band visualisation in the N-panel header.

    Args:
        freqs: List of cut-point frequencies in Hz.
        width: Total character width of the output string.
        low_hz: Lower Hz bound for the logarithmic scale.
        high_hz: Upper Hz bound for the logarithmic scale.

    Returns:
        A string of length *width* consisting of ``-`` and ``|`` characters.
    """
    if width <= 2:
        return ""
    marks = ["-"] * width
    for f in freqs:
        t = _splitter_hz_to_norm(float(f), low_hz=low_hz, high_hz=high_hz)
        idx = max(0, min(width - 1, int(round(t * (width - 1)))))
        marks[idx] = "|"
    return "".join(marks)


def _tree_pointer(tree) -> int:
    """Return a stable integer identity for *tree* usable as a dict key."""
    if tree is None:
        return 0
    try:
        return int(tree.as_pointer())
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._tree_pointer", exc)
        return id(tree)


def _ui_tree_topology_signature(tree) -> str:
    """Compute a short content hash representing the current GN tree topology.

    Hashes node identities, ACE metadata custom properties, attribute-ID
    socket values, link wiring, and socket counts.  The resulting digest is
    used to detect changes that should trigger a UI sync rebuild.

    Args:
        tree: A Blender GeometryNodeTree or None.

    Returns:
        A hex digest string, or "" when *tree* is None.
    """
    if tree is None:
        return ""

    attr_socket_names = {
        "Attribute ID",
        "Frames per Beat Attribute ID",
        "Beat Phase Attribute ID",
    }
    node_tokens = []
    for node in getattr(tree, "nodes", []):
        ptr = _tree_pointer(node)
        node_type = str(getattr(node, "type", "") or "")
        node_name = str(getattr(node, "name", "") or "")
        node_tree_name = str(getattr(getattr(node, "node_tree", None), "name", "") or "")

        ace_id = ""
        source_id = ""
        op_value = ""
        preview_enabled = True
        try:
            ace_id = str(node.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ui_tree_topology_signature", exc)
            ace_id = ""
        if ace_id:
            try:
                source_id = str(node.get("ace_source_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("n_panel_ui._ui_tree_topology_signature#2", exc)
                source_id = ""
            try:
                op_value = str(node.get("ace_op", "") or "")
            except Exception as exc:
                utils.note_swallowed("n_panel_ui._ui_tree_topology_signature#3", exc)
                op_value = ""
            try:
                preview_enabled = bool(node.get("ace_preview_enabled", True))
            except Exception as exc:
                utils.note_swallowed("n_panel_ui._ui_tree_topology_signature#4", exc)
                preview_enabled = True

        proxy_name = ""
        attr_defaults = []
        for sock in getattr(node, "inputs", []):
            sock_name = str(getattr(sock, "name", "") or "")
            if sock_name == "Proxy Object" and not bool(getattr(sock, "is_linked", False)):
                try:
                    proxy_name = str(getattr(getattr(sock, "default_value", None), "name", "") or "")
                except Exception as exc:
                    utils.note_swallowed("n_panel_ui._ui_tree_topology_signature#5", exc)
                    proxy_name = ""
            if sock_name not in attr_socket_names or bool(getattr(sock, "is_linked", False)):
                continue
            raw_value = getattr(sock, "default_value", "")
            if hasattr(raw_value, "name"):
                raw_value = getattr(raw_value, "name", "")
            attr_defaults.append((sock_name, str(raw_value)))
        attr_defaults.sort()

        node_tokens.append(
            (
                ptr,
                ace_id,
                node_type,
                node_tree_name,
                node_name,
                source_id,
                proxy_name,
                op_value,
                int(preview_enabled),
                tuple(attr_defaults),
                len(getattr(node, "inputs", [])),
                len(getattr(node, "outputs", [])),
            )
        )
    node_tokens.sort()

    link_tokens = []
    for link in getattr(tree, "links", []):
        from_node = getattr(link, "from_node", None)
        to_node = getattr(link, "to_node", None)
        if from_node is None or to_node is None:
            continue

        from_socket = getattr(link, "from_socket", None)
        to_socket = getattr(link, "to_socket", None)
        from_sock_token = str(
            getattr(from_socket, "identifier", "") or getattr(from_socket, "name", "") or ""
        )
        to_sock_token = str(
            getattr(to_socket, "identifier", "") or getattr(to_socket, "name", "") or ""
        )
        try:
            from_sock_idx = int(getattr(from_socket, "index", -1))
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ui_tree_topology_signature#6", exc)
            from_sock_idx = -1
        try:
            to_sock_idx = int(getattr(to_socket, "index", -1))
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ui_tree_topology_signature#7", exc)
            to_sock_idx = -1

        link_tokens.append(
            (
                _tree_pointer(from_node),
                _tree_pointer(to_node),
                from_sock_idx,
                to_sock_idx,
                from_sock_token,
                to_sock_token,
            )
        )
    link_tokens.sort()

    payload = repr((tuple(node_tokens), tuple(link_tokens))).encode("utf-8", "ignore")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _ui_nodes_match(active_source, node_ids: list[str]) -> bool:
    """Return True when the ui_nodes collection exactly matches *node_ids*.

    Used to decide whether to do a cheaper in-place update or a full clear
    and rebuild of the ui_nodes collection property.

    Args:
        active_source: An ACEAudioSource property-group instance.
        node_ids: The ordered list of node ID strings to compare against.

    Returns:
        True only if counts and all IDs match in order.
    """
    try:
        ui_nodes = getattr(active_source, "ui_nodes", None)
        if ui_nodes is None or len(ui_nodes) != len(node_ids):
            return False
        for idx, node_id in enumerate(node_ids):
            if str(getattr(ui_nodes[idx], "node_id", "") or "") != str(node_id or ""):
                return False
        return True
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._ui_nodes_match", exc)
        return False


def _sync_ui_dummy_tree(context, force: bool = False, allow_dirty_rebuild: bool = True):
    global _ui_sync_last_ts, _ui_sync_last_key, _ui_tree_cache

    ace = getattr(context.scene, "audiocore", None)
    if not ace:
        utils.log("UI sync skipped: no audiocore on scene.", "DEBUG")
        return
    _, active_source = utils._get_active_source(context)
    if not active_source:
        utils.log("UI sync skipped: no active source.", "DEBUG")
        return
    with utils.logged_errors("n_panel_ui._sync_ui_dummy_tree"):
        ACEIdManager.ensure_source_uid(active_source)
    # Abort if collection is not writable in the current context (e.g., during draw in some areas)
    if not hasattr(active_source, "ui_nodes"):
        utils.log("UI sync skipped: active_source has no ui_nodes property.", "DEBUG")
        return

    tree = _find_active_node_tree(context, editor_context_only=True)
    source_uid = getattr(active_source, "stable_id", "") or ""
    source_proxy = getattr(active_source, "proxy_object_name", "") or ""
    tree_ptr = _tree_pointer(tree)
    now = time.monotonic()
    cache_scope_match = (
        bool(tree)
        and int(_ui_tree_cache.get("tree_ptr", 0) or 0) == int(tree_ptr)
        and str(_ui_tree_cache.get("source_uid", "") or "") == source_uid
        and str(_ui_tree_cache.get("source_proxy", "") or "") == source_proxy
    )

    tree_dirty_hint = bool(force)
    engine = ACEAudioEngine.get_instance(context.scene)
    engine_abstract = None
    engine_tree_ready = False
    if tree and engine:
        tree_ptr = 0
        try:
            tree_ptr = int(tree.as_pointer())
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._sync_ui_dummy_tree#2", exc)
            tree_ptr = 0
        pending_tree_ptr = int(getattr(getattr(engine, "frontends", None), "_pending_tree_ptr", 0) or 0)
        dirty_tree_ptrs = getattr(getattr(engine, "frontends", None), "_dirty_tree_ptrs", set()) or set()
        dirty_tree_match = bool((tree_ptr and pending_tree_ptr == tree_ptr) or (tree_ptr and tree_ptr in dirty_tree_ptrs))
        if allow_dirty_rebuild:
            tree_dirty_hint = tree_dirty_hint or dirty_tree_match
        if not tree_dirty_hint:
            compiled_ptr = int(getattr(getattr(engine, "compiler", None), "_compiled_tree_ptr", 0) or 0)
            # In Multiple-Trees-Support, _compiled_tree_ptr only tracks the
            # preferred (first) tree. Check all compiled frontend pointers so
            # any compiled tree gets engine_tree_ready = True and uses the
            # merged abstract tree directly instead of falling back to reader.build().
            compiled_frontend_ptrs = set(
                (getattr(getattr(engine, "compiler", None), "_compiled_frontend_ptrs", {}) or {}).values()
            )
            if compiled_ptr == int(tree_ptr) or int(tree_ptr) in compiled_frontend_ptrs:
                engine_abstract = getattr(engine, "abstract_tree_ref", None)
                engine_tree_ready = bool(engine_abstract)

    rebuild_topology = bool(
        tree
        and (
            force
            or (not cache_scope_match)
            or tree_dirty_hint
        )
    )
    topology_sig = ""
    if tree:
        if rebuild_topology:
            if engine_tree_ready and engine is not None:
                # Avoid Python's process-randomized hash() so UI topology cache
                # keys remain stable across sessions.
                engine_sig_seed = str(getattr(getattr(engine, "compiler", None), "_last_tree_signature", None)).encode("utf-8")
                engine_sig = hashlib.blake2b(engine_sig_seed, digest_size=8).hexdigest()
                topology_sig = f"eng:{engine_sig}"
            else:
                topology_sig = _ui_tree_topology_signature(tree)
        else:
            topology_sig = str(_ui_tree_cache.get("topology_sig", "") or "")
    else:
        rebuild_topology = True
        topology_sig = ""
    sync_key = (
        f"{source_uid}|{source_proxy}|"
        f"{getattr(tree, 'name', '') if tree else ''}|{tree_ptr}"
    )
    if not force and sync_key == _ui_sync_last_key and (now - _ui_sync_last_ts) < _UI_SYNC_MIN_INTERVAL:
        return

    nodes = []
    node_lookup_ids = {}
    if (not rebuild_topology) and cache_scope_match:
        nodes = list(_ui_tree_cache.get("nodes", ()) or ())
        node_lookup_ids = dict(_ui_tree_cache.get("lookup_ids", {}) or {})
    elif tree:
        if engine_tree_ready and engine_abstract is not None:
            abstract = engine_abstract
            node_lookup_ids = _build_ui_node_lookup_ids(tree)
        else:
            if str(getattr(tree, "bl_idname", "") or "") == "ShaderNodeTree":
                reader = ACEShaderTreeReader(tree)
            else:
                reader = ACEDummyTreeReader(tree)
            abstract = reader.build()
            node_lookup_ids = _build_ui_node_lookup_ids(tree, reader=reader)
        nodes = list(abstract.nodes.values())
        if nodes:
            bl_order: dict[str, int] = {}
            if tree:
                for _idx, _bl_node in enumerate(getattr(tree, "nodes", [])):
                    try:
                        _ace_id = str(_bl_node.get("ace_id", "") or "")
                    except Exception as exc:
                        utils.note_swallowed("n_panel_ui._sync_ui_dummy_tree#3", exc)
                        _ace_id = ""
                    if _ace_id:
                        bl_order[_ace_id] = _idx

            def _creation_sort_key(n, _bl_order=bl_order, _lu=node_lookup_ids):
                resolved = str(_lu.get(n.id, "") or n.id or "")
                return _bl_order.get(resolved, _bl_order.get(n.id, 999999))

            nodes.sort(key=_creation_sort_key)
            nodes = [
                node
                for node in nodes
                if str(getattr(node, "category", "") or "") != "source_mngmt"
            ]
        _ui_tree_cache = {
            "tree_ptr": int(tree_ptr),
            "topology_sig": topology_sig,
            "sig_ts": float(now),
            "source_uid": source_uid,
            "source_proxy": source_proxy,
            "nodes": tuple(nodes),
            "lookup_ids": dict(node_lookup_ids),
        }
    else:
        _ui_tree_cache = {
            "tree_ptr": 0,
            "topology_sig": "",
            "sig_ts": float(now),
            "source_uid": "",
            "source_proxy": "",
            "nodes": (),
            "lookup_ids": {},
        }

    nodes = [
        node
        for node in nodes
        if str(getattr(node, "category", "") or "") != "source_mngmt"
    ]
    if not nodes:
        if not tree:
            utils.log("UI sync skipped: no active ACE node tree found.", "DEBUG")
            try:
                active_source.ui_nodes.clear()
            except Exception as e:
                utils.log(f"UI sync could not clear ui_nodes: {e}", "DEBUG")
            _ui_sync_last_key = sync_key
            _ui_sync_last_ts = now
            return

    utils.log(f"UI sync found {len(nodes)} nodes for active source.", "DEBUG")

    proxy = bpy.data.objects.get(getattr(active_source, "proxy_object_name", "")) if hasattr(active_source, "proxy_object_name") else None

    def _resolve_attr_name(node_obj):
        attr_id = None
        attr_custom = False
        if isinstance(getattr(node_obj, "parameters", None), dict):
            attr_id = node_obj.parameters.get("attr_id_out_0")
            attr_custom = bool(node_obj.parameters.get("attr_id_custom_out_0", False))
        if attr_custom and isinstance(attr_id, str) and attr_id.strip():
            name = attr_id.strip()
            if not name.startswith(ACEIdManager.PREFIX):
                name = ACEIdManager.PREFIX + ACEIdManager._clean_string(name)
            return name[:63]

        if not proxy:
            return None
        if not getattr(node_obj, "outputs", None):
            return None
        out = node_obj.outputs[0]
        return ACEIdManager.generate_compact_attr_name(
            proxy.name,
            node_obj.id or node_obj.type,
            out.name or out.id or "out0",
            0,
        )[:63]

    def _read_attr_value(mesh, name):
        import math
        attr = mesh.attributes.get(name) if mesh else None
        if not attr or not attr.data:
            return None
        try:
            if attr.data_type == 'FLOAT':
                return float(attr.data[0].value)
            if attr.data_type == 'FLOAT_VECTOR':
                vec = attr.data[0].vector
                return float(math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2))
            if attr.data_type == 'FLOAT_COLOR':
                col = attr.data[0].color
                return float(sum(col) / len(col))
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._sync_ui_dummy_tree._read_attr_value", exc)
            return None
        return None

    def _write_ui_item(item, node):
        item.name = node.name or node.id
        item.node_id = _ui_node_lookup_id(node, node_lookup_ids)
        item.node_type = _node_type_display(node.category, node.type)
        value = None
        if proxy and getattr(proxy, "data", None):
            attr_name = _resolve_attr_name(node)
            if attr_name:
                value = _read_attr_value(proxy.data, attr_name)
        item.current_value = float(value) if value is not None else 0.0
        item.description = _describe_node(node.category, node.type)

    node_ids = [_ui_node_lookup_id(node, node_lookup_ids) for node in nodes]
    reuse_ui_rows = _ui_nodes_match(active_source, node_ids)
    if reuse_ui_rows:
        for idx, node in enumerate(nodes):
            try:
                item = active_source.ui_nodes[idx]
            except Exception as exc:
                utils.note_swallowed("n_panel_ui._sync_ui_dummy_tree#4", exc)
                continue
            _write_ui_item(item, node)
    else:
        try:
            active_source.ui_nodes.clear()
        except Exception as e:
            utils.log(f"UI sync could not clear ui_nodes before repopulating: {e}", "DEBUG")
            return
        for node in nodes:
            try:
                item = active_source.ui_nodes.add()
            except Exception:
                utils.log("UI sync could not add a ui_nodes item (write restricted).", "DEBUG")
                break
            _write_ui_item(item, node)

    with utils.logged_errors("n_panel_ui._sync_ui_dummy_tree#5"):
        if active_source.active_ui_node_index >= len(active_source.ui_nodes):
            active_source.active_ui_node_index = max(0, len(active_source.ui_nodes) - 1)
    _ui_sync_last_key = sync_key
    _ui_sync_last_ts = now


def _refresh_active_source_metadata(scene, now: float | None = None):
    """Trigger an async metadata probe for the active file source if needed.

    Probes are rate-limited to _UI_METADATA_PROBE_INTERVAL seconds per source
    to avoid blocking the main thread.  Clears the probe record once both
    duration and native_sample_rate are available.

    Args:
        scene: The active Blender scene.
        now: Current monotonic timestamp in seconds, or None to read it here.
    """
    global _ui_source_probe_last
    if scene is None:
        return
    ace = getattr(scene, "audiocore", None)
    sources = getattr(ace, "audio_sources", None) if ace else None
    if not sources:
        return
    idx = int(getattr(ace, "active_audio_source_index", 0) or 0)
    if idx < 0 or idx >= len(sources):
        return
    src = sources[idx]
    if getattr(src, "source_type", "") != 'FILE':
        return
    filepath = str(getattr(src, "filepath", "") or "")
    if not filepath:
        return

    source_key = str(getattr(src, "stable_id", "") or filepath)
    ts_now = float(now if now is not None else time.monotonic())
    last_probe = float(_ui_source_probe_last.get(source_key, 0.0) or 0.0)
    if (ts_now - last_probe) < _UI_METADATA_PROBE_INTERVAL:
        return

    dur = float(getattr(src, "duration", 0.0) or 0.0)
    native_rate = int(getattr(src, "native_sample_rate", 0) or 0)
    if dur > 0.0 and native_rate > 0:
        _ui_source_probe_last.pop(source_key, None)
        return

    _ui_source_probe_last[source_key] = ts_now
    with utils.logged_errors("n_panel_ui._refresh_active_source_metadata"):
        engine = ACEAudioEngine.ensure_instance(scene)
        engine.probes._probe_source_metadata(src)


def _ui_quick_state_key(context, scene, tree) -> str:
    """Build a cheap string key capturing the most volatile UI state.

    Used by the sync timer to detect whether a full panel redraw/sync is
    needed without running the expensive tree-topology hash every tick.  The
    key encodes the active tree pointer, active source index, and source UID.
    """
    ace = getattr(scene, "audiocore", None)
    src_uid = ""
    src_idx = -1
    if ace is not None:
        try:
            src_idx = utils.as_int(getattr(ace, "active_audio_source_index", None), -1)
        except Exception as exc:
            utils.note_swallowed("n_panel_ui._ui_quick_state_key", exc)
            src_idx = -1
        sources = getattr(ace, "audio_sources", None) or []
        if 0 <= src_idx < len(sources):
            src_uid = str(getattr(sources[src_idx], "stable_id", "") or "")

    return f"{_tree_pointer(tree)}|{src_idx}|{src_uid}"


def _ui_has_pending_engine_changes(scene, tree) -> bool:
    """Return True when the engine has a pending recompile or dirty tree.

    Args:
        scene: The active Blender scene.
        tree: The currently viewed GeometryNodeTree, or None.

    Returns:
        True if the engine has a pending rebuild for *tree* or if *tree* is in
        the engine's dirty tree-pointer set.
    """
    engine = ACEAudioEngine.get_instance(scene)
    if not engine:
        return False

    pending_tree_ptr = int(getattr(getattr(engine, "frontends", None), "_pending_tree_ptr", 0) or 0)
    dirty_tree_ptrs = getattr(getattr(engine, "frontends", None), "_dirty_tree_ptrs", set()) or set()

    if pending_tree_ptr:
        if tree is None:
            return True
        with utils.logged_errors("n_panel_ui._ui_has_pending_engine_changes"):
            if int(tree.as_pointer()) == pending_tree_ptr:
                return True
    if not dirty_tree_ptrs:
        return False
    if tree is None:
        return True
    try:
        return int(tree.as_pointer()) in dirty_tree_ptrs
    except Exception as exc:
        utils.note_swallowed("n_panel_ui._ui_has_pending_engine_changes#2", exc)
        return False


def _engine_latency_metrics(context) -> dict[str, float | bool]:
    """Collect and cache per-stage latency metrics for the active source.

    Reads perf and diagnostics data from the engine and aggregates them into
    named sub-totals (input, DSP, preview, Blender geometry, render).  Results
    are cached for _UI_LATENCY_UPDATE_INTERVAL_S seconds to keep draw() calls
    cheap.

    Args:
        context: The current Blender context.

    Returns:
        A dict mapping metric names to float values or bool flags.  All keys
        are always present; values default to 0.0 / False when not available.
    """
    global _ui_latency_cache
    scene = getattr(context, "scene", None)
    engine = ACEAudioEngine.get_instance(scene)
    scene_key = int(ACEAudioEngine._scene_key(scene))
    now = time.monotonic()
    _, active_source = utils._get_active_source(context)
    source_uid = str(getattr(active_source, "stable_id", "") or getattr(active_source, "name", "") or "")
    cached_metrics = _ui_latency_cache.get("metrics")
    cached_scene_key = int(_ui_latency_cache.get("scene_key", 0) or 0)
    cached_source_uid = str(_ui_latency_cache.get("source_uid", "") or "")
    cached_ts = float(_ui_latency_cache.get("ts", 0.0) or 0.0)
    if (
        isinstance(cached_metrics, dict)
        and cached_scene_key == scene_key
        and cached_source_uid == source_uid
        and (now - cached_ts) < _UI_LATENCY_UPDATE_INTERVAL_S
    ):
        return dict(cached_metrics)
    metrics: dict[str, float | bool] = {
        "source_ms": 0.0,
        "dsp_ms": 0.0,
        "preview_ms": 0.0,
        "preview_branch_ms": 0.0,
        "preview_output_buffer_ms": 0.0,
        "preview_device_ms": 0.0,
        "preview_prefetch_ms": 0.0,
        "preview_active": False,
        "proxy_write_ms": 0.0,
        "file_buffer_ms": 0.0,
        "geometry_ms": 0.0,
        "geometry_effective_ms": 0.0,
        "render_ms": 0.0,
        "live_target_ms": 0.0,
        "input_total_ms": 0.0,
        "engine_total_ms": 0.0,
        "preview_total_ms": 0.0,
        "blender_total_ms": 0.0,
        "total_latency_ms": 0.0,
        "has_data": False,
    }
    if active_source is not None and getattr(active_source, "source_type", "") == 'LIVE':
        metrics["live_target_ms"] = float(int(getattr(active_source, "live_latency_ms", 0) or 0))
    if engine is None:
        _ui_latency_cache = {
            "scene_key": scene_key,
            "source_uid": source_uid,
            "ts": now,
            "metrics": dict(metrics),
        }
        return metrics

    perf = engine.get_perf_current()
    diag = engine.get_runtime_diagnostics_current()
    metrics["source_ms"] = float(perf.get("source_read", 0.0) or 0.0)
    metrics["dsp_ms"] = float(perf.get("dsp", 0.0) or 0.0)
    metrics["preview_ms"] = float(perf.get("preview", 0.0) or 0.0)
    metrics["preview_branch_ms"] = float(diag.get("preview_branch_latency_ms", 0.0) or 0.0)
    metrics["preview_output_buffer_ms"] = float(diag.get("preview_output_buffer_ms", 0.0) or 0.0)
    metrics["preview_device_ms"] = float(diag.get("preview_callback_ms", 0.0) or 0.0)
    metrics["proxy_write_ms"] = float(perf.get("proxy_write", 0.0) or 0.0)
    metrics["file_buffer_ms"] = float(diag.get("source_prefetch_ms", 0.0) or 0.0)
    metrics["preview_active"] = bool(
        float(metrics["preview_ms"]) > 0.0
        or float(metrics["preview_branch_ms"]) > 0.0
        or float(metrics["preview_output_buffer_ms"]) > 0.0
        or float(metrics["preview_device_ms"]) > 0.0
    )
    metrics["preview_prefetch_ms"] = (
        float(metrics["file_buffer_ms"])
        if (bool(metrics["preview_active"]) and float(metrics["preview_output_buffer_ms"]) > 0.0)
        else 0.0
    )
    metrics["geometry_ms"] = float(diag.get("blender_geometry_eval_ms", 0.0) or 0.0)
    metrics["geometry_effective_ms"] = (
        float(metrics["geometry_ms"])
        if float(metrics["geometry_ms"]) > 0.0
        else float(metrics["proxy_write_ms"])
    )
    metrics["render_ms"] = float(diag.get("blender_render_ms", 0.0) or 0.0)
    measured_total_ms = (
        float(metrics["source_ms"])
        + float(metrics["dsp_ms"])
        + float(metrics["preview_ms"])
        + float(metrics["preview_branch_ms"])
        + float(metrics["preview_output_buffer_ms"])
        + float(metrics["preview_device_ms"])
        + float(metrics["file_buffer_ms"])
        + float(metrics["proxy_write_ms"])
        + float(metrics["geometry_ms"])
        + float(metrics["render_ms"])
    )
    metrics["has_data"] = bool(measured_total_ms > 0.0)
    if not bool(metrics["has_data"]):
        return metrics

    metrics["input_total_ms"] = (
        float(metrics["live_target_ms"])
        + float(metrics["source_ms"])
        + float(metrics["file_buffer_ms"])
        - float(metrics["preview_prefetch_ms"])
    )
    metrics["engine_total_ms"] = (
        float(metrics["dsp_ms"])
    )
    metrics["preview_total_ms"] = (
        float(metrics["preview_branch_ms"])
        + float(metrics["preview_ms"])
        + float(metrics["preview_output_buffer_ms"])
        + float(metrics["preview_prefetch_ms"])
    )
    metrics["blender_total_ms"] = (
        float(metrics["proxy_write_ms"])
        + float(metrics["geometry_ms"])
        + float(metrics["render_ms"])
    )
    if float(metrics["geometry_ms"]) <= 0.0:
        metrics["blender_total_ms"] = (
            float(metrics["geometry_effective_ms"])
            + float(metrics["render_ms"])
        )
    metrics["total_latency_ms"] = (
        float(metrics["input_total_ms"])
        + float(metrics["engine_total_ms"])
        + float(metrics["blender_total_ms"])
    )
    _ui_latency_cache = {
        "scene_key": scene_key,
        "source_uid": source_uid,
        "ts": now,
        "metrics": dict(metrics),
    }
    return metrics


def _ui_sync_timer():
    """Blender persistent timer callback: drive background UI sync.

    Runs at ``_UI_SYNC_TIMER_INTERVAL_ACTIVE`` during playback or when engine
    changes are pending, and at ``_UI_SYNC_TIMER_INTERVAL_IDLE`` otherwise.
    Each tick optionally triggers ``_sync_ui_dummy_tree()`` when the quick
    state key changes or the idle interval has elapsed.

    Returns:
        Float interval in seconds for the next timer invocation.
    """
    global _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_timer_last_dirty_rebuild_ts
    with utils.logged_errors("n_panel_ui._ui_sync_timer"):
        context = getattr(bpy, "context", None)
        scene = getattr(context, "scene", None) if context else None
        if not scene:
            return _UI_SYNC_TIMER_INTERVAL_IDLE
        if _ui_is_blocked(context):
            return _UI_SYNC_TIMER_INTERVAL_IDLE

        _refresh_ace_nodes_in_trees_flag(scene)
        tree = _find_active_node_tree(context, editor_context_only=True)
        quick_key = _ui_quick_state_key(context, scene, tree)
        now = time.monotonic()
        engine_dirty = _ui_has_pending_engine_changes(scene, tree)
        is_playing = ACEAudioEngine._is_animation_playing()
        target_interval = (
            _UI_SYNC_TIMER_INTERVAL_ACTIVE
            if (is_playing or engine_dirty)
            else _UI_SYNC_TIMER_INTERVAL_IDLE
        )
        key_changed = quick_key != _ui_timer_last_quick_key
        dirty_rebuild_due = bool(
            engine_dirty
            and ((now - _ui_timer_last_dirty_rebuild_ts) >= _UI_DIRTY_REBUILD_MIN_INTERVAL)
        )
        force_sync = bool(key_changed or dirty_rebuild_due)
        allow_dirty_rebuild = bool(force_sync)

        _refresh_active_source_metadata(scene, now=now)
        with utils.logged_errors("n_panel_ui._ui_sync_timer#2"):
            engine = ACEAudioEngine.ensure_instance(scene)
            ace = getattr(scene, "audiocore", None)
            sources = list(getattr(ace, "audio_sources", []) or []) if ace else []
            engine.flush_source_tempo_detection_updates(sources)

        if key_changed or engine_dirty or ((now - _ui_timer_last_sync_ts) >= target_interval):
            t0 = time.perf_counter()
            _sync_ui_dummy_tree(
                context,
                force=force_sync,
                allow_dirty_rebuild=allow_dirty_rebuild,
            )
            engine = ACEAudioEngine.get_instance(scene)
            if engine is not None:
                with utils.logged_errors("n_panel_ui._ui_sync_timer#3"):
                    engine.diagnostics._perf_add("ui_sync", time.perf_counter() - t0)
            _ui_timer_last_quick_key = quick_key
            _ui_timer_last_sync_ts = now
            if dirty_rebuild_due:
                _ui_timer_last_dirty_rebuild_ts = now
        return target_interval
    return _UI_SYNC_TIMER_INTERVAL_IDLE


@persistent
def _ui_load_post(_dummy):
    """Blender load_post handler: reset all UI state caches after a file load."""
    global _ui_sync_last_ts, _ui_sync_last_key, _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_timer_last_dirty_rebuild_ts, _ui_source_probe_last, _ui_latency_cache, _ui_tree_cache, _has_ace_nodes_in_trees
    _ui_sync_last_ts = 0.0
    _ui_sync_last_key = ""
    _ui_timer_last_quick_key = ""
    _ui_timer_last_sync_ts = 0.0
    _ui_timer_last_dirty_rebuild_ts = 0.0
    _has_ace_nodes_in_trees = False
    _ui_source_probe_last = {}
    _ui_latency_cache = {
        "scene_key": 0,
        "source_uid": "",
        "ts": 0.0,
        "metrics": None,
    }
    _ui_tree_cache = {
        "tree_ptr": 0,
        "topology_sig": "",
        "sig_ts": 0.0,
        "source_uid": "",
        "source_proxy": "",
        "nodes": (),
        "lookup_ids": {},
    }

# --- Lists & Menus ---
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

# --- Panels & Sub-Panels ---
class ACE_PT_presentation_panel(bpy.types.Panel):
    """Panel displaying the add-on logo."""
    bl_label = "AudioCore Engine"
    bl_idname = "ACE_PT_presentation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AudioCore Engine"
    bl_order = 0

    def draw(self, context):
        from ..utils import _ensure_logo_icon
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
        from ..utils import _get_active_source

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
        from ..utils import _get_active_source
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

        # Parallel DSP disabled warning — shown when restart limit was exhausted.
        # _parallel_disabled_sources is keyed by source_uid (stable_id), with proxy_name
        # as a fallback when stable_id is absent, so check both.
        engine = ACEAudioEngine.get_instance(context.scene)
        disabled_srcs = set(getattr(getattr(engine, "parallel", None), "_parallel_disabled_sources", set()) or set())
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
        from ..utils import _get_active_source
        _, active_source = _get_active_source(context)
        return active_source is not None and not _ui_is_blocked(context)

    def draw(self, context):
        from ..utils import _get_active_source
        layout = self.layout
        _, src = _get_active_source(context)
        
        if not src:
            layout.label(text="No Source Selected")
            return

        panel_prop = getattr(layout, "panel_prop", None)
        if callable(panel_prop):
            with utils.logged_errors("n_panel_ui.ACE_PT_sub_tempo_meter.draw"):
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
            utils.note_swallowed("n_panel_ui.ACE_PT_sub_tempo_meter.draw#2", exc)
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
        from ..utils import _get_active_source
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
            and _has_ace_nodes_in_trees
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


# --- REGISTRATION ---
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
    global _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_timer_last_dirty_rebuild_ts, _ui_source_probe_last, _ui_latency_cache, _ui_tree_cache, _has_ace_nodes_in_trees
    _ui_timer_last_quick_key = ""
    _ui_timer_last_sync_ts = 0.0
    _ui_timer_last_dirty_rebuild_ts = 0.0
    _has_ace_nodes_in_trees = False
    _ui_source_probe_last = {}
    _ui_latency_cache = {
        "scene_key": 0,
        "source_uid": "",
        "ts": 0.0,
        "metrics": None,
    }
    _ui_tree_cache = {
        "tree_ptr": 0,
        "topology_sig": "",
        "sig_ts": 0.0,
        "source_uid": "",
        "source_proxy": "",
        "nodes": (),
    }
    if _ui_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_ui_load_post)
    if bpy.app.timers.is_registered(_ui_sync_timer):
        bpy.app.timers.unregister(_ui_sync_timer)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
