"""Scene frontend discovery and dirty tracking for recompiles."""

import bpy
import time
from typing import Any, Dict, Optional
from .. import utils
from .abstract_tree import ACEAbstractTree, ACEFrontendRef
from .id_management import ACEIdManager
from .proxy_management import ACEProxyManager
from .engine_common import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, _ACECompiledFrontendSnapshot, _ACEFrontendDiscovery


class FrontendTracker:
    """Scene frontend discovery and dirty tracking for recompiles."""

    def __init__(self, engine):
        self.engine = engine
        # Legacy compatibility slots: keep these unset so the engine never holds
        # long-lived RNA object references across undo/redo.
        self._pending_tree: Optional[bpy.types.NodeTree] = None
        self._pending_scene_rebuild: bool = False
        self._last_request_time: float = 0.0
        self._dirty_trees: set[bpy.types.NodeTree] = set()
        self._dirty_tree_ptrs: set[int] = set()
        self._dirty_tree_bl_idnames: Dict[int, str] = {}
        self._dirty_frontend_uids: set[str] = set()
        self._last_dirty_time: float = 0.0
        self._is_compiling: bool = False
        self._pending_tree_ptr: int = 0
        self._pending_tree_bl_idname: str = ""

    @staticmethod
    def _node_tree_pointer(tree) -> int:
        """Return a stable integer identity for a node tree (pointer or id())."""
        if tree is None:
            return 0
        if hasattr(tree, "as_pointer"):
            with utils.logged_errors("audio_engine.ACEAudioEngine._node_tree_pointer"):
                return int(tree.as_pointer())
        return id(tree)

    @staticmethod
    def _tree_frontend_uid(tree) -> str:
        if tree is None:
            return ""
        try:
            return str(tree.get("ace_frontend_uid", "") or "")
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._tree_frontend_uid", exc)
            return ""

    @staticmethod
    def _tree_is_internal_ace_group(tree) -> bool:
        if tree is None:
            return False
        name = str(getattr(tree, "name", "") or "")
        return bool(name.startswith(".ACE_"))

    @classmethod
    def _tree_is_ace_managed(cls, tree) -> bool:
        return bool(cls._tree_frontend_uid(tree) or cls._tree_has_ace_nodes(tree))

    @staticmethod
    def _tree_has_ace_nodes(tree: bpy.types.NodeTree | None) -> bool:
        if not tree:
            return False
        with utils.logged_errors("audio_engine.ACEAudioEngine._tree_has_ace_nodes"):
            if bool(tree.get("ace_frontend_uid", "")):
                return True
        try:
            for node in getattr(tree, "nodes", []) or []:
                get_prop = getattr(node, "get", lambda _k, d=None: d)
                if bool(get_prop("ace_id")):
                    return True
                source_id = str(get_prop("ace_source_id", "") or "")
                if source_id and ACEIdManager.parse_node_id(source_id):
                    return True
                if str(getattr(node, "type", "") or "") != "GROUP":
                    continue
                group = getattr(node, "node_tree", None)
                group_name = str(getattr(group, "name", "") or "")
                norm_group_name = group_name.lstrip(".")
                if group_name.startswith(".ACE_") or norm_group_name.startswith("ACE_"):
                    return True
            return False
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._tree_has_ace_nodes#2", exc)
            return False

    @staticmethod
    def _frontend_membership_signature(frontends: list[_ACEFrontendDiscovery]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (
                    str(rec.frontend_ref.uid),
                    str(rec.frontend_ref.kind),
                    str(rec.frontend_ref.tree_name),
                    int(rec.frontend_ref.tree_ptr),
                    tuple(rec.frontend_ref.owner_keys),
                    str(rec.frontend_ref.material_name),
                    tuple(rec.consumer_object_names),
                )
                for rec in (frontends or [])
            )
        )

    @staticmethod
    def _compiled_snapshot_from_discovery(rec: _ACEFrontendDiscovery) -> _ACECompiledFrontendSnapshot:
        frontend_ref = rec.frontend_ref
        return _ACECompiledFrontendSnapshot(
            uid=str(frontend_ref.uid or ""),
            kind=str(frontend_ref.kind or ""),
            tree_name=str(frontend_ref.tree_name or ""),
            tree_bl_idname=str(frontend_ref.tree_bl_idname or ""),
            owner_keys=tuple(str(v) for v in tuple(frontend_ref.owner_keys or ()) if str(v)),
            consumer_object_names=tuple(str(v) for v in tuple(rec.consumer_object_names or ()) if str(v)),
            material_name=str(frontend_ref.material_name or ""),
            tree_ptr=int(frontend_ref.tree_ptr or 0),
        )

    @staticmethod
    def _resolve_preferred_tree_from_discoveries(
        frontends: list[_ACEFrontendDiscovery],
        frontend_uid: str = "",
        tree_ptr: int = 0,
        tree_bl_idname: str = "",
    ):
        uid_token = str(frontend_uid or "")
        ptr_token = int(tree_ptr or 0)
        type_token = str(tree_bl_idname or "")
        if uid_token:
            for rec in frontends or []:
                if str(getattr(getattr(rec, "frontend_ref", None), "uid", "") or "") == uid_token:
                    return getattr(rec, "tree", None)
        if ptr_token:
            for rec in frontends or []:
                rec_ptr = int(getattr(getattr(rec, "frontend_ref", None), "tree_ptr", 0) or 0)
                if rec_ptr == ptr_token:
                    return getattr(rec, "tree", None)
        if type_token:
            for rec in frontends or []:
                if str(getattr(getattr(rec, "frontend_ref", None), "tree_bl_idname", "") or "") == type_token:
                    return getattr(rec, "tree", None)
        return None

    def discover_scene_frontends(self, scene) -> list[_ACEFrontendDiscovery]:
        """Discover managed ACE frontends used by the scene."""
        from ..nodes.dummy_tree import ensure_frontend_uid as _ensure_frontend_uid
        from ..nodes.shader_tree import ACEShaderTreeReader

        utils.assert_main_thread("discover_scene_frontends")

        if scene is None:
            return []

        by_ptr: Dict[int, Dict[str, Any]] = {}

        def _register_tree(tree, frontend_kind: str, owner_keys, material_name: str = "", consumer_object_names=()):
            if tree is None or self._tree_is_internal_ace_group(tree):
                return
            tree_type = str(getattr(tree, "bl_idname", "") or "")
            is_shader = str(frontend_kind or "") == FRONTEND_KIND_SHADER or tree_type == "ShaderNodeTree"
            if is_shader:
                managed = bool(ACEShaderTreeReader.is_managed_tree(tree) or self._tree_is_ace_managed(tree))
            else:
                managed = self._tree_is_ace_managed(tree)
            if not managed:
                return
            tree_ptr = self._node_tree_pointer(tree)
            owner_keys_tuple = tuple(sorted(str(v) for v in (owner_keys or []) if str(v)))
            consumer_keys_tuple = tuple(sorted(str(v) for v in (consumer_object_names or []) if str(v)))
            existing = by_ptr.get(tree_ptr)
            if existing is None:
                by_ptr[tree_ptr] = {
                    "tree": tree,
                    "frontend_kind": str(frontend_kind or ""),
                    "discovery_index": len(by_ptr),
                    "owner_keys": set(owner_keys_tuple),
                    "consumer_keys": set(consumer_keys_tuple),
                    "material_name": str(material_name or ""),
                }
                return
            existing["owner_keys"].update(owner_keys_tuple)
            existing["consumer_keys"].update(consumer_keys_tuple)
            if material_name and not existing.get("material_name"):
                existing["material_name"] = str(material_name)

        def _frontend_record_sort_key(rec: Dict[str, Any]) -> tuple[Any, ...]:
            tree = rec.get("tree")
            return (
                str(rec.get("frontend_kind", "") or ""),
                str(getattr(tree, "bl_idname", "") or ""),
                str(getattr(tree, "name", "") or ""),
                str(rec.get("material_name", "") or ""),
                tuple(sorted(str(v) for v in rec.get("owner_keys", set()) if str(v))),
                tuple(sorted(str(v) for v in rec.get("consumer_keys", set()) if str(v))),
                int(self._node_tree_pointer(tree)),
            )

        for obj in getattr(scene, "objects", []) or []:
            obj_name = str(getattr(obj, "name", "") or "")
            for mod in getattr(obj, "modifiers", []) or []:
                if getattr(mod, "type", None) != "NODES":
                    continue
                tree = getattr(mod, "node_group", None)
                if tree is None:
                    continue
                _register_tree(
                    tree,
                    FRONTEND_KIND_GEOMETRY,
                    owner_keys=(f"{obj_name}:{getattr(mod, 'name', '')}",),
                )
            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
                    continue
                _register_tree(
                    tree,
                    FRONTEND_KIND_SHADER,
                    owner_keys=(obj_name,),
                    material_name=str(getattr(material, "name", "") or ""),
                    consumer_object_names=(obj_name,),
                )

        discovered: list[_ACEFrontendDiscovery] = []
        existing_uids: list[str] = []
        for rec in sorted(by_ptr.values(), key=_frontend_record_sort_key):
            tree = rec.get("tree")
            frontend_kind = str(rec.get("frontend_kind", "") or "")
            frontend_uid = _ensure_frontend_uid(tree, frontend_kind, existing_uids)
            if not frontend_uid:
                continue
            if frontend_uid not in existing_uids:
                existing_uids.append(frontend_uid)
            owner_keys = tuple(sorted(str(v) for v in rec.get("owner_keys", set()) if str(v)))
            consumer_keys = tuple(sorted(str(v) for v in rec.get("consumer_keys", set()) if str(v)))
            discovered.append(
                _ACEFrontendDiscovery(
                    tree=tree,
                    frontend_ref=ACEFrontendRef(
                        uid=frontend_uid,
                        kind=frontend_kind,
                        tree_name=str(getattr(tree, "name", "") or ""),
                        tree_bl_idname=str(getattr(tree, "bl_idname", "") or ""),
                        owner_keys=owner_keys,
                        material_name=str(rec.get("material_name", "") or ""),
                        tree_ptr=int(self._node_tree_pointer(tree)),
                        discovery_index=int(rec.get("discovery_index", 0) or 0),
                    ),
                    consumer_object_names=consumer_keys,
                )
            )

        return sorted(discovered, key=lambda rec: str(rec.frontend_ref.uid or ""))

    @staticmethod
    def _find_scene_target_tree(scene) -> Optional[bpy.types.NodeTree]:
        def _tree_from_object(obj):
            if not obj:
                return (None, None)
            ace_tree = None
            fallback = None
            for mod in getattr(obj, "modifiers", []):
                if getattr(mod, "type", None) == 'NODES' and getattr(mod, "node_group", None):
                    tree = mod.node_group
                    if fallback is None:
                        fallback = tree
                    if FrontendTracker._tree_has_ace_nodes(tree):
                        ace_tree = tree
                        break
            return ace_tree, fallback

        ace = getattr(scene, "audiocore", None)
        target_obj = getattr(ace, "gn_target_object", None) if ace else None
        target_ace, target_fallback = _tree_from_object(target_obj)
        if target_ace:
            utils.log(
                f"ACE: target tree resolved via 'gn_target_object' → '{target_ace.name}'",
                "DEBUG",
            )
            return target_ace

        sonar = bpy.data.objects.get(ACEProxyManager.SONAR_OBJECT_NAME)
        sonar_ace, sonar_fallback = _tree_from_object(sonar)
        if sonar_ace:
            utils.log(
                f"ACE: target tree resolved via Sonar object → '{sonar_ace.name}'",
                "DEBUG",
            )
            return sonar_ace
        if target_fallback:
            utils.log(
                f"ACE: target tree resolved via gn_target_object fallback (no ace_id nodes) → '{target_fallback.name}'",
                "DEBUG",
            )
            return target_fallback
        if sonar_fallback:
            utils.log(
                f"ACE: target tree resolved via Sonar fallback (no ace_id nodes) → '{sonar_fallback.name}'",
                "DEBUG",
            )
        return sonar_fallback

    @staticmethod
    def _compute_tree_signature(abstract_tree: ACEAbstractTree):
        node_sig = []
        for nid, node in sorted(abstract_tree.nodes.items()):
            # repr(v) guards against NumPy scalars/arrays that would cause
            # TypeError when sorted() tries to compare values of mixed types.
            params = tuple(sorted(
                ((k, repr(v)) for k, v in node.parameters.items()),
                key=lambda x: x[0],
            ))
            inputs = tuple(sorted((s.socket_idx, s.data_type, s.name) for s in node.inputs))
            outputs = tuple(sorted((s.socket_idx, s.data_type, s.name) for s in node.outputs))
            roots = tuple(sorted(node.root_sources)) if getattr(node, "root_sources", None) else ()
            node_sig.append((
                nid,
                node.category,
                node.type,
                params,
                inputs,
                outputs,
                roots,
                str(getattr(node, "frontend_uid", "") or ""),
                str(getattr(node, "frontend_kind", "") or ""),
                str(getattr(node, "source_binding_id", "") or ""),
            ))
        links = tuple(sorted((l.source_node_id, l.source_socket_idx, l.target_node_id, l.target_socket_idx) for l in abstract_tree.links))
        output_links = tuple(
            sorted(
                (
                    str(getattr(l, "source_node_id", "")),
                    int(getattr(l, "source_socket_idx", 0)),
                    int(getattr(l, "target_socket_idx", 0)),
                    str(getattr(l, "target_socket_name", "") or ""),
                    str(getattr(l, "frontend_uid", "") or ""),
                    str(getattr(l, "frontend_kind", "") or ""),
                )
                for l in (getattr(abstract_tree, "group_output_links", []) or [])
            )
        )
        frontend_sig = tuple(
            sorted(
                (
                    str(uid),
                    str(getattr(frontend, "kind", "") or ""),
                    str(getattr(frontend, "tree_name", "") or ""),
                    str(getattr(frontend, "tree_bl_idname", "") or ""),
                    tuple(getattr(frontend, "owner_keys", ()) or ()),
                    str(getattr(frontend, "material_name", "") or ""),
                    int(getattr(frontend, "tree_ptr", 0) or 0),
                )
                for uid, frontend in (getattr(abstract_tree, "frontends", {}) or {}).items()
            )
        )
        frontends_by_uid = dict(getattr(abstract_tree, "frontends", {}) or {})
        consumer_sig = tuple(
            sorted(
                (
                    str(getattr(intent, "frontend_uid", "") or ""),
                    str(
                        getattr(
                            frontends_by_uid.get(str(getattr(intent, "frontend_uid", "") or "")),
                            "kind",
                            "",
                        )
                        or ""
                    ),
                    str(getattr(intent, "node_id", "") or ""),
                    int(getattr(intent, "socket_idx", 0) or 0),
                    str(getattr(intent, "sink_kind", "") or ""),
                    str(getattr(intent, "source_token", "") or ""),
                    tuple(
                        sorted(
                            str(v)
                            for v in (getattr(intent, "consumer_keys", ()) or ())
                            if str(v)
                        )
                    ),
                    str(getattr(intent, "output_name", "") or ""),
                    str(getattr(intent, "explicit_name", "") or ""),
                )
                for intent in (getattr(abstract_tree, "consumer_intents", []) or [])
            )
        )
        return (tuple(node_sig), links, output_links, frontend_sig, consumer_sig)

    def _maybe_compile_dirty(self, scene) -> bool:
        """Recompile the dirty scene frontends if the quiet window is met.

        Requires ``_last_dirty_time`` to have been stable for 350 ms AND
        ``_last_compile_time`` to be at least 350 ms ago.  Both guards are
        bypassed when the dirty state has persisted for more than 2 s (rapid
        link-editing bursts that keep resetting the quiet timer).

        Returns:
            True if a recompile was performed, False otherwise.
        """
        if self.engine._undo_recovery_active():
            return False
        now = time.time()
        discovered_frontends = self.discover_scene_frontends(scene)
        discovered_by_uid = {
            str(rec.frontend_ref.uid or ""): rec
            for rec in (discovered_frontends or [])
        }
        membership_sig = self._frontend_membership_signature(discovered_frontends)
        membership_changed = membership_sig != getattr(self.engine, "_last_frontend_membership_signature", ())
        if not self._dirty_frontend_uids and not self._pending_scene_rebuild and not membership_changed:
            return False
        shader_frontend_uids = {
            str(rec.frontend_ref.uid or "")
            for rec in (discovered_frontends or [])
            if str(getattr(rec.frontend_ref, "kind", "") or "") == FRONTEND_KIND_SHADER
        }
        touches_shader_frontends = bool(
            any(
                str(self._dirty_tree_bl_idnames.get(int(tree_ptr), "") or "") == "ShaderNodeTree"
                for tree_ptr in (self._dirty_tree_ptrs or set())
            )
            or str(getattr(self, "_pending_tree_bl_idname", "") or "") == "ShaderNodeTree"
            or bool(shader_frontend_uids.intersection({str(uid) for uid in (self._dirty_frontend_uids or set())}))
            or ((self._pending_scene_rebuild or membership_changed) and bool(shader_frontend_uids))
        )
        if touches_shader_frontends and not self.engine._structural_tree_writes_allowed(
            scene,
            frontend_kind=FRONTEND_KIND_SHADER,
            tree_bl_idname="ShaderNodeTree",
        ):
            utils.log_throttled(
                "shader_tree_preview_deferred",
                "Deferring ACE shader-tree rebuild while a live material/rendered viewport preview is active.",
                "WARNING",
                min_interval=5.0,
            )
            return False
        # Require a quiet window and minimum spacing between rebuilds.
        # Exception: bypass the quiet window if dirty state has persisted for >2 s
        # (catches rapid link-editing bursts that perpetually reset the quiet timer).
        first_dirty = getattr(self, "_first_dirty_time", 0.0)
        bypass_quiet = first_dirty > 0.0 and (now - first_dirty) >= 2.0
        if not bypass_quiet and (now - self._last_dirty_time) < 0.35:
            return False
        if (now - self.engine._last_compile_time) < 0.35:
            return False
        if self._is_compiling:
            return False
        if self._pending_scene_rebuild and not self._dirty_frontend_uids and not membership_changed:
            self._pending_scene_rebuild = False
            self._first_dirty_time = 0.0
            return False
        preferred_tree = None
        if self._dirty_frontend_uids:
            dirty_uid = next(iter(self._dirty_frontend_uids))
            preferred_tree = getattr(discovered_by_uid.get(dirty_uid), "tree", None)
        if preferred_tree is None:
            preferred_tree = self._resolve_preferred_tree_from_discoveries(
                discovered_frontends,
                tree_ptr=int(getattr(self, "_pending_tree_ptr", 0) or 0),
                tree_bl_idname=str(getattr(self, "_pending_tree_bl_idname", "") or ""),
            )
        self._pending_tree = None
        self._pending_tree_ptr = 0
        self._pending_tree_bl_idname = ""
        self._dirty_trees.clear()
        self._dirty_tree_ptrs.clear()
        self._dirty_tree_bl_idnames.clear()
        self._dirty_frontend_uids.clear()
        self._pending_scene_rebuild = False
        self._is_compiling = True
        compiled = False
        try:
            self.engine.compile_scene_frontends(scene, preferred_tree=preferred_tree)
            compiled = True
        finally:
            self._is_compiling = False
            self._first_dirty_time = 0.0  # reset after compile
        return compiled

    def request_recompile(self, bl_tree: bpy.types.NodeTree | None):
        """Mark scene frontends as needing recompilation on the next compile poll.

        Rate-limited: multiple calls within 150 ms of each other update only
        ``_last_dirty_time`` without re-triggering the cooldown so rapid source
        property edits do not spam recompiles.

        Args:
            bl_tree: Tree to mark dirty, or None to request a scene-wide rebuild.
        """
        now = time.time()
        if self.engine._undo_recovery_active():
            self._pending_scene_rebuild = True
            self._pending_tree = None
            self._pending_tree_ptr = 0
            self._pending_tree_bl_idname = ""
            self._last_dirty_time = now
            self._first_dirty_time = now
            return
        if bl_tree is None:
            self._pending_scene_rebuild = True
        if bl_tree:
            frontend_uid = self._tree_frontend_uid(bl_tree)
            tree_ptr = int(self._node_tree_pointer(bl_tree) or 0)
            tree_bl_idname = str(getattr(bl_tree, "bl_idname", "") or "")
            if frontend_uid:
                self._dirty_frontend_uids.add(frontend_uid)
            else:
                self._pending_scene_rebuild = True
            self._dirty_trees.add(tree_ptr)
            self._dirty_tree_ptrs.add(tree_ptr)
            self._dirty_tree_bl_idnames[tree_ptr] = tree_bl_idname
            self._pending_tree = None
            self._pending_tree_ptr = tree_ptr
            self._pending_tree_bl_idname = tree_bl_idname
        # Track when dirty state first appeared so _maybe_compile_dirty can enforce
        # a max wait (2 s) even when rapid changes keep resetting _last_dirty_time.
        if not getattr(self, "_first_dirty_time", 0.0):
            self._first_dirty_time = now
        if (now - getattr(self, "_last_request_time", 0.0)) < 0.15:
            self._last_dirty_time = now
            return
        self._last_dirty_time = now
        self._last_request_time = now
