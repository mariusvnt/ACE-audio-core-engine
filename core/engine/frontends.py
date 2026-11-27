"""Scene frontend discovery and dirty tracking for recompiles."""

import bpy
import hashlib
import time
from typing import Any, Dict, Optional
from ... import utils
from ..abstract_tree import ACEAbstractTree, ACEFrontendRef
from ..proxy.manager import ACEProxyManager
from ...nodes.identity import is_ace_group, tree_key
from .common import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, _ACECompiledFrontendSnapshot, _ACEFrontendDiscovery


class FrontendTracker:
    """Scene frontend discovery and dirty tracking for recompiles."""

    def __init__(self, engine):
        self.engine = engine
        # Empty between compiles: no RNA reference may outlive an undo.
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
        # Set by any recompile request that is not a depsgraph report: those always compile.
        self._forced_recompile: bool = False
        # Content signature of each compiled frontend tree, by tree pointer.
        self._compiled_content_sigs: Dict[int, str] = {}

    @staticmethod
    def _node_tree_pointer(tree) -> int:
        if tree is None:
            return 0
        if hasattr(tree, "as_pointer"):
            with utils.logged_errors("frontends.FrontendTracker._node_tree_pointer"):
                return int(tree.as_pointer())
        return id(tree)

    @staticmethod
    def _tree_frontend_uid(tree) -> str:
        """The tree's frontend uid for this session, computed: nothing is stored on the tree."""
        if tree is None:
            return ""
        kind = "shader" if str(getattr(tree, "bl_idname", "") or "") == "ShaderNodeTree" else "geometry"
        return f"ACE_FRONTEND_{kind}_{tree_key(tree)}"

    @staticmethod
    def _tree_is_internal_ace_group(tree) -> bool:
        return tree is not None and is_ace_group(tree)

    @classmethod
    def _tree_is_ace_managed(cls, tree) -> bool:
        return bool(cls._tree_has_ace_nodes(tree))

    @staticmethod
    def _tree_has_ace_nodes(tree: bpy.types.NodeTree | None) -> bool:
        """True when *tree* or a group it uses (at any depth) holds an ACE node."""
        from ...nodes.definitions import tree_uses_ace

        return bool(tree) and tree_uses_ace(tree)

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

    @staticmethod
    def _scene_objects(scene) -> list:
        """The scene's objects and those it instances through collections (other scenes' included), each once.

        Walked from the scene's own data, not from a depsgraph: a depsgraph not evaluated since
        an object was deleted still lists it, and reading it crashes Blender.
        """
        objects: list = []
        seen: set = set()
        stack = list(getattr(scene, "objects", []) or [])
        while stack:
            obj = stack.pop()
            marker = FrontendTracker._node_tree_pointer(obj)
            if marker in seen:
                continue
            seen.add(marker)
            objects.append(obj)
            collection = getattr(obj, "instance_collection", None)
            if collection is not None and str(getattr(obj, "instance_type", "") or "") == "COLLECTION":
                stack.extend(getattr(collection, "all_objects", []) or [])
        return sorted(objects, key=lambda o: str(getattr(o, "name_full", "") or getattr(o, "name", "")))

    @staticmethod
    def _modifier_bindings(modifier, tree) -> dict:
        """The values *modifier* gives the inputs of *tree* that reach an ACE node, as plain data."""
        from ...nodes.definitions import plain_value, relevant_inputs

        wanted = relevant_inputs(tree)
        if not wanted:
            return {}
        inputs = getattr(getattr(modifier, "properties", None), "inputs", None)
        bindings = {}
        for identifier in sorted(wanted):
            entry = getattr(inputs, identifier, None) if inputs is not None else None
            if entry is None or getattr(entry, "type", "VALUE") != "VALUE":
                continue
            bindings[identifier] = plain_value(entry.value)
        return bindings

    def discover_scene_frontends(self, scene) -> list[_ACEFrontendDiscovery]:
        """The roots of every ACE use in the scene, one per tree and distinct modifier values.

        Owners: objects' Geometry Nodes modifiers and material slots, lights and the World.
        A tree counts when it or a group it uses (at any depth) holds an ACE node.
        """
        from ...nodes.definitions import tree_uses_ace

        utils.assert_main_thread("discover_scene_frontends")
        if scene is None:
            return []
        memo: dict = {}
        roots: Dict[tuple, Dict[str, Any]] = {}

        def _register(tree, kind, owner_key, consumer="", material_name="", modifier=None, bindings=None):
            if tree is None or self._tree_is_internal_ace_group(tree) or not tree_uses_ace(tree, memo):
                return
            bindings = dict(bindings or {})
            digest = hashlib.blake2b(repr(sorted(bindings.items())).encode("utf-8"), digest_size=4).hexdigest() if bindings else ""
            key = (int(self._node_tree_pointer(tree)), digest)
            rec = roots.get(key)
            if rec is None:
                rec = roots[key] = {
                    "tree": tree, "kind": kind, "digest": digest, "bindings": bindings, "modifier": modifier,
                    "owner_keys": set(), "consumer_keys": set(), "material_name": material_name,
                    "discovery_index": len(roots),
                }
            rec["owner_keys"].add(owner_key)
            if consumer:
                rec["consumer_keys"].add(consumer)

        for obj in self._scene_objects(scene):
            obj_name = str(getattr(obj, "name", "") or "")
            for mod in getattr(obj, "modifiers", []) or []:
                tree = getattr(mod, "node_group", None)
                if getattr(mod, "type", None) != "NODES" or tree is None:
                    continue
                with utils.logged_errors("frontends.FrontendTracker.discover_scene_frontends#modifier"):
                    _register(tree, FRONTEND_KIND_GEOMETRY, f"{obj_name}:{getattr(mod, 'name', '')}",
                              modifier=mod, bindings=self._modifier_bindings(mod, tree))
            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
                    continue
                _register(tree, FRONTEND_KIND_SHADER, obj_name, consumer=obj_name,
                          material_name=str(getattr(material, "name", "") or ""))
            if str(getattr(obj, "type", "") or "") == "LIGHT":
                light = getattr(obj, "data", None)
                _register(getattr(light, "node_tree", None), FRONTEND_KIND_SHADER, f"light:{getattr(light, 'name', '')}",
                          consumer=obj_name)
        world = getattr(scene, "world", None)
        if world is not None:
            _register(getattr(world, "node_tree", None), FRONTEND_KIND_SHADER, f"world:{world.name}")

        # Uses of one tree that differ in their values: the first keeps the tree's own ids.
        firsts: Dict[int, str] = {}
        for (ptr, digest), rec in sorted(roots.items(), key=lambda kv: (kv[0][0], sorted(kv[1]["owner_keys"]))):
            firsts.setdefault(ptr, digest)

        discovered: list[_ACEFrontendDiscovery] = []
        for (ptr, digest), rec in roots.items():
            tree = rec["tree"]
            own = digest == firsts.get(ptr)
            uid = self._tree_frontend_uid(tree) + ("" if own else f"_{digest}")
            root_key = tree_key(tree) + ("" if own else f"#{digest}")
            discovered.append(_ACEFrontendDiscovery(
                tree=tree,
                frontend_ref=ACEFrontendRef(
                    uid=uid,
                    kind=rec["kind"],
                    tree_name=str(getattr(tree, "name", "") or ""),
                    tree_bl_idname=str(getattr(tree, "bl_idname", "") or ""),
                    owner_keys=tuple(sorted(rec["owner_keys"])),
                    material_name=str(rec["material_name"] or ""),
                    tree_ptr=int(ptr),
                    discovery_index=int(rec["discovery_index"]),
                ),
                consumer_object_names=tuple(sorted(rec["consumer_keys"])),
                bindings=rec["bindings"],
                root_key=root_key,
                modifier=rec["modifier"],
            ))
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
                f"ACE: target tree resolved via gn_target_object fallback (no ACE nodes) → '{target_fallback.name}'",
                "DEBUG",
            )
            return target_fallback
        if sonar_fallback:
            utils.log(
                f"ACE: target tree resolved via Sonar fallback (no ACE nodes) → '{sonar_fallback.name}'",
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

    @staticmethod
    def _tree_content_signature(tree) -> str:
        from ...nodes.definitions import tree_content_signature

        return tree_content_signature(tree)

    @staticmethod
    def _hierarchy_trees(frontends) -> Dict[int, Any]:
        """Each root tree and every group it uses, at any depth, by pointer."""
        from ...nodes.definitions import hierarchy_trees

        found: Dict[int, Any] = {}
        for rec in frontends or []:
            for tree in hierarchy_trees(rec.tree):
                found.setdefault(int(FrontendTracker._node_tree_pointer(tree)), tree)
        return found

    def record_compiled_content(self, frontends: list[_ACEFrontendDiscovery]) -> None:
        """Remember the content each tree had when it was compiled: the roots and the groups they use."""
        self._compiled_content_sigs = {
            ptr: self._tree_content_signature(tree) for ptr, tree in self._hierarchy_trees(frontends).items()
        }

    def _dirty_trees_unchanged(self, discovered_by_uid: Dict[str, _ACEFrontendDiscovery], membership_changed: bool) -> bool:
        """True when the only dirty marks are depsgraph reports and no marked tree's content changed.

        A compile writes proxy data; every tree reading a proxy is then re-evaluated and reported
        by the depsgraph like a user edit. Without this check that report recompiles again,
        forever, every ~0.4 s. Marks are by tree, groups included: a tree the last compile did not
        read counts as changed.
        """
        if self._forced_recompile or self._pending_scene_rebuild or membership_changed or not self.engine.graphs:
            return False
        dirty_ptrs = {int(p) for p in self._dirty_tree_ptrs}
        if not dirty_ptrs:
            return False
        trees_by_ptr = self._hierarchy_trees(discovered_by_uid.values())
        for ptr in dirty_ptrs:
            tree = trees_by_ptr.get(ptr)
            recorded = self._compiled_content_sigs.get(ptr)
            if tree is None or recorded is None or recorded != self._tree_content_signature(tree):
                return False
        return True

    def _clear_dirty_state(self) -> None:
        self._pending_tree = None
        self._pending_tree_ptr = 0
        self._pending_tree_bl_idname = ""
        self._dirty_trees.clear()
        self._dirty_tree_ptrs.clear()
        self._dirty_tree_bl_idnames.clear()
        self._dirty_frontend_uids.clear()
        self._pending_scene_rebuild = False
        self._forced_recompile = False
        self._first_dirty_time = 0.0

    def _maybe_compile_dirty(self, scene) -> bool:
        """Recompile the dirty scene frontends if the quiet window is met."""
        if self.engine._undo_recovery_active():
            return False
        now = time.time()
        discovered_frontends = self.discover_scene_frontends(scene)
        discovered_by_uid = {
            str(rec.frontend_ref.uid or ""): rec
            for rec in (discovered_frontends or [])
        }
        membership_sig = self._frontend_membership_signature(discovered_frontends)
        membership_changed = membership_sig != getattr(self.engine.compiler, "_last_frontend_membership_signature", ())
        if not self._dirty_frontend_uids and not self._pending_scene_rebuild and not membership_changed:
            return False
        # Rebuild after a quiet window, or after 2 s of continuous edits.
        first_dirty = getattr(self, "_first_dirty_time", 0.0)
        bypass_quiet = first_dirty > 0.0 and (now - first_dirty) >= 2.0
        if not bypass_quiet and (now - self._last_dirty_time) < 0.35:
            return False
        if (now - self.engine.compiler._last_compile_time) < 0.35:
            return False
        if self._is_compiling:
            return False
        if self._pending_scene_rebuild and not self._dirty_frontend_uids and not membership_changed:
            self._pending_scene_rebuild = False
            self._first_dirty_time = 0.0
            return False
        if self._dirty_trees_unchanged(discovered_by_uid, membership_changed):
            self._clear_dirty_state()
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
        self._clear_dirty_state()
        self._is_compiling = True
        compiled = False
        try:
            self.engine.compiler.compile_scene_frontends(scene, preferred_tree=preferred_tree)
            compiled = True
        finally:
            self._is_compiling = False
            self._first_dirty_time = 0.0  # reset after compile
        return compiled

    def request_recompile(self, bl_tree: bpy.types.NodeTree | None, verify_content: bool = False):
        """Mark scene frontends as needing recompilation on the next compile poll.

        *verify_content* is for depsgraph reports: the compile is skipped when no marked tree's
        content changed. Every other request compiles.
        """
        now = time.time()
        if not verify_content:
            self._forced_recompile = True
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
