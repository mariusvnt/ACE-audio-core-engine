"""Proxy objects per source and the writer that commits outputs to their sinks."""

import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any
from ..id_management import ACEIdManager
from ..abstract_tree import ACEAbstractTree
from ...nodes.frontend import FRONTEND_KIND_SHADER, ensure_frontend_uid
from ... import utils
from .attributes import AttributeSinks
from .objects import ObjectPropertySinks
from .sonar import Sonar


class ACEProxyManager(AttributeSinks, ObjectPropertySinks, Sonar):
    """Manage concrete Stage-C sinks for compiled ACE outputs."""

    # "Pro" name for the invisible collection
    COLLECTION_NAME = "ACE_Internals"
    SONAR_OBJECT_NAME = "ACE_Sonar"
    SONAR_MODIFIER_NAME = "ACE_Sonar_Mod"
    SONAR_TREE_NAME = "ACE_Sonar_Tree"
    SONAR_SHADER_MATERIAL_NAME = "ACE_Sonar_Material"
    _MIRROR_OBJECT_PROPERTIES_TO_MESH_DATA = False

    def __init__(self, abstract_tree: ACEAbstractTree | None = None):
        self.abstract_tree = abstract_tree
        self.output_defs: Dict[str, Dict[str, Any]] = {}
        self._attr_cache: Dict[str, Dict[str, Any]] = {}
        self._last_values: Dict[str, Dict[str, Any]] = {}
        self._object_prop_cache: Dict[str, Any] = {}
        self._last_object_prop_values: Dict[str, Dict[str, Any]] = {}
        self._write_buffer_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if abstract_tree:
            self._cache_output_defs()


    @staticmethod
    def _resolve_scene(scene=None):
        if scene is not None:
            return scene
        active_scene = getattr(getattr(bpy, "context", None), "scene", None)
        if active_scene is not None:
            return active_scene
        scenes = list(getattr(bpy.data, "scenes", []) or [])
        return scenes[0] if scenes else None

    @classmethod
    def _shader_tree_writes_are_unsafe(cls, scene=None) -> bool:
        scene = cls._resolve_scene(scene)
        try:
            from ..engine.engine import ACEAudioEngine
        except Exception as exc:
            utils.note_swallowed("manager.ACEProxyManager._shader_tree_writes_are_unsafe", exc)
            return False
        try:
            return not bool(
                ACEAudioEngine._structural_tree_writes_allowed(
                    scene,
                    frontend_kind=FRONTEND_KIND_SHADER,
                    tree_bl_idname="ShaderNodeTree",
                )
            )
        except Exception as exc:
            utils.note_swallowed("manager.ACEProxyManager._shader_tree_writes_are_unsafe#2", exc)
            return False

    @classmethod
    def _defer_shader_tree_sync(cls, tree, scene=None) -> None:
        if tree is None:
            return
        with utils.logged_errors("manager.ACEProxyManager._defer_shader_tree_sync"):
            ensure_frontend_uid(tree, FRONTEND_KIND_SHADER, [])
        with utils.logged_errors("manager.ACEProxyManager._defer_shader_tree_sync#2"):
            from ..engine.engine import ACEAudioEngine
            resolved_scene = cls._resolve_scene(scene)
            engine = ACEAudioEngine.get_instance(resolved_scene)
            if engine is None and resolved_scene is not None:
                engine = ACEAudioEngine.ensure_instance(resolved_scene)
            if engine is not None:
                engine.request_recompile(tree)

    @classmethod
    def _ensure_internal_collection(cls, scene=None) -> bpy.types.Collection:
        """Create or retrieve the hidden ACE_Internals collection."""
        utils.assert_main_thread("_ensure_internal_collection")
        if cls.COLLECTION_NAME in bpy.data.collections:
            col = bpy.data.collections[cls.COLLECTION_NAME]
        else:
            col = bpy.data.collections.new(cls.COLLECTION_NAME)
        target_scene = cls._resolve_scene(scene)
        if target_scene:
            children = target_scene.collection.children
            if children.get(col.name) is None:
                children.link(col)

        if col.color_tag != 'COLOR_05':
            col.color_tag = 'COLOR_05'

        if not col.hide_viewport:
            col.hide_viewport = True
        if not col.hide_render:
            col.hide_render = True

        return col


    @classmethod
    def ensure_proxy_for_source(cls, source_prop, scene=None) -> bpy.types.Object:
        """Create or retrieve the single-vertex proxy mesh for an audio source."""
        utils.assert_main_thread("ensure_proxy_for_source")
        with utils.logged_errors("manager.ACEProxyManager.ensure_proxy_for_source"):
            ACEIdManager.ensure_source_uid(source_prop)
        stable_id = getattr(source_prop, "stable_id", "") or ""
        source_name = getattr(source_prop, "name", "") or "Source"
        proxy_name = ACEIdManager.generate_source_proxy_name(
            source_name,
            stable_seed=stable_id or source_name,
        )

        existing_ref = getattr(source_prop, "proxy_object_name", "") or ""
        if existing_ref:
            existing_obj = bpy.data.objects.get(existing_ref)
            if existing_obj and existing_obj.type == 'MESH':
                if existing_obj.name != proxy_name:
                    existing_obj.name = proxy_name
                if source_prop.proxy_object_name != existing_obj.name:
                    source_prop.proxy_object_name = existing_obj.name
                cls._move_to_internal_collection(existing_obj, scene=scene)
                return existing_obj

        if proxy_name in bpy.data.objects:
            obj = bpy.data.objects[proxy_name]
            if obj.type == 'MESH':
                cls._move_to_internal_collection(obj, scene=scene)
                if source_prop.proxy_object_name != obj.name:
                    source_prop.proxy_object_name = obj.name
                return obj

        mesh = bpy.data.meshes.new(f"{proxy_name}_Mesh")
        obj = bpy.data.objects.new(proxy_name, mesh)

        target_col = cls._ensure_internal_collection(scene=scene)
        target_col.objects.link(obj)

        mesh.from_pydata([(0,0,0)], [], [])

        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)

        source_prop.proxy_object_name = obj.name

        return obj

    @classmethod
    def _move_to_internal_collection(cls, obj, scene=None):
        utils.assert_main_thread("_move_to_internal_collection")
        target_col = cls._ensure_internal_collection(scene=scene)
        if obj.name in target_col.objects:
            return

        target_col.objects.link(obj)
        for col in obj.users_collection:
            if col != target_col:
                col.objects.unlink(obj)

    @staticmethod
    def delete_proxy_for_source(source_prop):
        """Remove the proxy object and its orphaned mesh data block from Blender."""
        utils.assert_main_thread("delete_proxy_for_source")
        if not source_prop.proxy_object_name:
            return

        obj = bpy.data.objects.get(source_prop.proxy_object_name)
        if obj:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh and mesh.users == 0:
                bpy.data.meshes.remove(mesh, do_unlink=True)

        source_prop.proxy_object_name = ""


    def set_abstract_tree(self, abstract_tree: ACEAbstractTree):
        """Replace the active abstract tree and rebuild the output definition cache."""
        self.abstract_tree = abstract_tree
        self.output_defs.clear()
        self._object_prop_cache.clear()
        self._last_object_prop_values.clear()
        self._cache_output_defs()

    def clear_runtime_caches(self):
        """Discard all cached RNA handles, last-written values, and write buffers."""
        self._attr_cache.clear()
        self._last_values.clear()
        self._object_prop_cache.clear()
        self._last_object_prop_values.clear()
        self._write_buffer_cache.clear()

    @staticmethod
    def _mesh_data_idblock(obj):
        if obj is None:
            return None
        data = getattr(obj, "data", None)
        if data is None:
            return None
        if not hasattr(data, "keys"):
            return None
        return data

    def ensure_sink_layout(self, graph, active_keys: Set[str] | None = None):
        """Ensure all concrete sink targets for *graph* exist."""
        utils.assert_main_thread("ensure_sink_layout")
        proxy_keys_by_object: Dict[str, Set[str]] = {}
        object_bindings: Dict[str, list[Any]] = {}
        keys = set(self.output_defs.keys()) if active_keys is None else set(active_keys)
        for attr_key in keys:
            for binding in list(getattr(graph, "sink_bindings_by_attr", {}).get(attr_key, []) or []):
                sink_kind = str(getattr(binding, "sink_kind", "") or "")
                target_obj_name = str(getattr(binding, "target_object_name", "") or "")
                if sink_kind == "PROXY_ATTRIBUTE" and target_obj_name:
                    proxy_keys_by_object.setdefault(target_obj_name, set()).add(attr_key)
                elif sink_kind == "OBJECT_PROPERTY" and target_obj_name:
                    object_bindings.setdefault(target_obj_name, []).append(binding)

        geometry_proxy_names: set[str] = set()
        for proxy_name, attr_keys in proxy_keys_by_object.items():
            proxy = bpy.data.objects.get(proxy_name)
            if proxy is None:
                continue
            self.ensure_attributes_exist(proxy, self.abstract_tree, active_keys=set(attr_keys), prune_existing=True)
            geometry_proxy_names.add(str(proxy_name))
        for object_name, bindings in object_bindings.items():
            self.ensure_object_property_sinks(object_name, bindings, prune=True)
        return geometry_proxy_names

    def commit_results(self, graph, results: Dict[str, Any], active_keys: Set[str] | None = None):
        """Fan out compiled graph results into concrete sink targets."""
        utils.assert_main_thread("commit_results")
        keys = set(results.keys()) if active_keys is None else set(active_keys)
        proxy_results: Dict[str, Dict[str, Any]] = {}
        object_prop_values: Dict[str, Dict[str, Any]] = {}
        object_prop_types: Dict[str, Dict[str, str]] = {}
        for attr_key in keys:
            if attr_key not in results:
                continue
            value = results[attr_key]
            for binding in list(getattr(graph, "sink_bindings_by_attr", {}).get(attr_key, []) or []):
                sink_kind = str(getattr(binding, "sink_kind", "") or "")
                object_name = str(getattr(binding, "target_object_name", "") or "")
                if not object_name:
                    continue
                if sink_kind == "PROXY_ATTRIBUTE":
                    proxy_results.setdefault(object_name, {})[attr_key] = value
                elif sink_kind == "OBJECT_PROPERTY":
                    prop_name = str(getattr(binding, "target_data_name", "") or "")
                    if not prop_name:
                        continue
                    object_prop_values.setdefault(object_name, {})[prop_name] = value
                    object_prop_types.setdefault(object_name, {})[prop_name] = str(getattr(binding, "attr_type", "FLOAT") or "FLOAT")

        geometry_proxy_names: set[str] = set()
        for proxy_name, proxy_values in proxy_results.items():
            proxy = bpy.data.objects.get(proxy_name)
            if proxy is None:
                continue
            self.update_proxy_object(proxy, proxy_values, active_keys=set(proxy_values.keys()))
            geometry_proxy_names.add(str(proxy_name))
        for object_name, prop_values in object_prop_values.items():
            self.update_object_property_sinks(object_name, prop_values, object_prop_types.get(object_name, {}))
        return geometry_proxy_names

    @staticmethod
    def get_proxy_object(source_prop):
        """Safe getter"""
        if not source_prop.proxy_object_name:
            return None
        return bpy.data.objects.get(source_prop.proxy_object_name)
