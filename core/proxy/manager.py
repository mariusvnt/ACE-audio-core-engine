"""Proxy objects per source and the writer that commits outputs to their sinks."""

import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any
from ..id_management import ACEIdManager
from ..abstract_tree import ACEAbstractTree
from ... import utils
from .attributes import AttributeSinks
from .objects import ScenePropertySinks


class ACEProxyManager(AttributeSinks, ScenePropertySinks):
    """Manage concrete Stage-C sinks for compiled ACE outputs."""

    # "Pro" name for the invisible collection
    COLLECTION_NAME = "ACE_Internals"
    SONAR_OBJECT_NAME = "ACE_Sonar"  # made by earlier versions; still read as a frontend

    def __init__(self, abstract_tree: ACEAbstractTree | None = None):
        self.abstract_tree = abstract_tree
        self.output_defs: Dict[str, Dict[str, Any]] = {}
        # The mesh value nodes write (ensure_values_object), set by the compile when one is used.
        self.values_object_name: str = ""
        self._attr_cache: Dict[str, Dict[str, Any]] = {}
        self._last_values: Dict[str, Dict[str, Any]] = {}
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
                cls._mark_proxy(existing_obj, stable_id)
                return existing_obj

        if proxy_name in bpy.data.objects:
            obj = bpy.data.objects[proxy_name]
            if obj.type == 'MESH':
                cls._move_to_internal_collection(obj, scene=scene)
                if source_prop.proxy_object_name != obj.name:
                    source_prop.proxy_object_name = obj.name
                cls._mark_proxy(obj, stable_id)
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
        cls._mark_proxy(obj, stable_id)

        return obj

    @classmethod
    def ensure_values_object(cls, scene=None) -> bpy.types.Object:
        """The one-vertex mesh holding the values of nodes fed a value, not audio (Rolling Average,
        Stream Accumulator): they carry no source, so every source writes them here."""
        from ...nodes.identity import VALUES_OBJECT_NAME

        utils.assert_main_thread("ensure_values_object")
        obj = bpy.data.objects.get(VALUES_OBJECT_NAME)
        if obj is not None and obj.type == 'MESH' and obj.library is None:
            cls._move_to_internal_collection(obj, scene=scene)
            return obj
        mesh = bpy.data.meshes.new(f"{VALUES_OBJECT_NAME}_Mesh")
        mesh.from_pydata([(0, 0, 0)], [], [])
        obj = bpy.data.objects.new(VALUES_OBJECT_NAME, mesh)
        cls._ensure_internal_collection(scene=scene).objects.link(obj)
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
        return obj

    @staticmethod
    def _mark_proxy(obj, stable_id: str) -> None:
        """The proxy carries its source's id: a Source node reading this proxy reads that source."""
        if stable_id and str(obj.get("ace_source_id", "") or "") != stable_id:
            obj["ace_source_id"] = stable_id

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
        self._last_object_prop_values.clear()
        self._cache_output_defs()

    def clear_runtime_caches(self):
        """Discard all cached RNA handles, last-written values, and write buffers."""
        self._attr_cache.clear()
        self._last_values.clear()
        self._last_object_prop_values.clear()
        self._write_buffer_cache.clear()

    def ensure_sink_layout(self, graph, active_keys: Set[str] | None = None):
        """Ensure all concrete sink targets for *graph* exist."""
        utils.assert_main_thread("ensure_sink_layout")
        required_by_object: Dict[str, Dict[str, str]] = {}
        scene_bindings: Dict[str, list[Any]] = {}
        bindings_by_attr = getattr(graph, "sink_bindings_by_attr", {}) or {}
        keys = set(bindings_by_attr) if active_keys is None else set(active_keys)
        for attr_key in keys:
            for binding in list(bindings_by_attr.get(attr_key, []) or []):
                sink_kind = str(getattr(binding, "sink_kind", "") or "")
                target_obj_name = str(getattr(binding, "target_object_name", "") or "")
                name = str(getattr(binding, "target_data_name", "") or "")[:63]
                if target_obj_name and target_obj_name == self.values_object_name:
                    continue  # every source writes there: laid out once per compile (layout_values_object)
                if sink_kind == "PROXY_ATTRIBUTE" and target_obj_name and name:
                    required_by_object.setdefault(target_obj_name, {})[name] = str(binding.attr_type or "FLOAT")
                elif sink_kind == "SCENE_PROPERTY" and target_obj_name:
                    scene_bindings.setdefault(target_obj_name, []).append(binding)

        geometry_proxy_names: set[str] = set()
        for proxy_name, required in required_by_object.items():
            proxy = bpy.data.objects.get(proxy_name)
            if proxy is None:
                continue
            self.ensure_attributes_exist(proxy, required, prune_existing=True)
            geometry_proxy_names.add(str(proxy_name))
        for scene_name, bindings in scene_bindings.items():
            self.ensure_scene_property_sinks(scene_name, bindings)
        return geometry_proxy_names

    def layout_values_object(self, graphs) -> None:
        """The attributes of ACE_Values: what every graph's value nodes write, nothing else."""
        utils.assert_main_thread("layout_values_object")
        obj = bpy.data.objects.get(self.values_object_name) if self.values_object_name else None
        if obj is None:
            return
        required: Dict[str, str] = {}
        for graph in graphs:
            for bindings in (getattr(graph, "sink_bindings_by_attr", {}) or {}).values():
                for binding in bindings or []:
                    if binding.sink_kind == "PROXY_ATTRIBUTE" and binding.target_object_name == obj.name:
                        name = str(binding.target_data_name or "")[:63]
                        if name:
                            required[name] = str(binding.attr_type or "FLOAT")
        self.ensure_attributes_exist(obj, required, prune_existing=True)

    def commit_results(self, graph, results: Dict[str, Any], active_keys: Set[str] | None = None):
        """Fan out compiled graph results into concrete sink targets."""
        utils.assert_main_thread("commit_results")
        keys = set(results.keys()) if active_keys is None else set(active_keys)
        proxy_results: Dict[str, Dict[str, Any]] = {}
        proxy_pulses: Dict[str, Set[str]] = {}
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
                    name = str(getattr(binding, "target_data_name", "") or "")[:63]
                    if not name:
                        continue
                    proxy_results.setdefault(object_name, {})[name] = value
                    if getattr(binding, "pulse", False):
                        proxy_pulses.setdefault(object_name, set()).add(name)
                elif sink_kind == "SCENE_PROPERTY":
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
            self.update_proxy_object(proxy, proxy_values, pulses=proxy_pulses.get(proxy_name, set()))
            geometry_proxy_names.add(str(proxy_name))
        for scene_name, prop_values in object_prop_values.items():
            self.update_scene_property_sinks(scene_name, prop_values, object_prop_types.get(scene_name, {}))
        return geometry_proxy_names

    @staticmethod
    def get_proxy_object(source_prop):
        """Safe getter"""
        if not source_prop.proxy_object_name:
            return None
        return bpy.data.objects.get(source_prop.proxy_object_name)
