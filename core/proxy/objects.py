"""Object custom-property sinks for shader frontends."""

import bpy # type: ignore
import numpy as np
from typing import Dict, Set, Any
from ..id_management import ACEIdManager
from ... import utils


class ObjectPropertySinks:
    @classmethod
    def _normalize_object_prop_value(cls, value, attr_type: str):
        if attr_type == "FLOAT":
            try:
                return float(value)
            except Exception as exc:
                utils.note_swallowed("objects.ObjectPropertySinks._normalize_object_prop_value", exc)
                return 0.0
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            value = [float(value)] if isinstance(value, (int, float)) else []
        target_len = 4 if attr_type == "FLOAT_COLOR" else 3
        flat = np.zeros(target_len, dtype=np.float32)
        arr = np.asarray(list(value), dtype=np.float32).ravel() if value else np.zeros(0, dtype=np.float32)
        if arr.size > 0:
            flat[: min(target_len, arr.size)] = arr[:target_len]
        if attr_type == "FLOAT_COLOR" and arr.size < 4:
            flat[3] = 1.0
        return [float(v) for v in flat.tolist()]

    @classmethod
    def prune_object_ace_properties(cls, obj, active_prop_names: set[str]) -> bool:
        """Remove stale ACE_* custom properties from *obj* not in *active_prop_names*."""
        stale = [
            k for k in list(obj.keys() if hasattr(obj, "keys") else [])
            if str(k).startswith(ACEIdManager.PREFIX) and str(k) not in active_prop_names
        ]
        if not stale:
            return False
        for k in stale:
            with utils.logged_errors("objects.ObjectPropertySinks.prune_object_ace_properties"):
                del obj[k]
        try:
            obj.update_tag(refresh={"OBJECT"})
        except Exception as exc:
            utils.note_swallowed("objects.ObjectPropertySinks.prune_object_ace_properties#2", exc)
            with utils.logged_errors("objects.ObjectPropertySinks.prune_object_ace_properties#3"):
                obj.update_tag()
        return True

    def ensure_object_property_sinks(self, object_name: str, bindings, prune: bool = False) -> bool:
        utils.assert_main_thread("ensure_object_property_sinks")
        obj = self._object_prop_cache.get(object_name)
        if obj is not None and str(getattr(obj, "name", "") or "") != str(object_name or ""):
            obj = None
        if obj is None:
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                return False
            self._object_prop_cache[object_name] = obj
        mirror_to_data = bool(self._MIRROR_OBJECT_PROPERTIES_TO_MESH_DATA)
        mesh_data = self._mesh_data_idblock(obj) if mirror_to_data else None
        active_prop_names: set[str] = set()
        layout_changed = False
        for binding in bindings or []:
            prop_name = str(getattr(binding, "target_data_name", "") or "")
            if not prop_name:
                continue
            active_prop_names.add(prop_name)
            zero_value = self._zero_value_for_type(str(getattr(binding, "attr_type", "FLOAT") or "FLOAT"))
            already_exists = prop_name in obj.keys()
            if not already_exists:
                with utils.logged_errors("objects.ObjectPropertySinks.ensure_object_property_sinks"):
                    obj[prop_name] = zero_value
                    layout_changed = True
            if mesh_data is not None:
                with utils.logged_errors("objects.ObjectPropertySinks.ensure_object_property_sinks#2"):
                    if prop_name not in mesh_data.keys():
                        mesh_data[prop_name] = zero_value
                        layout_changed = True
        if prune:
            if self.prune_object_ace_properties(obj, active_prop_names):
                layout_changed = True
        if layout_changed:
            try:
                if mirror_to_data:
                    obj.update_tag(refresh={"OBJECT", "DATA"})
                else:
                    obj.update_tag(refresh={"OBJECT"})
            except Exception as exc:
                utils.note_swallowed("objects.ObjectPropertySinks.ensure_object_property_sinks#3", exc)
                with utils.logged_errors("objects.ObjectPropertySinks.ensure_object_property_sinks#4"):
                    obj.update_tag()
        return True

    def update_object_property_sinks(self, object_name: str, prop_values: Dict[str, Any], prop_types: Dict[str, str]) -> bool:
        utils.assert_main_thread("update_object_property_sinks")
        if not prop_values:
            return True
        obj = self._object_prop_cache.get(object_name)
        if obj is not None and str(getattr(obj, "name", "") or "") != str(object_name or ""):
            obj = None
        if obj is None:
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                return False
            self._object_prop_cache[object_name] = obj
        mirror_to_data = bool(self._MIRROR_OBJECT_PROPERTIES_TO_MESH_DATA)
        mesh_data = self._mesh_data_idblock(obj) if mirror_to_data else None
        last_vals = self._last_object_prop_values.setdefault(object_name, {})
        changed_specs: list[tuple[str, str, Any]] = []
        for prop_name, raw_value in prop_values.items():
            if not prop_name:
                continue
            attr_type = str((prop_types or {}).get(prop_name, "FLOAT") or "FLOAT")
            norm_value = self._normalize_object_prop_value(raw_value, attr_type)
            prev_value = last_vals.get(prop_name)
            if self._values_equal(prev_value, norm_value):
                continue
            changed_specs.append((prop_name, attr_type, norm_value))
        if not changed_specs:
            return True

        changed_values: Dict[str, Any] = {}
        for prop_name, attr_type, norm_value in changed_specs:
            wrote = False
            try:
                obj[prop_name] = norm_value
                wrote = True
            except Exception as exc:
                utils.note_swallowed("objects.ObjectPropertySinks.update_object_property_sinks", exc)
                with utils.logged_errors("objects.ObjectPropertySinks.update_object_property_sinks#2"):
                    zero_value = self._zero_value_for_type(attr_type)
                    obj[prop_name] = zero_value
                    norm_value = zero_value
                    wrote = True
            if wrote:
                changed_values[prop_name] = norm_value

        if mesh_data is not None and changed_values:
            for prop_name, norm_value in changed_values.items():
                try:
                    current_val = mesh_data.get(prop_name, None)
                except Exception as exc:
                    utils.note_swallowed("objects.ObjectPropertySinks.update_object_property_sinks#3", exc)
                    current_val = None
                if self._values_equal(current_val, norm_value):
                    continue
                with utils.logged_errors("objects.ObjectPropertySinks.update_object_property_sinks#4"):
                    mesh_data[prop_name] = norm_value

        if changed_values:
            last_vals.update(changed_values)
            try:
                if mirror_to_data:
                    obj.update_tag(refresh={"OBJECT", "DATA"})
                else:
                    obj.update_tag(refresh={"OBJECT"})
            except Exception as exc:
                utils.note_swallowed("objects.ObjectPropertySinks.update_object_property_sinks#5", exc)
                with utils.logged_errors("objects.ObjectPropertySinks.update_object_property_sinks#6"):
                    obj.update_tag()
        return True

    def prune_consumer_objects(self, active_props_by_object: Dict[str, Set[str]]) -> bool:
        """Remove stale ACE_* properties from all consumer objects in the scene."""
        utils.assert_main_thread("prune_consumer_objects")
        changed = False
        try:
            all_objects = bpy.data.objects
        except AttributeError:
            return False
        for obj in all_objects:
            obj_name = str(getattr(obj, "name", "") or "")
            try:
                ace_props = [k for k in list(obj.keys()) if str(k).startswith(ACEIdManager.PREFIX)]
            except Exception as exc:
                utils.note_swallowed("objects.ObjectPropertySinks.prune_consumer_objects", exc)
                ace_props = []
            if not ace_props:
                continue
            active = active_props_by_object.get(obj_name, set())
            if self.prune_object_ace_properties(obj, active):
                changed = True
        return changed
