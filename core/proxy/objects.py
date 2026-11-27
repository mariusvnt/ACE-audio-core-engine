"""Scene-property sinks for shader frontends.

A shader ACE node's sink group reads its values with Attribute nodes in View
Layer mode, which read the scene's custom properties: one write per value
serves every material and object. The evaluated scene only sees a property
write once the scene is tagged, so a frame with changes tags it once.
"""

import bpy # type: ignore
import numpy as np
from typing import Dict, Any
from ..id_management import ACEIdManager
from ... import utils


class ScenePropertySinks:
    @classmethod
    def _normalize_prop_value(cls, value, attr_type: str):
        if attr_type == "FLOAT":
            try:
                return float(value)
            except Exception as exc:
                utils.note_swallowed("objects.ScenePropertySinks._normalize_prop_value", exc)
                return 0.0
        if hasattr(value, "tolist"):
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

    @staticmethod
    def _scene_named(scene_name: str):
        return bpy.data.scenes.get(str(scene_name or "")) if scene_name else None

    def ensure_scene_property_sinks(self, scene_name: str, bindings) -> bool:
        """Create the scene properties *bindings* write, so the Attribute nodes find them."""
        utils.assert_main_thread("ensure_scene_property_sinks")
        scene = self._scene_named(scene_name)
        if scene is None:
            return False
        created = False
        for binding in bindings or []:
            prop_name = str(getattr(binding, "target_data_name", "") or "")
            if prop_name and prop_name not in scene.keys():
                scene[prop_name] = self._zero_value_for_type(str(getattr(binding, "attr_type", "FLOAT") or "FLOAT"))
                created = True
        if created:
            scene.update_tag()
        return True

    def update_scene_property_sinks(self, scene_name: str, prop_values: Dict[str, Any], prop_types: Dict[str, str]) -> bool:
        """Write changed values; tag the scene once if any changed."""
        utils.assert_main_thread("update_scene_property_sinks")
        scene = self._scene_named(scene_name)
        if scene is None or not prop_values:
            return False
        last_vals = self._last_object_prop_values.setdefault(scene_name, {})
        changed = False
        for prop_name, raw_value in prop_values.items():
            attr_type = str((prop_types or {}).get(prop_name, "FLOAT") or "FLOAT")
            value = self._normalize_prop_value(raw_value, attr_type)
            if self._values_equal(last_vals.get(prop_name), value):
                continue
            with utils.logged_errors("objects.ScenePropertySinks.update_scene_property_sinks"):
                scene[prop_name] = value
                last_vals[prop_name] = value
                changed = True
        if changed:
            scene.update_tag()
        return changed

    @staticmethod
    def prune_scene_properties(scene, active_prop_names: set[str], keep=()) -> bool:
        """Remove ACE_* scene properties no sink writes any more (and that are not in *keep*)."""
        if scene is None:
            return False
        stale = [
            k for k in list(scene.keys())
            if str(k).startswith(ACEIdManager.PREFIX) and str(k) not in active_prop_names and str(k) not in keep
        ]
        for k in stale:
            with utils.logged_errors("objects.ScenePropertySinks.prune_scene_properties"):
                del scene[k]
        if stale:
            scene.update_tag()
        return bool(stale)
