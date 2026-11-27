import importlib.util
import sys
import unittest
from pathlib import Path


def _has_bpy_module() -> bool:
    try:
        return importlib.util.find_spec("bpy") is not None
    except Exception:
        mod = sys.modules.get("bpy")
        if mod is None:
            return False
        context = getattr(mod, "context", None)
        preferences = getattr(context, "preferences", None) if context is not None else None
        return preferences is not None and hasattr(preferences, "addons")


_HAS_BPY = _has_bpy_module()


def _load_smoke_module():
    smoke_path = Path(__file__).with_name("blender_integration_smoke.py")
    spec = importlib.util.spec_from_file_location("ace_blender_integration_smoke", smoke_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load smoke module from {smoke_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_HAS_BPY, "requires Blender bpy module")
class TestBlenderIntegrationSmoke(unittest.TestCase):
    def test_smoke_script_runs_when_addon_enabled(self):
        import bpy  # type: ignore

        addon_keys = [k for k in bpy.context.preferences.addons.keys() if "AudioCore" in k or k.endswith("audio_core_engine")]
        if not addon_keys:
            self.skipTest("AudioCore add-on is not enabled in this Blender session")

        smoke = _load_smoke_module()
        smoke.main()


if __name__ == "__main__":
    unittest.main()
