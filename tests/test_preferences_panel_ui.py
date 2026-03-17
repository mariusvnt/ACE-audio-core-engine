import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import PKG_NAME, load_module


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


def _install_fake_bpy():
    if "bpy" in sys.modules:
        return sys.modules["bpy"]

    bpy_mod = types.ModuleType("bpy")
    handlers_mod = types.ModuleType("bpy.app.handlers")
    handlers_mod.persistent = lambda fn: fn
    handlers_mod.load_post = []

    app_mod = types.ModuleType("bpy.app")
    app_mod.handlers = handlers_mod

    def _prop_stub(**_kwargs):
        return None

    class _FakeCollection(list):
        def add(self):
            item = types.SimpleNamespace(
                library_id="",
                display_name="",
                description="",
                is_builtin=False,
            )
            self.append(item)
            return item

        def clear(self):
            del self[:]

    class _Props:
        @staticmethod
        def StringProperty(**_kwargs):
            return _prop_stub(**_kwargs)

        @staticmethod
        def BoolProperty(**_kwargs):
            return _prop_stub(**_kwargs)

        @staticmethod
        def IntProperty(**_kwargs):
            return _prop_stub(**_kwargs)

        @staticmethod
        def CollectionProperty(**_kwargs):
            return _FakeCollection()

    bpy_mod.app = types.SimpleNamespace(handlers=handlers_mod)
    bpy_mod.context = types.SimpleNamespace(preferences=None)
    bpy_mod.props = _Props()
    bpy_mod.types = types.SimpleNamespace(
        PropertyGroup=type("PropertyGroup", (), {}),
        UIList=type("UIList", (), {}),
        AddonPreferences=type("AddonPreferences", (), {}),
    )
    bpy_mod.utils = types.SimpleNamespace(
        register_class=lambda _cls: None,
        unregister_class=lambda _cls: None,
    )

    sys.modules["bpy"] = bpy_mod
    sys.modules["bpy.app"] = app_mod
    sys.modules["bpy.app.handlers"] = handlers_mod
    return bpy_mod


class TestPreferencesPanelUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        cls.bpy = _install_fake_bpy()
        load_module("utils")
        cls.prefs_ui = load_module("prefs_panel_ui")

    def test_preferences_bl_idname_uses_root_addon_package(self):
        self.assertEqual(self.prefs_ui.ACE_preferences.bl_idname, PKG_NAME)

    def test_get_addon_preferences_looks_up_root_addon_key(self):
        expected_prefs = object()
        self.bpy.context.preferences = types.SimpleNamespace(
            addons={
                PKG_NAME: types.SimpleNamespace(preferences=expected_prefs),
                f"{PKG_NAME}.ui": types.SimpleNamespace(preferences=object()),
            }
        )

        resolved = self.prefs_ui._get_addon_preferences()

        self.assertIs(resolved, expected_prefs)


if __name__ == "__main__":
    unittest.main()
