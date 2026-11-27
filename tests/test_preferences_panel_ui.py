import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import PKG_NAME, load_module

import fake_bpy


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


def _install_fake_bpy():
    return fake_bpy.install()


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

    def test_only_the_extended_logs_option(self):
        annotations = self.prefs_ui.ACE_preferences.__annotations__
        self.assertEqual(list(annotations), ["debug_logging"])


if __name__ == "__main__":
    unittest.main()
