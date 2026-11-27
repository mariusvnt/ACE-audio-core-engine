"""deps: no switches, the library report and its failures, and the spawn bootstrap."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import ROOT, forget_modules, load_module  # noqa: E402

import fake_bpy  # noqa: E402

class DepsTests(unittest.TestCase):
    def setUp(self):
        fake_bpy.install()
        forget_modules("deps")
        self.deps = load_module("deps")

    def tearDown(self):
        forget_modules("deps")

    def test_no_switches(self):
        # The libraries are always used when they import: nothing to turn off.
        for name in ("setting", "set_setting", "DEFAULT_SETTINGS"):
            self.assertFalse(hasattr(self.deps, name), name)
        self.assertEqual(self.deps.numba_enabled(), self.deps.available("numba"))
        self.assertEqual(self.deps.sounddevice_enabled(), self.deps.available("sounddevice"))

    def test_report_states_and_failures(self):
        libraries = {
            "json": {"name": "JSON", "desc": "always there"},
            "ace_no_such_library": {"name": "Missing", "desc": "never there", "missing_note": "Nothing works"},
        }
        with mock.patch.object(self.deps, "LIBRARIES", libraries):
            rows = {row["id"]: row for row in self.deps.report()}
            failures = self.deps.failures()
        self.assertEqual(rows["json"]["state"], "ok")
        self.assertEqual(rows["ace_no_such_library"]["state"], "missing")
        self.assertIn("Nothing works", rows["ace_no_such_library"]["detail"])
        self.assertIn("ModuleNotFoundError", rows["ace_no_such_library"]["detail"])
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("Missing could not be loaded: "))

    def test_no_user_folder_outside_an_extension(self):
        # The test package is not under bl_ext: Numba keeps its default cache folder.
        self.assertIsNone(self.deps._user_dir())


class SpawnBootstrapTests(unittest.TestCase):
    def test_child_can_import_a_bl_ext_package(self):
        """spawn_main.py lets a plain interpreter import bl_ext.<repo>.<pkg>, which exists only in Blender's memory."""
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "extensions" / "user_default" / "fake_ext"
            (package_dir / "core").mkdir(parents=True)
            (package_dir / "__init__.py").write_text("MARK = 'loaded'\n", encoding="utf-8")
            (package_dir / "core" / "__init__.py").write_text("", encoding="utf-8")
            env = dict(os.environ, ACE_SPAWN_PACKAGE="bl_ext.user_default.fake_ext", ACE_SPAWN_PACKAGE_DIR=str(package_dir))
            code = (
                "import runpy, sys\n"
                f"runpy.run_path({str(ROOT / 'core' / 'worker' / 'spawn_main.py')!r}, run_name='__mp_main__')\n"
                "import bl_ext.user_default.fake_ext.core as core\n"
                "import bl_ext.user_default.fake_ext as ext\n"
                "print(ext.MARK)\n"
            )
            result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "loaded")

    def test_top_level_package_needs_nothing(self):
        env = dict(os.environ, ACE_SPAWN_PACKAGE="AudioCoreEngine", ACE_SPAWN_PACKAGE_DIR=str(ROOT))
        code = (
            "import runpy, sys\n"
            f"runpy.run_path({str(ROOT / 'core' / 'worker' / 'spawn_main.py')!r}, run_name='__mp_main__')\n"
            "print('bl_ext' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout.strip(), "False", result.stderr)


if __name__ == "__main__":
    unittest.main()
