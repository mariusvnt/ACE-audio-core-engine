"""
Index 0 is a real index: ``int(x or -1)`` used to turn it into "none".
"""

import sys
import types
import unittest

import fake_bpy
from module_loader import PKG_NAME, load_module


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


class TestAsInt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils = load_module("utils")

    def test_zero_stays_zero(self):
        self.assertEqual(self.utils.as_int(0, -1), 0)
        self.assertEqual(self.utils.as_float(0.0, 1.5), 0.0)

    def test_missing_or_invalid_falls_back(self):
        for value in (None, "", "abc", object()):
            self.assertEqual(self.utils.as_int(value, -1), -1)
            self.assertEqual(self.utils.as_float(value, 2.5), 2.5)

    def test_numeric_strings_convert(self):
        self.assertEqual(self.utils.as_int("3", -1), 3)
        self.assertEqual(self.utils.as_float("0.25", 1.0), 0.25)


class _Source:
    def __init__(self, source_type="FILE"):
        self.source_type = source_type
        self.stable_id = "src_a"


class TestBakePollFirstSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _purge_testpkg_modules()
        fake_bpy.install()
        cls.ops = load_module("n_panel_operators")

    def _context(self, index, sources):
        ace = types.SimpleNamespace(active_audio_source_index=index, audio_sources=sources)
        return types.SimpleNamespace(scene=types.SimpleNamespace(audiocore=ace))

    def test_bake_is_available_for_the_first_source(self):
        poll = self.ops.ACE_OT_bake_active_source.poll
        self.assertTrue(poll(self._context(0, [_Source()])))

    def test_bake_needs_a_file_source_in_range(self):
        poll = self.ops.ACE_OT_bake_active_source.poll
        self.assertFalse(poll(self._context(0, [_Source("LIVE")])))
        self.assertFalse(poll(self._context(1, [_Source()])))
        self.assertFalse(poll(self._context(-1, [_Source()])))


if __name__ == "__main__":
    unittest.main()
