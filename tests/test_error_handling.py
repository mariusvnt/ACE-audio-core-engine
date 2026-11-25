"""
Errors are never absorbed silently (ARCHITECTURE.md §9).

A broad handler (bare ``except``, ``except Exception`` / ``BaseException``)
must record or log what it catches, or re-raise, unless its ``except`` line is
marked ``# noqa: BLE001 boundary`` with a reason. ``utils.logged_errors``
covers the "carry on" case.
"""

import ast
import unittest
from pathlib import Path

from module_loader import load_module

ROOT = Path(__file__).resolve().parents[1]
ADDON_DIRS = ("core", "nodes", "ui")
ADDON_ROOT_FILES = ("__init__.py", "utils.py", "properties.py", "vars.py")
HANDLED_CALLS = {"note_swallowed", "log", "log_throttled", "_log_runtime", "_log_error", "report"}
BOUNDARY_MARK = "# noqa: BLE001 boundary"


def _addon_files():
    files = [ROOT / name for name in ADDON_ROOT_FILES]
    for directory in ADDON_DIRS:
        files.extend(sorted((ROOT / directory).glob("*.py")))
    return [f for f in files if f.exists()]


def _is_broad(handler):
    t = handler.type
    if t is None:
        return True
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(isinstance(n, ast.Name) and n.id in ("Exception", "BaseException") for n in names)


def _records_or_raises(handler):
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in HANDLED_CALLS:
                return True
    return False


class TestNoSilentHandlers(unittest.TestCase):
    def test_every_broad_handler_records_raises_or_is_a_marked_boundary(self):
        offenders = []
        for path in _addon_files():
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
                    continue
                if BOUNDARY_MARK in lines[node.lineno - 1]:
                    continue
                if not _records_or_raises(node):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], "broad handlers that absorb errors silently")

    def test_boundary_marks_give_a_reason(self):
        missing = []
        for path in _addon_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if BOUNDARY_MARK in line and not line.split(BOUNDARY_MARK, 1)[1].strip(" :"):
                    missing.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(missing, [], "boundary marks need a reason after the colon")


class TestRecordingHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils = load_module("utils")

    def setUp(self):
        self.utils.reset_swallowed_errors()

    def test_logged_errors_suppresses_and_records(self):
        with self.utils.logged_errors("test.site"):
            raise ValueError("boom")
        count, last = self.utils.swallowed_errors()["test.site"]
        self.assertEqual(count, 1)
        self.assertIn("ValueError: boom", last)

    def test_logged_errors_lets_base_exceptions_through(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.utils.logged_errors("test.interrupt"):
                raise KeyboardInterrupt()
        self.assertNotIn("test.interrupt", self.utils.swallowed_errors())

    def test_note_swallowed_counts_repeats(self):
        for _ in range(3):
            self.utils.note_swallowed("test.repeat", RuntimeError("again"))
        self.assertEqual(self.utils.swallowed_errors()["test.repeat"][0], 3)

    def test_clean_block_records_nothing(self):
        with self.utils.logged_errors("test.clean"):
            pass
        self.assertEqual(self.utils.swallowed_errors(), {})

    def test_log_throttled_emits_once_per_interval(self):
        self.assertTrue(self.utils.log_throttled("test.throttle", "x", "DEBUG", min_interval=60.0))
        self.assertFalse(self.utils.log_throttled("test.throttle", "x", "DEBUG", min_interval=60.0))


if __name__ == "__main__":
    unittest.main()
