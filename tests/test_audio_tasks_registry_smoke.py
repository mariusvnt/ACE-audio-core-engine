import sys
import types
import unittest

from module_loader import load_module


def _load_tasks_registry_with_fake_numpy():
    # Drop cached modules so this test validates importability without real NumPy.
    sys.modules.pop("ace_addon_testpkg.audio_tasks_registry", None)
    sys.modules.pop("ace_addon_testpkg.audio_influx", None)

    had_numpy = "numpy" in sys.modules
    original_numpy = sys.modules.get("numpy")

    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = object
    sys.modules["numpy"] = fake_np

    try:
        module = load_module("audio_tasks_registry")
    finally:
        if had_numpy:
            sys.modules["numpy"] = original_numpy  # type: ignore[assignment]
        else:
            sys.modules.pop("numpy", None)
        # Do not leak fake-NumPy-loaded modules into later tests.
        sys.modules.pop("ace_addon_testpkg.audio_tasks_registry", None)
        sys.modules.pop("ace_addon_testpkg.audio_influx", None)

    return module


class TestAudioTasksRegistrySmoke(unittest.TestCase):
    def test_imports_with_fake_numpy_module(self):
        atr = _load_tasks_registry_with_fake_numpy()
        self.assertTrue(hasattr(atr, "ACEAudioTask"))
        self.assertTrue(hasattr(atr, "SourceTask"))
        self.assertTrue(hasattr(atr, "FrequencySplitterTask"))
        self.assertEqual(getattr(atr, "SAMPLE_RATE", None), 44100)


if __name__ == "__main__":
    unittest.main()
