import sys
import types
import unittest

from module_loader import forget_modules, load_tasks


_MISSING = object()


def _load_tasks_registry_with_fake_numpy():
    # Drop cached modules so this test validates importability without real NumPy.
    forget_modules("core.tasks", "core.audio_influx", "audio_influx")

    # Without NumPy the optional DSP packages that need it are unavailable too;
    # None in sys.modules makes their import raise ImportError.
    hidden = ("numpy", "audioflux", "numba", "scipy", "scipy.signal", "soundfile")
    originals = {name: sys.modules.get(name, _MISSING) for name in hidden}
    for name in hidden:
        sys.modules[name] = None  # type: ignore[assignment]

    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = object
    sys.modules["numpy"] = fake_np

    try:
        module = load_tasks()
    finally:
        for name, original in originals.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        # Do not leak fake-NumPy-loaded modules into later tests.
        forget_modules("core.tasks", "core.audio_influx", "audio_influx")

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
