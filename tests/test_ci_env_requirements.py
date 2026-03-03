import os
import unittest


def _ci_enabled() -> bool:
    raw = str(os.environ.get("CI", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class TestCIEnvRequirements(unittest.TestCase):
    def test_ci_has_numpy_available(self):
        if not _ci_enabled():
            self.skipTest("CI-only environment guard")
        try:
            import numpy as np  # noqa: F401
        except Exception as exc:
            self.fail(f"CI must provide NumPy for ACE perf/DSP tests: {exc}")


if __name__ == "__main__":
    unittest.main()
