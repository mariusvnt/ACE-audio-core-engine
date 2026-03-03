import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
except Exception:
    np = None

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from engine_perf_harness import _load_engine_stack


@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestSourceSampleRateMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ae, _, _ = _load_engine_stack()

    @staticmethod
    def _settings(sample_rate_cap: str):
        return types.SimpleNamespace(
            ace_scheduler_mode="demand",
            ace_file_working_sample_rate=str(sample_rate_cap),
        )

    def test_resolve_source_sample_rate_prefers_native_file_rate(self):
        source = types.SimpleNamespace(
            source_type="FILE",
            sample_rate=48000,
            native_sample_rate=96000,
        )

        resolved = self.ae.ACEAudioEngine._resolve_source_sample_rate(
            source,
            fallback_sr=44100,
            scene_or_engine_settings=self._settings("native"),
        )

        self.assertEqual(resolved, 96000)

    def test_collect_sample_rate_floor_uses_native_file_metadata(self):
        source = types.SimpleNamespace(
            source_type="FILE",
            filepath="",
            sample_rate=48000,
            native_sample_rate=96000,
            duration=12.0,
        )
        engine = self.ae.ACEAudioEngine(scene_key=1)

        sample_rate_floor = engine._collect_sample_rate_floor(
            [source],
            self._settings("native"),
            fps=24.0,
            allow_io=False,
        )

        self.assertEqual(sample_rate_floor, 96000)
