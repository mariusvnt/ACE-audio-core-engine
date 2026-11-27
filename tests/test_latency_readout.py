"""The latency readout stops growing once playback stops."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module_loader import load_module  # noqa: E402

import fake_bpy  # noqa: E402


class _Monitor:
    def current_snapshot(self):
        return {}


def _engine():
    return types.SimpleNamespace(
        current_frame=10,
        preview=types.SimpleNamespace(_preview_monitor=_Monitor()),
        _is_rendering_job=lambda: False,
    )


class LatencyReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.diag_mod = load_module("core.engine.diagnostics")

    def test_unfinished_geometry_evaluation_expires(self):
        diag = self.diag_mod.EngineDiagnostics(_engine())
        ttl = diag._BLENDER_LATENCY_EVENT_TTL_S
        diag._pending_geometry_events = [{"scene_key": 1, "frame": 10, "start_ts": 100.0, "target_tokens": {"x"}}]
        with mock.patch.object(self.diag_mod.time, "perf_counter", return_value=100.0 + ttl / 2):
            self.assertAlmostEqual(diag.get_runtime_diagnostics_current()["blender_geometry_eval_ms"], ttl / 2 * 1000.0)
        # Playback stopped: the evaluation never completes, and its age must not keep counting.
        with mock.patch.object(self.diag_mod.time, "perf_counter", return_value=100.0 + ttl * 50):
            self.assertEqual(diag.get_runtime_diagnostics_current()["blender_geometry_eval_ms"], 0.0)
        self.assertEqual(diag._pending_geometry_events, [])


if __name__ == "__main__":
    unittest.main()
