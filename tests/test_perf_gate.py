import os
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
except Exception:
    np = None

from perf_harness import generate_baseline_report
from perf_harness import (
    PERF_REQUIRED_STAGES,
    PERF_STRICT_P95_REGRESSION_RATIO,
    PERF_STRICT_P95_REGRESSION_FLOOR_MS,
    PERF_STRICT_PREVIEW_P95_MS,
    PERF_STRICT_PREVIEW_SCENARIOS,
    PERF_STRICT_PREVIEW_UNDERRUN_RATE_MAX,
    PERF_STRICT_PREVIEW_QUEUE_DROP_RATE_MAX,
    required_scenario_ids,
)


def _strict_mode_enabled() -> bool:
    raw = str(os.environ.get("ACE_PERF_GATE_STRICT", "")).strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    # CI defaults to strict unless explicitly disabled.
    ci_raw = str(os.environ.get("CI", "")).strip().lower()
    return ci_raw in {"1", "true", "yes", "on"}


# A baseline is only comparable on the machine that produced it, so it lives
# in the gitignored artifacts folder (written by run_quality_gate.py or
# perf_harness.py) and the comparison is skipped when it is missing.
BASELINE_PATH = Path(__file__).parent / "artifacts" / "perf_baseline_report.json"


class TestPerfGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        with tempfile.TemporaryDirectory(prefix="ace_perf_") as tmp_dir:
            cls.current = generate_baseline_report(frames=180, output_path=Path(tmp_dir) / "perf_current.json")

    def test_perf_harness_generates_required_scenarios(self):
        scenarios = self.current.get("scenarios", {})
        for sid in required_scenario_ids():
            self.assertIn(sid, scenarios)
            for stage in PERF_REQUIRED_STAGES:
                self.assertIn(stage, scenarios[sid], msg=f"missing stage={stage} in scenario={sid}")
                self.assertGreater(
                    float(scenarios[sid][stage].get("count", 0.0) or 0.0),
                    0.0,
                    msg=f"zero sample count sid={sid} stage={stage}",
                )

    def test_p95_within_local_baseline(self):
        if not _strict_mode_enabled():
            self.skipTest("perf gate in warning mode")
        if not BASELINE_PATH.exists():
            self.skipTest(f"no local baseline at {BASELINE_PATH}")
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for sid in required_scenario_ids():
            for stage in PERF_REQUIRED_STAGES:
                cur_p95 = float(self.current["scenarios"][sid][stage]["p95_ms"])
                base_p95 = float(baseline["scenarios"][sid][stage]["p95_ms"])
                allowed = max(
                    base_p95 * float(PERF_STRICT_P95_REGRESSION_RATIO),
                    float(PERF_STRICT_P95_REGRESSION_FLOOR_MS),
                )
                self.assertLessEqual(
                    cur_p95,
                    allowed,
                    msg=(
                        f"Perf regression sid={sid} stage={stage}: "
                        f"p95={cur_p95:.3f}ms baseline={base_p95:.3f}ms allowed={allowed:.3f}ms"
                    ),
                )

    def test_realtime_preview_limits(self):
        if not _strict_mode_enabled():
            self.skipTest("perf gate in warning mode")
        for sid in tuple(str(v) for v in PERF_STRICT_PREVIEW_SCENARIOS):
            scenario = self.current["scenarios"][sid]
            preview_p95 = float(scenario["preview"]["p95_ms"])
            self.assertLessEqual(
                preview_p95,
                float(PERF_STRICT_PREVIEW_P95_MS),
                msg=f"preview p95={preview_p95:.3f}ms > {float(PERF_STRICT_PREVIEW_P95_MS):.3f}ms sid={sid}",
            )
            diag = dict(scenario.get("realtime", {}) or {})
            underrun_rate = float(diag.get("preview_underrun_rate", 0.0) or 0.0)
            queue_drop_rate = float(diag.get("preview_queue_drop_rate", 0.0) or 0.0)
            self.assertLessEqual(
                underrun_rate,
                float(PERF_STRICT_PREVIEW_UNDERRUN_RATE_MAX),
                msg=f"underrun_rate={underrun_rate:.4f} sid={sid}",
            )
            self.assertLessEqual(
                queue_drop_rate,
                float(PERF_STRICT_PREVIEW_QUEUE_DROP_RATE_MAX),
                msg=f"queue_drop_rate={queue_drop_rate:.4f} sid={sid}",
            )


if __name__ == "__main__":
    unittest.main()
