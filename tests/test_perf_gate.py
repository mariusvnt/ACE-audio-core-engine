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


class TestPerfGate(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy unavailable in this environment")
    def test_perf_harness_generates_required_scenarios(self):
        fd, tmp_path = tempfile.mkstemp(
            prefix="ace_perf_smoke_",
            suffix=".json",
            dir=str(Path(__file__).parent),
        )
        os.close(fd)
        out_path = Path(tmp_path)
        try:
            report = generate_baseline_report(frames=180, output_path=out_path)
            scenarios = report.get("scenarios", {})
            required = required_scenario_ids()
            for sid in required:
                self.assertIn(sid, scenarios)
            for sid in required:
                for stage in PERF_REQUIRED_STAGES:
                    self.assertIn(stage, scenarios[sid], msg=f"missing stage={stage} in scenario={sid}")
        finally:
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass

    @unittest.skipIf(np is None, "numpy unavailable in this environment")
    def test_perf_gate_warning_mode_or_strict_threshold(self):
        strict = _strict_mode_enabled()
        baseline_path = Path(__file__).with_name("perf_baseline_report.json")
        fd, tmp_path = tempfile.mkstemp(
            prefix="ace_perf_current_",
            suffix=".json",
            dir=str(Path(__file__).parent),
        )
        os.close(fd)
        current_path = Path(tmp_path)
        current = generate_baseline_report(frames=180, output_path=current_path)

        try:
            if not baseline_path.exists():
                self.skipTest("Baseline report missing; run perf harness first")

            # Initial rollout is warning-only unless strict mode is enabled.
            if not strict:
                return

            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_scenarios = dict(baseline.get("scenarios", {}) or {})
            required = required_scenario_ids()
            for sid in required:
                self.assertIn(sid, baseline_scenarios, msg=f"Baseline missing required scenario={sid}")
                for stage in PERF_REQUIRED_STAGES:
                    self.assertIn(
                        stage,
                        baseline_scenarios[sid],
                        msg=f"Baseline missing required stage={stage} for scenario={sid}",
                    )
                    self.assertGreater(
                        float(current["scenarios"][sid][stage].get("count", 0.0) or 0.0),
                        0.0,
                        msg=f"Current strict perf metric has zero count sid={sid} stage={stage}",
                    )
                    self.assertGreater(
                        float(current["scenarios"][sid][stage].get("p95_ms", 0.0) or 0.0),
                        0.0,
                        msg=f"Current strict perf metric invalid/placeholder sid={sid} stage={stage}",
                    )

            for sid in required:
                for stage in PERF_REQUIRED_STAGES:
                    cur_p95 = float(current["scenarios"][sid][stage]["p95_ms"])
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
            for sid in tuple(str(v) for v in PERF_STRICT_PREVIEW_SCENARIOS):
                preview_p95 = float(current["scenarios"][sid]["preview"]["p95_ms"])
                self.assertLessEqual(
                    preview_p95,
                    float(PERF_STRICT_PREVIEW_P95_MS),
                    msg=(
                        f"Realtime preview gate failed sid={sid}: "
                        f"preview p95={preview_p95:.3f}ms > {float(PERF_STRICT_PREVIEW_P95_MS):.3f}ms"
                    ),
                )
                diag = dict(current["scenarios"][sid].get("realtime", {}) or {})
                underrun_rate = float(diag.get("preview_underrun_rate", 0.0) or 0.0)
                queue_drop_rate = float(diag.get("preview_queue_drop_rate", 0.0) or 0.0)
                self.assertLessEqual(
                    underrun_rate,
                    float(PERF_STRICT_PREVIEW_UNDERRUN_RATE_MAX),
                    msg=(
                        f"Realtime underrun gate failed sid={sid}: "
                        f"underrun_rate={underrun_rate:.4f} > {float(PERF_STRICT_PREVIEW_UNDERRUN_RATE_MAX):.4f}"
                    ),
                )
                self.assertLessEqual(
                    queue_drop_rate,
                    float(PERF_STRICT_PREVIEW_QUEUE_DROP_RATE_MAX),
                    msg=(
                        f"Realtime queue-drop gate failed sid={sid}: "
                        f"queue_drop_rate={queue_drop_rate:.4f} > {float(PERF_STRICT_PREVIEW_QUEUE_DROP_RATE_MAX):.4f}"
                    ),
                )
        finally:
            try:
                if current_path.exists():
                    current_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
