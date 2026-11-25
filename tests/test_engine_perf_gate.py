import os
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
except Exception:
    np = None

from engine_perf_harness import (
    ENGINE_PERF_REQUIRED_SCENARIOS,
    ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS,
    ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS,
    ENGINE_PERF_SLA_SAMPLE_RATE_HZ,
    ENGINE_PERF_STRICT_DSP_RATIO_MAX,
    ENGINE_PERF_STRICT_DSP_SCENARIOS,
    ENGINE_PERF_STRICT_FRESH_RESULT_RATE_MIN,
    ENGINE_PERF_STRICT_QUEUE_DROP_RATE_MAX,
    generate_engine_perf_report,
)


def _strict_mode_enabled() -> bool:
    raw = str(os.environ.get("ACE_ENGINE_PERF_GATE_STRICT", "")).strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    ci_raw = str(os.environ.get("CI", "")).strip().lower()
    return ci_raw in {"1", "true", "yes", "on"}


# Perf runs are opt-in (ACE_PERF=1): they are slow and measure the machine as
# much as the code, so the default unit run leaves them out.
@unittest.skipUnless(os.environ.get("ACE_PERF") == "1", "perf gates run with ACE_PERF=1")
@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestEnginePerfGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(prefix="ace_engine_perf_") as tmp_dir:
            cls.report = generate_engine_perf_report(output_path=Path(tmp_dir) / "engine_perf_report.json")

    def test_engine_perf_report_has_required_scenarios_and_modes(self):
        scenarios = dict(self.report.get("scenarios", {}) or {})
        for sid in tuple(str(v) for v in ENGINE_PERF_REQUIRED_SCENARIOS):
            self.assertIn(sid, scenarios)
            self.assertIn("threads_io", scenarios[sid])
            self.assertIn("processes_dsp", scenarios[sid])
            self.assertIn("comparison", scenarios[sid])
            proc = dict(scenarios[sid]["processes_dsp"] or {})
            self.assertIn("stages", proc)
            self.assertIn("realtime", proc)
            self.assertIn("parallel_meta", proc)
            self.assertGreater(float(proc["parallel_meta"].get("parallel_offload_attr_count", 0.0) or 0.0), 0.0)

    def test_sla_topology_report_exists(self):
        sla = dict(self.report.get("sla_topology", {}) or {})
        self.assertIn("modes", sla)
        self.assertIn("evaluation", sla)
        modes = dict(sla.get("modes", {}) or {})
        self.assertIn("threads_io", modes)
        self.assertIn("processes_dsp", modes)
        eval_rec = dict(sla.get("evaluation", {}) or {})
        self.assertEqual(int(eval_rec.get("sample_rate_hz", 0) or 0), int(ENGINE_PERF_SLA_SAMPLE_RATE_HZ))
        self.assertLessEqual(
            float(eval_rec.get("ideal_threshold_ms", 0.0) or 0.0),
            float(ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS),
        )
        self.assertLessEqual(
            float(eval_rec.get("pass_threshold_ms", 0.0) or 0.0),
            float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS),
        )
        self.assertGreater(float(eval_rec.get("total_latency_p95_ms", 0.0) or 0.0), 0.0)

    def test_engine_perf_strict_gate(self):
        if not _strict_mode_enabled():
            return
        scenarios = dict(self.report.get("scenarios", {}) or {})
        for sid in tuple(str(v) for v in ENGINE_PERF_STRICT_DSP_SCENARIOS):
            cmp_rec = dict(scenarios[sid].get("comparison", {}) or {})
            self.assertTrue(
                bool(cmp_rec.get("dsp_metrics_valid", False)),
                msg=(
                    f"Engine DSP strict gate invalid sid={sid}: "
                    f"threads/process DSP metrics are zero or missing "
                    f"(threads={float(cmp_rec.get('threads_dsp_p95_ms', 0.0) or 0.0):.3f}ms, "
                    f"processes={float(cmp_rec.get('processes_dsp_p95_ms', 0.0) or 0.0):.3f}ms)"
                ),
            )
            self.assertGreater(
                float(cmp_rec.get("threads_dsp_p95_ms", 0.0) or 0.0),
                0.0,
                msg=f"Invalid threads DSP metric sid={sid}: zero/placeholder value",
            )
            self.assertGreater(
                float(cmp_rec.get("processes_dsp_p95_ms", 0.0) or 0.0),
                0.0,
                msg=f"Invalid processes DSP metric sid={sid}: zero/placeholder value",
            )
            ratio = float(cmp_rec.get("dsp_p95_ratio_processes_vs_threads", 999.0) or 999.0)
            self.assertLessEqual(
                ratio,
                float(ENGINE_PERF_STRICT_DSP_RATIO_MAX),
                msg=(
                    f"Engine DSP strict gate failed sid={sid}: "
                    f"processes/threads dsp p95 ratio={ratio:.3f} > {float(ENGINE_PERF_STRICT_DSP_RATIO_MAX):.3f}"
                ),
            )

            proc = dict(scenarios[sid].get("processes_dsp", {}) or {})
            rt = dict(proc.get("realtime", {}) or {})
            self.assertLessEqual(
                float(rt.get("parallel_job_timeouts", 0.0) or 0.0),
                0.0,
                msg=f"Parallel timeouts detected sid={sid}",
            )
            self.assertLessEqual(
                float(rt.get("parallel_job_exceptions", 0.0) or 0.0),
                0.0,
                msg=f"Parallel exceptions detected sid={sid}",
            )
            queue_drop_rate = float(rt.get("parallel_queue_drop_rate", 0.0) or 0.0)
            self.assertLessEqual(
                queue_drop_rate,
                float(ENGINE_PERF_STRICT_QUEUE_DROP_RATE_MAX),
                msg=(
                    f"Parallel queue-drop rate too high sid={sid}: "
                    f"{queue_drop_rate:.3f} > {float(ENGINE_PERF_STRICT_QUEUE_DROP_RATE_MAX):.3f}"
                ),
            )
            fresh_result_rate = float(rt.get("parallel_fresh_result_rate", 0.0) or 0.0)
            self.assertGreaterEqual(
                fresh_result_rate,
                float(ENGINE_PERF_STRICT_FRESH_RESULT_RATE_MIN),
                msg=(
                    f"Parallel fresh-result rate too low sid={sid}: "
                    f"{fresh_result_rate:.3f} < {float(ENGINE_PERF_STRICT_FRESH_RESULT_RATE_MIN):.3f}"
                ),
            )

        sla = dict(self.report.get("sla_topology", {}) or {})
        eval_rec = dict(sla.get("evaluation", {}) or {})
        total_latency = float(eval_rec.get("total_latency_p95_ms", 0.0) or 0.0)
        self.assertGreater(total_latency, 0.0, msg="SLA total latency metric is invalid/placeholder")
        self.assertLess(
            total_latency,
            float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS),
            msg=(
                f"SLA strict gate failed: total latency p95={total_latency:.3f}ms "
                f">= {float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS):.3f}ms"
            ),
        )


if __name__ == "__main__":
    unittest.main()
