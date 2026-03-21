#!/usr/bin/env python3
"""
Run the full ACE test suite, capture results, and generate integrity/perf analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import unittest
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
ARTIFACTS_DIR = TESTS_DIR / "artifacts"

STATUS_ORDER = (
    "passed",
    "failed",
    "error",
    "skipped",
    "expected_failure",
    "unexpected_success",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _module_available(module_name: str) -> bool:
    """
    Robust module availability check.
    Some test harnesses install fake modules with __spec__ = None (notably bpy),
    which causes importlib.util.find_spec() to raise ValueError.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return module_name in sys.modules


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _module_filename(module_name: str) -> str:
    module = sys.modules.get(module_name)
    module_path = getattr(module, "__file__", None) if module is not None else None
    if module_path:
        return Path(module_path).name
    fallback = f"{module_name}.py"
    if (TESTS_DIR / fallback).exists():
        return fallback
    return module_name


def _category_for_test_file(filename: str) -> str:
    name = str(filename).lower()
    if "perf" in name:
        return "performance"
    if "smoke" in name or "blender" in name:
        return "integration"
    return "integrity"


@dataclass
class TestRecord:
    test_id: str
    module: str
    file: str
    category: str
    status: str
    duration_s: float
    message: str = ""


class CollectingTextResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: List[TestRecord] = []
        self._started: Dict[str, float] = {}

    def startTest(self, test):  # noqa: N802
        self._started[test.id()] = time.perf_counter()
        super().startTest(test)

    def _duration(self, test_id: str) -> float:
        start = self._started.pop(test_id, None)
        if start is None:
            return 0.0
        return max(time.perf_counter() - start, 0.0)

    def _record(self, test, status: str, message: str = ""):
        test_id = test.id()
        module_name = test_id.split(".", 1)[0]
        file_name = _module_filename(module_name)
        self.records.append(
            TestRecord(
                test_id=test_id,
                module=module_name,
                file=file_name,
                category=_category_for_test_file(file_name),
                status=status,
                duration_s=self._duration(test_id),
                message=message.strip(),
            )
        )

    def addSuccess(self, test):  # noqa: N802
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):  # noqa: N802
        super().addFailure(test, err)
        detail = self._exc_info_to_string(err, test).strip().splitlines()
        self._record(test, "failed", "\n".join(detail[-12:]))

    def addError(self, test, err):  # noqa: N802
        super().addError(test, err)
        detail = self._exc_info_to_string(err, test).strip().splitlines()
        self._record(test, "error", "\n".join(detail[-12:]))

    def addSkip(self, test, reason):  # noqa: N802
        super().addSkip(test, reason)
        self._record(test, "skipped", str(reason))

    def addExpectedFailure(self, test, err):  # noqa: N802
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure")

    def addUnexpectedSuccess(self, test):  # noqa: N802
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")


def _empty_counts() -> Dict[str, int]:
    counts = {key: 0 for key in STATUS_ORDER}
    counts["total"] = 0
    counts["executed"] = 0
    return counts


def _count_records(records: List[TestRecord]) -> Dict[str, int]:
    counts = _empty_counts()
    counts["total"] = len(records)
    for rec in records:
        counts[rec.status] = counts.get(rec.status, 0) + 1
    counts["executed"] = counts["total"] - counts["skipped"]
    return counts


def _pass_rate(counts: Dict[str, int]) -> float:
    executed = max(int(counts.get("executed", 0)), 0)
    if executed <= 0:
        return 0.0
    return (100.0 * float(counts.get("passed", 0)) / float(executed))


def _status_from_counts(counts: Dict[str, int], *, allow_blocked: bool = False) -> str:
    if int(counts.get("failed", 0)) > 0 or int(counts.get("error", 0)) > 0 or int(counts.get("unexpected_success", 0)) > 0:
        return "fail"
    total = int(counts.get("total", 0))
    executed = int(counts.get("executed", 0))
    if total <= 0:
        return "not_run"
    if executed <= 0:
        return "blocked" if allow_blocked else "pass"
    return "pass"


def _run_unittest_suite(verbosity: int) -> CollectingTextResult:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(
        stream=sys.stdout,
        verbosity=int(max(verbosity, 1)),
        resultclass=CollectingTextResult,
    )
    result = runner.run(suite)
    if not isinstance(result, CollectingTextResult):
        raise RuntimeError("Unexpected unittest result type")
    return result


def _run_perf_harnesses(artifacts_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ran": False,
        "status": "not_run",
        "errors": [],
        "outputs": {},
        "analysis": {},
    }
    numpy_available = _module_available("numpy")
    if not numpy_available:
        result["status"] = "blocked"
        result["errors"].append("NumPy unavailable; skipped direct perf harness generation")
        return result

    sys.path.insert(0, str(TESTS_DIR))
    try:
        from perf_harness import generate_baseline_report
        from engine_perf_harness import generate_engine_perf_report

        baseline_out = artifacts_dir / "perf_baseline_current.json"
        engine_out = artifacts_dir / "engine_perf_current.json"

        baseline_report = generate_baseline_report(frames=180, output_path=baseline_out)
        engine_report = generate_engine_perf_report(output_path=engine_out)
        result["ran"] = True
        result["outputs"] = {
            "baseline_report": str(baseline_out),
            "engine_report": str(engine_out),
        }
        result["analysis"] = {
            "baseline": _analyze_baseline_perf_report(baseline_report),
            "engine": _analyze_engine_perf_report(engine_report),
        }
        violations = []
        violations.extend(result["analysis"]["baseline"].get("violations", []))
        violations.extend(result["analysis"]["engine"].get("violations", []))
        result["status"] = "fail" if violations else "pass"
    except Exception as exc:
        result["status"] = "fail"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if sys.path and sys.path[0] == str(TESTS_DIR):
            sys.path.pop(0)

    return result


def _analyze_baseline_perf_report(report: Dict[str, Any]) -> Dict[str, Any]:
    contract = dict(report.get("contract", {}) or {})
    strict = dict(contract.get("strict_gate", {}) or {})
    scenarios = dict(report.get("scenarios", {}) or {})
    required_scenarios = [str(v) for v in contract.get("required_scenarios", [])]
    required_stages = [str(v) for v in contract.get("required_stages", [])]

    stage_peak_p95_ms: Dict[str, float] = {}
    missing_scenarios = [sid for sid in required_scenarios if sid not in scenarios]
    for stage in required_stages:
        peak = 0.0
        for sid in required_scenarios:
            rec = dict(scenarios.get(sid, {}) or {})
            stage_rec = dict(rec.get(stage, {}) or {})
            peak = max(peak, _as_float(stage_rec.get("p95_ms", 0.0)))
        stage_peak_p95_ms[stage] = float(round(peak, 6))

    violations: List[str] = []
    preview_limit = _as_float(strict.get("preview_p95_max_ms", 0.0))
    underrun_max = _as_float(strict.get("preview_underrun_rate_max", 0.0))
    queue_drop_max = _as_float(strict.get("preview_queue_drop_rate_max", 0.0))
    preview_scenarios = [str(v) for v in strict.get("preview_scenarios", [])]

    for sid in preview_scenarios:
        rec = dict(scenarios.get(sid, {}) or {})
        preview = dict(rec.get("preview", {}) or {})
        realtime = dict(rec.get("realtime", {}) or {})
        preview_p95 = _as_float(preview.get("p95_ms", 0.0))
        underrun = _as_float(realtime.get("preview_underrun_rate", 0.0))
        queue_drop = _as_float(realtime.get("preview_queue_drop_rate", 0.0))
        if preview_limit > 0.0 and preview_p95 > preview_limit:
            violations.append(
                f"baseline scenario {sid}: preview p95 {preview_p95:.3f}ms > {preview_limit:.3f}ms"
            )
        if underrun > underrun_max:
            violations.append(
                f"baseline scenario {sid}: underrun rate {underrun:.4f} > {underrun_max:.4f}"
            )
        if queue_drop > queue_drop_max:
            violations.append(
                f"baseline scenario {sid}: queue-drop rate {queue_drop:.4f} > {queue_drop_max:.4f}"
            )

    return {
        "contract_version": str(contract.get("version", "")),
        "required_scenarios": required_scenarios,
        "required_stages": required_stages,
        "missing_scenarios": missing_scenarios,
        "stage_peak_p95_ms": stage_peak_p95_ms,
        "violations": violations,
    }


def _analyze_engine_perf_report(report: Dict[str, Any]) -> Dict[str, Any]:
    contract = dict(report.get("contract", {}) or {})
    strict = dict(contract.get("strict_gate", {}) or {})
    scenarios = dict(report.get("scenarios", {}) or {})
    ratio_limit = _as_float(strict.get("dsp_p95_ratio_max", 0.0))
    queue_drop_limit = _as_float(strict.get("parallel_queue_drop_rate_max", 0.0))
    exact_hit_min = _as_float(strict.get("parallel_exact_frame_hit_rate_min", 0.0))
    sync_fallback_max = _as_float(strict.get("parallel_sync_fallback_rate_max", 0.0))
    ratio_scenarios = [str(v) for v in strict.get("dsp_ratio_scenarios", [])]

    max_ratio = 0.0
    max_queue_drop = 0.0
    min_exact_hit = 1.0
    max_sync_fallback = 0.0
    violations: List[str] = []

    for sid in ratio_scenarios:
        rec = dict(scenarios.get(sid, {}) or {})
        cmp_rec = dict(rec.get("comparison", {}) or {})
        proc_rec = dict(rec.get("processes_dsp", {}) or {})
        rt_rec = dict(proc_rec.get("realtime", {}) or {})

        ratio = _as_float(cmp_rec.get("dsp_p95_ratio_processes_vs_threads", 0.0))
        queue_drop = _as_float(rt_rec.get("parallel_queue_drop_rate", 0.0))
        exact_hit = _as_float(rt_rec.get("parallel_exact_frame_hit_rate", 0.0))
        sync_fallback = _as_float(rt_rec.get("parallel_sync_fallback_rate", 0.0))
        exceptions = _as_float(rt_rec.get("parallel_job_exceptions", 0.0))

        max_ratio = max(max_ratio, ratio)
        max_queue_drop = max(max_queue_drop, queue_drop)
        min_exact_hit = min(min_exact_hit, exact_hit)
        max_sync_fallback = max(max_sync_fallback, sync_fallback)

        if ratio_limit > 0.0 and ratio > ratio_limit:
            violations.append(f"engine scenario {sid}: dsp ratio {ratio:.3f} > {ratio_limit:.3f}")
        if queue_drop > queue_drop_limit:
            violations.append(
                f"engine scenario {sid}: queue-drop rate {queue_drop:.3f} > {queue_drop_limit:.3f}"
            )
        if exact_hit < exact_hit_min:
            violations.append(
                f"engine scenario {sid}: exact-frame hit rate {exact_hit:.3f} < {exact_hit_min:.3f}"
            )
        if sync_fallback > sync_fallback_max:
            violations.append(
                f"engine scenario {sid}: sync-fallback rate {sync_fallback:.3f} > {sync_fallback_max:.3f}"
            )
        if exceptions > 0.0:
            violations.append(f"engine scenario {sid}: parallel job exceptions {exceptions:.0f} > 0")

    sla_eval = dict(dict(report.get("sla_topology", {}) or {}).get("evaluation", {}) or {})
    sla_total = _as_float(sla_eval.get("total_latency_p95_ms", 0.0))
    sla_limit = _as_float(sla_eval.get("pass_threshold_ms", 0.0))
    if sla_total <= 0.0:
        violations.append("engine SLA total latency metric is invalid (<= 0)")
    elif sla_limit > 0.0 and sla_total >= sla_limit:
        violations.append(f"engine SLA total latency {sla_total:.3f}ms >= {sla_limit:.3f}ms")

    return {
        "contract_version": str(contract.get("version", "")),
        "scenarios_checked": ratio_scenarios,
        "max_dsp_ratio_processes_vs_threads": float(round(max_ratio, 6)),
        "max_queue_drop_rate": float(round(max_queue_drop, 6)),
        "min_exact_frame_hit_rate": float(round(min_exact_hit, 6)),
        "max_sync_fallback_rate": float(round(max_sync_fallback, 6)),
        "sla_total_latency_p95_ms": float(round(sla_total, 6)),
        "sla_threshold_ms": float(round(sla_limit, 6)),
        "violations": violations,
    }


def _top_slowest(records: List[TestRecord], limit: int = 10) -> List[Dict[str, Any]]:
    ordered = sorted(records, key=lambda rec: rec.duration_s, reverse=True)
    return [
        {
            "test_id": rec.test_id,
            "category": rec.category,
            "status": rec.status,
            "duration_s": float(round(rec.duration_s, 6)),
        }
        for rec in ordered[: int(max(limit, 0))]
    ]


def _build_markdown_report(report: Dict[str, Any]) -> str:
    summary = dict(report.get("summary", {}) or {})
    env = dict(report.get("environment", {}) or {})
    gates = dict(summary.get("gates", {}) or {})
    categories = dict(summary.get("categories", {}) or {})
    perf_harness = dict(report.get("performance_harness", {}) or {})
    failures = list(summary.get("failures", []) or [])
    slowest = list(summary.get("slowest_tests", []) or [])

    lines = []
    lines.append("# ACE Test Quality Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report.get('generated_utc', '')}`")
    lines.append(f"- Overall status: `{summary.get('overall_status', 'unknown')}`")
    lines.append(f"- Integrity gate: `{gates.get('integrity', 'unknown')}`")
    lines.append(f"- Performance gate: `{gates.get('performance', 'unknown')}`")
    lines.append(f"- Integration gate: `{gates.get('integration', 'unknown')}`")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: `{env.get('python_version', '')}`")
    lines.append(f"- Platform: `{env.get('platform', '')}`")
    lines.append(f"- NumPy available: `{env.get('numpy_available', False)}`")
    lines.append(f"- bpy available: `{env.get('bpy_available', False)}`")
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Status | Total | Executed | Passed | Failed | Error | Skipped | Pass Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for category in ("integrity", "performance", "integration"):
        rec = dict(categories.get(category, {}) or {})
        counts = dict(rec.get("counts", {}) or {})
        lines.append(
            f"| {category} | {rec.get('status', 'unknown')} | {counts.get('total', 0)} | "
            f"{counts.get('executed', 0)} | {counts.get('passed', 0)} | {counts.get('failed', 0)} | "
            f"{counts.get('error', 0)} | {counts.get('skipped', 0)} | {rec.get('pass_rate_pct', 0.0):.1f}% |"
        )
    lines.append("")
    lines.append("## Performance Harness")
    lines.append("")
    lines.append(f"- Harness status: `{perf_harness.get('status', 'not_run')}`")
    outputs = dict(perf_harness.get("outputs", {}) or {})
    if outputs:
        lines.append(f"- Baseline report: `{outputs.get('baseline_report', '')}`")
        lines.append(f"- Engine report: `{outputs.get('engine_report', '')}`")
    analysis = dict(perf_harness.get("analysis", {}) or {})
    baseline = dict(analysis.get("baseline", {}) or {})
    engine = dict(analysis.get("engine", {}) or {})
    if baseline:
        lines.append(f"- Baseline stage peak p95 (ms): `{baseline.get('stage_peak_p95_ms', {})}`")
    if engine:
        lines.append(
            "- Engine key metrics: "
            f"`ratio={engine.get('max_dsp_ratio_processes_vs_threads', 0.0):.3f}, "
            f"queue_drop={engine.get('max_queue_drop_rate', 0.0):.3f}, "
            f"fresh_min={engine.get('min_fresh_result_rate', 0.0):.3f}, "
            f"sla_p95_ms={engine.get('sla_total_latency_p95_ms', 0.0):.3f}`"
        )
    errors = list(perf_harness.get("errors", []) or [])
    if errors:
        lines.append("- Harness notes:")
        lines.extend([f"  - {item}" for item in errors])
    violations = []
    violations.extend(list(baseline.get("violations", []) or []))
    violations.extend(list(engine.get("violations", []) or []))
    if violations:
        lines.append("- Performance violations:")
        lines.extend([f"  - {item}" for item in violations])
    lines.append("")
    lines.append("## Failing Tests")
    lines.append("")
    if failures:
        for item in failures:
            lines.append(f"- `{item.get('test_id', '')}` ({item.get('status', '')})")
            if item.get("message"):
                lines.append(f"  - {item['message'].splitlines()[-1]}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Slowest Tests")
    lines.append("")
    if slowest:
        for item in slowest:
            lines.append(
                f"- `{item.get('test_id', '')}` "
                f"[{item.get('category', '')} | {item.get('status', '')}] "
                f"{item.get('duration_s', 0.0):.3f}s"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _build_summary(
    test_records: List[TestRecord],
    perf_harness: Dict[str, Any],
    fail_on_blocked: bool,
) -> Dict[str, Any]:
    by_category: Dict[str, List[TestRecord]] = {
        "integrity": [],
        "performance": [],
        "integration": [],
    }
    for rec in test_records:
        by_category.setdefault(rec.category, []).append(rec)

    category_summary: Dict[str, Dict[str, Any]] = {}
    for category, records in by_category.items():
        counts = _count_records(records)
        category_summary[category] = {
            "status": _status_from_counts(counts, allow_blocked=True),
            "counts": counts,
            "pass_rate_pct": float(round(_pass_rate(counts), 3)),
        }

    integrity_gate = str(category_summary.get("integrity", {}).get("status", "not_run"))
    integration_gate = str(category_summary.get("integration", {}).get("status", "not_run"))

    performance_tests_gate = str(category_summary.get("performance", {}).get("status", "not_run"))
    performance_harness_gate = str(perf_harness.get("status", "not_run"))
    if performance_tests_gate == "fail" or performance_harness_gate == "fail":
        performance_gate = "fail"
    elif performance_tests_gate == "blocked" or performance_harness_gate == "blocked":
        performance_gate = "blocked"
    elif performance_tests_gate == "not_run" and performance_harness_gate == "not_run":
        performance_gate = "not_run"
    else:
        performance_gate = "pass"

    overall_status = "pass"
    if integrity_gate == "fail" or integration_gate == "fail" or performance_gate == "fail":
        overall_status = "fail"
    elif performance_gate == "blocked" and fail_on_blocked:
        overall_status = "blocked"

    failures = []
    for rec in test_records:
        if rec.status in {"failed", "error", "unexpected_success"}:
            failures.append(asdict(rec))

    return {
        "overall_status": overall_status,
        "gates": {
            "integrity": integrity_gate,
            "performance": performance_gate,
            "integration": integration_gate,
            "performance_tests": performance_tests_gate,
            "performance_harness": performance_harness_gate,
        },
        "categories": category_summary,
        "failures": failures,
        "slowest_tests": _top_slowest(test_records, limit=10),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ACE tests and generate integrity/performance quality analysis."
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="unittest verbosity (1 or 2)",
    )
    parser.add_argument(
        "--skip-perf-harness",
        action="store_true",
        help="skip direct perf harness generation/analysis",
    )
    parser.add_argument(
        "--perf-warning-mode",
        action="store_true",
        help="do not force strict perf gate environment flags",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="allow blocked performance analysis without non-zero exit",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ARTIFACTS_DIR / "quality_report.json",
        help="path for JSON report output",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ARTIFACTS_DIR / "quality_report.md",
        help="path for Markdown report output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.perf_warning_mode:
        os.environ.setdefault("ACE_PERF_GATE_STRICT", "1")
        os.environ.setdefault("ACE_ENGINE_PERF_GATE_STRICT", "1")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    records_result = _run_unittest_suite(verbosity=int(max(args.verbosity, 1)))
    elapsed = max(time.perf_counter() - run_started, 0.0)

    perf_harness = {"status": "not_run", "errors": [], "analysis": {}, "outputs": {}, "ran": False}
    if not args.skip_perf_harness:
        perf_harness = _run_perf_harnesses(args.json_out.parent)

    test_records = [asdict(rec) for rec in records_result.records]
    summary = _build_summary(
        records_result.records,
        perf_harness,
        fail_on_blocked=(not bool(args.allow_blocked)),
    )

    environment = {
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_available": bool(_module_available("numpy")),
        "bpy_available": bool(_module_available("bpy")),
        "strict_perf_env": {
            "ACE_PERF_GATE_STRICT": str(os.environ.get("ACE_PERF_GATE_STRICT", "")),
            "ACE_ENGINE_PERF_GATE_STRICT": str(os.environ.get("ACE_ENGINE_PERF_GATE_STRICT", "")),
        },
    }

    report = {
        "generated_utc": _utc_now(),
        "duration_s": float(round(elapsed, 6)),
        "environment": environment,
        "summary": summary,
        "performance_harness": perf_harness,
        "tests": test_records,
    }

    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md_out.write_text(_build_markdown_report(report), encoding="utf-8")

    print("\nACE quality summary")
    print(f"- overall: {summary['overall_status']}")
    print(f"- integrity gate: {summary['gates']['integrity']}")
    print(f"- performance gate: {summary['gates']['performance']}")
    print(f"- integration gate: {summary['gates']['integration']}")
    print(f"- report json: {args.json_out}")
    print(f"- report md:   {args.md_out}")

    if summary["overall_status"] == "fail":
        return 1
    if summary["overall_status"] == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
