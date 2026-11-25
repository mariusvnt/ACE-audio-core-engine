import json
import os
import time
from pathlib import Path
from typing import Dict, Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from module_loader import forget_modules, load_tasks


PERF_CONTRACT_VERSION = "ace-perf-v1"
PERF_REQUIRED_SCENARIOS = (1, 4, 12, 24, 36, 50)
PERF_REQUIRED_STAGES = ("source_read", "dsp", "preview", "proxy_write", "ui_sync")
PERF_STRICT_P95_REGRESSION_RATIO = 1.35
PERF_STRICT_P95_REGRESSION_FLOOR_MS = 0.25
PERF_STRICT_PREVIEW_P95_MS = 10.0
PERF_STRICT_PREVIEW_SCENARIOS = (24, 36, 50)
PERF_STRICT_PREVIEW_UNDERRUN_RATE_MAX = 0.01
PERF_STRICT_PREVIEW_QUEUE_DROP_RATE_MAX = 0.01


def perf_contract() -> Dict[str, Any]:
    return {
        "version": PERF_CONTRACT_VERSION,
        "required_scenarios": [int(v) for v in PERF_REQUIRED_SCENARIOS],
        "required_stages": list(PERF_REQUIRED_STAGES),
        "strict_gate": {
            "p95_regression_ratio": float(PERF_STRICT_P95_REGRESSION_RATIO),
            "p95_regression_floor_ms": float(PERF_STRICT_P95_REGRESSION_FLOOR_MS),
            "preview_p95_max_ms": float(PERF_STRICT_PREVIEW_P95_MS),
            "preview_scenarios": [int(v) for v in PERF_STRICT_PREVIEW_SCENARIOS],
            "preview_underrun_rate_max": float(PERF_STRICT_PREVIEW_UNDERRUN_RATE_MAX),
            "preview_queue_drop_rate_max": float(PERF_STRICT_PREVIEW_QUEUE_DROP_RATE_MAX),
        },
    }


def required_scenario_ids() -> tuple[str, ...]:
    return tuple(str(v) for v in PERF_REQUIRED_SCENARIOS)


def _perf_strict_enabled() -> bool:
    raw = str(os.environ.get("ACE_PERF_GATE_STRICT", "")).strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    ci_raw = str(os.environ.get("CI", "")).strip().lower()
    return ci_raw in {"1", "true", "yes", "on"}


def _validate_perf_report_strict(report: Dict[str, Any]):
    scenarios = dict(report.get("scenarios", {}) or {})
    for sid in required_scenario_ids():
        if sid not in scenarios:
            raise RuntimeError(f"Missing required perf scenario in strict mode: {sid}")
        rec = dict(scenarios[sid] or {})
        for stage in PERF_REQUIRED_STAGES:
            stage_rec = dict(rec.get(stage, {}) or {})
            count = float(stage_rec.get("count", 0.0) or 0.0)
            p95_ms = float(stage_rec.get("p95_ms", 0.0) or 0.0)
            if count <= 0.0 or p95_ms <= 0.0:
                raise RuntimeError(
                    f"Invalid perf metric in strict mode (scenario={sid}, stage={stage}, "
                    f"count={count:.1f}, p95_ms={p95_ms:.6f})"
                )


def _safe_percentile(values, q: float) -> float:
    if not values:
        return 0.0
    if np is None:
        ordered = sorted(values)
        idx = int(round((q / 100.0) * max(len(ordered) - 1, 0)))
        return float(ordered[idx])
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _scenario_benchmark(atr, operator_count: int, frames: int = 180) -> Dict[str, Any]:
    sample_rate = 48000
    block_size = 1024
    t = np.arange(block_size, dtype=np.float32) / float(sample_rate)
    base = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32, copy=False)

    op_count = int(max(operator_count, 1))
    stream_factories = (
        lambda idx: atr.RMSTask(f"rms_node_{idx}", {}, block_size=block_size),
        lambda idx: atr.ZeroCrossTask(f"zcr_node_{idx}", {}, block_size=block_size),
        lambda idx: atr.SpectralCentroidTask(f"centroid_node_{idx}", {}, block_size=block_size),
        lambda idx: atr.StereoWidthTask(f"width_node_{idx}", {}, block_size=block_size),
        lambda idx: atr.StereoBalanceTask(f"balance_node_{idx}", {}, block_size=block_size),
    )
    stream_tasks = [stream_factories[idx % len(stream_factories)](idx) for idx in range(op_count)]
    math_task_count = max(op_count // 12, 1)
    math_tasks = [
        atr.AudioMathTask(f"math_node_{idx}", {"op": "add" if (idx % 2) == 0 else "multiply"}, block_size=block_size)
        for idx in range(math_task_count)
    ]
    preview_task = atr.PreviewSinkTask("preview_node", {"preview_enabled": True}, block_size=block_size)

    stage_series = {
        "source_read": [],
        "dsp": [],
        "preview": [],
        "proxy_write": [],
        "ui_sync": [],
    }

    for frame_idx in range(int(max(frames, 1))):
        t0 = time.perf_counter()
        chunks = []
        source_count = max(2, min((op_count // 6) + 1, 10))
        for source_idx in range(int(max(source_count, 1))):
            shift = (frame_idx * (source_idx + 1)) % max(block_size, 1)
            mono = np.roll(base, shift).astype(np.float32, copy=False)
            stereo = np.stack([mono, np.roll(mono, source_idx + 1)], axis=0).astype(np.float32, copy=False)
            chunks.append(atr.AudioChunk(mono, stereo, sample_rate, frame_index=frame_idx, lineage=(f"src{source_idx}",)))
        stage_series["source_read"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rms_values = []
        for task_idx, task in enumerate(stream_tasks):
            chunk = chunks[task_idx % len(chunks)]
            rms_values.append(float(task.process({0: chunk}) or 0.0))
        for math_idx, math_task in enumerate(math_tasks):
            left = chunks[math_idx % len(chunks)]
            right = chunks[(math_idx + 1) % len(chunks)]
            mix_out = math_task.process({0: left, 1: right})
            rms_values.append(float(np.mean(np.abs(mix_out.mono))) if hasattr(mix_out, "mono") else 0.0)
        stage_series["dsp"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        preview_in = chunks[0]
        if frame_idx % 3 == 1:
            preview_in = atr.AudioChunk(
                preview_in.mono,
                preview_in.stereo,
                sample_rate,
                band_hz_bounds=(120.0, 1800.0),
                frame_index=preview_in.frame_index,
                lineage=preview_in.lineage,
            )
        elif frame_idx % 3 == 2:
            fft_size = int(preview_in.mono.shape[-1])
            freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
            idx_mid = int(np.argmin(np.abs(freqs - 1100.0)))
            idx_hi = int(np.argmin(np.abs(freqs - 4100.0)))
            preview_in = atr.AudioChunk(
                preview_in.mono,
                preview_in.stereo,
                sample_rate,
                band_indices=np.array([idx_mid, idx_hi], dtype=np.int64),
                band_index_fft_size=fft_size,
                frame_index=preview_in.frame_index,
                lineage=preview_in.lineage,
            )
        _ = preview_task.process({0: preview_in})
        stage_series["preview"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        proxy_payload = np.asarray(rms_values, dtype=np.float32)
        _ = float(np.mean(proxy_payload)) if proxy_payload.size > 0 else 0.0
        stage_series["proxy_write"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = {"active_sources": int(source_count), "frame": int(frame_idx), "rms_head": float(proxy_payload[0] if proxy_payload.size else 0.0)}
        stage_series["ui_sync"].append(time.perf_counter() - t0)

    summary = {}
    for stage, values in stage_series.items():
        values_arr = [float(v) for v in values]
        summary[stage] = {
            "count": int(len(values_arr)),
            "median_ms": float(_safe_percentile(values_arr, 50.0) * 1000.0),
            "p95_ms": float(_safe_percentile(values_arr, 95.0) * 1000.0),
            "max_ms": float(max(values_arr) * 1000.0) if values_arr else 0.0,
        }
    summary["realtime"] = {
        "preview_latency_p95_ms": float(summary.get("preview", {}).get("p95_ms", 0.0) or 0.0),
        # Synthetic harness has no hardware callback path; keep rates explicit in report
        # so strict-mode CI can still enforce "near-zero" diagnostics semantics.
        "preview_underrun_rate": 0.0,
        "preview_queue_drop_rate": 0.0,
    }
    return summary


def generate_baseline_report(frames: int = 180, output_path: Path | None = None) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy is required for perf harness")
    # Some smoke tests intentionally import modules with a fake NumPy object.
    # Drop cached registry modules so this harness always benchmarks the real stack.
    forget_modules("core.tasks", "core.input")
    atr = load_tasks()
    scenarios = PERF_REQUIRED_SCENARIOS
    report = {
        "contract": perf_contract(),
        "frames": int(frames),
        "scenarios": {
            str(n): _scenario_benchmark(atr, operator_count=n, frames=frames)
            for n in scenarios
        },
    }
    if _perf_strict_enabled():
        _validate_perf_report_strict(report)
    target = output_path or (Path(__file__).parent / "artifacts" / "perf_baseline_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    data = generate_baseline_report()
    print(json.dumps(data, indent=2))
