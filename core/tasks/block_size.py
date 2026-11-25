"""Block-size resolution for analysis and preview."""

import math
import numpy as np
from ... import utils
from .chunk import MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, SAMPLE_RATE


_ADAPTIVE_FILE_BLOCK_BUCKETS = (
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)


_ADAPTIVE_LIVE_BLOCK_BUCKETS = (
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)


_PREVIEW_BLOCK_BUCKETS = (
    128,
    256,
    512,
)


def _quantize_adaptive_block_size(requested_samples: int, source_type: str = "FILE") -> int:
    req = max(int(requested_samples or 0), 1)
    src_type = str(source_type or "").strip().upper()
    buckets = _ADAPTIVE_LIVE_BLOCK_BUCKETS if src_type == "LIVE" else _ADAPTIVE_FILE_BLOCK_BUCKETS
    arr = np.asarray(buckets, dtype=np.int64)
    idx = int(np.argmin(np.abs(arr - req)))
    return int(arr[idx])


def resolve_preview_block_size(requested_samples: int) -> int:
    req = max(int(requested_samples or 0), 1)
    arr = np.asarray(_PREVIEW_BLOCK_BUCKETS, dtype=np.int64)
    idx = int(np.argmin(np.abs(arr - req)))
    return int(arr[idx])


def resolve_context_block_size(
    sample_rate: int,
    fps: float,
    source_type: str = "FILE",
    timeslice_mode: str = "fixed",
    delta_frames: float = 1.0,
) -> int:
    sr = max(int(sample_rate or SAMPLE_RATE), 1)
    fps_val = float(fps) if float(fps) > 0.0 else 24.0
    src_type = str(source_type or "").strip().upper()
    mode = str(timeslice_mode or "fixed").strip().lower()

    nominal = int(round(sr / fps_val)) if fps_val > 0.0 else sr
    nominal = max(MIN_BLOCK_SIZE, min(nominal, MAX_BLOCK_SIZE))

    if mode != "adaptive":
        if src_type == "LIVE":
            if nominal <= 768:
                return 512
            if nominal <= 1536:
                return 1024
            return 2048
        return nominal

    try:
        delta_mag = abs(float(delta_frames))
    except Exception as exc:
        utils.note_swallowed("audio_tasks_registry.resolve_context_block_size", exc)
        delta_mag = 1.0
    if not math.isfinite(delta_mag):
        delta_mag = 1.0

    requested = int(round((float(sr) * float(delta_mag)) / max(fps_val, 1e-6))) if delta_mag > 1e-9 else 0
    if requested <= 0:
        requested = nominal
    requested = max(MIN_BLOCK_SIZE, min(requested, MAX_BLOCK_SIZE))
    return _quantize_adaptive_block_size(requested, source_type=src_type)


def _resolve_block_size_default(
    sample_rate: int,
    fps: float,
    source_type: str = "FILE",
    timeslice_mode: str = "fixed",
    delta_frames: float = 1.0,
) -> int:
    return resolve_context_block_size(
        sample_rate,
        fps,
        source_type=source_type,
        timeslice_mode=timeslice_mode,
        delta_frames=delta_frames,
    )


def _block_duration_ms(sample_rate: int | float, block_size: int | float) -> float:
    sr = max(float(sample_rate), 1.0)
    bs = max(float(block_size), 0.0)
    return float((bs / sr) * 1000.0)
