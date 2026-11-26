"""Biquad, one-pole and envelope filter kernels."""

from typing import List, Tuple
import numpy as np
from ... import deps
NUMBA_AVAILABLE = deps.numba_enabled()
if NUMBA_AVAILABLE:
    from numba import njit  # type: ignore
try:
    from scipy.signal import lfilter as scipy_lfilter  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_lfilter = None
    SCIPY_AVAILABLE = False
from ... import utils


# Warn once per session when the pure-Python biquad runs (no numba, no scipy).
_BIQUAD_PYTHON_FALLBACK_WARNED: bool = False


def reset_biquad_fallback_warned() -> None:
    """Reset the one-time Tier-3 biquad fallback warning flag."""
    global _BIQUAD_PYTHON_FALLBACK_WARNED
    _BIQUAD_PYTHON_FALLBACK_WARNED = False


if NUMBA_AVAILABLE:
    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _signed_smooth_block(data: np.ndarray, attack_c: float, release_c: float, smooth_state: float):
        out = np.empty_like(data)
        smoothed = smooth_state
        for i in range(data.shape[0]):
            x = float(data[i])
            coeff = attack_c if abs(x) > abs(smoothed) else release_c
            smoothed = coeff * smoothed + (1.0 - coeff) * x
            out[i] = smoothed
        return out, smoothed

    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _biquad_filter_block(
        data: np.ndarray,
        b0: float,
        b1: float,
        b2: float,
        a1: float,
        a2: float,
        z1: float,
        z2: float,
    ):
        out = np.empty_like(data)
        s1 = z1
        s2 = z2
        for i in range(data.shape[0]):
            x = float(data[i])
            y = (b0 * x) + s1
            s1 = (b1 * x) - (a1 * y) + s2
            s2 = (b2 * x) - (a2 * y)
            out[i] = y
        return out, s1, s2

    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _iir1_filter_block(
        data: np.ndarray,
        b0: float,
        b1: float,
        a1: float,
        x_prev: float,
        y_prev: float,
    ):
        out = np.empty_like(data)
        x1 = x_prev
        y1 = y_prev
        for i in range(data.shape[0]):
            x = float(data[i])
            y = (b0 * x) + (b1 * x1) - (a1 * y1)
            out[i] = y
            x1 = x
            y1 = y
        return out, x1, y1
else:
    _biquad_filter_block = _signed_smooth_block = _iir1_filter_block = None


def _run_biquad(
    signal: np.ndarray,
    coeffs: Tuple[float, float, float, float, float],
    state: List[float],
    use_numba: bool = True,
) -> np.ndarray:
    """Apply a single Direct-Form-II biquad section to *signal* in-place on the state."""
    global _BIQUAD_PYTHON_FALLBACK_WARNED
    b0, b1, b2, a1, a2 = coeffs
    z1, z2 = float(state[0]), float(state[1])
    src = signal.astype(np.float32, copy=False)

    if use_numba and NUMBA_AVAILABLE:
        try:
            out, z1_new, z2_new = _biquad_filter_block(
                src,
                float(b0), float(b1), float(b2),
                float(a1), float(a2),
                float(z1), float(z2),
            )
            state[0] = float(z1_new)
            state[1] = float(z2_new)
            return out
        except Exception as exc:
            utils.note_swallowed("filters._run_biquad", exc)
            pass  # Fall through to Tier 2.

    # Converts Direct Form I coefficients to the [b0,b1,b2] / [1,a1,a2] vectors
    # scipy expects, and passes the DF-II state as initial conditions.
    if SCIPY_AVAILABLE:
        try:
            b_vec = np.array([b0, b1, b2], dtype=np.float64)
            a_vec = np.array([1.0, a1, a2], dtype=np.float64)
            zi = np.array([z1, z2], dtype=np.float64)  # zi shape must be (2,)
            out_64, zf = scipy_lfilter(b_vec, a_vec, src.astype(np.float64), zi=zi)
            state[0] = float(zf[0])
            state[1] = float(zf[1])
            return out_64.astype(np.float32, copy=False)
        except Exception as _exc:
            utils.log(
                f"[ACE-Preview] scipy biquad Tier-2 failed (falling to Python loop): {_exc!r}",
                "WARNING",
            )

    # Last resort.  At 512 samples/block this path takes ~10 ms on the preview
    # thread and will cause audible dropouts.  Emit a one-time WARNING.
    if not _BIQUAD_PYTHON_FALLBACK_WARNED:
        _BIQUAD_PYTHON_FALLBACK_WARNED = True
        utils.log(
            "ACE Layer-3 preview: Numba is off or failed to load. "
            "Using a slow Python biquad fallback that may cause audio dropouts. "
            "Turn Numba on in the add-on preferences (see its status there) to restore full preview performance.",
            "WARNING",
        )

    out = np.empty_like(src, dtype=np.float32)
    for i in range(src.shape[0]):
        x = float(src[i])
        y = (b0 * x) + z1
        z1 = (b1 * x) - (a1 * y) + z2
        z2 = (b2 * x) - (a2 * y)
        out[i] = y
    state[0] = z1
    state[1] = z2
    return out
