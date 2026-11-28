"""Cross-check the reference curves against independent tools and known signals.

Runs only where librosa and pyloudnorm are installed (tests/requirements-reference.txt);
they are test tools and never ship. The curves themselves are the committed goldens
in tests/reference/curves, so this needs no Blender.

Each check runs per frame rate. A check ACE fails today is listed in KNOWN_FAILURES
with what was measured, and runs as an expected failure: it shows up as
"unexpected success" once the cause is fixed, and then leaves the list.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "reference"
sys.path.insert(0, str(REFERENCE))

import fixtures  # noqa: E402
import spec  # noqa: E402

try:
    import librosa  # type: ignore
    import pyloudnorm  # type: ignore
    TOOLS = True
except ImportError:
    TOOLS = False

# (check, fps) -> what is measured today, for checks ACE fails. None at present.
KNOWN_FAILURES: dict = {}


def _golden(name: str, fps: int) -> dict:
    with np.load(REFERENCE / "curves" / spec.golden_name(name, fps)) as data:
        return {k: data[k] for k in data.files}


def _frame_centres(n: int, fps: int) -> np.ndarray:
    """Time at the middle of each frame's block of audio."""
    return (np.arange(n) + 0.5) / float(fps)


def _sweep_frequency(t: np.ndarray) -> np.ndarray:
    f0, f1 = 50.0, 12000.0
    return f0 * np.exp(np.log(f1 / f0) / fixtures.DURATION_S * t)


def _ace_stft(y: np.ndarray, sr: int, fps: int):
    """Magnitude spectra on ACE's grid: one per frame, FFT size the next power of two of the block.

    Each window ends with its frame's block and reaches back before it (zeros before the start).
    """
    hop = int(round(sr / fps))
    n_fft = 1 << int(np.ceil(np.log2(hop)))
    padded = np.concatenate((np.zeros(n_fft - hop, dtype=y.dtype), y))
    mags = np.abs(librosa.stft(padded, n_fft=n_fft, hop_length=hop, center=False))
    times = (np.arange(mags.shape[1]) * hop + hop / 2.0) / sr
    return mags, times


# The 0..1 log scales of Pitch and Brightness (core/tasks/stream.py), back to Hz; 0 is no frequency.
PITCH_RANGE_HZ = (20.0, 5000.0)
BRIGHTNESS_RANGE_HZ = (20.0, 20000.0)


def _hz(curve: np.ndarray, low: float, high: float) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    return np.where(curve <= 0.0, 0.0, low * (high / low) ** np.clip(curve, 0.0, 1.0))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# Each check: (fps) -> (passed, measured description).

def check_lufs_matches_pyloudnorm(fps):
    """LUFS over the 0.4 s window ending each frame, against pyloudnorm, on rising pink noise."""
    curve = _golden("pink_noise", fps)["audio.loudness.lufs/Loudness"]
    y, sr = fixtures.render("pink_noise")
    meter = pyloudnorm.Meter(sr, block_size=0.4)
    errors = []
    for i, value in enumerate(curve):
        end = int(round((i + 1) * sr / fps))
        start = end - int(0.4 * sr)
        if start < 0:
            continue
        ref = float(np.clip((meter.integrated_loudness(y[start:end, 0]) + 70.0) / 70.0, 0.0, 1.0))
        errors.append(abs(float(value) - ref) * 70.0)
    worst = max(errors)
    return worst <= 0.5, f"max error {worst:.2f} dB"


def check_transients_match_clicks(fps):
    """Transient peaks against the click times, within one frame."""
    curve = _golden("click_track", fps)["audio.transient/Transient"]
    clicks = np.floor(np.arange(0.25, fixtures.DURATION_S - 0.01, 60.0 / fixtures.BPM) * fps).astype(int)
    peaks = [
        i for i in range(len(curve))
        if curve[i] >= 0.5 and (i == 0 or curve[i] > curve[i - 1]) and (i == len(curve) - 1 or curve[i] >= curve[i + 1])
    ]
    precision = np.mean([np.any(np.abs(clicks - p) <= 1) for p in peaks]) if peaks else 0.0
    recall = np.mean([any(abs(p - c) <= 1 for p in peaks) for c in clicks])
    f = 2 * precision * recall / max(precision + recall, 1e-9)
    return f >= 0.95, f"F-measure {f:.2f} ({len(peaks)} peaks, {len(clicks)} clicks)"


def check_tempo_matches_click_track(fps):
    values = _golden("click_track", fps)["input.src/BPM"]
    worst = float(np.max(np.abs(values - fixtures.BPM)) / fixtures.BPM)
    return worst <= 0.02, f"BPM {np.unique(values)} for {fixtures.BPM}"


def check_tempo_matches_music(fps):
    values = _golden("music_synth", fps)["input.src/BPM"]
    worst = float(np.max(np.abs(values - fixtures.BPM)) / fixtures.BPM)
    return worst <= 0.02, f"BPM {np.unique(values)} for {fixtures.BPM}"


def check_meter_of_music(fps):
    """music_synth is two bars of 4/4."""
    values = _golden("music_synth", fps)["input.src/Beats per Bar"]
    return bool(np.all(values == 4)), f"beats per bar {np.unique(values)}"


def check_centroid_tracks_sweep(fps):
    """On a pure sweep the centroid is the sweep frequency."""
    curve = _hz(_golden("sine_sweep", fps)["audio.brightness/Brightness"], *BRIGHTNESS_RANGE_HZ)
    f = _sweep_frequency(_frame_centres(len(curve), fps))
    band = (f > 200.0) & (f < 8000.0)
    octaves = np.abs(np.log2(np.maximum(curve[band], 1e-9) / f[band]))
    median = float(np.median(octaves))
    return median <= 0.1, f"median error {median:.3f} octave"


def check_centroid_matches_librosa(fps):
    curve = _hz(_golden("music_synth", fps)["audio.brightness/Brightness"], *BRIGHTNESS_RANGE_HZ)
    y, sr = fixtures.render("music_synth")
    mags, times = _ace_stft(y.mean(axis=1), sr, fps)
    ref = librosa.feature.spectral_centroid(S=mags, sr=sr)[0]
    r = _pearson(np.interp(_frame_centres(len(curve), fps), times, ref), curve)
    return r >= 0.9, f"Pearson r {r:.3f}"


def check_pitch_tracks_sweep(fps):
    """At least 90% of frames between 80 Hz and 1 kHz within 50 cents of the sweep."""
    curve = _hz(_golden("sine_sweep", fps)["audio.pitch/Pitch"], *PITCH_RANGE_HZ)
    f = _sweep_frequency(_frame_centres(len(curve), fps))
    band = (f > 80.0) & (f < 1000.0)
    cents = 1200.0 * np.abs(np.log2(np.maximum(curve[band], 1e-9) / f[band]))
    share = float(np.mean(cents <= 50.0))
    return share >= 0.9, f"{share:.0%} of frames within 50 cents"


CHECKS = {
    name[len("check_"):]: fn for name, fn in sorted(globals().items()) if name.startswith("check_") and callable(fn)
}


@unittest.skipUnless(TOOLS, "librosa and pyloudnorm are not installed (tests/requirements-reference.txt)")
class ReferenceCrossCheck(unittest.TestCase):
    pass


def _make_test(check, fps):
    def test(self):
        passed, measured = CHECKS[check](fps)
        self.assertTrue(passed, f"{check} at {fps} fps: {measured}")
    test.__doc__ = f"{check} at {fps} fps"
    if (check, fps) in KNOWN_FAILURES:
        test = unittest.expectedFailure(test)
    return test


for _check in CHECKS:
    for _fps in spec.FPS:
        setattr(ReferenceCrossCheck, f"test_{_check}_{_fps}fps", _make_test(_check, _fps))


def measure() -> None:
    """Print every check's measurement (python tests/test_reference_crosscheck.py --measure)."""
    for check in CHECKS:
        for fps in spec.FPS:
            passed, measured = CHECKS[check](fps)
            known = " (known failure)" if (check, fps) in KNOWN_FAILURES else ""
            print(f"{'PASS' if passed else 'FAIL'} {check:28s} {fps:3d} fps  {measured}{known}")


if __name__ == "__main__":
    if "--measure" in sys.argv:
        measure()
    else:
        unittest.main()
