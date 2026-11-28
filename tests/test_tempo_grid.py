"""The beat grid fitted over a whole file (core/tasks/tempo.py): the BPM and downbeat that keep a beat clock on the beat.

A tempo estimated from the first 15 seconds is off by about 0.1 %: after 4 minutes, half a
beat. The fit runs a line through the onsets of the whole file.
"""

import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from module_loader import load_module
    import fake_bpy

STEP_S = 0.01  # the envelope's hop
ESTIMATE_STEPS = 1499  # what 15 s give


def _clicks(bpm: float, first_downbeat_s: float, seconds: float, beats_per_bar: int = 4, noise: float = 0.02, seed: int = 3,
            pickup: int = 0):
    """An onset envelope: a bump on every beat, twice as strong on each downbeat, after *pickup* beats before the first."""
    rng = np.random.default_rng(seed)
    env = rng.random(int(seconds / STEP_S)) * noise
    steps = np.arange(env.size, dtype=np.float64)
    period = 60.0 / bpm / STEP_S
    k = -int(pickup)
    while True:
        at = first_downbeat_s / STEP_S - 1.0 + k * period  # value i is the change at block i + 1
        if at >= env.size - 4:
            break
        lo, hi = max(int(at) - 6, 0), int(at) + 7
        strength = 2.0 if k % beats_per_bar == 0 else 1.0
        env[lo:hi] += strength * np.exp(-0.5 * ((steps[lo:hi] - at) / 1.5) ** 2)
        k += 1
    return env


@unittest.skipIf(np is None, "numpy unavailable")
class GridFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.tempo = load_module("core.tasks.tempo")

    def test_the_fit_keeps_a_four_minute_beat_clock_on_the_beat(self):
        bpm, downbeat = 120.36, 0.52  # 0.3 % off 120
        env = _clicks(bpm, downbeat, 240.0)
        found = self.tempo._tempo_of_envelope(env, STEP_S, ESTIMATE_STEPS)
        self.assertEqual(found["beats_per_bar"], 4)
        self.assertAlmostEqual(found["bpm"], bpm, delta=bpm * 2e-4)
        self.assertAlmostEqual(found["downbeat_s"], downbeat, delta=0.012)
        # Where the clock puts beat 470, near the end, against where it is: within a tenth of a beat.
        beat = 60.0 / bpm
        predicted = found["downbeat_s"] + 470 * 60.0 / found["bpm"]
        self.assertLess(abs(predicted - (downbeat + 470 * beat)), 0.1 * beat)

    def test_the_downbeat_is_the_first_of_the_file(self):
        # Bars of 2.4 s with downbeats at 0.52 s, 2.92 s...: the one reported is the first.
        env = _clicks(100.0, 2.92, 60.0, pickup=7)
        found = self.tempo._tempo_of_envelope(env, STEP_S, ESTIMATE_STEPS)
        self.assertAlmostEqual(found["downbeat_s"], 2.92 - 2.4, delta=0.03)

    def test_without_a_steady_pulse_the_estimate_stays(self):
        env = _clicks(120.0, 0.5, 60.0)
        rng = np.random.default_rng(9)
        tail = int(15.0 / STEP_S)
        env[tail:] = rng.random(env.size - tail) * 1.5  # the beat stops after 15 s: nothing to line up
        found = self.tempo._tempo_of_envelope(env, STEP_S, ESTIMATE_STEPS)
        estimate = self.tempo._tempo_of_envelope(env[:ESTIMATE_STEPS], STEP_S)
        self.assertAlmostEqual(found["bpm"], estimate["bpm"], delta=0.5)


if __name__ == "__main__":
    unittest.main()
