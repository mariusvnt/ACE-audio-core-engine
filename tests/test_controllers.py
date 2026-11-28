"""The controllers' maths (tests/controller_reference.py, which the node groups are checked against).

An input defined in seconds gives the same curve at any fps: every step is the exact solution
over its duration, for the input held during it. An input that changes at T0 is held from the
first frame after T0, so it is held over the same intervals at 24 and 48 fps when T0 is on both
grids.
"""

import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import controller_reference as ref  # noqa: E402

DURATION_S = 3.0
T0 = 0.5  # on the 24 and 48 fps grids


def _times(fps: int):
    return [frame / fps for frame in range(1, int(DURATION_S * fps) + 1)]


def _step(t: float) -> float:
    return 1.0 if t > T0 else 0.0


def _impulse(t: float) -> float:
    return 1.0 if T0 < t <= T0 + 1.0 / 24.0 else 0.0


def _at(fps: int, signal, **settings):
    times = _times(fps)
    out = ref.run_envelope([signal(t) for t in times], times, **settings)
    return dict(zip((round(t * 48) for t in times), out))  # keyed by the 48 fps frame


class EnvelopeTests(unittest.TestCase):
    def assert_same_curve(self, signal, **settings):
        slow, fast = _at(24, signal, **settings), _at(48, signal, **settings)
        for key, value in slow.items():
            self.assertAlmostEqual(value, fast[key], places=9, msg=f"{settings} at {key / 48:.4f} s")

    def test_follow_does_not_depend_on_fps(self):
        for signal in (_step, _impulse):
            for attack, hold, release in ((0.05, 0.0, 0.3), (0.0, 0.2, 0.5), (0.1, 0.1, 0.0)):
                self.assert_same_curve(signal, mode="Follow", attack=attack, hold=hold, release=release)

    def test_spring_does_not_depend_on_fps(self):
        for signal in (_step, _impulse):
            for bounce in (0.0, 0.5, 1.0):
                self.assert_same_curve(signal, mode="Spring", speed=3.0, bounce=bounce)

    def test_it_starts_at_rest_on_its_first_input(self):
        times = _times(24)
        for mode in ("Follow", "Spring"):
            out = ref.run_envelope([0.7] * len(times), times, mode=mode)
            self.assertTrue(all(abs(v - 0.7) < 1e-12 for v in out), mode)

    def test_attack_zero_is_a_peak_hold_then_release(self):
        times = _times(24)
        out = ref.run_envelope([_impulse(t) for t in times], times, attack=0.0, hold=0.25, release=0.1)
        peak = next(i for i, t in enumerate(times) if t > T0)
        self.assertEqual(out[peak], 1.0)
        held = [out[i] for i, t in enumerate(times) if T0 < t <= T0 + 1.0 / 24.0 + 0.25]
        self.assertTrue(all(v == 1.0 for v in held), held)
        later = out[next(i for i, t in enumerate(times) if t > T0 + 1.0 / 24.0 + 0.25 + 0.1)]
        self.assertLess(later, 0.5)

    def test_bounce_sets_the_overshoot(self):
        times = _times(48)
        overshoot = []
        for bounce in (0.0, 0.5, 1.0):
            out = ref.run_envelope([_step(t) for t in times], times, mode="Spring", speed=2.0, bounce=bounce)
            overshoot.append(max(out) - 1.0)
        self.assertLess(overshoot[0], 1e-3)  # critically damped: no visible overshoot
        self.assertLess(overshoot[0], overshoot[1])
        self.assertLess(overshoot[1], overshoot[2])
        self.assertGreater(overshoot[2], 0.8)  # Bounce 1 barely stops

    def test_a_skipped_frame_covers_the_time_that_passed(self):
        """A step measured in Scene Time: skipping a frame gives what playing it would have."""
        times = _times(24)
        inputs = [_step(t) for t in times]
        played = ref.run_envelope(inputs, times, mode="Spring", bounce=0.4)
        skip = 20
        skipped = ref.run_envelope(inputs[:skip] + inputs[skip + 1:], times[:skip] + times[skip + 1:], mode="Spring", bounce=0.4)
        # The input is constant over the skipped frame, so the frame after it matches.
        self.assertTrue(all(inputs[i] == inputs[skip] for i in (skip, skip + 1)))
        self.assertAlmostEqual(skipped[skip], played[skip + 1], places=9)


if __name__ == "__main__":
    unittest.main()
