"""The committed reference curves cover every node, fixture and frame rate.

No Blender needed: this reads tests/reference/curves. Recording and comparing
the curves happens in tests/reference/blender_reference.py.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "reference"
sys.path.insert(0, str(REFERENCE))

import fixtures  # noqa: E402
import spec  # noqa: E402

CURVES = REFERENCE / "curves"


class ReferenceGoldens(unittest.TestCase):
    def test_every_node_is_a_case_or_excluded(self):
        self.assertEqual(spec.uncovered_node_types(), [], "Add a case to tests/reference/spec.py, or an EXCLUDED reason")

    def test_metadata_matches_fixtures(self):
        meta = json.loads((CURVES / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["fixture_version"], fixtures.FIXTURE_VERSION, "Fixtures changed: record the goldens again")
        self.assertEqual(sorted(meta["fixtures"]), sorted(fixtures.FIXTURES))
        self.assertEqual(sorted(meta["fps"]), sorted(spec.FPS))

    def test_every_fixture_and_rate_has_every_curve(self):
        expected_cases = {case["id"] for case in spec.cases()}
        for name in fixtures.FIXTURES:
            for fps in spec.FPS:
                path = CURVES / spec.golden_name(name, fps)
                with self.subTest(golden=path.name):
                    self.assertTrue(path.is_file(), f"missing {path.name}")
                    with np.load(path) as data:
                        cases = {key.split("/", 1)[0] for key in data.files}
                        self.assertEqual(cases, expected_cases)
                        frames = int(fixtures.DURATION_S * fps)
                        for key in data.files:
                            curve = data[key]
                            self.assertEqual(curve.shape[0], frames, key)
                            self.assertTrue(np.all(np.isfinite(curve)), f"{key} has non-finite values")

    def test_fixtures_are_reproducible(self):
        """The generators still produce the audio the goldens were recorded from."""
        meta = json.loads((CURVES / "meta.json").read_text(encoding="utf-8"))
        import hashlib
        for name in fixtures.FIXTURES:
            sample_rate, generator = fixtures.FIXTURES[name]
            digest = hashlib.sha256(fixtures.pcm16(generator(sample_rate)).tobytes()).hexdigest()
            with self.subTest(fixture=name):
                self.assertEqual(digest, meta["fixtures"][name])


if __name__ == "__main__":
    unittest.main()
