"""Mode nodes (Loudness, Stereo, Spectral): task choice, old type names and the Loudness scale."""

import unittest

from module_loader import load_module

try:
    import numpy as np
except Exception:
    np = None

try:
    stream = load_module("core.tasks.stream") if np is not None else None
    chunk_mod = load_module("core.tasks.chunk") if np is not None else None
    ace_vars = load_module("vars")
except Exception:
    stream = None


def _tone(amp, n=2048, sr=48000, freq=1000.0):
    t = np.arange(n, dtype=np.float32) / float(sr)
    return chunk_mod.AudioChunk((amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32), None, sr, lineage=("src",))


@unittest.skipIf(stream is None or np is None, "numpy/stream tasks unavailable in this environment")
class TestStreamModes(unittest.TestCase):
    def test_mode_picks_the_task(self):
        cases = {
            ("loudness", "rms"): stream.RMSLoudnessTask,
            ("loudness", "lufs"): stream.LUFSTask,
            ("loudness", "peak"): stream.PeakTask,
            ("stereo", "width"): stream.StereoWidthTask,
            ("stereo", "balance"): stream.StereoBalanceTask,
            ("spectral", "centroid"): stream.SpectralCentroidTask,
            ("spectral", "flux"): stream.SpectralFluxTask,
            ("spectral", "zcr"): stream.ZeroCrossTask,
        }
        for (node_type, mode), task_cls in cases.items():
            task = stream.create_stream_task("n", node_type, {"mode": mode}, 256)
            self.assertIs(type(task), task_cls, f"{node_type}/{mode}")

    def test_missing_or_unknown_mode_uses_the_default(self):
        self.assertIs(type(stream.create_stream_task("n", "loudness", {}, 256)), stream.LUFSTask)
        self.assertIs(type(stream.create_stream_task("n", "loudness", {"mode": "bogus"}, 256)), stream.LUFSTask)
        self.assertIs(type(stream.create_stream_task("n", "stereo", {}, 256)), stream.StereoWidthTask)

    def test_old_type_names_resolve_to_their_mode(self):
        for old_type, (node_type, mode) in ace_vars.LEGACY_STREAM_TYPES.items():
            old = stream.create_stream_task("n", old_type, {}, 256)
            new = stream.create_stream_task("n", node_type, {"mode": mode}, 256)
            self.assertIs(type(old), type(new), old_type)

    def test_every_mode_node_is_declared_with_its_default(self):
        for node_type in ("loudness", "stereo", "spectral"):
            modes = [m[0] for m in ace_vars.node_modes("stream", node_type)]
            self.assertTrue(modes, node_type)
            self.assertIn(ace_vars.resolve_node_mode("stream", node_type, None), modes)

    def test_loudness_scale_is_linear_in_db_between_floor_and_zero(self):
        to_unit = stream._loudness_db_to_unit
        self.assertEqual(ace_vars.LOUDNESS_FLOOR_DB, -70.0)
        self.assertEqual(to_unit(0.0), 1.0)
        self.assertEqual(to_unit(6.0), 1.0)
        self.assertEqual(to_unit(-70.0), 0.0)
        self.assertEqual(to_unit(-90.0), 0.0)
        self.assertEqual(to_unit(float("-inf")), 0.0)
        self.assertAlmostEqual(to_unit(-35.0), 0.5)

    def test_rms_loudness_of_a_half_scale_sine(self):
        task = stream.create_stream_task("n", "loudness", {"mode": "rms"}, 2048)
        value = float(task.process({0: _tone(0.5)}))
        # RMS of a 0.5 sine is 0.3536, -9.03 dB: (70 - 9.03) / 70.
        self.assertAlmostEqual(value, (70.0 + 20.0 * np.log10(0.5 / np.sqrt(2.0))) / 70.0, places=3)

    def test_loudness_values_are_no_longer_squashed_near_zero(self):
        # A -20 dB signal used to read 0.1 (amplitude); on the dB scale it sits well up the range.
        for mode in ("rms", "lufs", "peak"):
            task = stream.create_stream_task("n", "loudness", {"mode": mode}, 2048)
            task._current_frame_request = 1
            value = float(task.process({0: _tone(0.1)}))
            self.assertGreater(value, 0.5, mode)
            self.assertLess(value, 1.0, mode)

    def test_floor_is_fixed_and_not_a_node_input(self):
        inputs = [name for _type, name in ace_vars.ACE_NODES["stream"]["nodes"]["loudness"]["inputs"]]
        self.assertNotIn("Floor (dB)", inputs)
        for mode in ("rms", "lufs", "peak"):
            plain = stream.create_stream_task("n", "loudness", {"mode": mode}, 2048)
            old_socket = stream.create_stream_task("n", "loudness", {"mode": mode, "in_2": -20.0}, 2048)
            plain._current_frame_request = old_socket._current_frame_request = 1
            self.assertEqual(float(plain.process({0: _tone(0.1)})), float(old_socket.process({0: _tone(0.1), 2: -20.0})), mode)

    def test_silence_reads_zero_in_every_loudness_mode(self):
        silent = chunk_mod.AudioChunk(np.zeros(1024, dtype=np.float32), None, 48000, lineage=("src",))
        for mode in ("rms", "lufs", "peak"):
            task = stream.create_stream_task("n", "loudness", {"mode": mode}, 1024)
            self.assertEqual(float(task.process({0: silent})), 0.0, mode)


if __name__ == "__main__":
    unittest.main()
