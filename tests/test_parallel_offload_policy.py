import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
except Exception:
    np = None

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from engine_perf_harness import _load_engine_stack


@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestParallelOffloadPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ae, _, _ = _load_engine_stack()

    @staticmethod
    def _node(category: str, node_type: str, mode=None):
        params = {} if mode is None else {"mode": mode}
        return types.SimpleNamespace(category=str(category), type=str(node_type), parameters=params)

    def test_stream_accumulator_is_not_parallel_offload_candidate(self):
        node = self._node("controllers", "stream_accumulator")
        allowed = self.ae.ParallelOffload.is_offload_candidate(node)
        self.assertFalse(allowed)

    def test_rms_remains_parallel_offload_candidate(self):
        node = self._node("audio", "loudness", "rms")
        allowed = self.ae.ParallelOffload.is_offload_candidate(node)
        self.assertTrue(allowed)

    def test_stateful_stream_nodes_stay_on_the_main_thread(self):
        """Each worker keeps its own task copy, so a stateful task offloaded would split its state."""
        offload = self.ae.ParallelOffload.is_offload_candidate
        self.assertFalse(offload(self._node("audio", "transient")))
        self.assertFalse(offload(self._node("audio", "loudness", "lufs")))
        self.assertFalse(offload(self._node("audio", "loudness", "peak")))
        self.assertTrue(offload(self._node("audio", "loudness", "rms")))
        self.assertTrue(offload(self._node("audio", "loudness")))  # RMS is the default
        self.assertTrue(offload(self._node("audio", "brightness")))
