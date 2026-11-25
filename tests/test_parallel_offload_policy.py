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
    def _node(category: str, node_type: str):
        return types.SimpleNamespace(category=str(category), type=str(node_type))

    def test_stream_accumulator_is_not_parallel_offload_candidate(self):
        node = self._node("utilities", "stream_accumulator")
        allowed = self.ae.ParallelOffload.is_offload_candidate(node)
        self.assertFalse(allowed)

    def test_rms_remains_parallel_offload_candidate(self):
        node = self._node("stream", "rms")
        allowed = self.ae.ParallelOffload.is_offload_candidate(node)
        self.assertTrue(allowed)
