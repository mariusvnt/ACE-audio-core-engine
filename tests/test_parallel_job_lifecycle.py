import concurrent.futures
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


class _FakeExecutor:
    def __init__(self, pids=()):
        self._processes = {pid: None for pid in pids}
        self.submitted = []

    def submit(self, fn, *args):
        future = concurrent.futures.Future()
        self.submitted.append((fn, future))
        return future


def _offload_graph():
    return types.SimpleNamespace(
        parallel_ir={"nodes": {}},
        parallel_offload_attr_keys=["n:0"],
        parallel_output_shards=[["n:0"]],
    )


@unittest.skipIf(np is None, "numpy unavailable in this environment")
class TestParallelJobLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ae, _, _ = _load_engine_stack()

    def _offload(self, graphs=None):
        engine = types.SimpleNamespace(graphs=dict(graphs or {}), undo_generation=0)
        return self.ae.ParallelOffload(engine)

    def _track(self, par, future, frame=1):
        slot_key, _ = par._parallel_reserve_payload_slot("g", frame, "src")
        par._parallel_future_meta[future] = {"queue_key": "g", "payload_slot_keys": [slot_key]}
        return slot_key

    def test_dropped_running_job_keeps_its_payload_slot_until_done(self):
        par = self._offload()
        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        slot_key = self._track(par, future)

        par._parallel_drop_future(future)
        par._parallel_reap_retired()
        self.assertEqual(par._parallel_payload_slot_refs[slot_key], 1)

        future.set_result({})
        par._parallel_reap_retired()
        self.assertEqual(par._parallel_payload_slot_refs[slot_key], 0)
        self.assertEqual(par._parallel_retired, {})

    def test_dropped_pending_job_is_cancelled_and_frees_its_slot(self):
        par = self._offload()
        future = concurrent.futures.Future()
        slot_key = self._track(par, future)

        par._parallel_drop_future(future)
        self.assertTrue(future.cancelled())
        self.assertEqual(par._parallel_payload_slot_refs[slot_key], 0)

    def test_graph_ir_is_sent_until_every_worker_has_it(self):
        par = self._offload()
        par._parallel_executor = _FakeExecutor(pids=(11, 12))
        par._parallel_require_ir("g")
        par._parallel_ack_ir("g", 11)
        self.assertIn("g", par._parallel_graph_ir_required)
        par._parallel_ack_ir("g", 12)
        self.assertNotIn("g", par._parallel_graph_ir_required)

    def test_new_generation_keeps_the_pool_and_retires_running_jobs(self):
        par = self._offload()
        executor = _FakeExecutor(pids=(11,))
        par._parallel_executor = executor
        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        slot_key = self._track(par, future)

        par._parallel_reset_generation()
        self.assertIs(par._parallel_executor, executor)
        self.assertEqual(par._parallel_future_meta, {})
        self.assertIn(slot_key, par._parallel_payload_slots)
        future.set_result({})
        par._parallel_reap_retired()
        self.assertEqual(par._parallel_payload_slot_refs[slot_key], 0)

    def test_jobs_wait_for_the_warm_up(self):
        par = self._offload(graphs={"g": _offload_graph()})
        executor = _FakeExecutor()
        par._parallel_executor = executor
        scene = None

        self.assertFalse(par._parallel_pool_warm(scene))
        expected = min(par._parallel_inflight_limit(), par._parallel_worker_count(scene))
        self.assertEqual(len(executor.submitted), expected)
        self.assertFalse(par._parallel_pool_warm(scene))
        for _fn, future in executor.submitted:
            future.set_result(0)
        self.assertTrue(par._parallel_pool_warm(scene))
        self.assertEqual(len(executor.submitted), expected)


if __name__ == "__main__":
    unittest.main()
