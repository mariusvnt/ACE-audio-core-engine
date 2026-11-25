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
        slot_key, _ = par.slots.reserve("g", frame, "src")
        par.future_meta[future] = {"queue_key": "g", "payload_slot_keys": [slot_key]}
        return slot_key

    def test_dropped_running_job_keeps_its_payload_slot_until_done(self):
        par = self._offload()
        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        slot_key = self._track(par, future)

        par.drop_future(future)
        par.reap_retired()
        self.assertEqual(par.slots.refs[slot_key], 1)

        future.set_result({})
        par.reap_retired()
        self.assertEqual(par.slots.refs[slot_key], 0)
        self.assertEqual(par.retired, {})

    def test_dropped_pending_job_is_cancelled_and_frees_its_slot(self):
        par = self._offload()
        future = concurrent.futures.Future()
        slot_key = self._track(par, future)

        par.drop_future(future)
        self.assertTrue(future.cancelled())
        self.assertEqual(par.slots.refs[slot_key], 0)

    def test_graph_ir_is_sent_until_every_worker_has_it(self):
        par = self._offload()
        par.pool.executor = _FakeExecutor(pids=(11, 12))
        par.require_ir("g")
        par.ack_ir("g", 11)
        self.assertIn("g", par.ir_required)
        par.ack_ir("g", 12)
        self.assertNotIn("g", par.ir_required)

    def test_new_generation_keeps_the_pool_and_retires_running_jobs(self):
        par = self._offload()
        executor = _FakeExecutor(pids=(11,))
        par.pool.executor = executor
        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        slot_key = self._track(par, future)

        par.reset_generation()
        self.assertIs(par.pool.executor, executor)
        self.assertEqual(par.future_meta, {})
        self.assertIn(slot_key, par.slots.slots)
        future.set_result({})
        par.reap_retired()
        self.assertEqual(par.slots.refs[slot_key], 0)

    def test_jobs_wait_for_the_warm_up(self):
        par = self._offload(graphs={"g": _offload_graph()})
        executor = _FakeExecutor()
        par.pool.executor = executor
        demand = par._warm_demand()

        self.assertFalse(par.pool.warm(demand))
        expected = min(par.inflight_limit(), par.pool.worker_count())
        self.assertEqual(len(executor.submitted), expected)
        self.assertFalse(par.pool.warm(demand))
        for _fn, future in executor.submitted:
            future.set_result(0)
        self.assertTrue(par.pool.warm(demand))
        self.assertEqual(len(executor.submitted), expected)

    def test_a_restart_that_is_still_warming_does_not_disable_the_source(self):
        par = self._offload(graphs={"g": _offload_graph()})
        par.pool.start = lambda: (setattr(par.pool, "executor", _FakeExecutor()), True)[1]
        par.mode = lambda _scene=None: "processes_dsp"

        self.assertFalse(par.restart_pool(None, "test", source_key="g"))
        self.assertNotIn("g", par.disabled_sources)


if __name__ == "__main__":
    unittest.main()
