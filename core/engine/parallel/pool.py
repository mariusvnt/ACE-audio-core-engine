"""The worker process pool: start, warm-up, restart budget and shutdown."""

import concurrent.futures
import multiprocessing
import os
import time
from typing import Iterable, Optional

from .... import utils
from ...worker.job import warm_worker, worker_pid
from ..common import _main_script_hidden_from_spawn


class WorkerPool:
    HEADROOM_CORES = 1
    RESTART_LIMIT = 3
    RESTART_BACKOFF_S = (0.5, 1.0, 2.0)

    def __init__(self):
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.failed = False
        self.restart_count = 0
        self.restart_backoff_until = 0.0
        # None: warm-up not started; empty: warm.
        self.warmup_futures: Optional[list] = None

    @classmethod
    def worker_count(cls) -> int:
        """ACE_PARALLEL_WORKERS, else all cores but one for Blender."""
        cpu_count = max(int(os.cpu_count() or 1), 1)
        override = utils.as_int(os.environ.get("ACE_PARALLEL_WORKERS", "0"), 0)
        if override > 0:
            return min(override, cpu_count)
        reserve = cls.HEADROOM_CORES if cpu_count > 1 else 0
        return max(1, cpu_count - reserve)

    def process_count(self) -> int:
        return max(len(getattr(self.executor, "_processes", None) or {}), 1)

    def start(self) -> bool:
        try:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.worker_count(),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=warm_worker,
            )
            return True
        except (OSError, ValueError, RuntimeError) as exc:
            self.executor = None
            self.failed = True
            utils.log_throttled(
                "parallel_init",
                f"Processes-DSP unavailable, reusing last-valid offload values: {exc}",
                "WARNING",
                min_interval=0.0,
            )
            return False

    def warm(self, demand: int) -> bool:
        """Submit one warm-up job per needed worker; True once all have answered."""
        if self.warmup_futures is None:
            if demand <= 0:
                return False
            count = min(demand, self.worker_count())
            self.warmup_futures = [self.submit(worker_pid) for _ in range(count)]
        if not all(f.done() for f in self.warmup_futures):
            return False
        pending, self.warmup_futures = self.warmup_futures, []
        for future in pending:
            if future.exception() is not None:
                raise future.exception()
        return True

    def submit(self, fn, *args):
        # Worker processes start inside submit().
        with _main_script_hidden_from_spawn():
            return self.executor.submit(fn, *args)

    def take_restart(self) -> Optional[float]:
        """Use one restart from the budget; the backoff delay, or None when none is allowed now."""
        now = time.monotonic()
        if self.restart_count >= self.RESTART_LIMIT or now < self.restart_backoff_until:
            return None
        self.restart_count += 1
        delay = float(self.RESTART_BACKOFF_S[min(self.restart_count - 1, len(self.RESTART_BACKOFF_S) - 1)])
        self.restart_backoff_until = now + delay
        return delay

    def restart_budget_left(self) -> bool:
        return self.restart_count < self.RESTART_LIMIT

    def shutdown(self, wait: bool, pending: Iterable = ()):
        executor, self.executor = self.executor, None
        self.warmup_futures = None
        if not wait:
            for future in pending:
                future.cancel()
            # Running workers would otherwise hold their shared memory and CPU until done.
            for proc in dict(getattr(executor, "_processes", {}) or {}).values():
                with utils.logged_errors("pool.WorkerPool.shutdown"):
                    proc.terminate()
        if executor is not None:
            with utils.logged_errors("pool.WorkerPool.shutdown#2"):
                executor.shutdown(wait=wait, cancel_futures=True)
