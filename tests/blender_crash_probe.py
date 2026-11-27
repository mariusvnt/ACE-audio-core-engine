"""Crash Blender on purpose while ACE's worker processes run: the run must end, not hang.

Run with ``python tests/run_in_blender.py --package --script tests/blender_crash_probe.py``.
Expected: exit status 2 and "Blender crashed" from the runner, within seconds, with no
worker process left behind (each one ends with its parent, ``core/worker/job.py``).
"""

import importlib
import os
import time

import bpy  # type: ignore

key = next(k for k in bpy.context.preferences.addons.keys() if k.endswith("audio_core_engine"))
pool = importlib.import_module(f"{key}.core.engine.parallel.pool").WorkerPool()
assert pool.start(), "the worker pool did not start"
deadline = time.monotonic() + 60.0
while not pool.warm(2) and time.monotonic() < deadline:
    time.sleep(0.1)
print("[ACE crash probe] workers:", sorted(p.pid for p in pool.executor._processes.values()), flush=True)
os.abort()  # a crash: no atexit handler runs, nothing shuts the pool down
