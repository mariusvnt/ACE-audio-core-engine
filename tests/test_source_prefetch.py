"""The file source's prefetch thread: it reads ahead, and never moves the reader under the main thread's read."""

import threading
import time
import types
import unittest

from module_loader import load_module

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

SR = 48000
FPS = 24
SPF = SR // FPS  # samples per frame, and the block size


class RampReception:
    """A file reader over a ramp: every sample's value is its index. Reads are slow, and overlapping ones are counted."""

    kind = "file"

    def __init__(self, n_samples: int, read_delay_s: float):
        self.data = np.arange(n_samples, dtype=np.float32)
        self.reader = object()
        self._cursor_sample = 0
        self._eof = False
        self.read_delay_s = read_delay_s
        self.sr = SR
        self.active = 0
        self.max_active = 0
        self._count_lock = threading.Lock()

    def seek(self, sample: int) -> bool:
        self._cursor_sample = int(sample)
        return True

    def history_before(self, n: int):
        start = max(self._cursor_sample - n, 0)
        mono = self.data[start:self._cursor_sample]
        return mono, np.stack((mono, mono))

    def read_block(self, n: int):
        with self._count_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            start = self._cursor_sample
            time.sleep(self.read_delay_s)  # widens the window in which another read could slip in
            mono = self.data[start:start + n].copy()
            self._cursor_sample = start + n
            return mono, np.stack((mono, mono)), self.sr
        finally:
            with self._count_lock:
                self.active -= 1

    def close(self):
        pass


@unittest.skipIf(np is None, "numpy is required")
class SourcePrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_module("core.tasks.source")

    def _task(self, read_delay_s: float):
        task = self.source.SourceTask("src", {}, block_size=SPF)
        task.source_context = types.SimpleNamespace(source_type="FILE", filepath="ramp.wav")
        task.scene_fps = float(FPS)
        task.scene_frame_start = 1
        task.sample_rate = SR
        task.resolve_block_size_fn = lambda *_a, **_k: SPF
        task.initialized = True
        task.source_ready = True
        task.reception = RampReception(SR * 4, read_delay_s)
        task._last_config_signature = task._config_signature()
        self.addCleanup(task._stop_prefetch_thread)
        return task

    def _play(self, task, frames, frame_gap_s=0.0):
        starts = {}
        for frame in frames:
            task._current_frame_request = frame
            chunk = task._read_next_block()
            self.assertIsNotNone(chunk, f"frame {frame}")
            starts[frame] = int(chunk.mono[0])
            if frame_gap_s:
                time.sleep(frame_gap_s)
        return starts

    def test_a_miss_reads_its_own_frame(self):
        # The main thread asks again before the prefetch is done: every frame is a miss.
        task = self._task(read_delay_s=0.01)
        starts = self._play(task, range(1, 25))
        self.assertEqual(starts, {f: (f - 1) * SPF for f in range(1, 25)})
        self.assertEqual(task.reception.max_active, 1, "two reads moved the reader at once")

    def test_hits_read_ahead_in_order(self):
        task = self._task(read_delay_s=0.001)
        starts = self._play(task, range(1, 25), frame_gap_s=0.02)
        self.assertEqual(starts, {f: (f - 1) * SPF for f in range(1, 25)})
        self.assertEqual(task.reception.max_active, 1)

    def test_every_frame_reads_from_its_own_start(self):
        """44.1 kHz at 24 fps: 1837.5 samples per frame. Blocks of 1838 must not drift from the frame clock."""
        task = self._task(read_delay_s=0.0)
        task.sample_rate = 44100
        task.reception.sr = 44100
        task.resolve_block_size_fn = lambda *_a, **_k: 1838
        task.block_size = 1838
        frames = list(range(1, 40))
        starts = self._play(task, frames)
        self.assertEqual(starts, {f: int(round((f - 1) * 44100 / FPS)) for f in frames})

    def test_a_forward_skip_reads_the_frame_it_lands_on(self):
        task = self._task(read_delay_s=0.0)
        starts = self._play(task, [1, 2, 4, 5, 7])
        self.assertEqual(starts, {f: (f - 1) * SPF for f in (1, 2, 4, 5, 7)})

    def test_a_restarted_thread_waits_for_a_new_request(self):
        """A stop wakes the thread and leaves a target behind; the next thread must not act on either."""
        task = self._task(read_delay_s=0.0)
        reads = []
        task._read_next_block_sync = lambda track_perf_stage=True, frame_request=None, view=None, prefetch=False: reads.append(frame_request)
        task._prefetch_view = task._context_view()
        task._prefetch_target_frame = 5
        task._start_prefetch_thread()
        task._stop_prefetch_thread()
        task._start_prefetch_thread()
        time.sleep(0.1)
        self.assertEqual(reads, [])
        task._prefetch_target_frame = 6
        task._prefetch_trigger.set()
        self.assertTrue(task._prefetch_ready.wait(1.0))
        self.assertEqual(reads, [6])

    def test_prefetch_leaves_the_frame_request_alone(self):
        task = self._task(read_delay_s=0.0)
        task._current_frame_request = 3
        seen = []
        original = task._read_next_block_sync

        def spy(track_perf_stage=True, frame_request=None, view=None, prefetch=False):
            seen.append(task._current_frame_request)
            return original(track_perf_stage, frame_request, view, prefetch)

        task._read_next_block_sync = spy
        task._prefetch_view = task._context_view()
        task._prefetch_target_frame = 4
        task._start_prefetch_thread()
        task._prefetch_trigger.set()
        self.assertTrue(task._prefetch_ready.wait(1.0))
        self.assertEqual(seen, [3])
        self.assertEqual(task._prefetch_for_frame, 4)
        self.assertEqual(int(task._prefetch_chunk.mono[0]), 3 * SPF)

    def test_the_thread_never_reads_the_source(self):
        """An undo frees the source's Blender data while the thread may be reading: it reads a snapshot.

        Blender crashed on undo when the thread read the source type (an EnumProperty) of freed data.
        """
        task = self._task(read_delay_s=0.002)
        values = {"source_type": "FILE", "filepath": "ramp.wav"}
        off_main = []

        class Guarded:
            def __getattr__(self, name):
                if threading.current_thread() is not threading.main_thread():
                    off_main.append(name)
                if name in values:
                    return values[name]
                raise AttributeError(name)

        task.source_context = Guarded()
        task._last_config_signature = task._config_signature()
        starts = self._play(task, range(1, 25), frame_gap_s=0.01)
        self.assertEqual(starts, {f: (f - 1) * SPF for f in range(1, 25)})
        self.assertGreater(task._prefetch_for_frame or 0, 1, "the thread read nothing")
        self.assertEqual(off_main, [], "the prefetch thread read the source's Blender data")

    def test_a_changed_source_is_read_on_the_main_thread(self):
        """After the source changes, the thread's read is left to the main thread, which reopens it."""
        task = self._task(read_delay_s=0.0)
        self._play(task, range(1, 4), frame_gap_s=0.02)
        reopened = []
        task._reopen_for_config_change = lambda: reopened.append(threading.current_thread() is threading.main_thread())
        task.source_context.filepath = "other.wav"
        task._current_frame_request = 4
        task._read_next_block()
        self.assertEqual(reopened, [True])


if __name__ == "__main__":
    unittest.main()
