import threading
import unittest

from module_loader import load_module

try:
    import numpy as np
except Exception:
    np = None

try:
    ai = load_module("audio_influx") if np is not None else None
except Exception:
    ai = None


@unittest.skipIf(ai is None or np is None, "numpy/audio_influx unavailable in this environment")
class TestAudioInflux(unittest.TestCase):
    def test_decode_wave_bytes_sign_extends_24bit_pcm(self):
        # Little-endian packed 24-bit PCM: -1, 0, +1
        frames = bytes([
            0xFF, 0xFF, 0xFF,
            0x00, 0x00, 0x00,
            0x01, 0x00, 0x00,
        ])
        mono, stereo = ai.AudioReception._decode_wave_bytes(frames, 3, 1)
        self.assertIsNone(stereo)
        self.assertAlmostEqual(float(mono[0]), -1.0 / float(1 << 23), places=8)
        self.assertEqual(float(mono[1]), 0.0)
        self.assertAlmostEqual(float(mono[2]), 1.0 / float(1 << 23), places=8)

    def test_decode_wave_bytes_16bit_stereo_outputs_stereo_and_mono(self):
        pcm = np.asarray(
            [
                -32768, 32767,  # frame 0: L, R
                0, 0,           # frame 1: L, R
            ],
            dtype="<i2",
        )
        mono, stereo = ai.AudioReception._decode_wave_bytes(pcm.tobytes(), 2, 2)
        self.assertIsNotNone(stereo)
        self.assertEqual(tuple(stereo.shape), (2, 2))
        self.assertAlmostEqual(float(stereo[0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(stereo[1, 0]), 32767.0 / 32768.0, places=6)
        self.assertAlmostEqual(float(mono[1]), 0.0, places=6)

    def test_decode_wave_bytes_24bit_ignores_incomplete_triplet(self):
        frames = bytes([
            0x01, 0x00, 0x00,  # +1 sample
            0xFF, 0xFF,        # trailing incomplete bytes should be dropped
        ])
        mono, stereo = ai.AudioReception._decode_wave_bytes(frames, 3, 1)
        self.assertIsNone(stereo)
        self.assertEqual(int(mono.shape[0]), 1)
        self.assertAlmostEqual(float(mono[0]), 1.0 / float(1 << 23), places=8)

    def test_decode_wave_bytes_multichannel_truncates_incomplete_frame(self):
        # 3 mono bytes at 8-bit decoded with channels=2 -> one full stereo frame + tail drop.
        frames = bytes([0, 255, 128])
        mono, stereo = ai.AudioReception._decode_wave_bytes(frames, 1, 2)
        self.assertIsNotNone(stereo)
        self.assertEqual(int(mono.shape[0]), 1)
        self.assertEqual(tuple(stereo.shape), (2, 1))


@unittest.skipIf(ai is None or np is None, "numpy/audio_influx unavailable in this environment")
class TestAudioInfluxCache(unittest.TestCase):
    """Unit and concurrency tests for the viewport-replay audio cache API.

    The cache is a thin, thread-safe accumulator on AudioReception that is
    activated by start_caching() and flushed/serialised off the hot path.
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _live_rec():
        """Return a fresh LIVE AudioReception without opening any IO resource."""
        return ai.AudioReception("LIVE", flow_id="sd:default")

    @staticmethod
    def _file_rec():
        return ai.AudioReception("FILE", filepath="dummy.wav")

    @staticmethod
    def _chunk(n_frames=256, n_ch=2):
        return np.random.randn(n_frames, n_ch).astype(np.float32)

    # ------------------------------------------------------------------ state / lifecycle

    def test_initial_state_is_inactive_and_buffer_is_empty(self):
        rec = self._live_rec()
        self.assertFalse(rec._cache_active)
        self.assertEqual(len(rec._cache_buffer), 0)
        self.assertIsNone(rec.get_cached_buffer())

    def test_start_caching_arms_flag_and_discards_stale_buffer(self):
        rec = self._live_rec()
        # Pre-populate with stale data from a previous capture
        rec._cache_buffer.append(self._chunk())
        rec.start_caching()
        self.assertTrue(rec._cache_active)
        self.assertEqual(
            len(rec._cache_buffer), 0,
            "start_caching must clear any stale buffer entries",
        )

    def test_stop_caching_disarms_flag_without_discarding_buffer(self):
        rec = self._live_rec()
        rec.start_caching()
        rec._cache_buffer.append(self._chunk(128))
        rec.stop_caching()
        self.assertFalse(rec._cache_active)
        self.assertEqual(
            len(rec._cache_buffer), 1,
            "stop_caching must NOT discard already-buffered chunks",
        )

    def test_clear_cache_resets_flag_and_empties_buffer(self):
        rec = self._live_rec()
        rec.start_caching()
        for _ in range(5):
            rec._cache_buffer.append(self._chunk())
        rec.clear_cache()
        self.assertFalse(rec._cache_active)
        self.assertEqual(len(rec._cache_buffer), 0)
        self.assertIsNone(rec.get_cached_buffer())

    def test_second_start_caching_clears_buffer_from_first_session(self):
        """Calling start_caching twice resets the buffer so sessions don't bleed."""
        rec = self._live_rec()
        rec.start_caching()
        rec._cache_buffer.append(self._chunk(64))
        # Simulate a second bake run without an intervening clear
        rec.start_caching()
        self.assertEqual(len(rec._cache_buffer), 0)

    # ------------------------------------------------------------------ get_cached_buffer

    def test_get_cached_buffer_concatenates_all_chunks_in_order(self):
        rec = self._live_rec()
        rec.start_caching()
        # Push 4 chunks, each filled with a distinct sentinel value
        for i in range(4):
            rec._cache_buffer.append(
                np.full((64, 2), float(i), dtype=np.float32)
            )
        buf = rec.get_cached_buffer()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape, (256, 2))
        self.assertEqual(buf.dtype, np.float32)
        # Verify ordering: first 64 rows must equal 0.0, next 64 must equal 1.0
        np.testing.assert_array_equal(buf[:64], 0.0)
        np.testing.assert_array_equal(buf[64:128], 1.0)

    def test_get_cached_buffer_total_frames_equals_sum_of_chunk_frames(self):
        rec = self._live_rec()
        rec.start_caching()
        frame_counts = [32, 64, 100, 200]
        for n in frame_counts:
            rec._cache_buffer.append(np.zeros((n, 1), dtype=np.float32))
        buf = rec.get_cached_buffer()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape[0], sum(frame_counts))

    def test_get_cached_buffer_returns_none_when_buffer_is_empty(self):
        rec = self._live_rec()
        rec.start_caching()
        self.assertIsNone(rec.get_cached_buffer())

    # ------------------------------------------------------------------ FILE source bypass

    def test_start_caching_is_noop_for_file_source(self):
        rec = self._file_rec()
        rec.start_caching()
        self.assertFalse(
            rec._cache_active,
            "start_caching must be a no-op for FILE sources",
        )

    def test_file_source_get_cached_buffer_returns_none(self):
        rec = self._file_rec()
        rec.start_caching()  # no-op
        # Buffer is always empty for FILE; no exception should occur
        self.assertIsNone(rec.get_cached_buffer())

    def test_file_source_stop_and_clear_cache_do_not_raise(self):
        rec = self._file_rec()
        try:
            rec.stop_caching()
            rec.clear_cache()
        except Exception as exc:
            self.fail(f"Cache methods raised on FILE source: {exc}")

    # ------------------------------------------------------------------ thread safety

    def test_concurrent_appends_do_not_crash_get_cached_buffer(self):
        """Producer thread appends chunks at high rate while main thread reads.

        Invariants:
        - No exception from either thread.
        - Every non-None buffer returned by get_cached_buffer() is (N, 2) float32.
        """
        rec = self._live_rec()
        rec.start_caching()

        stop = threading.Event()
        producer_errors = []

        def _producer():
            i = 0
            while not stop.is_set():
                try:
                    if rec._cache_active:
                        rec._cache_buffer.append(
                            np.full((64, 2), float(i % 10), dtype=np.float32)
                        )
                    i += 1
                except Exception as exc:
                    producer_errors.append(exc)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        try:
            for _ in range(60):
                buf = rec.get_cached_buffer()
                if buf is not None:
                    self.assertEqual(buf.ndim, 2)
                    self.assertEqual(buf.shape[1], 2)
                    self.assertEqual(buf.dtype, np.float32)
        finally:
            stop.set()
            t.join(timeout=3.0)

        self.assertFalse(t.is_alive(), "Producer thread did not terminate cleanly")
        self.assertFalse(producer_errors, f"Producer thread raised: {producer_errors}")

    def test_concurrent_clear_and_append_leaves_valid_state(self):
        """Main thread hammers clear_cache() while background thread appends.

        After each clear/restart cycle, get_cached_buffer() must either return
        None or a correctly-typed (N, 1) float32 array.  No exceptions allowed.
        """
        rec = self._live_rec()
        rec.start_caching()

        stop = threading.Event()
        appender_errors = []

        def _appender():
            while not stop.is_set():
                try:
                    if rec._cache_active:
                        rec._cache_buffer.append(
                            np.ones((32, 1), dtype=np.float32)
                        )
                except Exception as exc:
                    appender_errors.append(exc)

        t = threading.Thread(target=_appender, daemon=True)
        t.start()
        try:
            for _ in range(40):
                rec.clear_cache()
                rec.start_caching()
                buf = rec.get_cached_buffer()
                if buf is not None:
                    self.assertEqual(buf.dtype, np.float32)
                    self.assertEqual(buf.ndim, 2)
                    self.assertEqual(buf.shape[1], 1)
        finally:
            stop.set()
            t.join(timeout=3.0)

        self.assertFalse(appender_errors, f"Appender thread raised: {appender_errors}")

    def test_stop_from_background_thread_yields_consistent_buffer(self):
        """start_caching on main thread, stop_caching from background thread.

        All chunks appended before stop_caching() must be present and the
        buffer shape must be (N_CHUNKS * FRAMES, CH).
        """
        N_CHUNKS = 20
        FRAMES = 128
        CH = 2

        rec = self._live_rec()
        rec.start_caching()

        # Barrier ensures the background thread starts appending only after
        # the main thread is ready to observe the final state.
        barrier = threading.Barrier(2)

        def _producer_then_stop():
            barrier.wait()
            for i in range(N_CHUNKS):
                rec._cache_buffer.append(
                    np.full((FRAMES, CH), float(i), dtype=np.float32)
                )
            rec.stop_caching()

        t = threading.Thread(target=_producer_then_stop, daemon=True)
        t.start()
        barrier.wait()   # synchronise with producer start
        t.join(timeout=3.0)

        self.assertFalse(t.is_alive(), "Producer thread did not finish")
        self.assertFalse(rec._cache_active, "Cache should be inactive after stop_caching")

        buf = rec.get_cached_buffer()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape, (N_CHUNKS * FRAMES, CH))


if __name__ == "__main__":
    unittest.main()
