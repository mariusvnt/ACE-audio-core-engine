"""Live input from a sounddevice capture stream."""

from queue import Queue, Empty, Full
from typing import Any, Optional, Tuple
from ... import utils
import numpy as np
try:
    import sounddevice as sd  # type: ignore
    SOUNDDEVICE_AVAILABLE = True
except Exception:  # noqa: BLE001 boundary: a broken native library raises more than ImportError
    SOUNDDEVICE_AVAILABLE = False


class DeviceInput:
    @staticmethod
    def _query_sounddevice_input_info(device: Any) -> Optional[dict]:
        if not SOUNDDEVICE_AVAILABLE:
            return None
        attempts = [
            (device, "input"),
            (device,),
        ]
        if device is None:
            attempts.append(tuple())
        for args in attempts:
            try:
                info = sd.query_devices(*args)  # type: ignore[misc]
            except Exception as exc:
                utils.note_swallowed("audio_influx.AudioReception._query_sounddevice_input_info", exc)
                continue
            if isinstance(info, dict):
                return info
        return None

    def _open_live_sounddevice(self, cfg: dict, kind: str = "sounddevice") -> bool:
        """Open a sounddevice InputStream for live audio capture.

        Re-attempts sounddevice import at call time in case the module became
        available after add-on startup.  Applies WASAPI loopback settings on
        Windows when the selected device is an output-only device or when the
        caller sets cfg["loopback"]=True.  Falls back from stereo to mono if
        the device rejects the requested channel count.

        Args:
            cfg: Configuration dict with keys: device (str|int|None),
                channels (int), sample_rate (int, 0 to use device default),
                loopback (bool, optional).
            kind: String label stored in self.kind.

        Returns:
            True on success.  False on any sounddevice error, with
            self.last_error set to a description.
        """
        global sd, SOUNDDEVICE_AVAILABLE
        if not SOUNDDEVICE_AVAILABLE:
            # Re-attempt import — sounddevice may have been installed after startup
            # or the module-level import may have run before sys.path was fully set up.
            with utils.logged_errors("audio_influx.AudioReception._open_live_sounddevice"):
                import sounddevice as _sd  # type: ignore
                sd = _sd
                SOUNDDEVICE_AVAILABLE = True
        if not SOUNDDEVICE_AVAILABLE:
            self.kind = "live_stub"
            self.last_error = (
                "sounddevice is not installed in Blender's Python environment. "
                "Install it via the add-on preferences or run: "
                "import pip; pip.main(['install', 'sounddevice']) in Blender's Python."
            )
            return False

        try:
            device = cfg.get("device")
            if device is not None and device != "":
                try:
                    device = int(device)
                except ValueError:
                    device = str(device)
            else:
                device = None

            # Initialise before the queue/callback so the closure can write to it.
            self._live_stream_error: Optional[str] = None
            channels = int(cfg.get("channels", 2) or 2)
            channels = max(1, min(channels, 8))
            sample_rate = int(cfg.get("sample_rate", 0) or 0)
            _use_wasapi_loopback = bool(cfg.get("loopback", False))
            device_info = self._query_sounddevice_input_info(device)
            if device_info is not None:
                try:
                    max_input_channels = int(device_info.get("max_input_channels", 0) or 0)
                except Exception as exc:
                    utils.note_swallowed("audio_influx.AudioReception._open_live_sounddevice#3", exc)
                    max_input_channels = 0
                if max_input_channels > 0 and not _use_wasapi_loopback:
                    # Normal input device — clamp to its channel count.
                    channels = min(channels, max_input_channels)
                else:
                    # Either an explicit loopback request or an output-only device
                    # (e.g. speakers/headphones for WASAPI loopback capture).
                    try:
                        max_output_channels = int(device_info.get("max_output_channels", 0) or 0)
                    except Exception as exc:
                        utils.note_swallowed("audio_influx.AudioReception._open_live_sounddevice#4", exc)
                        max_output_channels = 0
                    if max_output_channels > 0:
                        channels = min(channels, max_output_channels)
                        _use_wasapi_loopback = True
                    elif max_input_channels > 0:
                        # Loopback flag on a normal input device — treat as regular input.
                        channels = min(channels, max_input_channels)
                        _use_wasapi_loopback = False
                    else:
                        raise ValueError(
                            "Selected capture device has no input or output channels. "
                            "Verify the selected device is a valid audio input or loopback-capable output device."
                        )
                try:
                    default_sample_rate = int(float(device_info.get("default_samplerate", 0) or 0))
                except Exception as exc:
                    utils.note_swallowed("audio_influx.AudioReception._open_live_sounddevice#5", exc)
                    default_sample_rate = 0
                if sample_rate <= 0 and default_sample_rate > 0:
                    sample_rate = default_sample_rate
            elif device is not None:
                # device_info query failed for a specific device — log and fall back
                # to letting sounddevice pick native params rather than aborting.
                self._live_stream_error = f"Could not query device info for '{device}'; using device defaults"

            # Freshness-first realtime policy: keep queue tiny to avoid latency drift.
            self._live_queue = Queue(maxsize=2)

            def _callback(indata, frames, time, status):  # type: ignore
                # Note: do NOT return early on status — sounddevice sets status
                # for minor events like input_overflow which are common at startup.
                # Returning early would silently drop valid audio frames.
                if status:
                    with utils.logged_errors("audio_influx.AudioReception._open_live_sounddevice._callback"):
                        self._live_stream_error = str(status)
                if not self._live_queue:
                    return
                block = indata.copy()
                try:
                    self._live_queue.put_nowait(block)
                    return
                except Full:
                    pass
                # Real-time policy: keep freshest block, drop oldest backlog.
                try:
                    _ = self._live_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self._live_queue.put_nowait(block)
                except Full:
                    pass

            blocksize = int(self.preferred_block_size) if self.preferred_block_size else 0
            latency_s = max(float(self.live_latency_ms) / 1000.0, 0.001)
            kwargs = {
                "callback": _callback,
                "device": device,
                "blocksize": blocksize,
                "latency": latency_s,
            }
            if sample_rate > 0:
                kwargs["samplerate"] = sample_rate
            # Build WASAPI loopback extra_settings when needed (Windows only).
            wasapi_extra: Optional[dict] = None
            if _use_wasapi_loopback:
                try:
                    wasapi_extra = {"extra_settings": sd.WasapiSettings(loopback=True)}  # type: ignore
                except Exception as exc:
                    utils.note_swallowed("audio_influx.AudioReception._open_live_sounddevice#6", exc)
                    wasapi_extra = None  # WASAPI not available on this platform
            channel_attempts = [channels]
            if channels > 1:
                channel_attempts.append(1)
            open_error = None
            for attempt_channels in channel_attempts:
                # Try WASAPI loopback settings first, then fall back to plain open.
                kw_variants = []
                if wasapi_extra:
                    kw_variants.append({**kwargs, "channels": int(attempt_channels), **wasapi_extra})
                kw_variants.append({**kwargs, "channels": int(attempt_channels)})
                for kw in kw_variants:
                    try:
                        self._live_stream = sd.InputStream(**kw)  # type: ignore
                        self._live_stream.start()
                        open_error = None
                        break
                    except Exception as e:
                        utils.note_swallowed("audio_influx.AudioReception._open_live_sounddevice#7", e)
                        open_error = e
                        self._live_stream = None
                if self._live_stream is not None:
                    break
            if self._live_stream is None:
                raise RuntimeError(str(open_error) or "unknown sounddevice error")
            self.sample_rate = int(self._live_stream.samplerate)
            self.channels = int(self._live_stream.channels)
            self.kind = kind
            self._eof = False
            self.last_error = ""
            self._last_live_block = None
            self._live_capture_sample_cursor = 0
            self._live_last_capture_sample = 0
            self._live_last_capture_block_samples = 0
            return True
        except Exception as e:
            utils.note_swallowed("audio_influx.AudioReception._open_live_sounddevice#2", e)
            self._live_stream = None
            self._live_queue = None
            self.kind = "live_stub"
            self.last_error = f"SoundDevice live stream failed to open: {e}"
            return False

    def _live_stream_active(self) -> bool:
        """Return True only if the sounddevice InputStream is still running."""
        stream = self._live_stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception as exc:
            utils.note_swallowed("audio_influx.AudioReception._live_stream_active", exc)
            return False

    def _read_live_block(self, block_size: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """Drain the sounddevice capture queue and return the freshest block.

        Keeps only the most recently arrived audio block to minimise latency —
        stale queue entries are discarded.  Returns the last valid block on a
        transient queue-empty condition so the engine never stalls on a
        scheduling gap.

        Args:
            block_size: Required number of samples in the returned block.

        Returns:
            (mono, stereo, sample_rate) or None if no audio has arrived yet.
        """
        if self.kind in {"udp_stream", "daw_udp", "instrument_udp"}:
            return self._read_live_udp_block(block_size)
        if self.kind not in {"sounddevice", "daw_sounddevice", "instrument_sounddevice"} or not self._live_queue:
            return None

        # If the stream stopped (device disconnected, driver error) signal EOF so
        # the calling task retries _load_source() rather than looping stale audio.
        if not self._live_stream_active():
            self._eof = True
            err = getattr(self, "_live_stream_error", None) or "sounddevice stream stopped"
            self.last_error = f"Live stream inactive: {err}"
            return None

        data_arr = None
        consumed_samples = 0
        while True:
            try:
                queued = self._live_queue.get_nowait()
            except Empty:
                break
            arr = np.asarray(queued, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                continue
            frames = int(arr.shape[0])
            if frames <= 0:
                continue
            consumed_samples += frames
            data_arr = arr

        if data_arr is None:
            # Queue is transiently empty (scheduling gap) — reuse last valid block
            # for this frame only so the engine keeps running without a dropout.
            if self._last_live_block:
                mono_prev, stereo_prev, sr_prev = self._last_live_block
                return mono_prev, stereo_prev, sr_prev
            return None

        if consumed_samples > 0:
            self._live_capture_sample_cursor += int(consumed_samples)
            self._live_last_capture_sample = int(self._live_capture_sample_cursor)
            self._live_last_capture_block_samples = int(data_arr.shape[0])

        arr = data_arr
        mono = np.mean(arr, axis=1)
        stereo = arr.T if arr.shape[1] >= 2 else None

        if mono.shape[0] < block_size:
            padded = np.zeros(block_size, dtype=np.float32)
            padded[:mono.shape[0]] = mono
            mono = padded
            if stereo is not None:
                padded_stereo = np.zeros((stereo.shape[0], block_size), dtype=np.float32)
                padded_stereo[:, :stereo.shape[1]] = stereo
                stereo = padded_stereo
        elif mono.shape[0] > block_size:
            mono = mono[:block_size]
            if stereo is not None:
                stereo = stereo[:, :block_size]

        # Cache latest usable block for future fallback
        self._last_live_block = (mono, stereo, self.sample_rate)

        return mono.astype(np.float32, copy=False), stereo, self.sample_rate
