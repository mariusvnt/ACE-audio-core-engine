"""
Audio input helpers (file + live capture).
Provides a unified reception object that prefers soundfile, then wave,
with live backends for sounddevice, UDP, DAW bridges, and instrument capture routes.
"""
import hashlib
import os
import socket
import tempfile
import threading
import wave
from collections import deque
from queue import Queue, Empty, Full
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ..utils import compose_live_flow_identifier as _compose_live_flow_identifier

try:
    import soundfile as sf  # type: ignore
    SF_AVAILABLE = True
except Exception:
    SF_AVAILABLE = False

try:
    import sounddevice as sd  # type: ignore
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


def build_live_flow_identifier(source_like: Any) -> str:
    """
    Compatibility wrapper around the shared live-route token builder.
    """
    return _compose_live_flow_identifier(source_like)


class _ACEStreamResampler:
    """Streaming linear resampler used for one-time file proxy generation.

    Processes audio in arbitrary-size chunks using linear interpolation and
    accumulates a small tail buffer between calls so the resampled output
    stream is continuous across chunk boundaries.  Call with final=True on the
    last chunk to flush remaining samples.
    """

    def __init__(self, source_sample_rate: int, target_sample_rate: int, channels: int):
        self.source_sample_rate = max(int(source_sample_rate), 1)
        self.target_sample_rate = max(int(target_sample_rate), 1)
        self.channels = max(int(channels), 1)
        self.step = float(self.source_sample_rate) / float(self.target_sample_rate)
        self.next_pos = 0.0
        self.consumed = 0
        self.tail = np.zeros((0, self.channels), dtype=np.float32)

    def process(self, chunk: np.ndarray, final: bool = False) -> np.ndarray:
        """Resample one chunk of interleaved audio frames.

        Args:
            chunk: Input audio as a (frames, channels) float32 array, or a
                1-D array that will be reshaped to (-1, channels).
            final: When True, flush the tail and emit any remaining output
                samples rather than holding them for the next call.

        Returns:
            Resampled audio as a (out_frames, channels) float32 array.  May be
            empty if the buffered input is not yet sufficient to produce output.
        """
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, self.channels)
        if arr.ndim != 2:
            arr = np.zeros((0, self.channels), dtype=np.float32)

        if self.tail.shape[0] > 0:
            if arr.shape[0] > 0:
                buf = np.concatenate((self.tail, arr), axis=0)
            else:
                buf = self.tail.copy()
        else:
            buf = arr.copy() if arr.shape[0] > 0 else np.zeros((0, self.channels), dtype=np.float32)

        buf_len = int(buf.shape[0] or 0)
        if buf_len <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        buf_start = int(self.consumed) - int(self.tail.shape[0] or 0)
        last_pos = float(buf_start + buf_len - 1)
        if (not final) and buf_len < 2:
            self.consumed += int(arr.shape[0] or 0)
            self.tail = buf.copy()
            return np.zeros((0, self.channels), dtype=np.float32)

        positions: list[float] = []
        pos = max(float(self.next_pos), float(buf_start))
        if final:
            while pos <= (last_pos + 1e-9):
                positions.append(pos - float(buf_start))
                pos += self.step
        else:
            while pos < (last_pos - 1e-9):
                positions.append(pos - float(buf_start))
                pos += self.step
        self.next_pos = pos
        self.consumed += int(arr.shape[0] or 0)

        if final:
            self.tail = np.zeros((0, self.channels), dtype=np.float32)
        else:
            keep = min(buf_len, 2)
            self.tail = buf[-keep:, :].copy()

        if not positions:
            return np.zeros((0, self.channels), dtype=np.float32)

        x_old = np.arange(buf_len, dtype=np.float64)
        x_new = np.asarray(positions, dtype=np.float64)
        out = np.empty((len(positions), self.channels), dtype=np.float32)
        for ch in range(self.channels):
            out[:, ch] = np.interp(
                x_new,
                x_old,
                buf[:, ch].astype(np.float64, copy=False),
            ).astype(np.float32, copy=False)
        return out


class AudioReception:
    """Unified audio reception for both file and live flow sources.

    Supports FILE sources (WAV via stdlib wave, plus additional formats via
    soundfile) and LIVE sources (sounddevice capture, UDP float32 streaming,
    DAW bridge, and instrument capture routes).  The public API is:

    - open() / close() / reset() for lifecycle management
    - read_block() to pull the next block of audio
    - seek() for frame-locked playback sync (file sources only)

    The source type is determined at construction time via *source_type*
    ("FILE" or "LIVE").  Live sources auto-detect the backend from the
    *flow_id* token format (see _parse_live_backend()).
    """

    @staticmethod
    def supported_file_extensions() -> tuple[str, ...]:
        """
        Conservative extension list supported by current file backends.
        `.wav/.wave` are always supported via Python `wave`.
        Additional formats require `soundfile`.
        """
        exts = [".wav", ".wave"]
        if SF_AVAILABLE:
            exts.extend([
                ".flac",
                ".ogg",
                ".oga",
                ".opus",
                ".aiff",
                ".aif",
                ".caf",
                ".w64",
                ".rf64",
                ".au",
                ".snd",
                ".mp3",
                ".m4a",
                ".aac",
            ])
        seen = set()
        ordered = []
        for ext in exts:
            key = str(ext).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return tuple(ordered)

    @staticmethod
    def supported_file_filter_glob() -> str:
        """Return a semicolon-separated glob string for Blender's file browser filter."""
        return ";".join(f"*{ext}" for ext in AudioReception.supported_file_extensions())

    @staticmethod
    def is_supported_file_path(filepath: str) -> bool:
        """Return True if *filepath* exists and can be opened by a supported backend.

        WAV/WAVE files are always considered valid (pure-Python wave fallback).
        All other formats require soundfile to be available and the file to open
        without error.

        Args:
            filepath: Absolute or relative path to the audio file.

        Returns:
            True when the path exists, has a supported extension, and the file
            can be opened by the best available backend.
        """
        if not filepath or not os.path.isfile(filepath):
            return False
        ext = str(os.path.splitext(filepath)[1] or "").lower()
        if ext not in AudioReception.supported_file_extensions():
            return False
        if ext in {".wav", ".wave"}:
            return True
        if not SF_AVAILABLE:
            return False
        try:
            handle = sf.SoundFile(filepath, mode="r")  # type: ignore
            handle.close()
            return True
        except Exception:
            return False

    def __init__(self, source_type: str, filepath: str = "", flow_id: str = "", live_latency_ms: int = 50):
        """Initialise a reception object without opening any IO resources.

        Args:
            source_type: "FILE" or "LIVE".
            filepath: Path to the audio file (FILE sources only).
            flow_id: Live routing token.  Format:
                ``sd:<device>`` | ``udp:<port>[:channels[:sample_rate]]`` |
                ``daw:<udp-or-sd-selector>`` |
                ``instrument:<sd-or-udp-selector>``
            live_latency_ms: Requested sounddevice stream latency in
                milliseconds (LIVE sources only).  Clamped to >=1 ms.
        """
        self.source_type = source_type
        self.filepath = filepath
        self.flow_id = flow_id
        self.live_latency_ms = max(1, int(live_latency_ms))
        self.preferred_block_size: Optional[int] = None
        self.sample_rate: int = 44100
        self.native_sample_rate: int = 0
        self.channels: int = 1
        self.kind: Optional[str] = None
        self.reader = None
        self.duration: float = 0.0
        self._eof = False
        self.last_error: str = ""
        self.error_kind: Optional[str] = None
        # Possible values: "not_found", "unsupported_format", "open_failed", "read_failed"
        self._cursor_sample: int = 0
        self.target_sample_rate: Optional[int] = None
        self.proxy_filepath: str = ""

        # Live capture state
        self._live_queue: Optional[Queue] = None
        self._live_stream = None
        self._live_udp_socket = None
        self._live_udp_host: str = "0.0.0.0"
        self._live_udp_port: int = 0
        self._live_udp_channels: int = 2
        self._live_udp_sample_rate: int = 44100
        self._last_live_block: Optional[Tuple[np.ndarray, Optional[np.ndarray], int]] = None
        self._live_capture_sample_cursor: int = 0
        self._live_last_capture_sample: int = 0
        self._live_last_capture_block_samples: int = 0

        # ── Viewport-replay audio cache (capture_audio_sources feature) ─────
        # Accumulates raw (frames, channels) float32 arrays from LIVE streams.
        # _cache_active is read in the sounddevice OS-thread callback: the CPython
        # GIL makes bool attribute reads atomic, so no lock is needed on the
        # hot read path.  deque.append() is likewise GIL-safe for single-producer
        # use.  The lock guards multi-step operations (clear, snapshot).
        self._cache_lock: threading.Lock = threading.Lock()
        self._cache_active: bool = False
        self._cache_buffer: deque = deque()  # each entry: (N, channels) float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open(self, block_size: Optional[int] = None, target_sample_rate: Optional[int] = None) -> bool:
        """Open the underlying IO resource and populate metadata fields.

        For FILE sources, builds a resampled proxy WAV at *target_sample_rate*
        if the file's native rate exceeds the target.  For LIVE sources,
        initialises the appropriate backend (sounddevice stream or UDP socket).

        Args:
            block_size: Preferred read block size in samples.  Passed as a hint
                to the sounddevice stream blocksize parameter.
            target_sample_rate: If > 0 and less than the file's native rate, a
                resampled WAV proxy is built and used for reading.

        Returns:
            True on success, False on any IO or configuration failure (see
            self.last_error for a description).
        """
        self.preferred_block_size = block_size
        self.target_sample_rate = (
            max(int(target_sample_rate), 1)
            if target_sample_rate is not None and int(target_sample_rate) > 0
            else None
        )
        self.last_error = ""
        self._cursor_sample = 0
        self.native_sample_rate = 0
        self.proxy_filepath = ""
        self._live_capture_sample_cursor = 0
        self._live_last_capture_sample = 0
        self._live_last_capture_block_samples = 0
        if self.source_type == "LIVE":
            return self._open_live()

        return self._open_file()

    def reset(self):
        """Reset the read cursor to the beginning without closing IO resources.

        For LIVE sources, flushes the capture queue and clears the last-block
        cache so the next read produces fresh data.  For FILE sources, seeks
        to frame 0.
        """
        if self.source_type == "LIVE":
            if self._live_queue:
                with self._live_queue.mutex:
                    self._live_queue.queue.clear()
            self._last_live_block = None
            self._live_capture_sample_cursor = 0
            self._live_last_capture_sample = 0
            self._live_last_capture_block_samples = 0
            return
        self._reset_file()

    def close(self):
        """Release any underlying IO resources."""
        # Close live stream
        if self._live_stream:
            try:
                self._live_stream.stop()
                self._live_stream.close()
            except Exception:
                pass
            self._live_stream = None
        if self._live_queue:
            with self._live_queue.mutex:
                self._live_queue.queue.clear()
            self._live_queue = None
        if self._live_udp_socket:
            try:
                self._live_udp_socket.close()
            except Exception:
                pass
            self._live_udp_socket = None

        # Close file readers
        self._close_file_reader()

    def seek(self, sample_index: int) -> bool:
        """Seek to absolute sample index for deterministic frame-lock sync."""
        if self.source_type == "LIVE":
            return False
        if sample_index < 0:
            return False

        try:
            if self.kind == "soundfile" and self.reader is not None:
                self.reader.seek(int(sample_index))
                self._eof = False
                self._cursor_sample = int(sample_index)
                return True
            if self.kind == "wave" and self.reader is not None:
                self.reader.setpos(int(sample_index))
                self._eof = False
                self._cursor_sample = int(sample_index)
                return True
        except Exception as e:
            self.last_error = f"seek failed: {e}"
            return False
        return False

    def read_block(self, block_size: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """Read the next block of audio from the source.

        Args:
            block_size: Number of samples to read per channel.

        Returns:
            A (mono, stereo, sample_rate) tuple where *mono* is a 1-D float32
            array of length *block_size*, *stereo* is a (2, block_size) float32
            array or None for mono sources, and *sample_rate* is an int.
            Returns None at EOF, on IO error, or if the source is not open.
        """
        if self.source_type == "LIVE":
            return self._read_live_block(block_size)

        if not self.reader:
            return None

        mono = None
        stereo = None
        reached_eof = False

        try:
            if self.kind == "soundfile":
                data = self.reader.read(frames=block_size, dtype="float32", always_2d=True)
                if data is None or len(data) == 0:
                    reached_eof = True
                else:
                    arr = np.asarray(data, dtype=np.float32)
                    mono = np.mean(arr, axis=1)
                    if arr.shape[1] >= 2:
                        stereo = arr.T
                    if arr.shape[0] < block_size:
                        reached_eof = True
            elif self.kind == "wave":
                frames = self.reader.readframes(block_size)
                if not frames:
                    reached_eof = True
                else:
                    sampwidth = self.reader.getsampwidth()
                    mono, stereo = self._decode_wave_bytes(frames, sampwidth, self.channels)
                    if mono is not None and mono.shape[-1] < block_size:
                        reached_eof = True
        except Exception as e:
            self.last_error = f"read_block failed ({self.kind}): {e}"
            self.error_kind = "read_failed"
            return None

        self._eof = bool(reached_eof)
        if self._eof and mono is None:
            return None

        if mono is None:
            return None

        # Pad if short
        if mono.shape[-1] < block_size:
            padded = np.zeros(block_size, dtype=np.float32)
            padded[: mono.shape[-1]] = mono
            mono = padded
            if stereo is not None:
                padded_stereo = np.zeros((stereo.shape[0], block_size), dtype=np.float32)
                padded_stereo[:, : stereo.shape[1]] = stereo
                stereo = padded_stereo

        out_mono = mono.astype(np.float32, copy=False)
        self._cursor_sample += int(out_mono.shape[-1])
        return out_mono, stereo, self.sample_rate

    # ------------------------------------------------------------------
    # Audio cache API (viewport-replay / capture_audio_sources)
    # ------------------------------------------------------------------

    def start_caching(self) -> None:
        """Activate raw-audio accumulation for LIVE sources.

        No-op for FILE sources — caching is only meaningful for real-time
        streams.  Any previously buffered data is discarded so each bake
        run starts with a clean buffer.
        """
        if self.source_type != "LIVE":
            return
        with self._cache_lock:
            self._cache_buffer.clear()
            self._cache_active = True

    def stop_caching(self) -> None:
        """Stop accumulation without discarding already-buffered data."""
        with self._cache_lock:
            self._cache_active = False

    def get_cached_buffer(self) -> Optional[np.ndarray]:
        """Return a single (N_total, channels) float32 array of all cached chunks.

        Takes a GIL-safe snapshot of the deque so the callback can still
        append while concatenation runs.  Returns None if the buffer is empty
        or if concatenation fails.
        """
        with self._cache_lock:
            chunks = list(self._cache_buffer)
        if not chunks:
            return None
        try:
            return np.concatenate(chunks, axis=0)
        except Exception:
            return None

    def clear_cache(self) -> None:
        """Discard all buffered chunks and deactivate caching."""
        with self._cache_lock:
            self._cache_active = False
            self._cache_buffer.clear()

    # ------------------------------------------------------------------
    # Internal helpers (file IO)
    # ------------------------------------------------------------------
    def _close_file_reader(self):
        """Close the active file reader (soundfile or wave) and clear the handle."""
        try:
            if self.kind == "soundfile" and self.reader:
                self.reader.close()  # type: ignore
            elif self.kind == "wave" and self.reader:
                self.reader.close()
        except Exception:
            pass
        self.reader = None

    def _open_file_path(self, path: str) -> bool:
        """Open a single file path, preferring soundfile then falling back to wave.

        Populates self.channels, self.sample_rate, self.duration, and self.kind
        on success.  Sets ``error_kind`` and ``last_error`` on failure:

        * ``"not_found"`` — file does not exist on disk.
        * ``"unsupported_format"`` — extension unsupported or soundfile refused it.
        * ``"open_failed"`` — supported extension but open raised an unexpected error.

        Args:
            path: Absolute path to the audio file.

        Returns:
            True if the file was opened successfully, False otherwise.
        """
        self._close_file_reader()

        # Fast existence check — avoids confusing "file not found" with codec errors.
        if not os.path.isfile(path):
            self.kind = None
            self.error_kind = "not_found"
            self.last_error = f"File not found: {path}"
            return False

        # Determine whether the extension is in the supported list before trying opens.
        ext = os.path.splitext(path)[-1].lower()
        if ext not in self.supported_file_extensions():
            self.kind = None
            self.error_kind = "unsupported_format"
            self.last_error = f"Unsupported audio format: '{ext}'"
            return False

        if SF_AVAILABLE and self.reader is None:
            try:
                self.reader = sf.SoundFile(path, mode="r")  # type: ignore
                self.channels = int(self.reader.channels)
                self.sample_rate = int(self.reader.samplerate)
                try:
                    self.duration = float(len(self.reader)) / float(self.sample_rate)
                except Exception:
                    self.duration = 0.0
                self.kind = "soundfile"
                self._eof = False
                self.last_error = ""
                self.error_kind = None
                return True
            except Exception as e:
                self.last_error = f"soundfile open failed: {e}"
                self.reader = None
                # Fall through to wave fallback — soundfile may refuse the format even
                # if the extension is nominally supported (e.g. a corrupt MP3 header).

        try:
            self.reader = wave.open(path, "rb")
            self.channels = self.reader.getnchannels()
            self.sample_rate = self.reader.getframerate()
            try:
                self.duration = float(self.reader.getnframes()) / float(self.sample_rate)
            except Exception:
                self.duration = 0.0
            self.kind = "wave"
            self._eof = False
            self.last_error = ""
            self.error_kind = None
            return True
        except Exception as e:
            self.reader = None
            self.last_error = f"wave open failed: {e}"

        # Both backends failed; classify as open_failed (file exists but is unreadable).
        self.kind = None
        self.error_kind = "open_failed"
        return False

    def format_error_summary(self) -> str:
        """Return a concise, user-facing description of the last error.

        Consults ``error_kind`` for context-aware messaging, falling back to the
        raw ``last_error`` string when no classification is available.

        Returns:
            A short string suitable for display in the N-panel.
        """
        _KIND_LABELS: Dict[str, str] = {
            "not_found": "File not found",
            "unsupported_format": "Unsupported audio format",
            "open_failed": "Cannot open file",
            "read_failed": "Read error",
        }
        kind = self.error_kind or ""
        base = _KIND_LABELS.get(kind, "")
        detail = self.last_error or ""
        # For "not_found" and "unsupported_format" the base label is self-explanatory;
        # suppress the raw backend error to keep the message concise.
        if base and kind in ("not_found", "unsupported_format"):
            return base
        if base and detail:
            return f"{base}: {detail}"
        return base or detail or "Unknown error"

    @staticmethod
    def _proxy_cache_dir() -> str:
        """Return the OS temp directory path used for resampled WAV proxy files."""
        return os.path.join(tempfile.gettempdir(), "ace_audio_proxy_cache")

    @staticmethod
    def _float_to_pcm16_bytes(data: np.ndarray) -> bytes:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        clipped = np.clip(arr, -1.0, 1.0)
        pcm = np.round(clipped * 32767.0).astype("<i2", copy=False)
        return pcm.tobytes()

    def _resampled_proxy_path(
        self,
        input_path: str,
        source_sample_rate: int,
        target_sample_rate: int,
        channels: int,
    ) -> str:
        stat = os.stat(input_path)
        stamp = int(getattr(stat, "st_mtime_ns", int(float(stat.st_mtime) * 1_000_000_000)))
        sig = hashlib.blake2b(
            (
                f"{os.path.abspath(input_path)}|{int(stat.st_size)}|{stamp}|"
                f"{int(source_sample_rate)}|{int(target_sample_rate)}|{int(channels)}"
            ).encode("utf-8", "ignore"),
            digest_size=12,
        ).hexdigest()
        return os.path.join(self._proxy_cache_dir(), f"{sig}_{int(target_sample_rate)}.wav")

    def _build_resampled_proxy(self, input_path: str, output_path: str, target_sample_rate: int) -> bool:
        """Stream-resample *input_path* to a WAV proxy at *target_sample_rate*.

        Uses soundfile for non-WAV inputs when available, and falls back to
        stdlib wave for plain WAV files.  Writes a PCM-16 WAV proxy to
        *output_path*.

        Args:
            input_path: Source audio file path.
            output_path: Destination WAV path for the resampled proxy.
            target_sample_rate: Target sample rate in Hz.

        Returns:
            True if the proxy was written successfully, False on any error.
        """
        block_frames = 65_536
        os.makedirs(self._proxy_cache_dir(), exist_ok=True)

        if SF_AVAILABLE:
            with sf.SoundFile(input_path, mode="r") as src:  # type: ignore
                source_sample_rate = int(src.samplerate)
                channels = int(src.channels)
                resampler = _ACEStreamResampler(source_sample_rate, target_sample_rate, channels)
                with sf.SoundFile(
                    output_path,
                    mode="w",
                    samplerate=int(target_sample_rate),
                    channels=channels,
                    format="WAV",
                    subtype="PCM_16",
                ) as dst:  # type: ignore
                    while True:
                        data = src.read(frames=block_frames, dtype="float32", always_2d=True)
                        if data is None or len(data) == 0:
                            break
                        out = resampler.process(np.asarray(data, dtype=np.float32), final=False)
                        if out.shape[0] > 0:
                            dst.write(out)
                    tail = resampler.process(np.zeros((0, channels), dtype=np.float32), final=True)
                    if tail.shape[0] > 0:
                        dst.write(tail)
                return True

        with wave.open(input_path, "rb") as src:
            channels = int(src.getnchannels())
            source_sample_rate = int(src.getframerate())
            sampwidth = int(src.getsampwidth())
            resampler = _ACEStreamResampler(source_sample_rate, target_sample_rate, channels)
            with wave.open(output_path, "wb") as dst:
                dst.setnchannels(channels)
                dst.setsampwidth(2)
                dst.setframerate(int(target_sample_rate))
                while True:
                    frames = src.readframes(block_frames)
                    if not frames:
                        break
                    mono, stereo = self._decode_wave_bytes(frames, sampwidth, channels)
                    if channels <= 1 or stereo is None:
                        in_block = np.asarray(mono, dtype=np.float32).reshape(-1, 1)
                    else:
                        in_block = stereo.T.astype(np.float32, copy=False)
                    out = resampler.process(in_block, final=False)
                    if out.shape[0] > 0:
                        dst.writeframes(self._float_to_pcm16_bytes(out))
                tail = resampler.process(np.zeros((0, channels), dtype=np.float32), final=True)
                if tail.shape[0] > 0:
                    dst.writeframes(self._float_to_pcm16_bytes(tail))
        return True

    def _ensure_resampled_proxy(
        self,
        input_path: str,
        source_sample_rate: int,
        target_sample_rate: int,
        channels: int,
    ) -> Optional[str]:
        """Return a path to a cached resampled WAV proxy, building it if missing.

        The proxy path is derived deterministically from a blake2b hash of the
        input file's absolute path, size, mtime, and the requested sample rate,
        so the cache is automatically invalidated when the source file changes.

        Args:
            input_path: Absolute path to the original audio file.
            source_sample_rate: Native sample rate of the source file.
            target_sample_rate: Desired sample rate for the proxy.
            channels: Channel count of the source file.

        Returns:
            Absolute path to the resampled WAV proxy, or None if the target
            rate is not lower than the source rate or if proxy build failed.
        """
        if int(target_sample_rate) <= 0 or int(target_sample_rate) >= int(source_sample_rate):
            return None

        proxy_path = self._resampled_proxy_path(
            input_path,
            source_sample_rate=source_sample_rate,
            target_sample_rate=target_sample_rate,
            channels=channels,
        )
        if os.path.isfile(proxy_path) and os.path.getsize(proxy_path) > 0:
            return proxy_path

        tmp_path = f"{proxy_path}.tmp"
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        try:
            if not self._build_resampled_proxy(input_path, tmp_path, int(target_sample_rate)):
                return None
            os.replace(tmp_path, proxy_path)
            return proxy_path
        except Exception as e:
            self.last_error = f"proxy build failed: {e}"
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return None

    def _open_file(self) -> bool:
        """Open self.filepath, optionally through a resampled WAV proxy.

        Sets self.proxy_filepath when a proxy is used so callers can
        distinguish proxy from native reads.

        Returns:
            True if a readable file handle is available after the call.
        """
        if not self.filepath or not os.path.isfile(self.filepath):
            self.last_error = "file path does not exist"
            return False

        if not self._open_file_path(self.filepath):
            return False

        native_sample_rate = int(self.sample_rate)
        native_duration = float(self.duration)
        native_channels = int(self.channels)
        self.native_sample_rate = native_sample_rate

        target_sr = int(self.target_sample_rate or 0)
        if target_sr > 0 and target_sr < native_sample_rate:
            proxy_path = self._ensure_resampled_proxy(
                self.filepath,
                source_sample_rate=native_sample_rate,
                target_sample_rate=target_sr,
                channels=native_channels,
            )
            if proxy_path:
                if self._open_file_path(proxy_path):
                    self.proxy_filepath = proxy_path
                    self.native_sample_rate = native_sample_rate
                    self.duration = native_duration
                    return True
                # Fall back to the original file if the proxy could not be opened.
                self._open_file_path(self.filepath)
                self.native_sample_rate = native_sample_rate

        return bool(self.reader is not None)

    def _reset_file(self):
        if self.kind == "soundfile" and self.reader is not None:
            try:
                self.reader.seek(0)
                self._eof = False
                self._cursor_sample = 0
            except Exception:
                self.reader = None
        elif self.kind == "wave" and self.reader is not None:
            try:
                self.reader.rewind()
                self._eof = False
                self._cursor_sample = 0
            except Exception:
                self.reader = None

    # ------------------------------------------------------------------
    # Internal helpers (live IO)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_udp_flow_identifier(flow_id: str):
        raw = (flow_id or "").strip()
        lower = raw.lower()
        if lower.startswith("udp://"):
            raw = raw[6:]
        elif lower.startswith("udp:"):
            raw = raw[4:]
        raw = raw.strip()
        if not raw:
            return None

        host = "0.0.0.0"
        port_str = ""
        channels = 2
        sample_rate = 44100

        if "@" in raw:
            raw, meta = raw.split("@", 1)
            meta = meta.strip().lower()
            if "x" in meta:
                sr_part, ch_part = meta.split("x", 1)
                if sr_part.strip().isdigit():
                    sample_rate = int(sr_part.strip())
                if ch_part.strip().isdigit():
                    channels = int(ch_part.strip())
            elif meta.isdigit():
                sample_rate = int(meta)

        parts = [p.strip() for p in raw.split(":") if p.strip()]
        if len(parts) == 1:
            port_str = parts[0]
        elif len(parts) == 2:
            if parts[0].isdigit():
                port_str = parts[0]
                if parts[1].isdigit():
                    channels = int(parts[1])
            else:
                host, port_str = parts[0], parts[1]
        elif len(parts) >= 3:
            if parts[0].isdigit():
                port_str = parts[0]
                if parts[1].isdigit():
                    channels = int(parts[1])
                if parts[2].isdigit():
                    sample_rate = int(parts[2])
            else:
                host, port_str = parts[0], parts[1]
                if parts[2].isdigit():
                    channels = int(parts[2])
                if len(parts) >= 4 and parts[3].isdigit():
                    sample_rate = int(parts[3])

        if not port_str.isdigit():
            return None
        port = int(port_str)
        if port < 1 or port > 65535:
            return None
        channels = max(1, min(int(channels), 8))
        sample_rate = max(8000, min(int(sample_rate), 192000))

        return {
            "host": host,
            "port": port,
            "channels": channels,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _parse_daw_flow_identifier(flow_id: str):
        payload = (flow_id or "").strip()
        if not payload:
            # Default DAW bridge: UDP stereo float stream on a conventional local port.
            return "daw_udp", {"host": "0.0.0.0", "port": 7001, "channels": 2, "sample_rate": 44100}

        lower = payload.lower()
        if lower.startswith("udp:") or lower.startswith("udp://"):
            cfg = AudioReception._parse_udp_flow_identifier(payload)
            if cfg is None:
                return "invalid", {"error": "Invalid DAW UDP format. Use daw:udp:<port>[:channels[:sample_rate]]"}
            return "daw_udp", cfg

        if lower.startswith("sd:"):
            device_token = payload[3:].strip()
            return "daw_sounddevice", {"device": device_token or None, "channels": 2}

        if lower.startswith("sounddevice:"):
            device_token = payload.split(":", 1)[1].strip()
            return "daw_sounddevice", {"device": device_token or None, "channels": 2}

        # Bare numeric/host payloads map to UDP by default for DAW emitters.
        cfg = AudioReception._parse_udp_flow_identifier(f"udp:{payload}")
        if cfg is not None:
            return "daw_udp", cfg

        # Fallback for loopback-capable DAW drivers exposed through sounddevice.
        return "daw_sounddevice", {"device": payload, "channels": 2}

    @staticmethod
    def _parse_instrument_flow_identifier(flow_id: str):
        payload = (flow_id or "").strip()
        if not payload:
            return "instrument_sounddevice", {"device": None, "channels": 1}

        lower = payload.lower()
        if lower.startswith("udp:") or lower.startswith("udp://"):
            cfg = AudioReception._parse_udp_flow_identifier(payload)
            if cfg is None:
                return "invalid", {"error": "Invalid instrument UDP format. Use instrument:udp:<port>[:channels[:sample_rate]]"}
            if int(cfg.get("channels", 1) or 1) < 1:
                cfg["channels"] = 1
            return "instrument_udp", cfg

        device_token = payload
        channels = 1
        if lower.startswith("sd:"):
            device_token = payload[3:].strip()
        elif lower.startswith("sounddevice:"):
            device_token = payload.split(":", 1)[1].strip()

        if "@" in device_token:
            dev, meta = device_token.split("@", 1)
            device_token = dev.strip()
            if meta.strip().isdigit():
                channels = int(meta.strip())

        channels = max(1, min(channels, 8))
        return "instrument_sounddevice", {"device": device_token or None, "channels": channels}

    def _parse_live_backend(self):
        """Determine the live backend type and configuration from self.flow_id.

        Dispatches on the token prefix:
        - ``daw:`` → DAW bridge (UDP or sounddevice)
        - ``instrument:`` → instrument capture (sounddevice or UDP)
        - ``udp:`` / ``udp://`` → raw UDP float32 stream
        - ``sd:`` / ``sounddevice:`` → sounddevice InputStream
        - loopback prefix → sounddevice with WASAPI loopback flag
        - bare token → sounddevice with device name/index

        Returns:
            A (backend_name, cfg_dict) tuple.  *backend_name* is one of
            "udp", "daw_udp", "instrument_udp", "sounddevice",
            "daw_sounddevice", "instrument_sounddevice", "invalid", or
            "unsupported".
        """
        token = (self.flow_id or "").strip()
        if not token:
            return "sounddevice", {"device": None, "channels": 2}

        lower = token.lower()
        if lower.startswith("daw:"):
            return self._parse_daw_flow_identifier(token.split(":", 1)[1].strip())
        if lower.startswith("instrument:"):
            return self._parse_instrument_flow_identifier(token.split(":", 1)[1].strip())
        if lower.startswith("udp:") or lower.startswith("udp://"):
            cfg = self._parse_udp_flow_identifier(token)
            if cfg is None:
                return "invalid", {"error": "Invalid UDP flow format. Use udp:<port>[:channels[:sample_rate]]"}
            return "udp", cfg
        if lower.startswith("sd:"):
            device_token = token[3:].strip()
            if device_token.lower().startswith("loopback:"):
                return "sounddevice", {"device": device_token[9:].strip() or None, "channels": 2, "loopback": True}
            return "sounddevice", {"device": device_token or None, "channels": 2}
        if lower.startswith("sounddevice:"):
            device_token = token.split(":", 1)[1].strip()
            if device_token.lower().startswith("loopback:"):
                return "sounddevice", {"device": device_token[9:].strip() or None, "channels": 2, "loopback": True}
            return "sounddevice", {"device": device_token or None, "channels": 2}

        # Keep truly unsupported host-api selectors explicit.
        if ":" in token and not token.isdigit():
            prefix = token.split(":", 1)[0].strip().lower()
            if prefix in {"asio", "jack", "coreaudio", "wasapi"}:
                return "unsupported", {"backend": prefix}
        return "sounddevice", {"device": token, "channels": 2}

    def _open_live_udp(self, cfg: dict, kind: str = "udp_stream") -> bool:
        """Bind a non-blocking UDP socket for float32 audio streaming.

        Args:
            cfg: Configuration dict with keys: host (str), port (int),
                channels (int), sample_rate (int).
            kind: String label stored in self.kind (e.g. "udp_stream",
                "daw_udp", "instrument_udp").

        Returns:
            True on success.  False on socket bind failure, with
            self.last_error set to a description.
        """
        try:
            host = str(cfg.get("host", "0.0.0.0") or "0.0.0.0")
            port = int(cfg.get("port", 0) or 0)
            channels = int(cfg.get("channels", 2) or 2)
            sample_rate = int(cfg.get("sample_rate", 44100) or 44100)
            if port <= 0:
                raise ValueError("Invalid UDP port")

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind((host, port))
            self._live_udp_socket = sock
            self._live_udp_host = host
            self._live_udp_port = port
            self._live_udp_channels = max(1, channels)
            self._live_udp_sample_rate = max(8000, sample_rate)
            self.sample_rate = self._live_udp_sample_rate
            self.channels = self._live_udp_channels
            self.kind = kind
            self._eof = False
            self.last_error = ""
            self._last_live_block = None
            self._live_capture_sample_cursor = 0
            self._live_last_capture_sample = 0
            self._live_last_capture_block_samples = 0
            return True
        except Exception as e:
            self._live_udp_socket = None
            self.kind = "live_stub"
            self.last_error = f"UDP live stream failed to open: {e}"
            return False

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
            except Exception:
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
            try:
                import sounddevice as _sd  # type: ignore
                sd = _sd
                SOUNDDEVICE_AVAILABLE = True
            except Exception:
                pass
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
                except Exception:
                    max_input_channels = 0
                if max_input_channels > 0 and not _use_wasapi_loopback:
                    # Normal input device — clamp to its channel count.
                    channels = min(channels, max_input_channels)
                else:
                    # Either an explicit loopback request or an output-only device
                    # (e.g. speakers/headphones for WASAPI loopback capture).
                    try:
                        max_output_channels = int(device_info.get("max_output_channels", 0) or 0)
                    except Exception:
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
                except Exception:
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
                    try:
                        self._live_stream_error = str(status)
                    except Exception:
                        pass
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
                except Exception:
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
            self._live_stream = None
            self._live_queue = None
            self.kind = "live_stub"
            self.last_error = f"SoundDevice live stream failed to open: {e}"
            return False

    def _open_live(self) -> bool:
        """
        Initialize a live backend.
        Supported:
        - sounddevice capture (`sd:` or plain device token)
        - UDP float32 streaming (`udp:<port>[:channels[:sample_rate]]`)
        - DAW bridge (`daw:` prefix routed to dedicated UDP/sounddevice handlers)
        - Instrument capture (`instrument:` prefix routed to dedicated sounddevice/UDP handlers)
        """
        backend, cfg = self._parse_live_backend()
        if backend == "udp":
            return self._open_live_udp(cfg, kind="udp_stream")
        if backend == "daw_udp":
            return self._open_live_udp(cfg, kind="daw_udp")
        if backend == "instrument_udp":
            return self._open_live_udp(cfg, kind="instrument_udp")
        if backend == "invalid":
            self.kind = "live_stub"
            self.last_error = str(cfg.get("error", "Invalid live flow identifier"))
            return False
        if backend == "unsupported":
            b = str(cfg.get("backend", "unknown"))
            self.kind = "live_stub"
            self.last_error = (
                f"Unsupported live backend '{b}'. "
                "Use sounddevice (`sd:`), UDP (`udp:`), DAW (`daw:`), or Instrument (`instrument:`)."
            )
            return False
        if backend == "daw_sounddevice":
            return self._open_live_sounddevice(cfg, kind="daw_sounddevice")
        if backend == "instrument_sounddevice":
            return self._open_live_sounddevice(cfg, kind="instrument_sounddevice")
        return self._open_live_sounddevice(cfg, kind="sounddevice")

    def _live_stream_active(self) -> bool:
        """Return True only if the sounddevice InputStream is still running."""
        stream = self._live_stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
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

        # Accumulate raw multi-channel block for viewport-replay capture.
        # data_arr is (frames, channels) float32 — the pre-downmix original.
        if self._cache_active and data_arr is not None:
            try:
                self._cache_buffer.append(data_arr.astype(np.float32, copy=False))
            except Exception:
                pass

        return mono.astype(np.float32, copy=False), stereo, self.sample_rate

    def _read_live_udp_block(self, block_size: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """Drain the non-blocking UDP socket and return the freshest packet.

        Reads all available UDP datagrams in a non-blocking loop and keeps
        only the last one (freshest-first policy).  Returns the last valid
        block on a transient receive-empty condition.

        Args:
            block_size: Required number of samples in the returned block.

        Returns:
            (mono, stereo, sample_rate) or None if no packet has arrived yet.
        """
        sock = self._live_udp_socket
        if sock is None:
            return None

        packet = None
        packet_frames = 0
        consumed_samples = 0
        channels = max(1, int(self._live_udp_channels))
        while True:
            try:
                data, _addr = sock.recvfrom(262144)
                if data:
                    raw = np.frombuffer(data, dtype=np.float32)
                    if raw.size <= 0:
                        continue
                    if channels == 1:
                        frames = int(raw.size)
                    else:
                        frames = int(raw.size // channels)
                    if frames <= 0:
                        continue
                    consumed_samples += frames
                    packet = raw
                    packet_frames = frames
            except BlockingIOError:
                break
            except Exception:
                self.last_error = "UDP live stream read failed"
                break

        if packet is None:
            if self._last_live_block:
                mono_prev, stereo_prev, sr_prev = self._last_live_block
                return mono_prev, stereo_prev, sr_prev
            return None

        raw = packet
        if raw.size == 0:
            return None
        if consumed_samples > 0:
            self._live_capture_sample_cursor += int(consumed_samples)
            self._live_last_capture_sample = int(self._live_capture_sample_cursor)
        self._live_last_capture_block_samples = max(int(packet_frames), 1)
        mono = None
        stereo = None
        if channels == 1:
            mono = raw
        else:
            frames = raw.size // channels
            if frames <= 0:
                return None
            raw = raw[:frames * channels]
            arr = raw.reshape(frames, channels)
            mono = np.mean(arr, axis=1)
            if channels >= 2:
                stereo = arr[:, :2].T.astype(np.float32, copy=False)

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

        sr = int(self._live_udp_sample_rate or self.sample_rate or 44100)
        self.sample_rate = sr
        self.channels = channels

        # Accumulate raw block for viewport-replay capture.
        # Rebuild the (frames, channels) interleaved array from what we have.
        if self._cache_active:
            try:
                if channels >= 2 and mono is not None:
                    # `arr` is (frames, channels) — defined in the channels>1 branch above.
                    _frames = int(mono.shape[0])
                    _raw_mc = raw[:_frames * channels].reshape(_frames, channels)
                    self._cache_buffer.append(_raw_mc.astype(np.float32, copy=False))
                elif mono is not None:
                    self._cache_buffer.append(
                        mono.reshape(-1, 1).astype(np.float32, copy=False)
                    )
            except Exception:
                pass

        self._last_live_block = (
            mono.astype(np.float32, copy=False),
            stereo.astype(np.float32, copy=False) if stereo is not None else None,
            sr,
        )
        return mono.astype(np.float32, copy=False), stereo, sr

    # ------------------------------------------------------------------
    # Internal helpers (byte decoding)
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_wave_bytes(frames: bytes, sampwidth: int, channels: int):
        """
        Decode PCM bytes to float32 mono/stereo numpy arrays.
        Supports 8/16/24/32-bit PCM and float32/float64.
        """
        mono = None
        stereo = None

        if sampwidth == 1:
            # WAV 8-bit PCM is unsigned.
            data_u8 = np.frombuffer(frames, dtype=np.uint8).astype(np.float32, copy=False)
            mono = (data_u8 - 128.0) / 128.0
        elif sampwidth == 2:
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32, copy=False)
            mono = data / float(1 << 15)
        elif sampwidth == 3:  # 24-bit packed
            raw = np.frombuffer(frames, dtype=np.uint8)
            if raw.size < 3:
                mono = np.zeros(0, dtype=np.float32)
            else:
                usable = (raw.size // 3) * 3
                raw = raw[:usable].reshape(-1, 3)
                # WAV 24-bit PCM is little-endian signed. Build int32 then sign-extend.
                data = (
                    raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16)
                )
                sign_bit = 1 << 23
                data = (data ^ sign_bit) - sign_bit
                data = data.astype(np.float32, copy=False)
                mono = data / float(1 << 23)
        elif sampwidth == 4:
            # Try float32 first, fall back to int32 PCM normalize
            data_f = np.frombuffer(frames, dtype="<f4")
            if data_f.size == 0:
                mono = data_f
            else:
                finite = data_f[np.isfinite(data_f)]
                max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0
                if finite.size != data_f.size or max_abs > 1.1:
                    data_i = np.frombuffer(frames, dtype="<i4").astype(np.float32, copy=False)
                    mono = data_i / float(1 << 31)
                else:
                    mono = data_f.astype(np.float32, copy=False)
        elif sampwidth == 8:
            # Ambiguous in WAV: could be float64 or int64 PCM. Use a bounded-float heuristic.
            data_f = np.frombuffer(frames, dtype="<f8")
            if data_f.size == 0:
                mono = np.zeros(0, dtype=np.float32)
            else:
                finite = data_f[np.isfinite(data_f)]
                max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0
                if finite.size == data_f.size and max_abs <= 1.1:
                    mono = data_f.astype(np.float32, copy=False)
                else:
                    data_i = np.frombuffer(frames, dtype="<i8").astype(np.float64, copy=False)
                    mono = (data_i / float(1 << 63)).astype(np.float32, copy=False)
        else:
            # Generic signed-PCM fallback for uncommon widths.
            try:
                dtype = np.dtype(f"<i{int(sampwidth)}")
                data = np.frombuffer(frames, dtype=dtype).astype(np.float64, copy=False)
                scale = float(1 << ((8 * int(sampwidth)) - 1))
                mono = (data / max(scale, 1.0)).astype(np.float32, copy=False)
            except Exception:
                mono = np.zeros(0, dtype=np.float32)

        if channels > 1 and mono is not None:
            usable = (int(mono.shape[0]) // int(channels)) * int(channels)
            if usable <= 0:
                return np.zeros(0, dtype=np.float32), None
            if usable != int(mono.shape[0]):
                mono = mono[:usable]
            mono = mono.reshape(-1, channels)
            stereo = mono.T
            mono = np.mean(mono, axis=1)

        return mono, stereo
