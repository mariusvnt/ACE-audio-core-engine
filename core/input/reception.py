"""Audio input boundary: one reader for a file or live source."""

import os
from queue import Queue
from typing import Any, Optional, Tuple
from ... import utils
import numpy as np
from ...utils import compose_live_flow_identifier as _compose_live_flow_identifier
try:
    import soundfile as sf  # type: ignore
    SF_AVAILABLE = True
except Exception:  # noqa: BLE001 boundary: a broken native library raises more than ImportError
    SF_AVAILABLE = False
from .device import DeviceInput
from .file import FileInput
from .udp import UdpInput


def build_live_flow_identifier(source_like: Any) -> str:
    """Compatibility wrapper around the shared live-route token builder."""
    return _compose_live_flow_identifier(source_like)
class AudioReception(FileInput, UdpInput, DeviceInput):
    """Unified audio reception for both file and live flow sources."""

    @staticmethod
    def supported_file_extensions() -> tuple[str, ...]:
        """Conservative extension list supported by current file backends."""
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
        """Return True if *filepath* exists and can be opened by a supported backend."""
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
        except Exception as exc:
            utils.note_swallowed("reception.AudioReception.is_supported_file_path", exc)
            return False

    def __init__(self, source_type: str, filepath: str = "", flow_id: str = "", live_latency_ms: int = 50):
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

    # Public API
    def open(self, block_size: Optional[int] = None, target_sample_rate: Optional[int] = None) -> bool:
        """Open the underlying IO resource and populate metadata fields."""
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
        """Reset the read cursor to the beginning without closing IO resources."""
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
            with utils.logged_errors("reception.AudioReception.close"):
                self._live_stream.stop()
                self._live_stream.close()
            self._live_stream = None
        if self._live_queue:
            with self._live_queue.mutex:
                self._live_queue.queue.clear()
            self._live_queue = None
        if self._live_udp_socket:
            with utils.logged_errors("reception.AudioReception.close#2"):
                self._live_udp_socket.close()
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
            utils.note_swallowed("reception.AudioReception.seek", e)
            self.last_error = f"seek failed: {e}"
            return False
        return False

    def read_block(self, block_size: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """Read the next block of audio from the source."""
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
            utils.note_swallowed("reception.AudioReception.read_block", e)
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

    def _parse_live_backend(self):
        """Determine the live backend type and configuration from self.flow_id."""
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

    def _open_live(self) -> bool:
        """Initialize a live backend."""
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
