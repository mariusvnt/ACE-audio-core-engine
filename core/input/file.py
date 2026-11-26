"""File input: the whole file decoded into memory (core/input/decode.py), read block by block."""

import os
from typing import Dict

import numpy as np

from ... import utils
from . import decode


class FileInput:
    """File side of AudioReception. ``reader`` holds the decoded samples [frames, channels]."""

    # Internal helpers (file IO)
    def _close_file_reader(self):
        self.reader = None

    def _open_file(self) -> bool:
        """Decode self.filepath, down to the working sample rate when the file's is higher."""
        self._close_file_reader()
        path = self.filepath
        if not path or not os.path.isfile(path):
            self.kind = None
            self.error_kind = "not_found"
            self.last_error = f"File not found: {path}"
            return False

        ext = os.path.splitext(path)[-1].lower()
        if ext not in self.supported_file_extensions():
            self.kind = None
            self.error_kind = "unsupported_format"
            self.last_error = f"Unsupported audio format: '{ext}'"
            return False

        try:
            decoded = decode.decode(path, target_sample_rate=self.target_sample_rate)
        except (decode.DecodeError, OSError) as exc:
            utils.note_swallowed("file.FileInput._open_file", exc)
            self.kind = None
            self.error_kind = "open_failed"
            self.last_error = str(exc) or type(exc).__name__
            return False

        self.reader = decoded.samples
        self.kind = "memory"
        self.channels = int(decoded.samples.shape[1])
        self.sample_rate = int(decoded.sample_rate)
        self.native_sample_rate = int(decoded.native_sample_rate)
        self.duration = float(decoded.native_frames) / float(max(decoded.native_sample_rate, 1))
        self._cursor_sample = 0
        self._eof = False
        self.last_error = ""
        self.error_kind = None
        return True

    def _reset_file(self):
        if self.reader is not None:
            self._cursor_sample = 0
            self._eof = False

    def _seek_file(self, sample_index: int) -> bool:
        if self.reader is None:
            return False
        self._cursor_sample = int(min(max(int(sample_index), 0), int(self.reader.shape[0])))
        self._eof = False
        return True

    def history_before(self, length: int):
        """(mono, channels [ch, n] or None): the *length* samples before the next block.

        Zeros stand for what comes before the start of the file. None without an open file.
        """
        if self.kind != "memory" or self.reader is None:
            return None
        end = int(self._cursor_sample)
        frames = self.reader[max(end - int(length), 0):end]
        mono = np.zeros(int(length), dtype=np.float32)
        stereo = None
        if frames.shape[1] >= 2:
            stereo = np.zeros((frames.shape[1], int(length)), dtype=np.float32)
        if frames.shape[0]:
            mono[-frames.shape[0]:] = np.mean(frames, axis=1, dtype=np.float32)
            if stereo is not None:
                stereo[:, -frames.shape[0]:] = frames.T
        return mono, stereo

    def _read_file_block(self, block_size: int):
        """(mono, channels [ch, n] or None, reached_eof) for the next block; samples may be fewer than block_size."""
        start = int(self._cursor_sample)
        frames = self.reader[start:start + int(block_size)]
        if frames.shape[0] == 0:
            return None, None, True
        mono = np.mean(frames, axis=1, dtype=np.float32)
        stereo = np.array(frames.T, dtype=np.float32) if frames.shape[1] >= 2 else None
        return mono, stereo, frames.shape[0] < int(block_size)

    def format_error_summary(self) -> str:
        """Return a concise, user-facing description of the last error."""
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

    # Kept for callers and tests that decode raw PCM.
    _decode_wave_bytes = staticmethod(decode.pcm_to_float)
