"""Streaming sample-rate converter."""

import numpy as np


class _ACEStreamResampler:
    """Streaming linear resampler used for one-time file proxy generation."""

    def __init__(self, source_sample_rate: int, target_sample_rate: int, channels: int):
        self.source_sample_rate = max(int(source_sample_rate), 1)
        self.target_sample_rate = max(int(target_sample_rate), 1)
        self.channels = max(int(channels), 1)
        self.step = float(self.source_sample_rate) / float(self.target_sample_rate)
        self.next_pos = 0.0
        self.consumed = 0
        self.tail = np.zeros((0, self.channels), dtype=np.float32)

    def process(self, chunk: np.ndarray, final: bool = False) -> np.ndarray:
        """Resample one chunk of interleaved audio frames."""
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
