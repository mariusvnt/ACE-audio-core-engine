"""Tempo and meter estimated from spectral flux."""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..input.reception import AudioReception
from ... import utils
from .block_size import _resolve_block_size_default
from .chunk import MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, SAMPLE_RATE
from .source import _get_file_open_lock


class TempoEstimation:
    @staticmethod
    def _flux_peak_indices(flux_arr: np.ndarray) -> np.ndarray:
        flux = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
        if flux.size < 4:
            return np.empty(0, dtype=np.int32)
        mean = float(np.mean(flux))
        std = float(np.std(flux))
        thresh = mean + (0.5 * std)
        mids = flux[1:-1]
        if mids.size <= 0:
            return np.empty(0, dtype=np.int32)
        peaks_mask = (mids > flux[:-2]) & (mids > flux[2:]) & (mids > thresh)
        return (np.nonzero(peaks_mask)[0] + 1).astype(np.int32, copy=False)

    @classmethod
    def _estimate_beats_per_bar_from_flux(
        cls,
        flux_arr: np.ndarray,
        peaks: np.ndarray,
        beat_period_blocks: float,
    ) -> int:
        flux = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
        peak_idx = np.asarray(peaks, dtype=np.int32).reshape(-1)
        if flux.size < 8 or peak_idx.size < 4 or beat_period_blocks <= 1e-6:
            return 4

        strengths = flux[peak_idx].astype(np.float64, copy=False)
        mean_strength = float(np.mean(strengths) or 0.0)
        if mean_strength <= 1e-9:
            return 4

        beat_numbers = np.rint(
            (peak_idx.astype(np.float64) - float(peak_idx[0])) / max(float(beat_period_blocks), 1e-6)
        ).astype(np.int32, copy=False)
        centered_flux = flux.astype(np.float64) - float(np.mean(flux))
        best_candidate = 4
        best_score = -1.0e12
        max_candidate = min(8, max(2, int(peak_idx.size // 2)))

        for candidate in range(2, max_candidate + 1):
            sums = np.zeros(candidate, dtype=np.float64)
            counts = np.zeros(candidate, dtype=np.float64)
            for beat_num, strength in zip(beat_numbers, strengths):
                idx = int(beat_num % candidate)
                sums[idx] += float(strength)
                counts[idx] += 1.0
            if np.count_nonzero(counts) < candidate:
                continue

            means = np.divide(sums, np.maximum(counts, 1.0))
            normalized = means / mean_strength
            variance_score = float(np.var(normalized))
            downbeat_score = float(normalized[0] - np.mean(normalized[1:])) if candidate > 1 else 0.0

            periodicity_score = 0.0
            lag = int(round(float(beat_period_blocks) * float(candidate)))
            if 0 < lag < centered_flux.size:
                lhs = centered_flux[:-lag]
                rhs = centered_flux[lag:]
                denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
                if denom > 1e-9:
                    periodicity_score = float(np.dot(lhs, rhs) / denom)

            score = variance_score + max(downbeat_score, 0.0) + (0.5 * max(periodicity_score, 0.0))
            score -= 0.05 * float(candidate - 2)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return int(max(best_candidate, 1))

    @classmethod
    def _estimate_tempo_from_flux(
        cls,
        flux_vals: list[float] | np.ndarray,
        block_size: int,
        sample_rate: float,
    ) -> Optional[Dict[str, Any]]:
        flux_arr = np.asarray(flux_vals, dtype=np.float32).reshape(-1)
        if flux_arr.size < 4:
            return None

        peaks = cls._flux_peak_indices(flux_arr)
        if peaks.size < 2:
            return None

        intervals = np.diff(peaks).astype(np.float64)
        intervals = intervals[intervals > 0.5]
        if intervals.size == 0:
            return None

        beat_period_blocks = float(np.median(intervals))
        seconds_per_beat = beat_period_blocks * (float(block_size) / max(float(sample_rate), 1.0))
        if seconds_per_beat <= 1e-6:
            return None

        bpm_est = 60.0 / seconds_per_beat
        if not np.isfinite(bpm_est):
            return None

        beats_per_bar = cls._estimate_beats_per_bar_from_flux(flux_arr, peaks, beat_period_blocks)
        return {
            "bpm": float(np.clip(bpm_est, 20.0, 300.0)),
            "beats_per_bar": int(max(1, beats_per_bar)),
        }

    @classmethod
    def _estimate_tempo_for_target(cls, target: Tuple[str, float]) -> Optional[Dict[str, Any]]:
        filepath, _mtime = target
        if not filepath or not os.path.isfile(filepath):
            return None

        rec = AudioReception("FILE", filepath=filepath)
        with _get_file_open_lock(filepath):
            opened = rec.open()
        if not opened:
            return None

        try:
            sr = int(getattr(rec, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            block = _resolve_block_size_default(sr, 24.0)
            block = int(max(MIN_BLOCK_SIZE, min(block, MAX_BLOCK_SIZE)))
            max_samples = int(max(float(sr), 1.0) * cls._LONG_TERM_MAX_SECONDS)

            flux_vals: list[float] = []
            prev_mag: Optional[np.ndarray] = None
            total_samples = 0

            while total_samples < max_samples and not cls._BG_STOP_EVENT.is_set():
                payload = rec.read_block(block)
                if payload is None:
                    break
                mono, _stereo, sr_val = payload
                mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
                if mono_arr.size <= 0:
                    break
                sr = int(max(int(sr_val or sr), 1))
                total_samples += int(mono_arr.size)

                mag = np.abs(np.fft.rfft(mono_arr, n=block)).astype(np.float32, copy=False)
                if prev_mag is not None and prev_mag.shape == mag.shape:
                    diff = mag - prev_mag
                    flux_vals.append(float(np.sum(np.maximum(diff, 0.0), dtype=np.float64)))
                prev_mag = mag

            return cls._estimate_tempo_from_flux(flux_vals, block, sr)
        finally:
            with utils.logged_errors("tempo.TempoEstimation._estimate_tempo_for_target"):
                rec.close()

    def _apply_detected_tempo(
        self,
        ctx: Optional[Any],
        bpm_val: float,
        beats_per_bar: int,
        *,
        respect_override: bool = True,
    ):
        # Update the instance cache (task-local state).
        self._long_term_cache["bpm"] = float(bpm_val)
        self._long_term_cache["beats_per_bar"] = int(max(1, beats_per_bar))
        self._long_term_done = True
        # Do not write bpy properties here: this may be called from background work.
        # Queue the write for main-thread flush instead.
        if ctx and (not respect_override or not self._user_overrode_tempo(ctx)):
            target = self._long_term_target
            if target is not None:
                with utils.logged_errors("tempo.TempoEstimation._apply_detected_tempo"):
                    self._PENDING_CTX_WRITES.put_nowait(
                        (
                            target,
                            float(bpm_val),
                            int(max(1, beats_per_bar)),
                            int(getattr(self, "undo_generation", 0) or 0),
                        )
                    )
