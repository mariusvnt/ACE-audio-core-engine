"""Tempo and meter estimated from spectral flux."""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..input.reception import AudioReception
from ... import utils
from .chunk import SAMPLE_RATE
from .source import _get_file_open_lock


# Onset envelope: one value every 10 ms, from a 40 ms window.
_ENVELOPE_HOP_S = 0.01
_ENVELOPE_WINDOW_HOPS = 4
# Magnitudes are compressed so quiet onsets (hats) count next to loud ones (kicks).
_ENVELOPE_COMPRESSION = 1000.0
_TEMPO_RANGE_BPM = (40.0, 240.0)
# Half and double tempo fit the audio about as well as the tempo itself; this preference, a
# bell over tempo in octaves, decides between them.
_TEMPO_PREFERRED_BPM = 120.0
_TEMPO_PREFERENCE_OCTAVES = 0.7
# A period also has to repeat at 2 and 4 times its length: weights of those against 1 for itself.
_TEMPO_MULTIPLES = ((2, 0.5), (4, 0.25))
_MIN_PERIODICITY = 0.05


class TempoEstimation:
    @staticmethod
    def _beat_period(envelope: np.ndarray, step_s: float) -> Optional[float]:
        """Beat period of an onset envelope in steps (fractional), or None without a steady pulse.

        Autocorrelation of the envelope, with the tempo preference choosing
        among its peaks, then refined on the peaks at multiples of the period.
        """
        env = np.asarray(envelope, dtype=np.float64).reshape(-1)
        n = env.size
        low_bpm, high_bpm = _TEMPO_RANGE_BPM
        lag_min = max(int(np.ceil(60.0 / (high_bpm * step_s))), 2)
        lag_max = min(int(60.0 / (low_bpm * step_s)), n // 2)
        if lag_max <= lag_min:
            return None
        env = env - np.mean(env)
        size = 1 << int(np.ceil(np.log2(2 * n)))
        corr = np.fft.irfft(np.abs(np.fft.rfft(env, size)) ** 2, size)[:n]
        if corr[0] <= 1e-12:
            return None
        # Per pair of samples, relative to lag 0: long lags are not penalized for overlapping less.
        corr = (corr / np.arange(n, 0, -1)) / (corr[0] / n)

        lags = np.arange(lag_min, lag_max + 1)
        is_peak = (corr[lags] > corr[lags - 1]) & (corr[lags] >= corr[lags + 1]) & (corr[lags] > _MIN_PERIODICITY)
        peaks = lags[is_peak]
        if peaks.size == 0:
            return None
        reach = (3 * n) // 4  # beyond this too few pairs remain to trust the correlation

        def score(lag: int) -> float:
            total, weight = float(corr[lag]), 1.0
            for multiple, share in _TEMPO_MULTIPLES:
                at = multiple * lag
                if at + 2 < reach:
                    total += share * max(float(np.max(corr[at - 2:at + 3])), 0.0)
                    weight += share
            octaves = np.log2(60.0 / (lag * step_s) / _TEMPO_PREFERRED_BPM)
            return (total / weight) * float(np.exp(-0.5 * (octaves / _TEMPO_PREFERENCE_OCTAVES) ** 2))

        def refine(lag: int) -> float:
            before, at, after = corr[lag - 1], corr[lag], corr[lag + 1]
            curve = before - 2.0 * at + after
            return lag + (0.5 * (before - after) / curve if curve < 0.0 else 0.0)

        period = refine(int(max(peaks, key=lambda lag: score(int(lag)))))
        for multiple in range(2, 9):
            centre = int(round(multiple * period))
            if centre + 3 >= reach:
                break
            lag = centre - 2 + int(np.argmax(corr[centre - 2:centre + 3]))
            if corr[lag] > 0.0 and corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]:
                period = refine(lag) / multiple
        return float(period)

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

        step_s = float(block_size) / max(float(sample_rate), 1.0)
        beat_period_blocks = cls._beat_period(flux_arr, step_s)
        if beat_period_blocks is None:
            return None
        peaks = cls._flux_peak_indices(flux_arr)

        seconds_per_beat = beat_period_blocks * (float(block_size) / max(float(sample_rate), 1.0))
        if seconds_per_beat <= 1e-6:
            return None

        bpm_est = 60.0 / seconds_per_beat
        if not np.isfinite(bpm_est):
            return None

        beats_per_bar = cls._estimate_beats_per_bar_from_flux(flux_arr, peaks, beat_period_blocks)
        return {
            "bpm": float(np.clip(bpm_est, *_TEMPO_RANGE_BPM)),
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
            hop = max(int(round(sr * _ENVELOPE_HOP_S)), 1)
            window = np.hanning(hop * _ENVELOPE_WINDOW_HOPS).astype(np.float32)
            scale = _ENVELOPE_COMPRESSION / max(float(np.sum(window)) / 2.0, 1.0)  # 1000 for a full-scale tone
            max_samples = int(max(float(sr), 1.0) * cls._LONG_TERM_MAX_SECONDS)

            flux_vals: list[float] = []
            recent = np.zeros(window.size, dtype=np.float32)
            prev_mag: Optional[np.ndarray] = None
            total_samples = 0

            while total_samples < max_samples and not cls._BG_STOP_EVENT.is_set():
                payload = rec.read_block(hop)
                if payload is None:
                    break
                mono_arr = np.asarray(payload[0], dtype=np.float32).reshape(-1)
                if mono_arr.size <= 0:
                    break
                total_samples += int(mono_arr.size)

                recent = np.concatenate((recent[mono_arr.size:], mono_arr))
                mag = np.log1p(scale * np.abs(np.fft.rfft(recent * window)))
                if prev_mag is not None:
                    flux_vals.append(float(np.sum(np.maximum(mag - prev_mag, 0.0), dtype=np.float64)))
                prev_mag = mag

            return cls._estimate_tempo_from_flux(flux_vals, hop, sr)
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
        if not ctx:
            return
        if respect_override and self._user_overrode_tempo(ctx):
            # Forget what was written, so the detected tempo returns when the override is removed.
            self._last_queued_tempo = None
            return
        target = self._long_term_target
        if target is None:
            return
        payload = (
            target,
            float(bpm_val),
            int(max(1, beats_per_bar)),
            int(getattr(self, "undo_generation", 0) or 0),
        )
        # The cached result is applied on every frame; write the source only when it changes.
        if payload == getattr(self, "_last_queued_tempo", None):
            return
        with utils.logged_errors("tempo.TempoEstimation._apply_detected_tempo"):
            self._PENDING_CTX_WRITES.put_nowait(payload)
            self._last_queued_tempo = payload
