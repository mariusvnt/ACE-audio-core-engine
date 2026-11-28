"""Tempo and meter of a file source, estimated in the background from its onset envelope."""

import os
import threading
from collections import OrderedDict
from queue import Empty, Full, Queue
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
# How much of a file the tempo and meter are estimated on, and how much the beat grid is then
# fitted over: a phase computed from a fixed BPM drifts, by half a beat after 4 minutes at 120
# BPM for an error of 0.1 %, which is what 15 seconds of audio gives.
_ANALYSIS_MAX_SECONDS = 15.0
_GRID_FIT_MAX_SECONDS = 600.0
# A beat's onset is looked for within this share of a beat around where the grid puts it, and
# the fit is kept only when its beats lie within this share of a beat of the line, on average.
_GRID_REACH = 1.0 / 6.0
_GRID_MAX_RESIDUAL = 1.0 / 8.0
_GRID_MIN_BEATS = 8
_TEMPO_RANGE_BPM = (40.0, 240.0)
# Half and double tempo fit the audio about as well as the tempo itself; this preference, a
# bell over tempo in octaves, decides between them.
_TEMPO_PREFERRED_BPM = 120.0
_TEMPO_PREFERENCE_OCTAVES = 0.7
# A period also has to repeat at longer lags: at 2 and 4 times its length (bars of 2 or 4),
# or at 3 and 6 (bars of 3), with these weights against 1 for itself. The better family counts.
_TEMPO_MULTIPLES = (((2, 0.5), (4, 0.25)), ((3, 0.5), (6, 0.25)))
_MIN_PERIODICITY = 0.05

# Meter: the bar lengths tried, the beat phases tried per beat, and the score below which
# no accent pattern is trusted (random accents score up to about 0.25, real bars 0.7 and more).
_METER_RANGE = range(2, 9)
_METER_PHASES = 12
_METER_FLOOR = 0.4
_DEFAULT_BEATS_PER_BAR = 4


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

    def repeats(lag: int, multiples) -> float:
        total, weight = float(corr[lag]), 1.0
        for multiple, share in multiples:
            at = multiple * lag
            if at + 2 < reach:
                total += share * max(float(np.max(corr[at - 2:at + 3])), 0.0)
                weight += share
        return total / weight

    def score(lag: int) -> float:
        octaves = np.log2(60.0 / (lag * step_s) / _TEMPO_PREFERRED_BPM)
        fit = max(repeats(lag, family) for family in _TEMPO_MULTIPLES)
        return fit * float(np.exp(-0.5 * (octaves / _TEMPO_PREFERENCE_OCTAVES) ** 2))

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


def _meter(envelope: np.ndarray, period: float) -> Tuple[int, Optional[float]]:
    """Beats per bar and the first downbeat (in steps) of an onset envelope with this beat period.

    Each beat's strength is the envelope's peak within a sixth of a beat, so
    off-beats never count. The strengths are folded by beat number modulo each
    bar length: a bar of m beats has one fold clearly above the others, and
    fits the strengths better than other lengths do. The beat phase is the one
    that shows the clearest bar. Without a clear accent pattern the result is
    4 beats per bar and no downbeat.
    """
    env = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if period <= 1.0 or env.size < 2:
        return _DEFAULT_BEATS_PER_BAR, None
    reach = max(int(round(period / 6.0)), 1)
    best_score, best_bar, best_first = 0.0, _DEFAULT_BEATS_PER_BAR, 0.0
    for p in range(_METER_PHASES):
        phase = p * period / _METER_PHASES
        centres = np.rint(np.arange(phase, env.size - reach, period)).astype(np.int64)
        if centres.size < 2 * _METER_RANGE.start:
            continue
        strengths = np.array([env[max(c - reach, 0):c + reach + 1].max() for c in centres])
        typical = float(np.median(strengths))
        spread = float(np.sum(np.abs(strengths - typical)))
        if spread <= 0.05 * typical * strengths.size:
            continue  # every beat alike: no accent to count
        beat = np.arange(strengths.size)
        for bar in _METER_RANGE:
            if strengths.size < 3 * bar:
                break  # fewer than 3 beats per fold
            folds = np.array([np.median(strengths[beat % bar == j]) for j in range(bar)])
            j = int(np.argmax(folds))
            explained = 1.0 - float(np.sum(np.abs(strengths - folds[beat % bar]))) / spread
            contrast = (folds[j] - np.max(np.delete(folds, j))) / max(float(folds[j]), 1e-12)
            # A bar of 4 also folds as 8 with nearly the same score; shorter bars win ties.
            score = explained * contrast * (1.0 - 0.03 * (bar - 2))
            if score > best_score:
                best_score, best_bar, best_first = score, bar, phase + j * period
    if best_score < _METER_FLOOR:
        return _DEFAULT_BEATS_PER_BAR, None
    # Phases whose window only grazes the accents score nearly as well: the downbeat is where
    # the envelope summed over all downbeats peaks, within a third of a beat of the grid.
    offsets = np.arange(-2 * reach, 2 * reach + 1)
    summed = np.zeros(offsets.size)
    for centre in np.rint(np.arange(best_first, env.size, best_bar * period)).astype(np.int64):
        at = centre + offsets
        inside = (at >= 0) & (at < env.size)
        summed[inside] += env[at[inside]]
    downbeat = float(np.rint(best_first) + offsets[int(np.argmax(summed))])
    return int(best_bar), downbeat


def _onset_peaks(env: np.ndarray, period: float, origin: float, last: float):
    """(beat number, position in steps, strength) of the onset nearest each beat of the grid up to *last*."""
    reach = max(int(round(period * _GRID_REACH)), 1)
    first = -int(np.floor(origin / period))
    peaks = []
    for k in range(first, int(np.floor((last - origin) / period)) + 1):
        centre = int(round(origin + k * period))
        lo, hi = max(centre - reach, 1), min(centre + reach, env.size - 2)
        if hi <= lo:
            continue
        i = lo + int(np.argmax(env[lo:hi + 1]))
        before, at, after = env[i - 1], env[i], env[i + 1]
        curve = before - 2.0 * at + after
        offset = 0.5 * (before - after) / curve if curve < 0.0 else 0.0
        peaks.append((k, i + float(np.clip(offset, -0.5, 0.5)), float(at)))
    return peaks


def _fit_grid(envelope: np.ndarray, period: float, downbeat: float) -> Optional[Tuple[float, float]]:
    """(period, downbeat) in steps, refined by a weighted least-squares line through the onsets of the whole envelope.

    The grid is extended in spans that double, each fitted before the next, so the onsets it
    looks for never drift out of reach. None when the onsets don't line up: the estimate stays.
    """
    env = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if period <= 1.0 or env.size < 4:
        return None
    span = 16.0 * period
    while True:
        last = min(downbeat + span, env.size - 2.0)
        peaks = _onset_peaks(env, period, downbeat, last)
        strengths = np.array([s for _k, _p, s in peaks])
        if strengths.size < _GRID_MIN_BEATS:
            return None
        keep = strengths >= float(np.median(strengths))
        k = np.array([b for b, _p, _s in peaks], dtype=np.float64)[keep]
        pos = np.array([p for _b, p, _s in peaks], dtype=np.float64)[keep]
        w = strengths[keep]
        if k.size < _GRID_MIN_BEATS or np.ptp(k) <= 0:
            return None
        a = np.vstack([np.ones_like(k), k]).T * np.sqrt(w)[:, None]
        (origin, fitted), *_ = np.linalg.lstsq(a, pos * np.sqrt(w), rcond=None)
        residual = float(np.sqrt(np.sum(w * (pos - origin - fitted * k) ** 2) / np.sum(w)))
        if not np.isfinite(fitted) or fitted <= 1.0 or residual > _GRID_MAX_RESIDUAL * fitted:
            return None
        period, downbeat = float(fitted), float(origin)
        if last >= env.size - 2.0:
            return period, downbeat
        span *= 2.0


def _tempo_of_envelope(envelope: np.ndarray, step_s: float, estimate_steps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """BPM, beats per bar and first downbeat (seconds) of an onset envelope, or None without a beat.

    The tempo and meter are estimated on the first *estimate_steps* values (all of them when
    None); when a downbeat is found, the beat grid is then fitted over the whole envelope.
    """
    env = np.asarray(envelope, dtype=np.float64).reshape(-1)
    head = env if estimate_steps is None else env[:int(estimate_steps)]
    if head.size < 4:
        return None
    period = _beat_period(head, step_s)
    if period is None:
        return None
    seconds_per_beat = period * step_s
    if seconds_per_beat <= 1e-6 or not np.isfinite(60.0 / seconds_per_beat):
        return None
    beats_per_bar, downbeat = _meter(head, period)
    if downbeat is not None:
        fitted = _fit_grid(env, period, downbeat)
        if fitted is not None:
            period, downbeat = fitted
            seconds_per_beat = period * step_s
            bar = beats_per_bar * period
            downbeat -= bar * np.floor((downbeat + 1.0) / bar)  # the first downbeat of the file
    return {
        "bpm": float(np.clip(60.0 / seconds_per_beat, *_TEMPO_RANGE_BPM)),
        "beats_per_bar": int(beats_per_bar),
        # Value i of the envelope is the change the block starting i + 1 hops in brings.
        "downbeat_s": None if downbeat is None else float(max(downbeat + 1.0, 0.0) * step_s),
    }


def estimate_file_tempo(filepath: str, stop: Optional[threading.Event] = None) -> Optional[Dict[str, Any]]:
    """Tempo of the first seconds of an audio file, or None when it has no steady beat."""
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
        max_samples = int(max(float(sr), 1.0) * _GRID_FIT_MAX_SECONDS)
        estimate_samples = int(max(float(sr), 1.0) * _ANALYSIS_MAX_SECONDS)
        estimate_steps = None

        flux: list[float] = []
        recent = np.zeros(window.size, dtype=np.float32)
        prev_mag: Optional[np.ndarray] = None
        total_samples = 0
        while total_samples < max_samples and not (stop is not None and stop.is_set()):
            payload = rec.read_block(hop)
            if payload is None:
                break
            mono = np.asarray(payload[0], dtype=np.float32).reshape(-1)
            if mono.size <= 0:
                break
            if total_samples < estimate_samples <= total_samples + int(mono.size):
                estimate_steps = len(flux)  # what the first 15 s gave: the estimate is made on these
            total_samples += int(mono.size)
            recent = np.concatenate((recent[mono.size:], mono))
            mag = np.log1p(scale * np.abs(np.fft.rfft(recent * window)))
            if prev_mag is not None:
                flux.append(float(np.sum(np.maximum(mag - prev_mag, 0.0), dtype=np.float64)))
            prev_mag = mag
        if estimate_steps is not None and estimate_steps < len(flux):
            # The block that reached 15 s gave its own value, which the estimate saw too.
            estimate_steps += 1
        return _tempo_of_envelope(np.asarray(flux), hop / float(sr), estimate_steps)
    finally:
        with utils.logged_errors("tempo.estimate_file_tempo"):
            rec.close()


Target = Tuple[str, float]  # (file path, modification time)


class TempoService:
    """Tempo detection of file sources on one background thread, with results kept per file version.

    Nothing here touches Blender data: the main thread asks for a file's
    tempo, then reads the result once it is there (see core/engine/probes.py).
    """

    _CACHE_MAX = 128
    _QUEUE_MAX = 16
    _NO_BEAT = "No steady beat found."

    _lock = threading.Lock()
    _results: "OrderedDict[Target, Dict[str, Any]]" = OrderedDict()
    _queue: "Queue[Target]" = Queue(maxsize=_QUEUE_MAX)
    _waiting: set = set()  # queued or being analysed
    _stop = threading.Event()
    _thread: Optional[threading.Thread] = None

    @staticmethod
    def target(source: Optional[Any]) -> Optional[Target]:
        """The file version a FILE source plays, or None."""
        if source is None or str(getattr(source, "source_type", "") or "").upper() != "FILE":
            return None
        filepath = str(getattr(source, "filepath", "") or "").strip()
        if not filepath or not os.path.isfile(filepath):
            return None
        try:
            return (filepath, float(os.path.getmtime(filepath)))
        except OSError:
            return None

    @classmethod
    def result(cls, target: Optional[Target]) -> Optional[Dict[str, Any]]:
        """The finished detection: bpm, beats_per_bar, downbeat_s, or error. None while unknown."""
        if target is None:
            return None
        with cls._lock:
            found = cls._results.get(target)
            return dict(found) if found is not None else None

    @classmethod
    def waiting(cls, target: Optional[Target]) -> bool:
        with cls._lock:
            return target in cls._waiting

    @classmethod
    def request(cls, target: Target) -> bool:
        """Queue the detection of *target* unless it is known or under way; False when the queue is full."""
        with cls._lock:
            if target in cls._results or target in cls._waiting:
                return True
        cls._ensure_thread()
        with cls._lock:
            try:
                cls._queue.put_nowait(target)
            except Full:
                return False
            cls._waiting.add(target)
        return True

    @classmethod
    def busy(cls) -> str:
        """What is still being detected ("" when nothing)."""
        with cls._lock:
            return ", ".join(sorted(os.path.basename(path) for path, _ in cls._waiting))

    @classmethod
    def _store(cls, target: Target, found: Dict[str, Any]) -> None:
        with cls._lock:
            cls._results[target] = found
            cls._results.move_to_end(target)
            while len(cls._results) > cls._CACHE_MAX:
                cls._results.popitem(last=False)

    @classmethod
    def _ensure_thread(cls) -> None:
        with cls._lock:
            if cls._thread is not None and cls._thread.is_alive():
                return
            cls._stop.clear()
            cls._thread = threading.Thread(target=cls._run, name="ACE_TempoDetection", daemon=True)
            cls._thread.start()

    @classmethod
    def _run(cls) -> None:
        while not cls._stop.is_set():
            try:
                target = cls._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                found = estimate_file_tempo(target[0], cls._stop)
                if not cls._stop.is_set():
                    cls._store(target, found if found else {"error": cls._NO_BEAT})
            except Exception as exc:
                utils.note_swallowed("tempo.TempoService._run", exc)
                cls._store(target, {"error": "Tempo detection failed."})
            finally:
                with cls._lock:
                    cls._waiting.discard(target)
                cls._queue.task_done()

    @classmethod
    def shutdown(cls) -> None:
        """Stop the thread and forget queued work; finished results are kept."""
        cls._stop.set()
        thread = cls._thread
        if thread is not None and thread.is_alive():
            with utils.logged_errors("tempo.TempoService.shutdown"):
                thread.join(timeout=0.25)
        with cls._lock:
            cls._thread = None
            cls._waiting.clear()
            try:
                while True:
                    cls._queue.get_nowait()
                    cls._queue.task_done()
            except Empty:
                pass

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._results.clear()
