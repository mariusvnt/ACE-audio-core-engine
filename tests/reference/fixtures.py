"""Reference audio for the reference-curve tests, generated rather than stored.

Only numpy is used, so the same code runs inside Blender (where the curves are
recorded) and in plain Python (where independent tools cross-check them).
Random numbers come from PCG64's raw output, which NumPy keeps stable across
versions, and every signal is quantized to 16-bit PCM, so tiny float
differences between NumPy versions never reach the file.

Bump FIXTURE_VERSION whenever a signal changes: the goldens record it.
"""

import hashlib
import wave
from typing import Callable, Dict, Tuple

import numpy as np

FIXTURE_VERSION = 1
DURATION_S = 4.0
BPM = 120.0


def _uniform(seed: int, n: int) -> np.ndarray:
    """Uniform floats in [0, 1) from the raw PCG64 stream."""
    raw = np.random.PCG64(seed).random_raw(n)
    return (raw >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def _white(seed: int, n: int) -> np.ndarray:
    return 2.0 * _uniform(seed, n) - 1.0


def _time(sample_rate: int) -> np.ndarray:
    return np.arange(int(DURATION_S * sample_rate), dtype=np.float64) / float(sample_rate)


def _log_sweep(sample_rate: int, f0: float = 50.0, f1: float = 12000.0, amp: float = 0.5) -> np.ndarray:
    t = _time(sample_rate)
    k = np.log(f1 / f0) / DURATION_S
    phase = 2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    return amp * np.sin(phase)


def sine_sweep(sample_rate: int = 48000) -> np.ndarray:
    """Log sweep 50 Hz to 12 kHz, mono, constant level."""
    return _log_sweep(sample_rate)[:, None]


def click_track(sample_rate: int = 48000) -> np.ndarray:
    """A 5 ms decaying 2 kHz burst on every beat at 120 BPM, silence between."""
    t = _time(sample_rate)
    out = np.zeros_like(t)
    burst_n = int(0.005 * sample_rate)
    bt = np.arange(burst_n) / float(sample_rate)
    burst = 0.8 * np.sin(2.0 * np.pi * 2000.0 * bt) * np.exp(-bt / 0.0015)
    beat = 60.0 / BPM
    start = 0.25  # first click after a short silence, so onsets have a clean baseline
    while start + 0.005 < DURATION_S:
        i = int(round(start * sample_rate))
        out[i:i + burst_n] += burst
        start += beat
    return out[:, None]


def pink_noise(sample_rate: int = 48000) -> np.ndarray:
    """Pink noise rising from -40 dBFS to -6 dBFS peak over the file."""
    n = int(DURATION_S * sample_rate)
    spectrum = np.fft.rfft(_white(1, n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    shaping = np.ones_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    shaping[0] = 0.0
    pink = np.fft.irfft(spectrum * shaping, n=n)
    pink /= np.max(np.abs(pink))
    gain_db = np.linspace(-40.0, -6.0, n)
    return (pink * 10.0 ** (gain_db / 20.0))[:, None]


def stereo_pan(sample_rate: int = 48000) -> np.ndarray:
    """A 440 Hz tone panned left to right (equal power); uncorrelated noise widens the second half."""
    t = _time(sample_rate)
    tone = 0.4 * np.sin(2.0 * np.pi * 440.0 * t)
    pan = np.linspace(0.0, 1.0, t.size) * (np.pi / 2.0)
    left, right = tone * np.cos(pan), tone * np.sin(pan)
    half = t >= DURATION_S / 2.0
    left = left + 0.1 * _white(2, t.size) * half
    right = right + 0.1 * _white(3, t.size) * half
    return np.stack([left, right], axis=1)


def music_synth(sample_rate: int = 48000) -> np.ndarray:
    """Two bars at 120 BPM: kick on beats, hats on off-beats (right), bass eighths, a chord per half bar."""
    t = _time(sample_rate)
    n = t.size
    beat = 60.0 / BPM
    left = np.zeros(n)
    right = np.zeros(n)

    kick_n = int(0.25 * sample_rate)
    kt = np.arange(kick_n) / float(sample_rate)
    kick_freq = 50.0 + 70.0 * np.exp(-kt / 0.03)
    kick = 0.7 * np.sin(2.0 * np.pi * np.cumsum(kick_freq) / sample_rate) * np.exp(-kt / 0.12)
    hat_n = int(0.04 * sample_rate)
    hat = 0.15 * _white(4, hat_n) * np.exp(-np.arange(hat_n) / (0.008 * sample_rate))
    for b in range(int(DURATION_S / beat)):
        i = int(round(b * beat * sample_rate))
        seg = kick[: max(0, min(kick_n, n - i))]
        left[i:i + seg.size] += seg
        right[i:i + seg.size] += seg
        j = int(round((b + 0.5) * beat * sample_rate))
        seg = hat[: max(0, min(hat_n, n - j))]
        right[j:j + seg.size] += seg
        left[j:j + seg.size] += 0.3 * seg

    # Am, F, C, G: one chord per half bar (two beats), bass on the root an octave down.
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (261.63, 329.63, 392.0), (196.0, 246.94, 293.66)]
    span = 2.0 * beat
    for idx, chord in enumerate(chords):
        lo, hi = int(round(idx * span * sample_rate)), int(round((idx + 1) * span * sample_rate))
        seg_t = t[lo:hi] - t[lo]
        env = np.minimum(1.0, seg_t / 0.02) * np.exp(-seg_t / 1.5)
        pad = sum(np.sin(2.0 * np.pi * f * seg_t) for f in chord) * 0.08 * env
        left[lo:hi] += pad
        right[lo:hi] += pad
        root = chord[0] / 4.0
        eighth = beat / 2.0
        bass_env = np.exp(-np.mod(seg_t, eighth) / 0.08)
        bass = 0.25 * bass_env * sum(np.sin(2.0 * np.pi * root * h * seg_t) / h for h in (1, 2, 3, 4))
        left[lo:hi] += bass
        right[lo:hi] += bass

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    scale = 0.9 / peak if peak > 0 else 1.0
    return np.stack([left * scale, right * scale], axis=1)


# name -> (sample rate, generator). The generator takes the sample rate.
FIXTURES: Dict[str, Tuple[int, Callable[[int], np.ndarray]]] = {
    "sine_sweep": (48000, sine_sweep),
    # 44.1 kHz: one frame of audio is not a whole number of samples at 24 fps.
    "sine_sweep_44k1": (44100, sine_sweep),
    "click_track": (48000, click_track),
    "pink_noise": (48000, pink_noise),
    "stereo_pan": (48000, stereo_pan),
    "music_synth": (48000, music_synth),
    # Above the default 48 kHz working rate: goes through the downsampling path.
    "sweep_96k": (96000, sine_sweep),
}


def pcm16(samples: np.ndarray) -> np.ndarray:
    """The int16 samples the WAV file holds."""
    return np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def render(name: str) -> Tuple[np.ndarray, int]:
    """(float32 samples [frames, channels] as decoded from the 16-bit file, sample rate)."""
    sample_rate, generator = FIXTURES[name]
    ints = pcm16(generator(sample_rate))
    return ints.astype(np.float32) / 32768.0, sample_rate


def write_wav(name: str, path: str) -> str:
    """Write the fixture as 16-bit PCM WAV; return the sha256 of the sample data."""
    sample_rate, generator = FIXTURES[name]
    ints = pcm16(generator(sample_rate))
    data = ints.tobytes()
    with wave.open(path, "wb") as wav:
        wav.setnchannels(int(ints.shape[1]))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(data)
    return hashlib.sha256(data).hexdigest()
