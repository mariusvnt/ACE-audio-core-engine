"""Audio files to samples in memory, through Blender's audio module.

``aud`` decodes with Blender's own FFmpeg, so ACE reads every format Blender
can with nothing to install: ``aud.Sound(path).data()`` returns float32
samples [frames, channels] at the file's rate. Outside Blender (unit tests),
or when ``aud`` fails to import (another extension shadowing Blender's NumPy),
only WAV files are read, with the standard ``wave`` module.

A file is decoded once and shared: the source, the tempo analysis and the bake
all open the same file, and each gets its own cursor over one read-only array.
The cache holds a few files by total size and drops the least recently used.

Scale: ``aud`` maps 16-bit PCM to float by 1/32767, ``wave`` decoding here by
1/32768, so the two paths differ by 3e-5 relative.
"""

import os
import threading
import wave
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .resampler import resample_linear

try:
    import aud  # type: ignore
    AUD_AVAILABLE = True
except Exception:  # noqa: BLE001 boundary: missing outside Blender, and fails when another extension shadows Blender's NumPy
    aud = None
    AUD_AVAILABLE = False

WAVE_EXTENSIONS = (".wav", ".wave")
# What ACE offers in the file browser when Blender's FFmpeg is available.
FFMPEG_EXTENSIONS = WAVE_EXTENSIONS + (
    ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac",
    ".aiff", ".aif", ".caf", ".w64", ".rf64", ".au", ".snd",
)

CACHE_MAX_BYTES = 768 * 1024 * 1024


class DecodeError(Exception):
    """The file exists but cannot be decoded."""


class FileInfo(NamedTuple):
    sample_rate: int
    channels: int
    frames: int


class Decoded(NamedTuple):
    samples: np.ndarray        # float32 [frames, channels], read-only, shared
    sample_rate: int           # rate of ``samples``
    native_sample_rate: int    # rate of the file
    native_frames: int         # length of the file, in its own frames


def supported_extensions() -> Tuple[str, ...]:
    return FFMPEG_EXTENSIONS if AUD_AVAILABLE else WAVE_EXTENSIONS


def _is_wave(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in WAVE_EXTENSIONS


def probe(path: str) -> FileInfo:
    """Sample rate, channels and length, without decoding the samples."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if AUD_AVAILABLE:
        try:
            sound = aud.Sound(path)
            rate, channels = sound.specs
            return FileInfo(int(round(float(rate))), int(channels), int(sound.length))
        except Exception as exc:  # noqa: BLE001 boundary: aud raises its own error type for unreadable files
            raise DecodeError(str(exc) or "unreadable audio file") from exc
    if not _is_wave(path):
        raise DecodeError("Only WAV files can be read without Blender's audio module")
    try:
        with wave.open(path, "rb") as reader:
            return FileInfo(int(reader.getframerate()), int(reader.getnchannels()), int(reader.getnframes()))
    except (wave.Error, EOFError, OSError) as exc:
        raise DecodeError(f"wave: {exc}") from exc


def pcm_to_float(frames: bytes, sampwidth: int, channels: int):
    """Decode interleaved PCM bytes to (mono [n], channels [ch, n] or None) float32."""
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
        except TypeError:
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


def _decode_native(path: str) -> Tuple[np.ndarray, int]:
    if AUD_AVAILABLE:
        try:
            sound = aud.Sound(path)
            rate = int(round(float(sound.specs[0])))
            samples = np.asarray(sound.data(), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 boundary: aud raises its own error type for unreadable files
            raise DecodeError(str(exc) or "unreadable audio file") from exc
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        return samples, rate
    if not _is_wave(path):
        raise DecodeError("Only WAV files can be read without Blender's audio module")
    try:
        with wave.open(path, "rb") as reader:
            channels = int(reader.getnchannels())
            rate = int(reader.getframerate())
            sampwidth = int(reader.getsampwidth())
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise DecodeError(f"wave: {exc}") from exc
    mono, per_channel = pcm_to_float(frames, sampwidth, channels)
    samples = mono.reshape(-1, 1) if per_channel is None else per_channel.T
    return np.ascontiguousarray(samples, dtype=np.float32), rate


_cache: "OrderedDict[tuple, Decoded]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(path: str, target_sample_rate: Optional[int]):
    stat = os.stat(path)
    mtime = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    return (os.path.abspath(path), int(stat.st_size), mtime, int(target_sample_rate or 0), AUD_AVAILABLE)


def decode(path: str, target_sample_rate: Optional[int] = None) -> Decoded:
    """The whole file as float32 [frames, channels], resampled down to ``target_sample_rate`` when lower."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    key = _cache_key(path, target_sample_rate)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    native, native_rate = _decode_native(path)
    samples, rate = native, native_rate
    target = int(target_sample_rate or 0)
    if 0 < target < native_rate:
        samples, rate = resample_linear(native, native_rate, target), target
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples.flags.writeable = False
    result = Decoded(samples, int(rate), int(native_rate), int(native.shape[0]))
    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        total = sum(entry.samples.nbytes for entry in _cache.values())
        while len(_cache) > 1 and total > CACHE_MAX_BYTES:
            _, dropped = _cache.popitem(last=False)
            total -= dropped.samples.nbytes
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
