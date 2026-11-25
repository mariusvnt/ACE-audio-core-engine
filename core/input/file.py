"""File input: decoding, seeking and resampled proxy files."""

import hashlib
import os
import tempfile
import wave
from typing import Dict, Optional
from ... import utils
import numpy as np
try:
    import soundfile as sf  # type: ignore
    SF_AVAILABLE = True
except Exception:  # noqa: BLE001 boundary: a broken native library raises more than ImportError
    SF_AVAILABLE = False
from .resampler import _ACEStreamResampler


class FileInput:
    # Internal helpers (file IO)
    def _close_file_reader(self):
        with utils.logged_errors("file.FileInput._close_file_reader"):
            if self.kind == "soundfile" and self.reader:
                self.reader.close()  # type: ignore
            elif self.kind == "wave" and self.reader:
                self.reader.close()
        self.reader = None

    def _open_file_path(self, path: str) -> bool:
        """Open a single file path, preferring soundfile then falling back to wave."""
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
                except Exception as exc:
                    utils.note_swallowed("file.FileInput._open_file_path", exc)
                    self.duration = 0.0
                self.kind = "soundfile"
                self._eof = False
                self.last_error = ""
                self.error_kind = None
                return True
            except Exception as e:
                utils.note_swallowed("file.FileInput._open_file_path#2", e)
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
            except Exception as exc:
                utils.note_swallowed("file.FileInput._open_file_path#3", exc)
                self.duration = 0.0
            self.kind = "wave"
            self._eof = False
            self.last_error = ""
            self.error_kind = None
            return True
        except Exception as e:
            utils.note_swallowed("file.FileInput._open_file_path#4", e)
            self.reader = None
            self.last_error = f"wave open failed: {e}"

        # Both backends failed; classify as open_failed (file exists but is unreadable).
        self.kind = None
        self.error_kind = "open_failed"
        return False

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

    @staticmethod
    def _proxy_cache_dir() -> str:
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
        """Stream-resample *input_path* to a WAV proxy at *target_sample_rate*."""
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
        """Return a path to a cached resampled WAV proxy, building it if missing."""
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
        with utils.logged_errors("file.FileInput._ensure_resampled_proxy"):
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

        try:
            if not self._build_resampled_proxy(input_path, tmp_path, int(target_sample_rate)):
                return None
            os.replace(tmp_path, proxy_path)
            return proxy_path
        except Exception as e:
            utils.note_swallowed("file.FileInput._ensure_resampled_proxy#2", e)
            self.last_error = f"proxy build failed: {e}"
            with utils.logged_errors("file.FileInput._ensure_resampled_proxy#3"):
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            return None

    def _open_file(self) -> bool:
        """Open self.filepath, optionally through a resampled WAV proxy."""
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
            except Exception as exc:
                utils.note_swallowed("file.FileInput._reset_file", exc)
                self.reader = None
        elif self.kind == "wave" and self.reader is not None:
            try:
                self.reader.rewind()
                self._eof = False
                self._cursor_sample = 0
            except Exception as exc:
                utils.note_swallowed("file.FileInput._reset_file#2", exc)
                self.reader = None

    # Internal helpers (byte decoding)
    @staticmethod
    def _decode_wave_bytes(frames: bytes, sampwidth: int, channels: int):
        """Decode PCM bytes to float32 mono/stereo numpy arrays."""
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
            except Exception as exc:
                utils.note_swallowed("file.FileInput._decode_wave_bytes", exc)
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
