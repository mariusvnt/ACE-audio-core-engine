"""Takes: a live session recorded, to be rendered later (FUTURE.md 1.6).

For each live input of a recording session:

- **the audio**, as a WAV file, from every block the device delivers (the analysis keeps only the
  latest block, so it can't be recorded from there), written on a thread. It starts with silence
  from the scene's first frame to the frame the session started on, so the file plays from the
  scene's start like any file source;
- **the values computed live** on each frame, into the live source's own bake store (origin
  "take"). Live frames are consecutive, so frame n of the session is start + n exactly. With
  *Use Take* on, the live source reads them back: a render shows the show as the audience saw
  it, latency included.

The WAV is added as a new file source, so the same performance can also be baked with values
exactly on the audio (ideal sync), by pointing the Source nodes at it. Only device inputs record
audio today; MIDI and OSC come once ACE reads them.
"""

import os
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .. import utils


@dataclass
class _Track:
    """One live input being recorded."""

    source_uid: str
    source_name: str
    path: str
    sample_rate: int
    channels: int
    pad_frames: int  # silence before the first block, in samples per channel
    blocks: "queue.Queue[Optional[np.ndarray]]" = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    samples_written: int = 0
    error: str = ""

    def push(self, block: np.ndarray) -> None:
        """Called from the device callback: hand a copy of the block to the writer thread."""
        self.blocks.put_nowait(np.array(block, dtype=np.float32, copy=True))

    def _write(self) -> None:
        try:
            with wave.open(self.path, "wb") as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                if self.pad_frames > 0:
                    wav.writeframes(np.zeros(self.pad_frames * self.channels, dtype="<i2").tobytes())
                    self.samples_written += self.pad_frames
                while True:
                    block = self.blocks.get()
                    if block is None:
                        return
                    frames = block.reshape(block.shape[0], -1) if block.ndim > 1 else block.reshape(-1, 1)
                    if frames.shape[1] != self.channels:
                        frames = np.repeat(frames[:, :1], self.channels, axis=1) if frames.shape[1] < self.channels else frames[:, : self.channels]
                    pcm = np.clip(np.nan_to_num(frames) * 32767.0, -32768.0, 32767.0).astype("<i2")
                    wav.writeframes(pcm.tobytes())
                    self.samples_written += int(frames.shape[0])
        except Exception as exc:  # noqa: BLE001 boundary: the writer thread reports instead of dying silently
            self.error = str(exc) or type(exc).__name__
            utils.note_swallowed("takes._Track._write", exc)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._write, name=f"ACE_Take_{self.source_uid}", daemon=True)
        self.thread.start()

    def finish(self, timeout_s: float = 10.0) -> None:
        self.blocks.put_nowait(None)
        if self.thread is not None:
            self.thread.join(timeout=timeout_s)


@dataclass
class TakeValues:
    """The values a live source's outputs had on each frame of the session."""

    by_frame: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def add(self, frame: int, values: Dict[str, Any]) -> None:
        self.by_frame.setdefault(int(frame), {}).update(values)

    def series(self) -> tuple:
        """(first frame, last frame, {output key: array}); a frame missing holds the one before."""
        if not self.by_frame:
            return 0, -1, {}
        first, last = min(self.by_frame), max(self.by_frame)
        keys = sorted({k for values in self.by_frame.values() for k in values})
        out: Dict[str, np.ndarray] = {}
        for key in keys:
            column: List[Any] = []
            current: Any = None
            for frame in range(first, last + 1):
                current = self.by_frame.get(frame, {}).get(key, current)
                column.append(current)
            fill = next((v for v in column if v is not None), 0.0)
            column = [fill if v is None else v for v in column]
            out[key] = np.asarray(column, dtype=np.float32)
        return first, last, out


def takes_folder() -> str:
    """Next to the .blend (//ACE_Takes), or in the temporary folder for an unsaved file."""
    import bpy  # type: ignore

    if bpy.data.filepath:
        folder = bpy.path.abspath("//ACE_Takes")
    else:
        import tempfile

        folder = os.path.join(tempfile.gettempdir(), "ACE_Takes")
    os.makedirs(folder, exist_ok=True)
    return folder


class TakeRecorder:
    """Records the live inputs of a session: audio to WAV, values to the live sources' stores."""

    def __init__(self, engine, scene, start_frame: int, fps: float):
        self.engine = engine
        self.scene_name = str(scene.name)
        self.start_frame = int(start_frame)
        self.fps = float(fps)
        self.frame_start = int(scene.frame_start)
        self.tracks: Dict[str, _Track] = {}
        self.values: Dict[str, TakeValues] = {}
        self._receptions: List[Any] = []
        self.stamp = time.strftime("%Y%m%d_%H%M%S")

    def add_track(self, source_uid: str, source_name: str, sample_rate: int, channels: int, folder: str) -> _Track:
        clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name) or "live"
        pad_s = max(self.start_frame - self.frame_start, 0) / max(self.fps, 1.0)
        track = _Track(
            source_uid=str(source_uid),
            source_name=str(source_name),
            path=os.path.join(folder, f"{clean}_{self.stamp}.wav"),
            sample_rate=int(sample_rate),
            channels=max(int(channels), 1),
            pad_frames=int(round(pad_s * int(sample_rate))),
        )
        track.start()
        self.tracks[str(source_uid)] = track
        return track

    def start(self) -> "TakeRecorder":
        """Tap the device of every live input of the compiled graphs."""
        from .tasks.source import SourceTask

        folder = takes_folder()
        for uid, graph in self.engine.graphs.items():
            ctx = getattr(graph, "source_context", None)
            if str(getattr(ctx, "source_type", "") or "").upper() != "LIVE":
                continue
            task = next((t for t in graph.tasks.values() if isinstance(t, SourceTask)), None)
            reception = getattr(task, "reception", None)
            if reception is None or not hasattr(reception, "capture_tap"):
                utils.log(f"Take: '{getattr(ctx, 'name', uid)}' has no device to record from.", "WARNING")
                continue
            channels = int(getattr(reception, "channels", 0) or getattr(reception, "_live_channels", 0) or 1)
            track = self.add_track(uid, str(getattr(ctx, "name", uid) or uid), int(task.sample_rate), channels, folder)
            reception.capture_tap = track.push
            self._receptions.append(reception)
        return self

    def on_frame(self, source_uid: str, frame: int, values: Dict[str, Any]) -> None:
        """The values a live source's outputs had on a frame of the session."""
        self.values.setdefault(str(source_uid), TakeValues()).add(frame, values)

    def stop(self, scene) -> List[str]:
        """Finish the files, store the values, add the WAVs as file sources; their names."""
        for reception in self._receptions:
            reception.capture_tap = None
        added: List[str] = []
        for track in self.tracks.values():
            track.finish()
            if track.error:
                utils.log(f"Take of '{track.source_name}' failed: {track.error}", "ERROR")
        store = self.engine.bake.store
        for uid, take in self.values.items():
            first, last, series = take.series()
            if not series:
                continue
            output_defs = getattr(self.engine.writer, "output_defs", {}) or {}
            types = {k: str((output_defs.get(k) or {}).get("type", "FLOAT") or "FLOAT") for k in series}
            store.write(uid, first, last, series, types, identity={}, origin="take")
        ace = getattr(scene, "audiocore", None)
        if ace is not None:
            from .id_management import ACEIdManager

            for track in self.tracks.values():
                if track.error or not os.path.isfile(track.path):
                    continue
                source = ace.audio_sources.add()
                ACEIdManager.ensure_source_uid(source, existing_ids=[s.stable_id for s in ace.audio_sources if s.stable_id])
                source.source_type = "FILE"
                source.name = f"Take {self.stamp} ({track.source_name})"
                source.filepath = track.path
                added.append(source.name)
        self.engine.bake.layout()
        return added
