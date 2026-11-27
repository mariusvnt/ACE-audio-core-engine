"""Source audio sent to workers: payload dicts and the shared-memory slots that carry them."""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .... import utils
from ..common import _SHARED_MEMORY_AVAILABLE, mp_shared_memory


def _fit(mono: np.ndarray, size: int) -> np.ndarray:
    if mono.shape[0] < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[: mono.shape[0]] = mono
        return padded
    return mono[:size]


def source_payload(chunk, block_size: int, sample_rate: int) -> Dict[str, Any]:
    """A chunk as a picklable dict: mono of exactly block_size samples, stereo channels-first.

    The audio before the block, when the chunk has it, sits in front of both
    arrays; history_len says how many samples that is.
    """
    mono = _fit(np.asarray(getattr(chunk, "mono", chunk), dtype=np.float32).reshape(-1), max(int(block_size), 1))
    stereo = getattr(chunk, "stereo", None)
    stereo_payload = None
    if isinstance(stereo, np.ndarray) and stereo.ndim == 2:
        arr = np.asarray(stereo, dtype=np.float32)
        if arr.shape[1] == mono.shape[0]:
            stereo_payload = arr
        elif arr.shape[0] == mono.shape[0]:
            stereo_payload = arr.T
    payload: Dict[str, Any] = {
        "mono": mono,
        "stereo": stereo_payload,
        "sample_rate": max(int(sample_rate), 1),
        "latency_ms": float(getattr(chunk, "latency_ms", 0.0) or 0.0),
    }
    history = getattr(chunk, "history_mono", None)
    if isinstance(history, np.ndarray) and history.size:
        payload["history_len"] = int(history.size)
        payload["mono"] = np.concatenate((history.astype(np.float32, copy=False), mono))
        if stereo_payload is not None:
            before = getattr(chunk, "history_stereo", None)
            if not isinstance(before, np.ndarray) or before.shape != (stereo_payload.shape[0], history.size):
                before = np.zeros((stereo_payload.shape[0], history.size), dtype=np.float32)
            payload["stereo"] = np.concatenate((before, stereo_payload), axis=1).astype(np.float32, copy=False)
    band_indices = getattr(chunk, "band_indices", None)
    if band_indices is not None:
        with utils.logged_errors("slots.source_payload#2"):
            payload["band_indices"] = np.asarray(band_indices, dtype=np.int64).reshape(-1)
            payload["band_index_fft_size"] = max(int(getattr(chunk, "band_index_fft_size", 0) or 0), 1)
    band_hz_bounds = getattr(chunk, "band_hz_bounds", None)
    if isinstance(band_hz_bounds, (list, tuple)) and len(band_hz_bounds) >= 2:
        with utils.logged_errors("slots.source_payload#3"):
            low_hz, high_hz = float(band_hz_bounds[0]), float(band_hz_bounds[1])
            if high_hz > low_hz:
                payload["band_hz_bounds"] = (low_hz, high_hz)
    return payload


def _close_unlink(shm_obj):
    if shm_obj is None:
        return
    with utils.logged_errors("slots._close_unlink"):
        shm_obj.close()
    try:
        shm_obj.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        utils.note_swallowed("slots._close_unlink#2", exc)


class PayloadSlots:
    """Shared-memory blocks per (graph, source), reference-counted and reused across frames."""

    def __init__(self):
        self.slots: Dict[str, Dict[str, Any]] = {}
        self.refs: Dict[str, int] = {}
        self.counters: Dict[str, int] = {}
        self.free: Dict[str, deque] = {}
        self.frame_slots: Dict[Tuple[str, int, str], str] = {}

    def reserve(self, graph_key: str, frame_index: int, src_id: str) -> Tuple[str, bool]:
        """The slot for this frame's source audio, and whether its content must be written."""
        frame_key = (str(graph_key or ""), int(frame_index), str(src_id or ""))
        existing = self.frame_slots.get(frame_key)
        if existing and existing in self.slots:
            self.refs[existing] = self.refs.get(existing, 0) + 1
            return existing, False
        prefix = f"{frame_key[0]}:{frame_key[2]}"
        chosen = ""
        free = self.free.get(prefix)
        while free and not chosen:
            candidate = str(free.popleft())
            if candidate in self.slots:
                chosen = candidate
        if free is not None and not free:
            self.free.pop(prefix, None)
        if not chosen:
            index = self.counters.get(prefix, 0)
            chosen = f"{prefix}#{index}"
            self.counters[prefix] = index + 1
            self.slots.setdefault(chosen, {})
            self.refs[chosen] = 0
        self.frame_slots[frame_key] = chosen
        self.refs[chosen] = self.refs.get(chosen, 0) + 1
        return chosen, True

    def release(self, slot_key: str):
        key = str(slot_key)
        if key not in self.slots:
            return
        self.refs[key] = max(self.refs.get(key, 0) - 1, 0)
        if self.refs[key] > 0:
            return
        self._unmap_frames(key)
        free = self.free.setdefault(key.split("#", 1)[0], deque())
        if key not in free:
            free.append(key)

    def release_keys(self, slot_keys: List[str]):
        for slot_key in slot_keys:
            self.release(slot_key)

    def destroy(self, slot_key: str):
        key = str(slot_key)
        self.refs.pop(key, None)
        prefix = key.split("#", 1)[0]
        free = self.free.get(prefix)
        if free is not None:
            remaining = deque(v for v in free if str(v) != key)
            if remaining:
                self.free[prefix] = remaining
            else:
                self.free.pop(prefix, None)
        self._unmap_frames(key)
        entry = self.slots.pop(key, None)
        if isinstance(entry, dict):
            _close_unlink(entry.get("mono_shm"))
            _close_unlink(entry.get("stereo_shm"))

    def destroy_idle(self):
        for slot_key in list(self.slots):
            if self.refs.get(slot_key, 0) <= 0:
                self.destroy(slot_key)

    def destroy_all(self):
        for slot_key in list(self.slots):
            self.destroy(slot_key)
        self.free.clear()
        self.refs.clear()
        self.counters.clear()
        self.frame_slots.clear()

    def pack(self, source_payloads: Dict[str, Dict[str, Any]], graph_key: str, frame_index: int) -> Tuple[Dict[str, Any], List[str]]:
        """Job fields for the source payloads, with audio moved to shared memory when available."""
        chunks = {str(src_id): dict(p or {}) for src_id, p in dict(source_payloads or {}).items()}
        if not _SHARED_MEMORY_AVAILABLE:
            return {"source_chunks": chunks}, []
        slot_keys: List[str] = []
        shm_map: Dict[str, Dict[str, Any]] = {}
        try:
            for src_id, payload in chunks.items():
                mono = payload.get("mono")
                if not isinstance(mono, np.ndarray):
                    continue
                slot_key, needs_upload = self.reserve(graph_key, frame_index, src_id)
                slot_keys.append(slot_key)
                meta = {"mono": self._field(slot_key, "mono", mono, needs_upload)}
                stereo = payload.get("stereo")
                if isinstance(stereo, np.ndarray):
                    meta["stereo"] = self._field(slot_key, "stereo", stereo, needs_upload)
                elif needs_upload:
                    self._clear_field(slot_key, "stereo")
                payload.pop("mono", None)
                payload.pop("stereo", None)
                shm_map[src_id] = meta
            packed: Dict[str, Any] = {"source_chunks": chunks}
            if shm_map:
                packed["source_chunks_shm"] = shm_map
            return packed, slot_keys
        except (OSError, ValueError, BufferError) as exc:
            self.release_keys(slot_keys)
            utils.log_throttled(
                "parallel_shm_ipc",
                f"Shared-memory IPC setup failed, falling back to pickle transport: {exc}",
                "WARNING",
                min_interval=4.0,
            )
            return {"source_chunks": chunks}, []

    def _unmap_frames(self, slot_key: str):
        for frame_key, mapped in list(self.frame_slots.items()):
            if mapped == slot_key:
                self.frame_slots.pop(frame_key, None)

    def _field(self, slot_key: str, field: str, array: np.ndarray, needs_upload: bool) -> Dict[str, Any]:
        if not needs_upload:
            descriptor = self._descriptor(slot_key, field)
            if descriptor is not None:
                return descriptor
        return self._upload(slot_key, field, array)

    def _descriptor(self, slot_key: str, field: str) -> Optional[Dict[str, Any]]:
        entry = self.slots.get(slot_key)
        if not isinstance(entry, dict):
            return None
        shm_obj = entry.get(f"{field}_shm")
        shape = tuple(int(v) for v in entry.get(f"{field}_shape", ()) or ())
        if shm_obj is None or not shape:
            return None
        return {"name": str(shm_obj.name), "shape": shape, "dtype": "float32"}

    def _upload(self, slot_key: str, field: str, array: np.ndarray) -> Dict[str, Any]:
        entry = self.slots.setdefault(slot_key, {})
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        required = max(int(arr.nbytes), 4)
        shm_obj = entry.get(f"{field}_shm")
        if shm_obj is None or int(entry.get(f"{field}_bytes", 0) or 0) < required:
            _close_unlink(shm_obj)
            shm_obj = mp_shared_memory.SharedMemory(create=True, size=required)
            entry[f"{field}_shm"] = shm_obj
            entry[f"{field}_bytes"] = required
        shape = tuple(int(v) for v in arr.shape)
        entry[f"{field}_shape"] = shape
        np.ndarray(shape, dtype=np.float32, buffer=shm_obj.buf)[:] = arr
        return {"name": str(shm_obj.name), "shape": shape, "dtype": "float32"}

    def _clear_field(self, slot_key: str, field: str):
        entry = self.slots.get(slot_key)
        if not isinstance(entry, dict):
            return
        _close_unlink(entry.pop(f"{field}_shm", None))
        entry.pop(f"{field}_bytes", None)
        entry.pop(f"{field}_shape", None)
