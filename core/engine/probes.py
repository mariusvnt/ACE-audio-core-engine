"""Background probing of source metadata and tempo detection."""

import threading
from typing import Any, Dict, Optional
from ... import utils
from ...properties import ACEAudioSource
from ..input import decode
from ..tasks.chunk import SAMPLE_RATE
from ..tasks.tempo import TempoService


class SourceProbes:
    """Background probing of source metadata and tempo detection."""

    _META_CACHE_MAX_ENTRIES = 256

    def __init__(self, engine):
        self.engine = engine
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self._meta_cache_ttl = 30.0  # seconds
        # Async metadata probe: background thread writes here; main thread flushes to bpy props.
        self._probe_pending_updates: Dict[str, Dict[str, Any]] = {}
        self._probe_thread_lock = threading.Lock()
        self._probe_in_flight: set[str] = set()

    def _clear_probe_runtime_state(self) -> None:
        with self._probe_thread_lock:
            self._probe_pending_updates.clear()
            self._probe_in_flight.clear()

    def _probe_source_metadata(self, source: ACEAudioSource, allow_io: bool = True):
        """Populate ``source.sample_rate``, ``native_sample_rate``, and ``duration``."""
        if getattr(source, "source_type", "") != "FILE":
            return
        fp = getattr(source, "filepath", "")
        if not fp:
            return
        import os, time
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = None
        current_duration = float(getattr(source, "duration", 0.0) or 0.0)
        current_sample_rate = int(getattr(source, "sample_rate", 0) or 0)
        should_refresh_working_rate = current_duration <= 0.0 or current_sample_rate <= 0
        cache_key = f"FILE:{fp}"
        cached = self._meta_cache.get(cache_key)
        if cached and cached.get("mtime") == mtime and (time.time() - cached.get("ts", 0.0) <= self._meta_cache_ttl):
            with utils.logged_errors("probes.SourceProbes._probe_source_metadata"):
                cached_sample_rate = int(cached.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
                if should_refresh_working_rate:
                    source.sample_rate = cached_sample_rate
                source.native_sample_rate = int(cached.get("native_sample_rate", cached_sample_rate) or cached_sample_rate)
                source.duration = cached.get("duration", 0.0)
            return
        # Cache miss: dispatch background probe to avoid blocking the main thread.
        # Results are flushed into bpy props by _flush_probe_updates() on the next frame.
        self._dispatch_async_probe(fp)

    def _dispatch_async_probe(self, fp: str):
        """Schedule a background file-metadata probe."""
        if not fp:
            return
        undo_generation = int(getattr(self.engine, "undo_generation", 0) or 0)
        with self._probe_thread_lock:
            if fp in self._probe_in_flight or fp in self._probe_pending_updates:
                return
            self._probe_in_flight.add(fp)
        t = threading.Thread(
            target=self._run_probe_in_bg,
            args=(fp, undo_generation),
            name="ACEProbeThread",
            daemon=True,
        )
        t.start()

    def _run_probe_in_bg(self, fp: str, undo_generation: int):
        """Worker: open the file, read metadata, write to _probe_pending_updates (no bpy access)."""
        try:
            import os as _os, time as _time
            try:
                mtime = _os.path.getmtime(fp)
            except OSError:
                mtime = None
            # Header only: decoding the samples is left to the source that reads them.
            try:
                info = decode.probe(fp)
            except (decode.DecodeError, OSError):
                info = None  # missing or unreadable: the source reports it when it opens
            if info is not None:
                native_sr = int(info.sample_rate or SAMPLE_RATE)
                duration = float(info.frames) / float(max(native_sr, 1))
                with self._probe_thread_lock:
                    if int(undo_generation) != int(getattr(self.engine, "undo_generation", 0) or 0):
                        return
                    self._probe_pending_updates[fp] = {
                        "sample_rate": native_sr,
                        "native_sample_rate": native_sr,
                        "duration": duration,
                        "mtime": mtime,
                        "ts": _time.time(),
                        "undo_generation": int(undo_generation),
                    }
        except Exception as exc:
            utils.note_swallowed("probes.SourceProbes._run_probe_in_bg#2", exc)
        finally:
            with self._probe_thread_lock:
                self._probe_in_flight.discard(fp)

    def _flush_probe_updates(self, sources, scene=None, preferred_tree_ptr: int = 0) -> bool:
        """Flush background-probe results into bpy source props and meta_cache."""
        with self._probe_thread_lock:
            if not self._probe_pending_updates:
                return False
            updates = dict(self._probe_pending_updates)
            self._probe_pending_updates.clear()
        any_updated = False
        # A probe repeats for a file it already measured (a fresh engine, an expired
        # cache entry); only a changed sample rate or duration needs a recompile.
        metadata_changed = False
        active_generation = int(getattr(self.engine, "undo_generation", 0) or 0)
        for source in (sources or []):
            if getattr(source, "source_type", "") != "FILE":
                continue
            fp = getattr(source, "filepath", "")
            if not fp or fp not in updates:
                continue
            result = updates[fp]
            if int(result.get("undo_generation", active_generation) or active_generation) != active_generation:
                continue
            with utils.logged_errors("probes.SourceProbes._flush_probe_updates"):
                native_sr = int(result.get("native_sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
                duration = float(result.get("duration", 0.0) or 0.0)
                if (
                    int(getattr(source, "native_sample_rate", 0) or 0) != native_sr
                    or abs(float(getattr(source, "duration", 0.0) or 0.0) - duration) > 1e-6
                    or not int(getattr(source, "sample_rate", 0) or 0)
                ):
                    metadata_changed = True
                source.native_sample_rate = native_sr
                source.duration = duration
                if not int(getattr(source, "sample_rate", 0) or 0):
                    source.sample_rate = native_sr
            cache_key = f"FILE:{fp}"
            self._meta_cache[cache_key] = result
            if len(self._meta_cache) > self._META_CACHE_MAX_ENTRIES:
                stale_keys = sorted(
                    self._meta_cache.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)) if isinstance(kv[1], dict) else 0.0,
                )[: max(len(self._meta_cache) - self._META_CACHE_MAX_ENTRIES, 0)]
                for stale_key, _ in stale_keys:
                    self._meta_cache.pop(stale_key, None)
            any_updated = True
        if metadata_changed:
            with utils.logged_errors("probes.SourceProbes._flush_probe_updates#2"):
                preferred_tree = self.engine.frontends._resolve_preferred_tree_from_discoveries(
                    self.engine.frontends.discover_scene_frontends(scene) if scene is not None else [],
                    frontend_uid=str(getattr(self.engine.compiler, "_compiled_tree_frontend_uid", "") or ""),
                    tree_ptr=int(preferred_tree_ptr or 0),
                    tree_bl_idname=str(getattr(self.engine.compiler, "_compiled_tree_bl_idname", "") or ""),
                )
                self.engine.frontends.request_recompile(preferred_tree)
        return any_updated

    @staticmethod
    def _user_set_tempo(source) -> bool:
        """True once the user typed a BPM or Beats per Bar: automatic detection leaves the source alone."""
        try:
            return bool(source.get("ace_bpm_user_override", False) or source.get("ace_meter_user_override", False))
        except Exception as exc:
            utils.note_swallowed("probes.SourceProbes._user_set_tempo", exc)
            return False

    @staticmethod
    def _set_detect_status(source, status: str, message: str) -> None:
        with utils.logged_errors("probes.SourceProbes._set_detect_status"):
            source["ace_tempo_detect_status"] = status
            source["ace_tempo_detect_message"] = message

    def _apply_tempo_detection_to_source(self, source: ACEAudioSource, detected: Dict[str, Any]) -> bool:
        bpm = float(detected.get("bpm", 0.0) or 0.0)
        beats_per_bar = int(detected.get("beats_per_bar", 0) or 0)
        if bpm <= 0.0 or beats_per_bar <= 0:
            return False
        try:
            source.bpm = float(bpm)
            source.beats_per_bar = int(beats_per_bar)
            # Writing the values ran their update callbacks, which mark a user override.
            source["ace_bpm_user_override"] = False
            source["ace_meter_user_override"] = False
            source["ace_tempo_detect_status"] = "ready"
            source["ace_tempo_detect_message"] = f"Detected {float(bpm):.2f} BPM / {int(beats_per_bar)} BPB"
            return True
        except Exception as exc:
            utils.note_swallowed("probes.SourceProbes._apply_tempo_detection_to_source", exc)
            return False

    def request_source_tempo_detection(self, source: Optional[ACEAudioSource], automatic: bool = False) -> tuple[bool, str]:
        """Queue non-blocking BPM / beats-per-bar detection for a FILE source.

        The Auto Detect button asks with automatic=False: its result replaces
        values the user typed. An automatic request leaves those alone.
        """
        utils.assert_main_thread("request_source_tempo_detection")
        if source is None:
            return False, "No source selected."

        target = TempoService.target(source)
        if target is None:
            self._set_detect_status(source, "failed", "Select a valid file source before detecting tempo.")
            return False, "Select a valid file source before detecting tempo."

        filepath, mtime = target
        with utils.logged_errors("probes.SourceProbes.request_source_tempo_detection"):
            source["ace_tempo_detect_status"] = "pending"
            source["ace_tempo_detect_message"] = "Analyzing source tempo in the background..."
            source["ace_tempo_detect_filepath"] = str(filepath)
            source["ace_tempo_detect_mtime"] = float(mtime)
            source["ace_tempo_detect_auto"] = bool(automatic)

        found = TempoService.result(target)
        if found:
            if self._take_result(source, found):
                return True, "Tempo detection updated from cache."
            return found.get("bpm") is not None, str(found.get("error", "") or "Tempo detection skipped.")

        if not TempoService.request(target):
            self._set_detect_status(source, "failed", "Tempo analysis queue is full. Try again in a moment.")
            if automatic:
                # Forget the file, so the next automatic pass asks again.
                with utils.logged_errors("probes.SourceProbes.request_source_tempo_detection#2"):
                    del source["ace_tempo_detect_filepath"]
            return False, "Tempo analysis queue is full. Try again in a moment."

        return True, "Tempo detection queued."

    def _take_result(self, source, found: Dict[str, Any]) -> bool:
        """Apply a finished detection to *source*; True when it wrote the tempo."""
        if found.get("bpm", None) is None:
            self._set_detect_status(source, "failed", str(found.get("error", "") or "Tempo detection failed."))
            return False
        if bool(source.get("ace_tempo_detect_auto", False)) and self._user_set_tempo(source):
            self._set_detect_status(source, "", "")  # the user typed a tempo while it ran
            return False
        return self._apply_tempo_detection_to_source(source, found)

    def _wants_automatic_detection(self, source) -> bool:
        """A FILE source with no typed tempo whose current file was not detected yet, or whose detection was lost."""
        target = TempoService.target(source)
        if target is None or self._user_set_tempo(source):
            return False
        try:
            recorded = (str(source.get("ace_tempo_detect_filepath", "") or ""),
                        float(source.get("ace_tempo_detect_mtime", 0.0) or 0.0))
            status = str(source.get("ace_tempo_detect_status", "") or "")
        except Exception as exc:
            utils.note_swallowed("probes.SourceProbes._wants_automatic_detection", exc)
            return False
        if recorded[0] != target[0] or abs(recorded[1] - target[1]) > 1e-6:
            return True
        # Pending, but the request is gone (an undo or reset stopped the detection thread).
        return status == "pending" and not TempoService.waiting(target) and TempoService.result(target) is None

    def flush_source_tempo_detection_updates(self, sources) -> bool:
        """Start automatic detections and apply finished ones to FILE source properties."""
        utils.assert_main_thread("flush_source_tempo_detection_updates")
        any_updated = False
        for source in (sources or []):
            if getattr(source, "source_type", "") != "FILE":
                continue
            if self._wants_automatic_detection(source):
                self.request_source_tempo_detection(source, automatic=True)
                any_updated = True
            try:
                status = str(source.get("ace_tempo_detect_status", "") or "")
                filepath = str(source.get("ace_tempo_detect_filepath", "") or "")
                mtime = float(source.get("ace_tempo_detect_mtime", 0.0) or 0.0)
            except Exception as exc:
                utils.note_swallowed("probes.SourceProbes.flush_source_tempo_detection_updates", exc)
                continue
            if status != "pending" or not filepath or mtime <= 0.0:
                continue

            current = TempoService.target(source)
            if current is None or str(current[0]) != filepath or abs(float(current[1]) - mtime) > 1e-6:
                self._set_detect_status(source, "failed", "Source changed. Run tempo detection again.")
                any_updated = True
                continue

            found = TempoService.result((filepath, mtime))
            if found:
                self._take_result(source, found)
                any_updated = True
        return any_updated
