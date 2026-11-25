"""Background probing of source metadata and tempo detection."""

import threading
from typing import Any, Dict, Optional
from ... import utils
from ...properties import ACEAudioSource
from ..input.reception import AudioReception
from ..tasks.chunk import SAMPLE_RATE
from ..tasks.specs import SourceSpecsTask


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
            rec = AudioReception("FILE", filepath=fp)
            if rec.open():
                try:
                    native_sr = int(
                        getattr(rec, "native_sample_rate", 0)
                        or getattr(rec, "sample_rate", SAMPLE_RATE)
                        or SAMPLE_RATE
                    )
                    duration = float(getattr(rec, "duration", 0.0) or 0.0)
                finally:
                    with utils.logged_errors("probes.SourceProbes._run_probe_in_bg"):
                        rec.close()
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
                source.native_sample_rate = native_sr
                source.duration = float(result.get("duration", 0.0) or 0.0)
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
        if any_updated:
            with utils.logged_errors("probes.SourceProbes._flush_probe_updates#2"):
                preferred_tree = self.engine.frontends._resolve_preferred_tree_from_discoveries(
                    self.engine.frontends.discover_scene_frontends(scene) if scene is not None else [],
                    frontend_uid=str(getattr(self.engine.compiler, "_compiled_tree_frontend_uid", "") or ""),
                    tree_ptr=int(preferred_tree_ptr or 0),
                    tree_bl_idname=str(getattr(self.engine.compiler, "_compiled_tree_bl_idname", "") or ""),
                )
                self.engine.frontends.request_recompile(preferred_tree)
        return any_updated

    def _apply_tempo_detection_to_source(self, source: ACEAudioSource, detected: Dict[str, Any]) -> bool:
        bpm = float(detected.get("bpm", 0.0) or 0.0)
        beats_per_bar = int(detected.get("beats_per_bar", 0) or 0)
        if bpm <= 0.0 or beats_per_bar <= 0:
            return False
        try:
            source["ace_bpm_auto"] = float(bpm)
            source["ace_meter_auto"] = int(beats_per_bar)
            source.bpm = float(bpm)
            source.beats_per_bar = int(beats_per_bar)
            source["ace_bpm_user_override"] = False
            source["ace_meter_user_override"] = False
            source["ace_tempo_detect_status"] = "ready"
            source["ace_tempo_detect_message"] = f"Detected {float(bpm):.2f} BPM / {int(beats_per_bar)} BPB"
            return True
        except Exception as exc:
            utils.note_swallowed("probes.SourceProbes._apply_tempo_detection_to_source", exc)
            return False

    def request_source_tempo_detection(self, source: Optional[ACEAudioSource]) -> tuple[bool, str]:
        """Queue non-blocking BPM / beats-per-bar detection for a FILE source."""
        utils.assert_main_thread("request_source_tempo_detection")
        if source is None:
            return False, "No source selected."

        target = SourceSpecsTask.resolve_analysis_target(source)
        if target is None:
            with utils.logged_errors("probes.SourceProbes.request_source_tempo_detection"):
                source["ace_tempo_detect_status"] = "failed"
                source["ace_tempo_detect_message"] = "Select a valid file source before detecting tempo."
            return False, "Select a valid file source before detecting tempo."

        filepath, mtime = target
        with utils.logged_errors("probes.SourceProbes.request_source_tempo_detection#2"):
            source["ace_tempo_detect_status"] = "pending"
            source["ace_tempo_detect_message"] = "Analyzing source tempo in the background..."
            source["ace_tempo_detect_filepath"] = str(filepath)
            source["ace_tempo_detect_mtime"] = float(mtime)

        cached = SourceSpecsTask.get_cached_analysis(target)
        if cached:
            if cached.get("bpm", None) is not None and self._apply_tempo_detection_to_source(source, cached):
                return True, "Tempo detection updated from cache."
            error_msg = str(cached.get("error", "") or "Tempo analysis failed.")
            with utils.logged_errors("probes.SourceProbes.request_source_tempo_detection#3"):
                source["ace_tempo_detect_status"] = "failed"
                source["ace_tempo_detect_message"] = error_msg
            return False, error_msg

        queued = SourceSpecsTask._queue_background_analysis(target)
        if not queued:
            with utils.logged_errors("probes.SourceProbes.request_source_tempo_detection#4"):
                source["ace_tempo_detect_status"] = "failed"
                source["ace_tempo_detect_message"] = "Tempo analysis queue is full. Try again in a moment."
            return False, "Tempo analysis queue is full. Try again in a moment."

        return True, "Tempo detection queued."

    def flush_source_tempo_detection_updates(self, sources) -> bool:
        """Apply finished background tempo detections to FILE source properties."""
        utils.assert_main_thread("flush_source_tempo_detection_updates")
        any_updated = False
        for source in (sources or []):
            if getattr(source, "source_type", "") != "FILE":
                continue
            try:
                status = str(source.get("ace_tempo_detect_status", "") or "")
            except Exception as exc:
                utils.note_swallowed("probes.SourceProbes.flush_source_tempo_detection_updates", exc)
                status = ""
            if status != "pending":
                continue

            try:
                filepath = str(source.get("ace_tempo_detect_filepath", "") or "")
                mtime = float(source.get("ace_tempo_detect_mtime", 0.0) or 0.0)
            except Exception as exc:
                utils.note_swallowed("probes.SourceProbes.flush_source_tempo_detection_updates#2", exc)
                filepath = ""
                mtime = 0.0
            if not filepath or mtime <= 0.0:
                continue

            current_target = SourceSpecsTask.resolve_analysis_target(source)
            if current_target is None or str(current_target[0]) != filepath or abs(float(current_target[1]) - mtime) > 1e-6:
                with utils.logged_errors("probes.SourceProbes.flush_source_tempo_detection_updates#3"):
                    source["ace_tempo_detect_status"] = "failed"
                    source["ace_tempo_detect_message"] = "Source changed. Run tempo detection again."
                any_updated = True
                continue

            cached = SourceSpecsTask.get_cached_analysis((filepath, mtime))
            if not cached:
                continue

            if cached.get("bpm", None) is None:
                with utils.logged_errors("probes.SourceProbes.flush_source_tempo_detection_updates#4"):
                    source["ace_tempo_detect_status"] = "failed"
                    source["ace_tempo_detect_message"] = str(
                        cached.get("error", "") or "Tempo analysis failed."
                    )
                any_updated = True
                continue

            if self._apply_tempo_detection_to_source(source, cached):
                any_updated = True
        return any_updated
