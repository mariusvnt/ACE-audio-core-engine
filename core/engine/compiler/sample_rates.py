"""Working sample rate per source: probe floor and file caps."""

from typing import Optional
from .... import utils
from ...input.reception import build_live_flow_identifier
from ...tasks.chunk import SAMPLE_RATE
from ...tasks.source import SourceTask
from ..common import engine_settings


class SampleRates:
    def _collect_sample_rate_floor(self, sources, engine_settings, fps: float, allow_io: bool = True) -> int:
        min_sr = None
        for source in sources or []:
            source_type = str(getattr(source, "source_type", "") or "").upper()
            if source_type == "FILE":
                try:
                    sr = int(getattr(source, "native_sample_rate", 0) or 0)
                except Exception as exc:
                    utils.note_swallowed("sample_rates.SampleRates._collect_sample_rate_floor", exc)
                    sr = 0
            else:
                sr = int(getattr(source, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            cache_key = ""
            mtime = None
            with utils.logged_errors("sample_rates.SampleRates._collect_sample_rate_floor#2"):
                if source_type == "LIVE":
                    live_flow_id = build_live_flow_identifier(source)
                    with utils.logged_errors("sample_rates.SampleRates._collect_sample_rate_floor#3"):
                        source.flow_identifier = live_flow_id
                    # Compile path must not touch live devices/sockets.
                    live_transport = str(getattr(source, "live_transport", "") or "").upper()
                    udp_sr = int(getattr(source, "live_udp_sample_rate", 0) or 0)
                    if live_transport == "UDP" and udp_sr > 0:
                        sr = udp_sr
                    elif sr <= 0:
                        sr = SAMPLE_RATE
                    with utils.logged_errors("sample_rates.SampleRates._collect_sample_rate_floor#4"):
                        source.sample_rate = int(sr)
                    if min_sr is None or sr < min_sr:
                        min_sr = sr
                    continue

                # Fast path: trust existing metadata when available
                if source_type == "FILE" and sr > 0 and getattr(source, "duration", 0.0) > 0.0:
                    working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
                    if min_sr is None or working_sr < min_sr:
                        min_sr = working_sr
                    continue
                if source_type == "FILE":
                    import os
                    cache_key = f"FILE:{getattr(source, 'filepath', '')}"
                    try:
                        mtime = os.path.getmtime(getattr(source, "filepath", ""))
                    except OSError:
                        mtime = None
                    cached = self.engine.probes._meta_cache.get(cache_key)
                    ts_ok = cached and (getattr(__import__("time"), "time")() - cached.get("ts", 0.0) <= self.engine.probes._meta_cache_ttl)
                    if cached and cached.get("mtime") == mtime and ts_ok:
                        sr = int(cached.get("native_sample_rate", cached.get("sample_rate", sr)) or sr)
                        with utils.logged_errors("sample_rates.SampleRates._collect_sample_rate_floor#5"):
                            source.native_sample_rate = int(sr)
                        if getattr(source, "duration", 0.0) == 0.0:
                            with utils.logged_errors("sample_rates.SampleRates._collect_sample_rate_floor#6"):
                                source.duration = float(cached.get("duration", 0.0))
                        working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
                        if min_sr is None or working_sr < min_sr:
                            min_sr = working_sr
                        continue
                if source_type == "FILE":
                    # Probe in the background; the result triggers a recompile.
                    self.engine.probes._dispatch_async_probe(getattr(source, "filepath", ""))
                # Use current sr as fallback; updated on recompile after probe completes.
            working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
            if min_sr is None or working_sr < min_sr:
                min_sr = working_sr

        if min_sr is None:
            min_sr = SAMPLE_RATE
        return int(min_sr)

    @classmethod
    def _resolve_source_sample_rate(cls, source, fallback_sr: int, scene_or_engine_settings=None) -> int:
        sr = cls._source_original_sample_rate(source)
        if sr <= 0:
            sr = int(fallback_sr)
        source_type = str(getattr(source, "source_type", "") or "").strip().upper()
        return cls._cap_source_sample_rate(source_type, sr, scene_or_engine_settings)

    @classmethod
    def _file_working_sample_rate_cap(cls, scene_or_engine_settings=None) -> int:
        settings = engine_settings(scene_or_engine_settings)
        default_cap = int(cls._FILE_WORKING_SAMPLE_RATE_CAP)
        try:
            raw = str(
                getattr(settings, "ace_file_working_sample_rate", str(default_cap)) or str(default_cap)
            ).strip().lower()
        except Exception as exc:
            utils.note_swallowed("sample_rates.SampleRates._file_working_sample_rate_cap", exc)
            raw = str(default_cap)
        if raw in {"native", "off", "0"}:
            return 0
        try:
            return max(int(raw), 0)
        except Exception as exc:
            utils.note_swallowed("sample_rates.SampleRates._file_working_sample_rate_cap#2", exc)
            return default_cap

    @classmethod
    def _cap_source_sample_rate(
        cls,
        source_type: str,
        sample_rate: int,
        scene_or_engine_settings=None,
    ) -> int:
        sr = max(int(sample_rate or SAMPLE_RATE), 1)
        if str(source_type or "").strip().upper() == "FILE":
            cap_sr = cls._file_working_sample_rate_cap(scene_or_engine_settings)
            if cap_sr > 0:
                sr = min(sr, int(cap_sr))
        return sr

    @staticmethod
    def _source_original_sample_rate(source) -> int:
        source_type = str(getattr(source, "source_type", "") or "").strip().upper()
        if source_type == "FILE":
            try:
                sr = int(getattr(source, "native_sample_rate", 0) or 0)
            except Exception as exc:
                utils.note_swallowed("sample_rates.SampleRates._source_original_sample_rate", exc)
                sr = 0
            if sr > 0:
                return sr
        try:
            return int(getattr(source, "sample_rate", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("sample_rates.SampleRates._source_original_sample_rate#2", exc)
            return 0

    def _source_task_target_sample_rate(self, task: Optional[SourceTask]) -> Optional[int]:
        if not isinstance(task, SourceTask):
            return None
        ctx = getattr(task, "source_context", None)
        source_type = str(getattr(ctx, "source_type", "") or "").strip().upper()
        if source_type != "FILE":
            return None
        cap_sr = self._file_working_sample_rate_cap(getattr(task, "engine_settings", None))
        return int(cap_sr) if cap_sr > 0 else None
