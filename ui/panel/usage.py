"""Engine Status meters, averaged over a short window: latency, whole-machine CPU and GPU load.

CPU: Windows and Linux. GPU: NVIDIA cards (NVML) and, on Linux, AMD cards. Elsewhere unknown.
"""

import ctypes
import glob
import sys
import time
from collections import deque
from ... import utils

_INTERVAL = 0.25
_WINDOW = 2.0  # seconds averaged, to keep the meters calm

_last_sample = 0.0
_last_cpu_times: tuple[float, float] | None = None  # (busy, total)
_last_cpu = None
_history: dict[str, deque] = {}  # key -> (time, value)
_nvml = None
_nvml_tried = False


def average(key: str) -> float | None:
    """Mean of the recent measures of *key* ("cpu", "gpu" as 0..1, latencies in ms)."""
    samples = _history.get(key)
    return sum(value for _time, value in samples) / len(samples) if samples else None


def sample(now: float | None = None, extra: dict[str, float] | None = None) -> bool:
    """Measure again if the last measure is old enough; True when a shown value changed.

    *extra* values (the latencies) are recorded with the machine's load.
    """
    global _last_sample
    now = time.monotonic() if now is None else now
    if now - _last_sample < _INTERVAL:
        return False
    _last_sample = now
    before = _shown()
    measures = dict(extra or {})
    for key, read in (("cpu", _cpu_load), ("gpu", _gpu_load)):
        try:
            measures[key] = read()
        except Exception as exc:  # noqa: BLE001 boundary: a driver or OS query must not break the UI
            utils.note_swallowed("usage.sample", exc)
            measures[key] = None
    for key, value in measures.items():
        samples = _history.setdefault(key, deque())
        if value is not None:
            samples.append((now, float(value)))
        while samples and now - samples[0][0] > _WINDOW:
            samples.popleft()
    return _shown() != before


def _shown() -> tuple:
    return tuple(
        (key, None if average(key) is None else round(average(key) * (100.0 if key in ("cpu", "gpu") else 10.0)))
        for key in sorted(_history)
    )


def _cpu_times() -> tuple[float, float] | None:
    if sys.platform == "win32":
        times = [ctypes.c_ulonglong() for _ in range(3)]  # idle, kernel (includes idle), user
        if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(t) for t in times)):
            return None
        idle, kernel, user = (float(t.value) for t in times)
        return kernel + user - idle, kernel + user
    if sys.platform.startswith("linux"):
        with open("/proc/stat", encoding="ascii") as handle:
            values = [float(v) for v in handle.readline().split()[1:9]]
        return sum(values) - values[3] - values[4], sum(values)
    return None


def _cpu_load() -> float | None:
    """Share of all cores in use since the previous call."""
    global _last_cpu_times, _last_cpu
    current = _cpu_times()
    previous, _last_cpu_times = _last_cpu_times, current
    if current is not None and previous is not None and current[1] > previous[1]:
        _last_cpu = min(max((current[0] - previous[0]) / (current[1] - previous[1]), 0.0), 1.0)
    return _last_cpu


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _open_nvml():
    global _nvml, _nvml_tried
    if not _nvml_tried:
        _nvml_tried = True
        try:
            library = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
            if library.nvmlInit_v2() == 0:
                _nvml = library
        except OSError:
            _nvml = None  # no NVIDIA driver
    return _nvml


def _gpu_load() -> float | None:
    """Busiest GPU."""
    library = _open_nvml()
    loads: list[float] = []
    if library is not None:
        count = ctypes.c_uint()
        library.nvmlDeviceGetCount_v2(ctypes.byref(count))
        for index in range(count.value):
            handle, used = ctypes.c_void_p(), _Utilization()
            if library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                continue
            if library.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(used)) == 0:
                loads.append(used.gpu / 100.0)
    elif sys.platform.startswith("linux"):
        for path in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
            with open(path, encoding="ascii") as handle:
                loads.append(float(handle.read().strip()) / 100.0)
    return max(loads) if loads else None


def close() -> None:
    """Release the NVIDIA library (add-on unregister)."""
    global _nvml, _nvml_tried
    if _nvml is not None:
        with utils.logged_errors("usage.close"):
            _nvml.nvmlShutdown()
    _nvml, _nvml_tried = None, False
