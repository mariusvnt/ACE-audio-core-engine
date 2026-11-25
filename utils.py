"""utils.py contains helper functions used across the add-on."""
import sys
import os
import threading
import time
import importlib.util
from typing import Any

# Preview cache to avoid leaking ImagePreviewCollection instances
_preview_collections = {}
_logo_icon_id = None
_DEBUG_LOG_ENABLED = False
_DEBUG_STRICT = True  # Guardrail defaults to hard-fail to block unsafe off-main-thread bpy writes.
_module_presence_cache = {}


def set_debug_strict(enabled: bool) -> None:
    """Enable strict guardrail mode."""
    global _DEBUG_STRICT
    _DEBUG_STRICT = bool(enabled)


def assert_main_thread(context: str = "") -> None:
    """Assert that the current call is on the Blender main thread."""
    if threading.current_thread() is not threading.main_thread():
        caller = context or "unknown"
        msg = (
            f"assert_main_thread: bpy write attempted off main thread in '{caller}'. "
            "This is a guardrail violation — results are undefined."
        )
        if _DEBUG_STRICT:
            raise RuntimeError(msg)
        log(msg, "ERROR")


def as_int(value: Any, default: int) -> int:
    """Return *value* as an int, or *default* when it is None, empty or not a number."""
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def as_float(value: Any, default: float) -> float:
    """Return *value* as a float, or *default* when it is None, empty or not a number."""
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def set_debug_logging(enabled: bool):
    """Enable or disable add-on debug logging."""
    global _DEBUG_LOG_ENABLED
    _DEBUG_LOG_ENABLED = bool(enabled)

def log(message, type='INFO', debug_active: bool | None = None):
    """Print *message* to the console subject to the current debug mode."""
    # ERROR always prints
    if type == 'ERROR':
        print(f"[ACE {type}] {message}")
        return

    # All other types (INFO, WARNING, DEBUG) only print when debug is enabled
    flag = _DEBUG_LOG_ENABLED if debug_active is None else bool(debug_active)
    if flag:
        print(f"[ACE {type}] {message}")


_THROTTLE_LAST: dict = {}
_THROTTLE_MAX_KEYS = 512
_SWALLOWED: dict = {}
_SWALLOWED_LOCK = threading.Lock()


def log_throttled(key: str, message: str, level: str = "ERROR", min_interval: float = 2.0) -> bool:
    """Log *message* at most once per *min_interval* seconds per *key*."""
    now = time.monotonic()
    last = _THROTTLE_LAST.get(key)
    if last is not None and (now - last) < float(min_interval):
        return False
    if len(_THROTTLE_LAST) >= _THROTTLE_MAX_KEYS:
        _THROTTLE_LAST.clear()
    _THROTTLE_LAST[key] = now
    log(message, level)
    return True


def note_swallowed(key: str, exc: BaseException, level: str = "DEBUG") -> None:
    """Record an exception a fallback path absorbs, and log it (throttled)."""
    summary = f"{type(exc).__name__}: {exc}"
    with _SWALLOWED_LOCK:
        entry = _SWALLOWED.get(key)
        if entry is None:
            _SWALLOWED[key] = [1, summary]
        else:
            entry[0] += 1
            entry[1] = summary
    log_throttled(f"swallowed:{key}", f"{key} absorbed {summary}", level, min_interval=5.0)


def swallowed_errors() -> dict:
    """Return ``{key: (count, last_error)}`` for every suppressed error so far."""
    with _SWALLOWED_LOCK:
        return {k: (v[0], v[1]) for k, v in _SWALLOWED.items()}


def reset_swallowed_errors() -> None:
    with _SWALLOWED_LOCK:
        _SWALLOWED.clear()


class logged_errors:
    """Context manager that suppresses ``Exception`` and records it."""

    __slots__ = ("key", "level")

    def __init__(self, key: str, level: str = "DEBUG"):
        self.key = key
        self.level = level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb):
        if exc_type is None or not issubclass(exc_type, Exception):
            return False
        note_swallowed(self.key, exc, self.level)
        return True

def get_python_executable():
    """Return the path to the Python executable used by Blender."""
    return sys.executable


def dependency_vendor_path(create: bool = False) -> str:
    """Return the path to the add-on's local _vendor directory."""
    path = os.path.join(os.path.dirname(__file__), "_vendor")
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            note_swallowed("utils.dependency_vendor_path", exc)
    return path


def ensure_dependency_vendor_path(create: bool = False) -> str:
    """Ensure the _vendor directory is on sys.path and return its path."""
    path = dependency_vendor_path(create=create)
    if path and os.path.isdir(path) and path not in sys.path:
        # Keep add-on-local dependencies ahead of ambient interpreter paths.
        sys.path.insert(0, path)
    return path

def is_module_installed(module_name):
    """Check whether a module is importable in Blender's Python environment."""
    key = str(module_name or "").strip()
    if not key:
        return False
    cached = _module_presence_cache.get(key)
    if cached is not None:
        return bool(cached)

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        # Module files are not on sys.path at all.
        _module_presence_cache[key] = False
        return False

    # Module files were found — verify they actually load (native extensions,
    # DLL dependencies, etc.).
    try:
        __import__(module_name)
        _module_presence_cache[key] = True
        return True
    except Exception as exc:  # noqa: BLE001 boundary: any import failure means "not installed"
        # Native extension failed to initialise (missing DLL, ABI mismatch …).
        log(f"Module '{module_name}' found but failed to import: {type(exc).__name__}: {exc}", "DEBUG")
        _module_presence_cache[key] = False
        return False

def is_module_in_vendor(module_name: str) -> bool:
    """Return True if the module can be found inside ACE's local _vendor directory."""
    vendor_dir = dependency_vendor_path(create=False)
    if not vendor_dir or not os.path.isdir(vendor_dir):
        return False
    token = str(module_name or "").strip().lower().replace("-", "_")
    if not token:
        return False
    try:
        for entry in os.listdir(vendor_dir):
            normalized = entry.lower().replace("-", "_")
            if normalized == token or normalized.startswith(token + "-") or normalized.startswith(token + "."):
                return True
    except OSError as exc:
        note_swallowed("utils.is_module_in_vendor", exc)
    return False


def refresh_import_caches():
    """Invalidate importlib caches to reflect recent installs/uninstalls."""
    importlib.invalidate_caches()
    _module_presence_cache.clear()


def compose_live_flow_identifier(source_like: Any) -> str:
    """Build the internal live routing token from structured source settings."""
    if source_like is None:
        return "sd:"

    transport = str(getattr(source_like, "live_transport", "") or "").strip().upper()

    # Backward-compat fallback for older data with only flow_identifier.
    legacy_flow = str(getattr(source_like, "flow_identifier", "") or "").strip()
    if not transport:
        return legacy_flow or "sd:"
    if transport not in {"SOUNDDEVICE", "UDP"}:
        transport = "SOUNDDEVICE"

    if transport == "UDP":
        port = as_int(getattr(source_like, "live_udp_port", 7001) or 7001, 7001)
        channels = as_int(getattr(source_like, "live_udp_channels", 2) or 2, 2)
        sample_rate = as_int(getattr(source_like, "live_udp_sample_rate", 44100) or 44100, 44100)

        port = min(max(port, 1), 65535)
        channels = min(max(channels, 1), 8)
        sample_rate = min(max(sample_rate, 8000), 192000)
        payload = f"udp:{port}:{channels}:{sample_rate}"
    else:
        token = str(getattr(source_like, "live_device_selector", "") or "").strip()
        if token and token not in {"__AUTO__", "__MISSING__"}:
            payload = f"sd:{token}"
        else:
            payload = "sd:"
    return payload

def _ensure_logo_icon():
    """Load the ACE logo image into a preview collection and return its icon ID."""
    import os
    from .vars import ACE_LOGO_FILENAME

    global _logo_icon_id
    if _logo_icon_id is not None:
        return _logo_icon_id

    addon_dir = os.path.dirname(__file__)
    logo_path = os.path.join(addon_dir, ACE_LOGO_FILENAME)
    if not os.path.exists(logo_path): return None

    import bpy.utils.previews                                                                                                               # type: ignore

    # Reuse a single preview collection to avoid leaks
    pcoll = _preview_collections.get("ace_logo")
    if pcoll is None:
        pcoll = bpy.utils.previews.new()
        _preview_collections["ace_logo"] = pcoll
        pcoll.load("ace_logo", logo_path, 'IMAGE')
    _logo_icon_id = pcoll["ace_logo"].icon_id
    return _logo_icon_id


def unregister():
    """Cleanup preview collections to prevent ResourceWarnings on reload/unload."""
    import bpy.utils.previews                                                                                                               # type: ignore
    for pcoll in _preview_collections.values():
        with logged_errors("utils.unregister"):
            bpy.utils.previews.remove(pcoll)
    _preview_collections.clear()
    global _logo_icon_id
    _logo_icon_id = None

def _get_active_source(context):
    ace = getattr(context.scene, "audiocore", None)
    if not ace or not ace.audio_sources: return ace, None
    idx = getattr(ace, "active_audio_source_index", 0)
    if idx < 0 or idx >= len(ace.audio_sources): return ace, None
    return ace, ace.audio_sources[idx]
