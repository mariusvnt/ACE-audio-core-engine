"""
utils.py contains helper functions used across the add-on.
"""
import sys
import os
import threading
import importlib.util
from typing import Any

# Preview cache to avoid leaking ImagePreviewCollection instances
_preview_collections = {}
_logo_icon_id = None
_DEBUG_LOG_ENABLED = False
_DEBUG_STRICT = True  # Guardrail defaults to hard-fail to block unsafe off-main-thread bpy writes.
_module_presence_cache = {}


# --- Log and environment helpers ---
def set_debug_strict(enabled: bool) -> None:
    """Enable strict guardrail mode.

    When enabled, assert_main_thread() raises RuntimeError on violations instead
    of logging an ERROR. Disable only for controlled test/debug scenarios.

    Default: True.
    """
    global _DEBUG_STRICT
    _DEBUG_STRICT = bool(enabled)


def assert_main_thread(context: str = "") -> None:
    """Assert that the current call is on the Blender main thread.

    In strict mode (_DEBUG_STRICT=True, default): raises RuntimeError so write
    paths fail-safe instead of continuing with undefined Blender state.
    In non-strict mode: logs an ERROR only.
    Call this at the entry points of any function that writes bpy state.
    """
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
    """Return *value* as an int, or *default* when it is None, empty or not a number.

    Unlike ``int(value or default)``, a real 0 stays 0.
    """
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def as_float(value: Any, default: float) -> float:
    """Return *value* as a float, or *default* when it is None, empty or not a number.

    Unlike ``float(value or default)``, a real 0.0 stays 0.0.
    """
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def set_debug_logging(enabled: bool):
    """Enable or disable add-on debug logging.

    When disabled, only ERROR messages are printed.  When enabled, INFO,
    WARNING, and DEBUG messages are also printed.

    Args:
        enabled: True to enable debug output, False to suppress it.
    """
    global _DEBUG_LOG_ENABLED
    _DEBUG_LOG_ENABLED = bool(enabled)

def log(message, type='INFO', debug_active: bool | None = None):
    """Print *message* to the console subject to the current debug mode.

    ERROR messages are always printed regardless of debug state.  All other
    types (INFO, WARNING, DEBUG) are printed only when debug logging is active.

    Args:
        message: The message string to print.
        type: Severity level string — "INFO", "WARNING", "DEBUG", or "ERROR".
        debug_active: Override the global _DEBUG_LOG_ENABLED flag for this call.
            Pass None (default) to use the global flag.
    """
    # ERROR always prints
    if type == 'ERROR':
        print(f"[ACE {type}] {message}")
        return

    # All other types (INFO, WARNING, DEBUG) only print when debug is enabled
    flag = _DEBUG_LOG_ENABLED if debug_active is None else bool(debug_active)
    if flag:
        print(f"[ACE {type}] {message}")

def get_python_executable():
    """Return the path to the Python executable used by Blender.

    Returns:
        Absolute path string from sys.executable.
    """
    return sys.executable


def dependency_vendor_path(create: bool = False) -> str:
    """Return the path to the add-on's local _vendor directory.

    Args:
        create: When True, create the directory if it does not exist.

    Returns:
        Absolute path to the _vendor directory alongside this module.
    """
    path = os.path.join(os.path.dirname(__file__), "_vendor")
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    return path


def ensure_dependency_vendor_path(create: bool = False) -> str:
    """Ensure the _vendor directory is on sys.path and return its path.

    Inserts the directory at the front of sys.path so add-on-local packages
    take precedence over system-wide or Blender-bundled packages with the same
    name.

    Args:
        create: When True, create the directory if it does not exist.

    Returns:
        Absolute path to the _vendor directory.
    """
    path = dependency_vendor_path(create=create)
    if path and os.path.isdir(path) and path not in sys.path:
        # Keep add-on-local dependencies ahead of ambient interpreter paths.
        sys.path.insert(0, path)
    return path

def is_module_installed(module_name):
    """
    Check whether a module is importable in Blender's Python environment.

    find_spec() alone is not sufficient — it only locates the module files but
    does NOT verify that native extensions (e.g. sounddevice's _portaudio*.pyd)
    can actually be loaded. We therefore do a real import test when find_spec
    succeeds, catching all Exception subclasses (not just ImportError) to also
    catch OSError / DLL-load failures on Windows.
    """
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
    except Exception:
        # Native extension failed to initialise (missing DLL, ABI mismatch …).
        _module_presence_cache[key] = False
        return False

def is_module_in_vendor(module_name: str) -> bool:
    """
    Return True if the module can be found inside ACE's local _vendor directory.
    Used by the uninstall operator to distinguish between modules installed via
    AudioCore's installer and those installed by other means (system pip,
    Blender Extensions, etc.).
    """
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
    except Exception:
        pass
    return False


def refresh_import_caches():
    """
    Invalidate importlib caches to reflect recent installs/uninstalls.
    """
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass
    _module_presence_cache.clear()


def compose_live_flow_identifier(source_like: Any) -> str:
    """Build the internal live routing token from structured source settings.

    Constructs a canonical token string from the live_transport,
    live_device_selector, and UDP parameter attributes on *source_like*.
    Falls back to the legacy flow_identifier string for backward compatibility
    with older .blend files that stored only the raw token.

    Args:
        source_like: An ACEAudioSource property-group instance, or any object
            that exposes the relevant attributes.  None produces "sd:".

    Returns:
        A routing token string such as "sd:", "sd:<device>", or
        "udp:<port>:<channels>:<sample_rate>".
    """
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
        try:
            port = int(getattr(source_like, "live_udp_port", 7001) or 7001)
        except Exception:
            port = 7001
        try:
            channels = int(getattr(source_like, "live_udp_channels", 2) or 2)
        except Exception:
            channels = 2
        try:
            sample_rate = int(getattr(source_like, "live_udp_sample_rate", 44100) or 44100)
        except Exception:
            sample_rate = 44100

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

# --- Logo Helper ---
def _ensure_logo_icon():
    """Load the ACE logo image into a preview collection and return its icon ID.

    Uses a module-level cache so the preview is loaded only once.  Returns
    None if the logo file is missing or bpy.utils.previews is unavailable.

    Returns:
        Integer Blender icon ID, or None.
    """
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
        try:
            bpy.utils.previews.remove(pcoll)
        except Exception:
            pass
    _preview_collections.clear()
    global _logo_icon_id
    _logo_icon_id = None

# --- UI Helper ---
def _get_active_source(context):
    """Return the (ace_settings, active_source) pair for the current context.

    Args:
        context: The current Blender context.

    Returns:
        A (ACEProjectSettings, ACEAudioSource) tuple.  The source element is
        None when there are no audio sources or the active index is out of
        range.  The ace_settings element is None when the scene has no
        audiocore property.
    """
    ace = getattr(context.scene, "audiocore", None)
    if not ace or not ace.audio_sources: return ace, None
    idx = getattr(ace, "active_audio_source_index", 0)
    if idx < 0 or idx >= len(ace.audio_sources): return ace, None
    return ace, ace.audio_sources[idx]
