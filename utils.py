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
_module_presence_cache = {}

# --- Log and environment helpers ---
def assert_main_thread(context: str = "") -> None:
    """
    Assert that the current call is on the Blender main thread.
    Logs an ERROR if violated — does NOT raise, to avoid crashing the host.
    Call this at the entry points of any function that writes bpy state.
    """
    if threading.current_thread() is not threading.main_thread():
        caller = context or "unknown"
        log(
            f"assert_main_thread: bpy write attempted off main thread in '{caller}'. "
            "This is a guardrail violation — results are undefined.",
            "ERROR",
        )


def set_debug_logging(enabled: bool):
    """
    Enable or disable the add-on debug logging mode.
    """
    global _DEBUG_LOG_ENABLED
    _DEBUG_LOG_ENABLED = bool(enabled)

def log(message, type='INFO', debug_active: bool | None = None):
    """
    Prints to console if debug is enabled or type is ERROR.
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
    """
    Return the path to the Python executable used by Blender.
    """
    return sys.executable


def dependency_vendor_path(create: bool = False) -> str:
    path = os.path.join(os.path.dirname(__file__), "_vendor")
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    return path


def ensure_dependency_vendor_path(create: bool = False) -> str:
    path = dependency_vendor_path(create=create)
    if path and os.path.isdir(path) and path not in sys.path:
        # Keep add-on-local dependencies ahead of ambient interpreter paths.
        sys.path.insert(0, path)
    return path

def is_module_installed(module_name):
    """
    Check whether a module exists without importing it (avoids cascading
    ImportError when optional deps are missing).
    """
    key = str(module_name or "").strip()
    if not key:
        return False
    cached = _module_presence_cache.get(key)
    if cached is not None:
        return bool(cached)
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        _module_presence_cache[key] = True
        return True
    try:
        __import__(module_name)
        _module_presence_cache[key] = True
        return True
    except ImportError:
        _module_presence_cache[key] = False
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
    """
    Build the internal live routing token from structured source settings.
    Falls back to legacy `flow_identifier` data for backward compatibility.
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
    ace = getattr(context.scene, "audiocore", None)
    if not ace or not ace.audio_sources: return ace, None
    idx = getattr(ace, "active_audio_source_index", 0)
    if idx < 0 or idx >= len(ace.audio_sources): return ace, None
    return ace, ace.audio_sources[idx]

