bl_info = {
    "name": "AudioCore Engine",
    "author": "Marius Viennot",
    "version": (0, 5, 0),
    "blender": (5, 0, 0), # 5.0 needed for 'NodeSocketBundle'
    "location": "Node Editor > Add Menu / N-Panel",
    "description": (
        "Drive Geometry Node Trees with information streams extracted from audio sources in real-time. "
        "A performance engine to map and drive audio-reactive object motion."
    ),
    "category": "Animation",
}

import importlib
import sys
import os

# --- CONFIGURATION ---
PACKAGE_NAME = __name__

# --- DEPENDENCY HANDLING ---
def _ensure_dependency_paths():
    """
    Ensure add-on local vendored packages are importable.
    """
    try:
        from . import utils as _ace_utils
        _ace_utils.ensure_dependency_vendor_path(create=True)
    except ImportError:
        vendor_path = os.path.join(os.path.dirname(__file__), "_vendor")
        if vendor_path and os.path.isdir(vendor_path) and vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

_ensure_dependency_paths()

# --- MODULE REGISTRY ---
# Order matters: Primitives -> Core DSP -> Nodes -> UI -> Orchestration
# Sub-package __init__ modules are listed first so they are in sys.modules
# before their children are imported/reloaded (required for correct hot-reload).
MODULE_NAMES = [
    # ── Root ────────────────────────────────────────────────────────────────
    "vars",                          # 1.  Constants / definitions
    "utils",                         # 2.  Global helpers
    "properties",                    # 3.  Blender property groups (Scene data)
    # ── core/ sub-package ───────────────────────────────────────────────────
    "core",                          # 4.  Package marker (core/__init__.py)
    "core.abstract_tree",            # 5.  Abstract IR data structures
    "core.id_management",            # 6.  ID / hash utilities
    "core.audio_influx",             # 7.  Audio I/O (file + live)
    "core.audio_tasks_registry",     # 8.  DSP task class registry
    "core.parallel_dsp",             # 9.  Process-worker DSP helpers
    "core.proxy_management",         # 10. Bridge to Blender mesh attributes
    "core.engine_common",
    "core.engine_preview",
    "core.engine_diagnostics",
    "core.engine_probes",
    "core.engine_bake",
    "core.engine_parallel",
    "core.engine_frontends",
    "core.audio_engine",             # 11. Engine orchestration
    "core.engine_handlers",          # Blender handlers and timers
    # ── nodes/ sub-package ──────────────────────────────────────────────────
    "nodes",                         # 12. Package marker (nodes/__init__.py)
    "nodes.dummy_tree",              # 13. GN node factory & tree reader
    "nodes.shader_tree",             # 14. Shader node factory & tree reader
    "nodes.dynamic_audio_operators", # 15. Dynamic splitter/math variants
    # ── ui/ sub-package ─────────────────────────────────────────────────────
    "ui",                            # 16. Package marker (ui/__init__.py)
    "ui.gn_menu",                    # 17. Geometry Node Editor add menus
    "ui.n_panel_operators",          # 18. N-Panel operators
    "ui.n_panel_ui",                 # 19. N-Panel UI panels
    "ui.prefs_library_manager",      # 20. Dependency installer
    "ui.prefs_panel_ui",             # 21. Add-on preferences UI
]

modules = []


def _import_or_reload_submodule(module_name: str):
    qualified_name = f"{PACKAGE_NAME}.{module_name}"
    if qualified_name in sys.modules:
        return importlib.reload(sys.modules[qualified_name])
    return importlib.import_module(f".{module_name}", PACKAGE_NAME)

def _remove_handlers_and_timers(log):
    """Remove ACE handlers and timers; safe to call when they were never added."""
    try:
        from .core.engine_handlers import unregister_handlers
        unregister_handlers()
    except Exception as exc:  # noqa: BLE001 boundary: teardown must reach the timer below
        log(f"Removing ACE handlers failed: {exc}", 'ERROR')
    try:
        import bpy  # type: ignore
        from .properties import _live_device_cache_timer
        if bpy.app.timers.is_registered(_live_device_cache_timer):
            bpy.app.timers.unregister(_live_device_cache_timer)
    except Exception as exc:  # noqa: BLE001 boundary: teardown
        log(f"Removing the live-device timer failed: {exc}", 'ERROR')


# --- MAIN REGISTRATION ---

def register():
    """Register all submodules in the correct order."""

    from .utils import log
    modules.clear()

    imported_modules = []
    registered_modules = []

    try:
        for module_name in MODULE_NAMES:
            module = _import_or_reload_submodule(module_name)
            imported_modules.append(module)

        for module in imported_modules:
            register_fn = getattr(module, "register", None)
            if callable(register_fn):
                register_fn()
                registered_modules.append(module)

        # A failure here is a registration failure: without handlers the
        # engine never runs, so it goes to the rollback below.
        from .core.engine_handlers import register_handlers
        register_handlers()
        import bpy  # type: ignore
        from .properties import _live_device_cache_timer
        if not bpy.app.timers.is_registered(_live_device_cache_timer):
            bpy.app.timers.register(_live_device_cache_timer, first_interval=0.1, persistent=True)

        modules[:] = imported_modules
        log(f"Registered successfully ({len(modules)} modules).")
    except Exception as e:  # noqa: BLE001 boundary: roll back, then re-raise
        log(f"Registration failed: {e}", 'ERROR')
        _remove_handlers_and_timers(log)
        for module in reversed(registered_modules):
            unregister_fn = getattr(module, "unregister", None)
            if callable(unregister_fn):
                try:
                    unregister_fn()
                except Exception as rollback_error:  # noqa: BLE001 boundary: keep rolling back
                    log(
                        f"Rollback failed for module '{module.__name__}': {rollback_error}",
                        'ERROR',
                    )
        modules.clear()
        raise

def unregister():
    """Unregister all submodules in reverse order."""

    from .utils import log

    # 1. Remove Handlers
    _remove_handlers_and_timers(log)

    # 2. Unregister Blender Classes
    for module in reversed(list(modules)):
        if hasattr(module, "unregister"):
            try:
                module.unregister()
            except Exception as e:  # noqa: BLE001 boundary: unregister every module
                log(f"Failed to unregister module '{module.__name__}': {e}", 'ERROR')
    
    modules.clear()
    log("Unregistered.")
