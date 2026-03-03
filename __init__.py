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
    except Exception:
        vendor_path = os.path.join(os.path.dirname(__file__), "_vendor")
        if vendor_path and os.path.isdir(vendor_path) and vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

_ensure_dependency_paths()

# --- MODULE REGISTRY ---
# Order matters: Primitives -> Logic -> UI -> Orchestration
MODULE_NAMES = [
    "vars",                     # 1. Definitions / Constants
    "utils",                    # 2. Global Helpers
    "id_management",            # 2. ID Utilities
    "abstract_tree",            # 3. Data Structures
    "properties",               # 4. Blender Properties (Scene Data)
    "proxy_management",         # 5. Bridge to Blender Data
    "audio_influx",             # 6. Audio IO helpers
    "parallel_dsp",             # 6b. Process-worker DSP helpers
    "dynamic_audio_operators",  # 7. Dynamic Variant Logic (modular splitters)
    "dummy_tree",               # 8. Node Factory & Tree Reader
    "audio_engine",             # 9. DSP Orchestration
    "gn_menu",                  # 10. Geometry Node Editor Menus
    "n_panel_operators",        # 11. N-Panel Operators
    "n_panel_ui",               # 11. N-Panel UI
    "prefs_library_manager",    # 12. Dependency Installer
    "prefs_panel_ui",           # 13. Add-on Preferences UI
]

modules = []


def _import_or_reload_submodule(module_name: str):
    qualified_name = f"{PACKAGE_NAME}.{module_name}"
    if qualified_name in sys.modules:
        return importlib.reload(sys.modules[qualified_name])
    return importlib.import_module(f".{module_name}", PACKAGE_NAME)

# --- MAIN REGISTRATION ---

def register():
    """Register all submodules in the correct order."""

    from .utils import log
    global modules
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

        try:
            from .audio_engine import register_handlers
            register_handlers()
        except (ImportError, AttributeError):
            pass

        modules[:] = imported_modules
        log(f"Registered successfully ({len(modules)} modules).")
    except Exception as e:
        log(f"Registration failed: {e}", 'ERROR')
        for module in reversed(registered_modules):
            unregister_fn = getattr(module, "unregister", None)
            if callable(unregister_fn):
                try:
                    unregister_fn()
                except Exception as rollback_error:
                    log(
                        f"Rollback failed for module '{module.__name__}': {rollback_error}",
                        'ERROR',
                    )
        modules.clear()
        raise

def unregister():
    """Unregister all submodules in reverse order."""

    from .utils import log
    global modules
    
    # 1. Remove Handlers
    try:
        from .audio_engine import unregister_handlers
        unregister_handlers()
    except (ImportError, AttributeError):
        pass

    # 2. Unregister Blender Classes
    for module in reversed(list(modules)):
        if hasattr(module, "unregister"):
            try:
                module.unregister()
            except Exception as e:
                log(f"Failed to unregister module '{module.__name__}': {e}", 'ERROR')
    
    modules.clear()
    log("Unregistered.")
