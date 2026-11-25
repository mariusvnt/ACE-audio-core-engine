import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

# Numba caches compiled kernels next to the source, keyed to the importing
# module. Under the test package name those entries break the same checkout
# when Blender loads it as the add-on, so tests cache elsewhere.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "ace_tests_numba_cache"))


ROOT = Path(__file__).resolve().parents[1]
PKG_NAME = "ace_addon_testpkg"

_MODULE_PATHS = {
    "audio_engine": ("core", ROOT / "core" / "audio_engine.py"),
    "engine_common": ("core", ROOT / "core" / "engine_common.py"),
    "engine_preview": ("core", ROOT / "core" / "engine_preview.py"),
    "engine_diagnostics": ("core", ROOT / "core" / "engine_diagnostics.py"),
    "engine_probes": ("core", ROOT / "core" / "engine_probes.py"),
    "engine_bake": ("core", ROOT / "core" / "engine_bake.py"),
    "engine_parallel": ("core", ROOT / "core" / "engine_parallel.py"),
    "engine_frontends": ("core", ROOT / "core" / "engine_frontends.py"),
    "engine_sinks": ("core", ROOT / "core" / "engine_sinks.py"),
    "engine_compiler": ("core", ROOT / "core" / "engine_compiler.py"),
    "engine_frame": ("core", ROOT / "core" / "engine_frame.py"),
    "engine_handlers": ("core", ROOT / "core" / "engine_handlers.py"),
    "audio_influx": ("core", ROOT / "core" / "audio_influx.py"),
    "abstract_tree": ("core", ROOT / "core" / "abstract_tree.py"),
    "id_management": ("core", ROOT / "core" / "id_management.py"),
    "parallel_dsp": ("core", ROOT / "core" / "parallel_dsp.py"),
    "proxy_management": ("core", ROOT / "core" / "proxy_management.py"),
    "dummy_tree": ("nodes", ROOT / "nodes" / "dummy_tree.py"),
    "shader_tree": ("nodes", ROOT / "nodes" / "shader_tree.py"),
    "dynamic_audio_operators": ("nodes", ROOT / "nodes" / "dynamic_audio_operators.py"),
    "gn_menu": ("ui", ROOT / "ui" / "gn_menu.py"),
    "n_panel_operators": ("ui", ROOT / "ui" / "n_panel_operators.py"),
    "n_panel_ui": ("ui", ROOT / "ui" / "n_panel_ui.py"),
    "prefs_library_manager": ("ui", ROOT / "ui" / "prefs_library_manager.py"),
    "prefs_panel_ui": ("ui", ROOT / "ui" / "prefs_panel_ui.py"),
    "properties": ("", ROOT / "properties.py"),
    "utils": ("", ROOT / "utils.py"),
    "vars": ("", ROOT / "vars.py"),
}


def _ensure_package(name: str, path: Path):
    pkg = sys.modules.get(name)
    if pkg is None:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = pkg
    return pkg


TASK_MODULES = (
    "chunk", "filters", "frontend", "block_size", "base", "spectral", "analyzer",
    "source", "specs", "delta", "stream", "preview", "utility", "graph",
)


def forget_modules(*names: str):
    """Drop loaded add-on modules (and their submodules) so the next load imports them again."""
    for name in names:
        full = f"{PKG_NAME}.{name}"
        for key in [k for k in sys.modules if k == full or k.startswith(full + ".")]:
            sys.modules.pop(key, None)


def load_tasks():
    """All task modules' top-level names in one namespace."""
    merged = types.SimpleNamespace()
    for name in TASK_MODULES:
        module = load_module(f"core.tasks.{name}")
        for attr, value in vars(module).items():
            if not attr.startswith("__") and not hasattr(merged, attr):
                setattr(merged, attr, value)
    return merged


def load_module(module_name: str):
    """
    Load add-on modules from this folder under a synthetic package namespace so
    relative imports (e.g. from .audio_influx import ...) keep working.
    A dotted name ("core.tasks.chunk") is imported as a submodule of that namespace.
    """
    _ensure_package(PKG_NAME, ROOT)
    _ensure_package(f"{PKG_NAME}.core", ROOT / "core")
    _ensure_package(f"{PKG_NAME}.nodes", ROOT / "nodes")
    _ensure_package(f"{PKG_NAME}.ui", ROOT / "ui")
    if "." in module_name:
        return importlib.import_module(f"{PKG_NAME}.{module_name}")

    subpkg, module_path = _MODULE_PATHS.get(module_name, ("", ROOT / f"{module_name}.py"))
    package_name = f"{PKG_NAME}.{subpkg}" if subpkg else PKG_NAME
    full_name = f"{package_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_name}' from {module_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    legacy_alias = f"{PKG_NAME}.{module_name}"
    sys.modules.setdefault(legacy_alias, module)
    return module
