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
    "abstract_tree": ("core", ROOT / "core" / "abstract_tree.py"),
    "id_management": ("core", ROOT / "core" / "id_management.py"),
    "dynamic_audio_operators": ("nodes", ROOT / "nodes" / "dynamic_audio_operators.py"),
    "gn_menu": ("ui", ROOT / "ui" / "gn_menu.py"),
    "n_panel_operators": ("ui", ROOT / "ui" / "n_panel_operators.py"),
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
    "chunk", "filters", "frontend", "block_size", "base", "spectral",
    "source_frontend", "source_clock", "source", "tempo", "specs", "delta", "stream", "preview", "utility", "graph",
)


def forget_modules(*names: str):
    """Drop loaded add-on modules (and their submodules) so the next load imports them again."""
    for name in names:
        full = f"{PKG_NAME}.{name}"
        for key in [k for k in sys.modules if k == full or k.startswith(full + ".")]:
            sys.modules.pop(key, None)
            # "from .. import name" reads the parent package's attribute before sys.modules.
            parent_name, _, child = key.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, child):
                delattr(parent, child)


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
    relative imports (e.g. from .abstract_tree import ...) keep working.
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
