import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_NAME = "ace_addon_testpkg"

_MODULE_PATHS = {
    "audio_engine": ("core", ROOT / "core" / "audio_engine.py"),
    "audio_influx": ("core", ROOT / "core" / "audio_influx.py"),
    "audio_tasks_registry": ("core", ROOT / "core" / "audio_tasks_registry.py"),
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


def load_module(module_name: str):
    """
    Load add-on modules from this folder under a synthetic package namespace so
    relative imports (e.g. from .audio_influx import ...) keep working.
    """
    _ensure_package(PKG_NAME, ROOT)
    _ensure_package(f"{PKG_NAME}.core", ROOT / "core")
    _ensure_package(f"{PKG_NAME}.nodes", ROOT / "nodes")
    _ensure_package(f"{PKG_NAME}.ui", ROOT / "ui")

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
