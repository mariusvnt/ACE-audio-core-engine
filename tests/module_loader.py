import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_NAME = "ace_addon_testpkg"


def load_module(module_name: str):
    """
    Load add-on modules from this folder under a synthetic package namespace so
    relative imports (e.g. from .audio_influx import ...) keep working.
    """
    if PKG_NAME not in sys.modules:
        pkg = types.ModuleType(PKG_NAME)
        pkg.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
        sys.modules[PKG_NAME] = pkg

    full_name = f"{PKG_NAME}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = ROOT / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_name}' from {module_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = PKG_NAME
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module
