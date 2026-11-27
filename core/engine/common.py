"""Shared imports, optional-library probes and small types for the ACE engine modules."""

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Any
import bpy  # type: ignore
import numpy as np
from ... import deps
_NUMBA_AVAILABLE = deps.numba_enabled()
if _NUMBA_AVAILABLE:
    from numba import njit  # type: ignore
try:
    from bpy.app.handlers import persistent  # type: ignore
except ImportError:  # pragma: no cover - Blender provides this at runtime
    def persistent(fn):
        return fn
from ..abstract_tree import ACEFrontendRef
try:
    from concurrent.futures.process import BrokenProcessPool
except ImportError:  # pragma: no cover - platform/runtime dependent
    BrokenProcessPool = RuntimeError
try:
    from multiprocessing import shared_memory as mp_shared_memory
    _SHARED_MEMORY_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime/platform dependent
    mp_shared_memory = None
    _SHARED_MEMORY_AVAILABLE = False
FRONTEND_KIND_GEOMETRY = "GEOMETRY_NODES"
FRONTEND_KIND_SHADER = "SHADER_NODES"
@dataclass
class _ACEFrontendDiscovery:
    """One root of ACE uses: a tree with its owners and the values they bind (live refs: one compile only)."""
    tree: Any
    frontend_ref: ACEFrontendRef
    consumer_object_names: tuple[str, ...] = ()
    bindings: dict = None  # Group Input socket identifier -> plain value
    root_key: str = ""
    modifier: Any = None
@dataclass(frozen=True)
class _ACECompiledFrontendSnapshot:
    uid: str
    kind: str
    tree_name: str
    tree_bl_idname: str
    owner_keys: tuple[str, ...] = ()
    consumer_object_names: tuple[str, ...] = ()
    material_name: str = ""
    tree_ptr: int = 0
if _NUMBA_AVAILABLE:
    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _preview_limiter_block_numba(
        frames: np.ndarray,
        ceiling: float,
        release_step: float,
        gain: float,
    ):
        out = np.empty_like(frames)
        g = gain
        channels = int(frames.shape[1]) if frames.ndim == 2 else 1
        for i in range(frames.shape[0]):
            frame_peak = 0.0
            if channels <= 1:
                frame_peak = abs(float(frames[i, 0])) if frames.ndim == 2 else abs(float(frames[i]))
            else:
                for c in range(channels):
                    val = abs(float(frames[i, c]))
                    if val > frame_peak:
                        frame_peak = val
            desired = 1.0 if frame_peak <= 1e-9 else min(1.0, ceiling / frame_peak)
            if desired < g:
                g = desired
            else:
                g += (desired - g) * release_step
            if frames.ndim == 2:
                for c in range(channels):
                    out[i, c] = frames[i, c] * g
            else:
                out[i] = frames[i] * g
        return out, g
else:
    _preview_limiter_block_numba = None
_ADDON_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ADDON_PACKAGE = (__package__ or "").rsplit(".core.engine", 1)[0]
_SPAWN_MAIN = os.path.join(_ADDON_DIR, "core", "worker", "spawn_main.py")
_MISSING = object()


@contextlib.contextmanager
def _spawn_bootstrap_main():
    """While worker processes spawn, make each child run core/worker/spawn_main.py as its __main__.

    Spawn re-runs the parent's __main__ in every child. Under Blender that is a
    bpy script (or nothing), and an extension's package name starts with
    bl_ext, which a plain interpreter cannot import; spawn_main.py fixes the
    latter and runs nothing else.
    """
    main_module = sys.modules.get("__main__")
    if main_module is None:
        yield
        return
    saved_file = getattr(main_module, "__file__", _MISSING)
    saved_spec = getattr(main_module, "__spec__", _MISSING)
    os.environ["ACE_SPAWN_PACKAGE"] = _ADDON_PACKAGE
    os.environ["ACE_SPAWN_PACKAGE_DIR"] = _ADDON_DIR
    main_module.__file__ = _SPAWN_MAIN
    main_module.__spec__ = None  # spawn would run a named __main__ module instead of the file
    try:
        yield
    finally:
        for name, saved in (("__file__", saved_file), ("__spec__", saved_spec)):
            if saved is _MISSING:
                with contextlib.suppress(AttributeError):
                    delattr(main_module, name)
            else:
                setattr(main_module, name, saved)

def engine_settings(scene_or_engine_settings=None):
    if scene_or_engine_settings is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        return getattr(getattr(scene, "audiocore", None), "engine_settings", None)
    # Accept direct engine settings object
    if hasattr(scene_or_engine_settings, "ace_scheduler_mode"):
        return scene_or_engine_settings
    scene = scene_or_engine_settings
    return getattr(getattr(scene, "audiocore", None), "engine_settings", None)


def rna_snapshot(obj) -> Any:
    """A plain copy of *obj*'s values (an RNA struct, or any object with attributes).

    Work off the main thread (the bake) reads its source and settings from a snapshot, never
    from Blender data, and whatever it writes back lands on the snapshot.
    """
    import types

    if obj is None:
        return None
    values: dict = {}
    rna = getattr(obj, "bl_rna", None)
    if rna is not None:
        for prop in rna.properties:
            ident = str(getattr(prop, "identifier", "") or "")
            if not ident or ident == "rna_type" or getattr(prop, "type", "") in {"POINTER", "COLLECTION"}:
                continue
            try:
                value = getattr(obj, ident)
            except (AttributeError, TypeError, ValueError):
                continue
            if hasattr(value, "__len__") and not isinstance(value, str):
                value = tuple(value)
            values[ident] = value
    else:
        values.update({k: v for k, v in vars(obj).items() if not k.startswith("__")} if hasattr(obj, "__dict__") else {})
    return types.SimpleNamespace(**values)


def close_task(task) -> None:
    """Close *task* (files, streams, threads), when it has anything to close."""
    from ... import utils

    close_fn = getattr(task, "close", None)
    if callable(close_fn):
        with utils.logged_errors("common.close_task"):
            close_fn()
