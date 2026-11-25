"""Shared imports, optional-library probes and small types for the ACE engine modules."""

import contextlib
import sys
from dataclasses import dataclass
from typing import Any
import bpy  # type: ignore
import numpy as np
try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001 boundary: a broken native library raises more than ImportError
    _NUMBA_AVAILABLE = False
try:
    from bpy.app.handlers import persistent  # type: ignore
except ImportError:  # pragma: no cover - Blender provides this at runtime
    def persistent(fn):
        return fn
from .abstract_tree import ACEFrontendRef
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
    tree: Any
    frontend_ref: ACEFrontendRef
    consumer_object_names: tuple[str, ...] = ()
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
@contextlib.contextmanager
def _main_script_hidden_from_spawn():
    """Keep spawned DSP workers from re-running Blender's ``--python`` script.

    ``spawn`` re-imports ``__main__`` from its file in each new worker; under
    ``blender --python script.py`` that is the script, which imports bpy and
    kills the worker. Workers start inside ``submit()``, so the file is hidden
    only around it.
    """
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_module is None or main_file is None:
        yield
        return
    del main_module.__file__
    try:
        yield
    finally:
        main_module.__file__ = main_file

def engine_settings(scene_or_engine_settings=None):
    if scene_or_engine_settings is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        return getattr(getattr(scene, "audiocore", None), "engine_settings", None)
    # Accept direct engine settings object
    if hasattr(scene_or_engine_settings, "ace_scheduler_mode"):
        return scene_or_engine_settings
    scene = scene_or_engine_settings
    return getattr(getattr(scene, "audiocore", None), "engine_settings", None)
