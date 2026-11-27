"""Bundled libraries: which load here, and why one did not.

The libraries ship as wheels inside the extension (blender_manifest.toml), and
Blender installs them with it; nothing is installed at runtime. ACE always
uses them: there is no switch. A library that fails to import (SoundDevice on
Linux without PortAudio, or any of them when another extension shadows
Blender's NumPy) leaves ACE on its fallback paths, and registration logs the
reason as an error (failures()).

No imports from the add-on: everything else imports this first.
"""

import importlib
import importlib.metadata
import os
import sys
from typing import Dict, List, Optional

PACKAGE = __package__ or ""

# id -> display name, what it gives, extra hint per platform.
LIBRARIES: Dict[str, dict] = {
    # Blender's own module, not a bundled wheel; it can fail when another extension shadows Blender's NumPy.
    "aud": {
        "name": "Blender Audio",
        "desc": "Reads audio files: MP3, FLAC, Ogg and every format Blender can",
        "version_label": "Blender",
        "missing_note": "Only WAV files can be read",
    },
    "numba": {
        "name": "Numba",
        "desc": "Compiled filters and envelopes",
        "missing_note": "Filters and envelopes run on slower NumPy and Python fallbacks",
    },
    "sounddevice": {
        "name": "SoundDevice",
        "desc": "Live input and preview playback",
        "missing_note": "Live input and preview playback are unavailable",
        "linux_hint": "Install the PortAudio system library (libportaudio2 on Debian/Ubuntu)",
    },
}

_import_results: Dict[str, Optional[str]] = {}  # id -> None when it imported, else the error


def _user_dir() -> Optional[str]:
    """The extension's writable folder, or None outside Blender or as a legacy add-on."""
    try:
        import bpy  # type: ignore
        path_fn = getattr(getattr(bpy, "utils", None), "extension_path_user", None)
        if path_fn is None or not PACKAGE.startswith("bl_ext."):
            return None
        return str(path_fn(PACKAGE, path="", create=True))
    except Exception:  # noqa: BLE001 boundary: no user folder means Numba's default cache
        return None


def _try_import(lib_id: str) -> Optional[str]:
    """None when *lib_id* imports, else why not. Each library is tried once per session."""
    if lib_id in _import_results:
        return _import_results[lib_id]
    try:
        importlib.import_module(lib_id)
        error = None
    except Exception as exc:  # noqa: BLE001 boundary: a broken native library raises more than ImportError
        error = f"{type(exc).__name__}: {exc}".strip()
    _import_results[lib_id] = error
    return error


def available(lib_id: str) -> bool:
    return _try_import(lib_id) is None


def numba_enabled() -> bool:
    """Whether this session uses Numba: whether it imports (kernels are compiled at import)."""
    return available("numba")


def sounddevice_enabled() -> bool:
    return available("sounddevice")


def _version(lib_id: str) -> str:
    try:
        return importlib.metadata.version(lib_id)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(sys.modules.get(lib_id), "__version__", "") or "")


def report() -> List[dict]:
    """One row per library: id, name, desc, state ("ok", "missing"), version, detail."""
    rows = []
    for lib_id, info in LIBRARIES.items():
        row = {"id": lib_id, "name": info["name"], "desc": info["desc"], "version": "", "detail": ""}
        error = _try_import(lib_id)
        if error is None:
            row["state"] = "ok"
            row["version"] = info.get("version_label") or _version(lib_id)
        else:
            row["state"] = "missing"
            hint = info.get("linux_hint") if sys.platform.startswith("linux") else None
            note = info.get("missing_note")
            row["detail"] = "; ".join(part for part in (note, hint, error) if part)
        rows.append(row)
    return rows


def failures() -> List[str]:
    """One message per library that did not load, for the registration log."""
    return [f"{row['name']} could not be loaded: {row['detail']}" for row in report() if row["state"] != "ok"]


def _prepare_environment() -> None:
    """Before any library loads: Numba caches its kernels in the extension's user folder."""
    folder = _user_dir()
    if folder and "NUMBA_CACHE_DIR" not in os.environ:
        os.environ["NUMBA_CACHE_DIR"] = os.path.join(folder, "numba_cache")


_prepare_environment()
