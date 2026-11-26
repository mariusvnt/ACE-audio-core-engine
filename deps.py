"""Optional libraries: which ship with ACE, which load here, and which the user turned off.

The libraries ship as wheels inside the extension (blender_manifest.toml), and
Blender installs them with it; nothing is installed at runtime. This module is
the one place that decides whether ACE uses one:

- it is importable, and
- its switch in the add-on preferences is on (Numba, SoundDevice).

The switches are kept in a small JSON file in the extension's user folder, not
in the preferences datablock, so they can be read before any class is
registered and, through environment variables, by worker processes, which
have no bpy. Numba's switch applies at the next start (its kernels compile at
import); SoundDevice's applies the next time a live source or the preview
monitor opens.

No imports from the add-on: everything else imports this first.
"""

import importlib
import importlib.metadata
import json
import os
import sys
import threading
from typing import Dict, List, Optional

PACKAGE = __package__ or ""

# id -> display name, what it gives, preference switch (None: always used), extra hint per platform.
LIBRARIES: Dict[str, dict] = {
    # Blender's own module, not a bundled wheel; it can fail when another extension shadows Blender's NumPy.
    "aud": {
        "name": "Blender Audio",
        "desc": "Reads audio files: MP3, FLAC, Ogg and every format Blender can",
        "switch": None,
        "version_label": "Blender",
        "missing_note": "Only WAV files can be read",
    },
    "audioflux": {
        "name": "AudioFlux",
        "desc": "Spectrum, pitch and spectral features",
        "switch": None,
    },
    "numba": {
        "name": "Numba",
        "desc": "Compiled filters and envelopes",
        "switch": "use_numba",
    },
    "sounddevice": {
        "name": "SoundDevice",
        "desc": "Live input and preview playback",
        "switch": "use_sounddevice",
        "linux_hint": "Install the PortAudio system library (libportaudio2 on Debian/Ubuntu)",
    },
}

DEFAULT_SETTINGS = {"use_numba": True, "use_sounddevice": True}
_ENV = {"use_numba": "ACE_USE_NUMBA", "use_sounddevice": "ACE_USE_SOUNDDEVICE"}
_SETTINGS_FILE = "settings.json"

_lock = threading.Lock()
_settings: Optional[dict] = None
_import_results: Dict[str, Optional[str]] = {}  # id -> None when it imported, else the error


def _user_dir() -> Optional[str]:
    """The extension's writable folder, or None outside Blender or as a legacy add-on."""
    try:
        import bpy  # type: ignore
        path_fn = getattr(getattr(bpy, "utils", None), "extension_path_user", None)
        if path_fn is None or not PACKAGE.startswith("bl_ext."):
            return None
        return str(path_fn(PACKAGE, path="", create=True))
    except Exception:  # noqa: BLE001 boundary: no user folder means in-memory settings
        return None


def _load_settings() -> dict:
    data = dict(DEFAULT_SETTINGS)
    folder = _user_dir()
    if folder:
        path = os.path.join(folder, _SETTINGS_FILE)
        try:
            with open(path, encoding="utf-8") as fh:
                stored = json.load(fh)
            data.update({k: bool(v) for k, v in stored.items() if k in DEFAULT_SETTINGS})
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            print(f"[ACE ERROR] Could not read {path}: {exc}")
    # A worker process gets the parent's choice through the environment.
    for key, env in _ENV.items():
        if env in os.environ:
            data[key] = os.environ[env] == "1"
    return data


def setting(key: str) -> bool:
    global _settings
    with _lock:
        if _settings is None:
            _settings = _load_settings()
        return bool(_settings.get(key, DEFAULT_SETTINGS.get(key, False)))


def set_setting(key: str, value: bool) -> None:
    """Store a switch. Returns nothing; Numba's takes effect at the next start."""
    global _settings
    if key not in DEFAULT_SETTINGS:
        raise KeyError(key)
    with _lock:
        if _settings is None:
            _settings = _load_settings()
        _settings[key] = bool(value)
        snapshot = dict(_settings)
    if key == "use_sounddevice":
        os.environ[_ENV[key]] = "1" if value else "0"
    folder = _user_dir()
    if folder:
        path = os.path.join(folder, _SETTINGS_FILE)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(snapshot, fh, indent=2)
        except OSError as exc:
            print(f"[ACE ERROR] Could not write {path}: {exc}")


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


_numba_in_use: Optional[bool] = None


def numba_enabled() -> bool:
    """Whether this session uses Numba. Decided once: kernels are compiled at import."""
    global _numba_in_use
    if _numba_in_use is None:
        _numba_in_use = setting("use_numba") and available("numba")
    return _numba_in_use


def sounddevice_enabled() -> bool:
    return setting("use_sounddevice") and available("sounddevice")


def _version(lib_id: str) -> str:
    try:
        return importlib.metadata.version(lib_id)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(sys.modules.get(lib_id), "__version__", "") or "")


def report() -> List[dict]:
    """One row per library: id, name, desc, state ("ok", "off", "missing"), version, detail."""
    rows = []
    for lib_id, info in LIBRARIES.items():
        switch = info.get("switch")
        row = {"id": lib_id, "name": info["name"], "desc": info["desc"], "switch": switch, "version": "", "detail": ""}
        if switch and not setting(switch):
            row["state"] = "off"
            row["detail"] = "Turned off below"
        else:
            error = _try_import(lib_id)
            if error is None:
                row["state"] = "ok"
                row["version"] = info.get("version_label") or _version(lib_id)
            else:
                row["state"] = "missing"
                hint = info.get("linux_hint") if sys.platform.startswith("linux") else None
                note = info.get("missing_note")
                row["detail"] = hint or (f"{note}: {error}" if note else error)
        rows.append(row)
    return rows


def _prepare_environment() -> None:
    """Before any library loads: Numba caches in the user folder, and worker processes inherit the switches."""
    folder = _user_dir()
    if folder and "NUMBA_CACHE_DIR" not in os.environ:
        os.environ["NUMBA_CACHE_DIR"] = os.path.join(folder, "numba_cache")
    for key, env in _ENV.items():
        os.environ.setdefault(env, "1" if setting(key) else "0")


_prepare_environment()
