#!/usr/bin/env python3
"""
Run the Blender-side smoke script headless against this checkout.

The repo folder name is not a valid Python module name, so a temporary user
scripts folder gets an ``addons/AudioCoreEngine`` link to the repo, Blender is
started with it as ``BLENDER_USER_SCRIPTS``, and the add-on is enabled through
``addon_utils`` before the script runs (``--addons`` is ignored next to
``--factory-startup``). An enable failure stops the run.

Usage (from the repo root, with the system Python):
  python tests/run_in_blender.py [--blender PATH] [--script tests/other_script.py]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON_MODULE = "AudioCoreEngine"
DEFAULT_SCRIPT = ROOT / "tests" / "blender_integration_smoke.py"
ENABLE_EXPR = (
    "import addon_utils\n"
    "def _raise(exc):\n"
    "    raise exc\n"
    f"addon_utils.enable({ADDON_MODULE!r}, default_set=True, handle_error=_raise)\n"
)
WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe")


def _find_blender(explicit: str) -> str:
    candidates = [explicit, os.environ.get("BLENDER", ""), shutil.which("blender") or ""]
    if WINDOWS_BLENDER.exists():
        candidates.append(str(WINDOWS_BLENDER))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Blender not found: pass --blender or set BLENDER.")


def _link_dir(link: Path, target: Path) -> None:
    if os.name == "nt":
        # A junction needs no admin rights, unlike a symlink.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--blender", default="", help="path to the Blender executable")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="script to run inside Blender")
    args = parser.parse_args()

    blender = _find_blender(args.blender)
    with tempfile.TemporaryDirectory(prefix="ace_blender_") as tmp:
        addons_dir = Path(tmp) / "addons"
        addons_dir.mkdir()
        link = addons_dir / ADDON_MODULE
        _link_dir(link, ROOT)
        env = dict(os.environ, BLENDER_USER_SCRIPTS=tmp, PYTHONDONTWRITEBYTECODE="1")
        cmd = [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code", "1",
            "--python-expr", ENABLE_EXPR,
            "--python", args.script,
        ]
        try:
            result = subprocess.run(cmd, env=env)
        finally:
            # Remove the junction itself, never the repo it points to.
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()
    return int(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
