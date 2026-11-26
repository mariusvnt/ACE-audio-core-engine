#!/usr/bin/env python3
"""
Run a Blender-side script headless against this checkout, with ACE enabled as an extension.

Blender starts with temporary ``BLENDER_USER_EXTENSIONS`` and
``BLENDER_USER_SCRIPTS`` folders, so nothing installed on this machine takes
part: installed extensions share one site-packages, and a wheel one of them
ships (NumPy 2 for Blender 5.0, for instance) would shadow Blender's own
modules. The extension is enabled through ``addon_utils`` before the script
runs (``--addons`` is ignored next to ``--factory-startup``); an enable
failure stops the run.

Two ways to provide the extension:

- default: the checkout is linked into the user_default repository, so edits
  apply without rebuilding. Its wheels are not installed; pass the libraries
  with ``--extra-path``.
- ``--package``: the extension is built for this platform (wheels from
  tools/fetch_wheels.py) and installed with ``extension install-file``, exactly
  as a user installs the zip. This checks that the shipped wheels load.

Usage (from the repo root, with the system Python):
  python tests/run_in_blender.py [--blender PATH] [--package | --extra-path LIBS]
                                 [--script tests/other_script.py] [--root OTHER_CHECKOUT] [-- script args]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ID = "audio_core_engine"
REPOSITORY = "user_default"
ADDON_MODULE = f"bl_ext.{REPOSITORY}.{EXTENSION_ID}"
DEFAULT_SCRIPT = ROOT / "tests" / "blender_integration_smoke.py"
ENABLE_EXPR = (
    "import addon_utils\n"
    "def _raise(exc):\n"
    "    raise exc\n"
    f"addon_utils.enable({ADDON_MODULE!r}, default_set=True, handle_error=_raise)\n"
)
WINDOWS_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def _enable_expr(extra_path: str) -> str:
    if not extra_path:
        return ENABLE_EXPR
    return f"import sys\nsys.path.insert(0, {str(Path(extra_path).resolve())!r})\n" + ENABLE_EXPR


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


def _unlink_dir(link: Path) -> None:
    """Remove the link itself, never the checkout it points to."""
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def _platform_tag() -> str:
    """This machine's platform as extension zips name it (windows_x64, macos_arm64, linux_x64)."""
    system = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}[platform.system()]
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return f"{system}_{arch}"


def _build_and_install(blender: str, root: Path, env: dict, tmp: Path) -> None:
    out_dir = tmp / "dist"
    out_dir.mkdir()
    subprocess.run(
        [blender, "--factory-startup", "--command", "extension", "build", "--split-platforms",
         "--source-dir", str(root), "--output-dir", str(out_dir)],
        env=env, check=True,
    )
    zips = sorted(out_dir.glob(f"*-{_platform_tag()}.zip"))
    if not zips:
        raise SystemExit(f"No {_platform_tag()} zip was built in {out_dir}: is that platform in the manifest?")
    subprocess.run(
        [blender, "--factory-startup", "--command", "extension", "install-file", "-r", REPOSITORY, str(zips[0])],
        env=env, check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--blender", default="", help="path to the Blender executable")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="script to run inside Blender")
    parser.add_argument("--root", default=str(ROOT), help="checkout to run (default: this one)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--package", action="store_true", help="build the extension for this platform and install the zip")
    mode.add_argument("--extra-path", default="", help="folder put first on sys.path before ACE loads (its libraries, in the default mode)")
    parser.add_argument("script_args", nargs="*", help="arguments passed to the script after --")
    args = parser.parse_args()

    blender = _find_blender(args.blender)
    root = Path(args.root).resolve()
    with tempfile.TemporaryDirectory(prefix="ace_blender_") as tmp_name:
        tmp = Path(tmp_name)
        extensions = tmp / "extensions"
        (extensions / REPOSITORY).mkdir(parents=True)
        (tmp / "scripts").mkdir()
        env = dict(
            os.environ,
            BLENDER_USER_SCRIPTS=str(tmp / "scripts"),
            BLENDER_USER_EXTENSIONS=str(extensions),
            PYTHONDONTWRITEBYTECODE="1",
        )
        link = None
        if args.package:
            _build_and_install(blender, root, env, tmp)
        else:
            link = extensions / REPOSITORY / EXTENSION_ID
            _link_dir(link, root)
        cmd = [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code", "1",
            "--python-expr", _enable_expr(args.extra_path),
            "--python", args.script,
        ]
        if args.script_args:
            cmd += ["--", *args.script_args]
        try:
            result = subprocess.run(cmd, env=env)
        finally:
            if link is not None:
                _unlink_dir(link)
    return int(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
