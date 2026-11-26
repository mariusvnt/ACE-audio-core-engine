#!/usr/bin/env python3
"""Repackage the AudioFlux wheel for one platform, for bundling in the extension.

The PyPI wheel (py3-none-any) carries the native library of every OS (200 MB
of Intel MKL for Linux among them), and importing it loads SciPy, SoundFile
and matplotlib for its file-reading and plotting helpers, which ACE never
calls. This writes ``audioflux-<version>+ace.<n>-py3-none-<tag>.whl`` that:

- keeps only the libraries the target platform loads;
- imports ``audioflux.audio`` and ``audioflux.display`` on first use, so
  ``import audioflux`` needs NumPy only.

Nothing else in the package changes. AudioFlux is MIT-licensed; its license
file stays in the wheel.

Usage:
  python tools/trim_audioflux.py UPSTREAM.whl --platform windows-x64 --out wheels/
"""

import argparse
import ast
import base64
import hashlib
import io
import os
import sys
import zipfile

LOCAL_VERSION = "ace.1"

# Blender platform -> (wheel platform tag, files kept under audioflux/lib/).
PLATFORMS = {
    "windows-x64": ("win_amd64", {"libaudioflux.dll", "libfftw3f-3.dll"}),
    "macos-arm64": ("macosx_11_0_arm64", {"libaudioflux.dylib", "lib/libomp.dylib"}),
    "linux-x64": (
        "manylinux_2_17_x86_64",
        {
            "libaudioflux.so",
            "lib/libomp.so.5",
            "lib/libmkl_core.so.2",
            "lib/libmkl_def.so.2",
            "lib/libmkl_intel_lp64.so.2",
            "lib/libmkl_intel_thread.so.2",
        },
    ),
}

EAGER_IMPORTS = ("from .audio import *\n", "from . import display\n")


def _lazy_block(audio_names):
    names = ", ".join(repr(n) for n in audio_names)
    return (
        "\n"
        "# ACE: audio (SciPy, SoundFile) and display (matplotlib) load on first use,\n"
        "# so importing audioflux needs only NumPy.\n"
        f"_LAZY_AUDIO = ({names},)\n"
        "\n"
        "\n"
        "def __getattr__(name):\n"
        "    import importlib\n"
        "    if name == 'display':\n"
        "        return importlib.import_module('.display', __name__)\n"
        "    if name == 'audio' or name in _LAZY_AUDIO:\n"
        "        module = importlib.import_module('.audio', __name__)\n"
        "        return module if name == 'audio' else getattr(module, name)\n"
        "    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')\n"
    )


def _audio_exports(source: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("audioflux/audio.py has no __all__")


def _record_line(path, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def trim(upstream: str, platform: str, out_dir: str) -> str:
    tag, keep_libs = PLATFORMS[platform]
    with zipfile.ZipFile(upstream) as src:
        names = src.namelist()
        dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/WHEEL"))
        version = dist_info[len("audioflux-"):-len(".dist-info")]
        new_version = f"{version}+{LOCAL_VERSION}"
        new_dist_info = f"audioflux-{new_version}.dist-info"
        files = {}
        for name in names:
            if name.startswith(dist_info + "/"):
                continue
            if name.startswith("audioflux/lib/"):
                rel = name[len("audioflux/lib/"):]
                if rel not in keep_libs:
                    continue
            files[name] = src.read(name)
        missing = sorted(f"audioflux/lib/{rel}" for rel in keep_libs if f"audioflux/lib/{rel}" not in files)
        if missing:
            raise SystemExit(f"Upstream wheel lacks {missing}")

        init = files["audioflux/__init__.py"].decode("utf-8")
        newline = "\r\n" if "\r\n" in init else "\n"
        text = init.replace("\r\n", "\n")
        for line in EAGER_IMPORTS:
            if line not in text:
                raise SystemExit(f"audioflux/__init__.py no longer has {line.strip()!r}: check the patch")
            text = text.replace(line, "")
        text += _lazy_block(_audio_exports(files["audioflux/audio.py"].decode("utf-8")))
        files["audioflux/__init__.py"] = text.replace("\n", newline).encode("utf-8")

        metadata = src.read(f"{dist_info}/METADATA").decode("utf-8")
        metadata = metadata.replace(f"\nVersion: {version}\n", f"\nVersion: {new_version}\n", 1)
        # Only NumPy is needed at import now; the rest stay optional extras.
        metadata = "".join(
            line for line in metadata.splitlines(keepends=True)
            if not (line.startswith("Requires-Dist:") and "numpy" not in line)
        )
        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: ace-trim-audioflux\n"
            "Root-Is-Purelib: false\n"
            f"Tag: py3-none-{tag}\n"
        )
        meta_files = {f"{new_dist_info}/METADATA": metadata.encode("utf-8"), f"{new_dist_info}/WHEEL": wheel_meta.encode()}
        for name in names:
            if name.startswith(dist_info + "/") and name.rsplit("/", 1)[-1] in ("LICENSE", "LICENSE.md", "top_level.txt"):
                meta_files[f"{new_dist_info}/{name.rsplit('/', 1)[-1]}"] = src.read(name)

    out_name = f"audioflux-{new_version}-py3-none-{tag}.whl"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    record_name = f"{new_dist_info}/RECORD"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        record = []
        for name, data in sorted({**files, **meta_files}.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if name.endswith((".so", ".so.2", ".so.5", ".dylib", ".dll")) else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)
            record.append(_record_line(name, data))
        record.append(f"{record_name},,")
        info = zipfile.ZipInfo(record_name, date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info, "\n".join(record) + "\n")
    with open(out_path, "wb") as fh:
        fh.write(buf.getvalue())
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", help="the audioflux wheel from PyPI")
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    parser.add_argument("--out", default="wheels", help="output folder")
    args = parser.parse_args()
    path = trim(args.wheel, args.platform, args.out)
    print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
