#!/usr/bin/env python3
"""Download the wheels the extension bundles and list them in blender_manifest.toml.

For every platform in the manifest (or those given with --platform), downloads
each pinned wheel of tools/wheels.txt built for Blender's Python into
./wheels/, and rewrites the manifest's ``wheels = [...]`` list. Fails when a wheel is
missing for a platform, or when a wheel would bring NumPy.

Then: blender --command extension build --split-platforms

Usage: python tools/fetch_wheels.py [--platform windows-x64 ...] [--no-manifest]

--no-manifest only downloads (CI installs the Linux wheels to run unit tests with them).
"""

import argparse
import base64
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
WHEELS_DIR = os.path.join(ROOT, "wheels")
MANIFEST = os.path.join(ROOT, "blender_manifest.toml")
PYTHON_VERSION = "3.13"  # Blender 5.2

# Blender platform -> pip --platform tags, newest first (pip accepts any of them).
PIP_PLATFORMS = {
    "windows-x64": ["win_amd64"],
    "windows-arm64": ["win_arm64"],
    "macos-arm64": ["macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_13_0_arm64", "macosx_14_0_arm64"],
    "linux-x64": ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"],
}


def _record_line(path, data):
    """A wheel RECORD line: path, urlsafe base64 SHA-256 and size."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def _requirements():
    reqs = []
    with open(os.path.join(TOOLS, "wheels.txt"), encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line):
                raise SystemExit(f"tools/wheels.txt: pin an exact version: {line!r}")
            reqs.append(line)
    if any(r.lower().startswith("numpy==") for r in reqs):
        raise SystemExit("tools/wheels.txt must not list NumPy: Blender ships its own")
    return reqs


def _pip_download(requirement, platform, dest):
    cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--disable-pip-version-check",
           "--no-deps", "--only-binary=:all:", "--python-version", PYTHON_VERSION,
           "--implementation", "cp", "--dest", dest, requirement]
    for tag in PIP_PLATFORMS[platform]:
        cmd += ["--platform", tag]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"No {platform} wheel for {requirement}:\n{result.stderr.strip()}")


def _runtime_requirements(wheel_path):
    with zipfile.ZipFile(wheel_path) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        text = zf.read(meta).decode("utf-8", "replace")
    out = []
    for line in text.splitlines():
        if line.startswith("Requires-Dist:") and "extra ==" not in line:
            out.append(re.split(r"[ ;<>=!~(\[]", line[len("Requires-Dist:"):].strip(), maxsplit=1)[0].lower())
    return out


def _retag(src, tag, dest_dir):
    """Copy a pure-Python wheel under one platform's tag.

    ``extension build --split-platforms`` puts a ``none-any`` wheel in every
    platform's zip. When a project ships a native wheel for some platforms and a
    pure one for others (SoundDevice: PortAudio is bundled on Windows and macOS,
    a system library on Linux), the pure one is tagged for the platforms it is
    for, so no zip holds two copies of the project.
    """
    name = os.path.basename(src)
    stem = name[: -len("-py3-none-any.whl")]
    out_path = os.path.join(dest_dir, f"{stem}-py3-none-{tag}.whl")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        record_name = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
        wheel_name = record_name.replace("RECORD", "WHEEL")
        wheel_text = zin.read(wheel_name).decode("utf-8")
        wheel_text = re.sub(r"^Tag: .*$", f"Tag: py3-none-{tag}", wheel_text, flags=re.M)
        wheel_text = wheel_text.replace("Root-Is-Purelib: true", "Root-Is-Purelib: false")
        records = []
        for item in zin.infolist():
            if item.filename == record_name:
                continue
            data = wheel_text.encode("utf-8") if item.filename == wheel_name else zin.read(item.filename)
            zout.writestr(item, data)
            records.append(_record_line(item.filename, data))
        records.append(f"{record_name},,")
        zout.writestr(record_name, "\n".join(records) + "\n")
    return out_path


def fetch(platforms):
    reqs = _requirements()
    bundled = {r.split("==")[0].lower().replace("_", "-") for r in reqs}
    os.makedirs(WHEELS_DIR, exist_ok=True)
    downloads = {}
    for platform in platforms:
        if platform not in PIP_PLATFORMS:
            raise SystemExit(f"Unknown platform {platform!r}")
        tmp = tempfile.mkdtemp(prefix=f"ace_wheels_{platform}_")
        for req in reqs:
            _pip_download(req, platform, tmp)
        downloads[platform] = tmp
    # Projects whose wheel differs between platforms: their pure wheels get a platform tag.
    def project(name):
        return name.split("-", 1)[0].lower()
    per_project = {}
    for tmp in downloads.values():
        for name in os.listdir(tmp):
            per_project.setdefault(project(name), set()).add(name)
    mixed = {p for p, names in per_project.items() if len(names) > 1}
    produced = []
    for platform, tmp in downloads.items():
        try:
            for name in sorted(os.listdir(tmp)):
                src = os.path.join(tmp, name)
                if project(name) in mixed and name.endswith("-py3-none-any.whl"):
                    path = _retag(src, PIP_PLATFORMS[platform][0], WHEELS_DIR)
                else:
                    path = os.path.join(WHEELS_DIR, name)
                    shutil.copyfile(src, path)
                for dep in _runtime_requirements(path):
                    dep = dep.replace("_", "-")
                    if dep == "numpy" or dep in bundled:
                        continue
                    print(f"warning: {os.path.basename(path)} requires {dep}, which is not bundled")
                produced.append(os.path.basename(path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return sorted(set(produced))


def remove_unlisted(keep):
    """Delete wheels in ./wheels that a full fetch did not produce (older pins)."""
    for name in os.listdir(WHEELS_DIR):
        if name.endswith(".whl") and name not in keep:
            os.remove(os.path.join(WHEELS_DIR, name))


def write_manifest(wheel_names):
    with open(MANIFEST, encoding="utf-8") as fh:
        text = fh.read()
    listing = "wheels = [\n" + "".join(f'  "./wheels/{n}",\n' for n in wheel_names) + "]"
    new_text, count = re.subn(r"^wheels = \[.*?\]", listing, text, count=1, flags=re.S | re.M)
    if count != 1:
        raise SystemExit("blender_manifest.toml has no wheels = [...] line")
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", action="append", help="Blender platform (default: every one in the manifest)")
    parser.add_argument("--no-manifest", action="store_true", help="download only; leave blender_manifest.toml and other wheels alone")
    args = parser.parse_args()
    with open(MANIFEST, "rb") as fh:
        manifest = tomllib.load(fh)
    platforms = args.platform or list(manifest.get("platforms", []))
    names = fetch(platforms)
    if args.no_manifest:
        print(f"{len(names)} wheels for {', '.join(platforms)} in {WHEELS_DIR}")
        return 0
    if args.platform:
        # Keep what the manifest already lists for the other platforms.
        names = sorted(set(names) | {os.path.basename(w) for w in manifest.get("wheels", [])
                                     if os.path.isfile(os.path.join(WHEELS_DIR, os.path.basename(w)))})
    else:
        remove_unlisted(set(names))
    write_manifest(names)
    total = sum(os.path.getsize(os.path.join(WHEELS_DIR, n)) for n in names)
    print(f"{len(names)} wheels for {', '.join(platforms)} ({total / 1e6:.0f} MB) listed in blender_manifest.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
