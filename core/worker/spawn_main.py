"""The first thing every spawned worker process runs, as its ``__main__``.

``multiprocessing`` spawn starts a fresh interpreter, runs the parent's
``__main__`` file there, then unpickles the job, which imports the add-on by
its full name. Installed as an extension that name is
``bl_ext.<repository>.audio_core_engine``, and Blender creates ``bl_ext`` and
``bl_ext.<repository>`` in memory only, so a plain interpreter cannot import
them. While the pool spawns, ``core/engine/common.py`` points the parent's
``__main__`` at this file and names the package in the environment; this
registers those parent packages with a real ``__path__`` so the import works.

No bpy, no imports of the add-on: this runs before the add-on can be found.
"""

import os
import sys
import types


def _register_parent_packages() -> None:
    package = os.environ.get("ACE_SPAWN_PACKAGE", "")
    package_dir = os.environ.get("ACE_SPAWN_PACKAGE_DIR", "")
    parts = [p for p in package.split(".") if p]
    if len(parts) < 2 or not package_dir:
        return  # a top-level package (legacy add-on) imports from sys.path as is
    directory = os.path.dirname(os.path.abspath(package_dir))
    for depth in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:depth])
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [directory]
            sys.modules[name] = module
        directory = os.path.dirname(directory)


_register_parent_packages()
