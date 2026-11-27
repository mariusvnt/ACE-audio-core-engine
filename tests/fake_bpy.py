"""
One fake ``bpy`` for every test that loads add-on modules outside Blender.

``install()`` puts the same complete module tree into ``sys.modules`` whichever
test file runs first; ``reset()`` restores empty data, handlers and timers.
ID-type stand-ins (Object, Material, Mesh, ...) have no ``bl_idname``, as in
real Blender: only registrable types (node trees, nodes, operators, panels,
menus) carry one.
"""

import sys
import types

HANDLER_NAMES = (
    "load_post",
    "undo_post",
    "redo_post",
    "frame_change_pre",
    "frame_change_post",
    "render_init",
    "render_complete",
    "render_pre",
    "render_post",
    "render_cancel",
    "depsgraph_update_post",
)

_ID_TYPES = ("ID", "Object", "Scene", "Collection", "Mesh", "Material")
_REGISTRABLE_TYPES = (
    "NodeTree",
    "Node",
    "Operator",
    "Panel",
    "Menu",
    "UIList",
    "PropertyGroup",
    "AddonPreferences",
)
_OTHER_TYPES = ("UILayout", "OperatorFileListElement")
_MENU_TYPES = ("NODE_MT_add", "NODE_MT_category_shader_add")


class TruthyDict(dict):
    """Base for fake datablocks that hold ID properties (``obj["key"]``).

    A plain dict subclass is falsy while empty; Blender datablocks never are,
    and the add-on relies on that (``if node_tree:``).
    """

    def __bool__(self):
        return True


class FakeCollection(list):
    """Stand-in for a bpy CollectionProperty value."""

    def add(self):
        item = types.SimpleNamespace(
            library_id="",
            display_name="",
            description="",
            is_builtin=False,
        )
        self.append(item)
        return item

    def clear(self):
        del self[:]


class _Props:
    """bpy.props: every property factory returns None except CollectionProperty."""

    @staticmethod
    def CollectionProperty(**_kwargs):
        return FakeCollection()

    def __getattr__(self, name):
        if name.endswith("Property"):
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)


class _Timers:
    def __init__(self):
        self.registered = {}

    def register(self, fn, first_interval=0.0, persistent=False):
        self.registered[fn] = (first_interval, persistent)

    def unregister(self, fn):
        self.registered.pop(fn, None)

    def is_registered(self, fn):
        return fn in self.registered


def _persistent(fn):
    # Blender's decorator marks the function the same way.
    fn._bpy_persistent = None
    return fn


def _menu_type(name):
    def _noop(_cls, *_args, **_kwargs):
        return None

    return type(name, (), {"append": classmethod(_noop), "prepend": classmethod(_noop), "remove": classmethod(_noop)})


def _build():
    bpy_mod = types.ModuleType("bpy")
    bpy_mod.__dict__["_ace_fake_bpy"] = True

    handlers_mod = types.ModuleType("bpy.app.handlers")
    handlers_mod.persistent = _persistent

    app_mod = types.ModuleType("bpy.app")
    app_mod.handlers = handlers_mod
    app_mod.timers = _Timers()
    app_mod.is_job_running = lambda _name: False
    app_mod.version = (5, 0, 0)
    app_mod.background = True

    types_ns = types.SimpleNamespace()
    for name in _ID_TYPES + _OTHER_TYPES:
        setattr(types_ns, name, type(name, (), {}))
    for name in _REGISTRABLE_TYPES:
        setattr(types_ns, name, type(name, (), {"bl_idname": ""}))
    for name in _MENU_TYPES:
        setattr(types_ns, name, _menu_type(name))

    utils_mod = types.ModuleType("bpy.utils")
    utils_mod.register_class = lambda _cls: None
    utils_mod.unregister_class = lambda _cls: None
    previews_mod = types.ModuleType("bpy.utils.previews")
    previews_mod.new = lambda: {}
    previews_mod.remove = lambda _pcoll: None
    utils_mod.previews = previews_mod

    bpy_mod.app = app_mod
    bpy_mod.types = types_ns
    bpy_mod.props = _Props()
    bpy_mod.utils = utils_mod

    extras_mod = types.ModuleType("bpy_extras")
    io_utils_mod = types.ModuleType("bpy_extras.io_utils")
    io_utils_mod.ImportHelper = type("ImportHelper", (), {})
    extras_mod.io_utils = io_utils_mod

    return {
        "bpy": bpy_mod,
        "bpy.app": app_mod,
        "bpy.app.handlers": handlers_mod,
        "bpy.utils": utils_mod,
        "bpy.utils.previews": previews_mod,
        "bpy_extras": extras_mod,
        "bpy_extras.io_utils": io_utils_mod,
    }


def reset(bpy_mod=None):
    """Empty the fake's data, context, handlers and timers."""
    bpy_mod = bpy_mod or sys.modules["bpy"]
    bpy_mod.data = types.SimpleNamespace(
        objects={},
        node_groups={},
        collections={},
        meshes={},
        materials={},
        scenes=[],
    )
    bpy_mod.context = types.SimpleNamespace(
        scene=None,
        preferences=None,
        window_manager=types.SimpleNamespace(windows=[]),
    )
    for name in HANDLER_NAMES:
        setattr(bpy_mod.app.handlers, name, [])
    bpy_mod.app.timers.registered.clear()
    return bpy_mod


def install():
    """Install the fake (once) and return the ``bpy`` module."""
    existing = sys.modules.get("bpy")
    if existing is not None:
        if not getattr(existing, "_ace_fake_bpy", False):
            raise RuntimeError("A real or foreign bpy module is already loaded; fake_bpy cannot replace it.")
        return existing
    modules = _build()
    sys.modules.update(modules)
    return reset(modules["bpy"])
