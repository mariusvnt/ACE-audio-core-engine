"""Audio Core Engine: a Blender extension (see blender_manifest.toml)."""

import importlib
import sys

PACKAGE_NAME = __name__

# Dependency order; packages come before their modules so a reload re-imports parents first.
# deps comes first: it sets up the optional libraries' environment before any of them loads.
MODULE_NAMES = [
    "deps",
    "vars",
    "utils",
    "properties",
    "core",
    "core.abstract_tree",
    "core.id_management",
    "core.input",
    "core.input.resampler",
    "core.input.decode",
    "core.input.file",
    "core.input.udp",
    "core.input.device",
    "core.input.reception",
    "core.tasks",
    "core.tasks.chunk",
    "core.tasks.filters",
    "core.tasks.frontend",
    "core.tasks.block_size",
    "core.tasks.base",
    "core.tasks.spectral",
    "core.tasks.analyzer",
    "core.tasks.source_frontend",
    "core.tasks.source_clock",
    "core.tasks.source",
    "core.tasks.tempo",
    "core.tasks.specs_analysis",
    "core.tasks.specs",
    "core.tasks.delta",
    "core.tasks.stream",
    "core.tasks.preview",
    "core.tasks.utility",
    "core.tasks.graph",
    "core.worker",
    "core.worker.graph",
    "core.worker.job",
    "core.proxy",
    "core.proxy.attributes",
    "core.proxy.objects",
    "core.proxy.manager",
    "core.engine",
    "core.engine.common",
    "core.engine.preview",
    "core.engine.preview.feeder",
    "core.engine.preview.monitor",
    "core.engine.preview.router",
    "core.engine.diagnostics",
    "core.engine.probes",
    "core.engine.bake",
    "core.engine.parallel",
    "core.engine.parallel.ir",
    "core.engine.parallel.slots",
    "core.engine.parallel.pool",
    "core.engine.parallel.offload",
    "core.engine.frontends",
    "core.engine.sinks",
    "core.engine.compiler",
    "core.engine.compiler.sample_rates",
    "core.engine.compiler.required_nodes",
    "core.engine.compiler.graph_build",
    "core.engine.compiler.compiler",
    "core.engine.frame",
    "core.engine.frame.block_size",
    "core.engine.frame.evaluation",
    "core.engine.frame.runner",
    "core.engine.engine",
    "core.engine.handlers",
    "nodes",
    "nodes.frontend",
    "nodes.geometry",
    "nodes.geometry.factory",
    "nodes.geometry.identity",
    "nodes.geometry.reader",
    "nodes.shader",
    "nodes.shader.interface",
    "nodes.shader.layout",
    "nodes.shader.helpers",
    "nodes.shader.factory",
    "nodes.shader.reader",
    "nodes.dynamic_audio_operators",
    "nodes.driven_values",
    "nodes.split_freqs",
    "nodes.source_nodes",
    "ui",
    "ui.gn_menu",
    "ui.n_panel_operators",
    "ui.panel",
    "ui.panel.helpers",
    "ui.panel.splitter",
    "ui.panel.usage",
    "ui.panel.sync",
    "ui.panel.selection",
    "ui.panel.panels",
    "ui.prefs_panel_ui",
]

modules = []


def _import_or_reload_submodule(module_name: str):
    qualified_name = f"{PACKAGE_NAME}.{module_name}"
    if qualified_name in sys.modules:
        return importlib.reload(sys.modules[qualified_name])
    return importlib.import_module(f".{module_name}", PACKAGE_NAME)

def _remove_handlers_and_timers(log):
    try:
        from .core.engine.handlers import unregister_handlers
        unregister_handlers()
    except Exception as exc:  # noqa: BLE001 boundary: teardown must reach the timer below
        log(f"Removing ACE handlers failed: {exc}", 'ERROR')
    try:
        import bpy  # type: ignore
        from .properties import _live_device_cache_timer
        if bpy.app.timers.is_registered(_live_device_cache_timer):
            bpy.app.timers.unregister(_live_device_cache_timer)
    except Exception as exc:  # noqa: BLE001 boundary: teardown
        log(f"Removing the live-device timer failed: {exc}", 'ERROR')



def register():
    """Register all submodules in the correct order."""

    from .utils import log
    modules.clear()

    imported_modules = []
    registered_modules = []

    try:
        for module_name in MODULE_NAMES:
            module = _import_or_reload_submodule(module_name)
            imported_modules.append(module)

        for module in imported_modules:
            register_fn = getattr(module, "register", None)
            if callable(register_fn):
                register_fn()
                registered_modules.append(module)

        # A failure here is a registration failure: without handlers the
        # engine never runs, so it goes to the rollback below.
        from .core.engine.handlers import register_handlers
        register_handlers()
        import bpy  # type: ignore
        from .properties import _live_device_cache_timer
        if not bpy.app.timers.is_registered(_live_device_cache_timer):
            bpy.app.timers.register(_live_device_cache_timer, first_interval=0.1, persistent=True)

        modules[:] = imported_modules
        log(f"Registered successfully ({len(modules)} modules).")
    except Exception as e:  # noqa: BLE001 boundary: roll back, then re-raise
        log(f"Registration failed: {e}", 'ERROR')
        _remove_handlers_and_timers(log)
        for module in reversed(registered_modules):
            unregister_fn = getattr(module, "unregister", None)
            if callable(unregister_fn):
                try:
                    unregister_fn()
                except Exception as rollback_error:  # noqa: BLE001 boundary: keep rolling back
                    log(
                        f"Rollback failed for module '{module.__name__}': {rollback_error}",
                        'ERROR',
                    )
        modules.clear()
        raise

def unregister():
    """Unregister all submodules in reverse order."""

    from .utils import log

    _remove_handlers_and_timers(log)

    for module in reversed(list(modules)):
        if hasattr(module, "unregister"):
            try:
                module.unregister()
            except Exception as e:  # noqa: BLE001 boundary: unregister every module
                log(f"Failed to unregister module '{module.__name__}': {e}", 'ERROR')
    
    modules.clear()
    log("Unregistered.")
