"""The ACE engine: one ``ACEAudioEngine`` per Blender scene."""

import bpy
import queue
import time
from typing import Dict, Optional
from ... import utils
from ...properties import ACEAudioSource
from ..abstract_tree import ACEAbstractTree
from ..tasks.chunk import SAMPLE_RATE
from ..tasks.filters import reset_biquad_fallback_warned
from ..tasks.graph import ACEAudioGraph
from ..tasks.source import _clear_file_open_locks
from ..tasks.specs import SourceSpecsTask
from ..proxy.manager import ACEProxyManager
from .common import FRONTEND_KIND_SHADER
from .bake import BakeManager
from .compiler.compiler import GraphCompiler
from .diagnostics import EngineDiagnostics
from .frame.runner import FrameRunner
from .frontends import FrontendTracker
from .parallel.offload import ParallelOffload
from .preview.router import PreviewRouter
from .probes import SourceProbes
from .sinks import SinkResolver


class ACEAudioEngine:
    """Scene-scoped ACE runtime and the owner of its components."""

    _instances: Dict[int, "ACEAudioEngine"] = {}

    def __init__(self, scene_key: int):
        self.scene_key = int(scene_key)
        self.scene_name: str = ""
        self.graphs: Dict[str, ACEAudioGraph] = {}
        # Latest value of every operator output, by 'node_id:socket' (linked split frequencies read it).
        self.latest_values: dict = {}
        self.current_frame = 0
        self.last_scene_frame = -1
        self.writer: Optional[ACEProxyManager] = None
        self.abstract_tree_ref: Optional[ACEAbstractTree] = None
        self.effective_sample_rate: int = SAMPLE_RATE
        self._last_attr_sync_frame: Dict[str, int] = {}
        self._bootstrap_retry_at: float = 0.0
        self._bootstrap_fail_count: int = 0
        self._current_delta_frames: float = 1.0
        # Tracks scene FPS so time-based constants can be converted to frame counts.
        self._current_scene_fps: float = 24.0
        # Dirty-flag set for attribute schema resync: populated on compile,
        # cleared per-graph after ensure_attributes_exist() runs.
        self._attr_schema_dirty: set[str] = set()
        self.undo_generation: int = 0
        self.undo_recovery_until: float = 0.0
        # Components own their own state; build them last so they can read the fields above.
        self.diagnostics = EngineDiagnostics(self)
        self.probes = SourceProbes(self)
        self.preview = PreviewRouter(self)
        self.bake = BakeManager(self)
        self.parallel = ParallelOffload(self)
        self.frontends = FrontendTracker(self)
        self.sinks = SinkResolver(self)
        self.compiler = GraphCompiler(self)
        self.frame = FrameRunner(self)

    @staticmethod
    def _scene_key(scene) -> int:
        if scene is not None and hasattr(scene, "as_pointer"):
            with utils.logged_errors("engine.ACEAudioEngine._scene_key"):
                return int(scene.as_pointer())
        return 0

    @staticmethod
    def _scene_name(scene) -> str:
        if scene is None:
            return ""
        try:
            return str(getattr(scene, "name_full", "") or getattr(scene, "name", "") or "")
        except Exception as exc:
            utils.note_swallowed("engine.ACEAudioEngine._scene_name", exc)
            return ""

    def _clear_blender_runtime_refs(self) -> None:
        self.abstract_tree_ref = None
        self.compiler._compiled_tree_ref = None
        self.compiler._compiled_tree_ptr = 0
        self.compiler._compiled_tree_frontend_uid = ""
        self.compiler._compiled_tree_bl_idname = ""
        self.compiler._compiled_frontend_refs.clear()
        self.compiler._compiled_frontend_snapshots.clear()
        self.compiler._compiled_frontend_ptrs.clear()
        self.compiler._compiled_frontend_uids = ()
        self.frontends._dirty_trees.clear()
        self.frontends._dirty_tree_ptrs.clear()
        self.frontends._dirty_tree_bl_idnames.clear()
        self.frontends._dirty_frontend_uids.clear()
        self.frontends._pending_tree = None
        self.frontends._pending_tree_ptr = 0
        self.frontends._pending_tree_bl_idname = ""
        self.frontends._pending_scene_rebuild = False
        self.compiler._last_frontend_membership_signature = ()

    def _undo_recovery_active(self) -> bool:
        recovery_until = float(getattr(self, "undo_recovery_until", 0.0) or 0.0)
        return recovery_until > 0.0 and time.monotonic() < recovery_until

    @classmethod
    def _undo_recovery_active_for_scene(cls, scene=None) -> bool:
        engine = cls.get_instance(scene)
        if engine is None:
            return False
        try:
            return bool(engine._undo_recovery_active())
        except Exception as exc:
            utils.note_swallowed("engine.ACEAudioEngine._undo_recovery_active_for_scene", exc)
            return False

    @classmethod
    def _structural_tree_writes_allowed(
        cls,
        scene=None,
        tree=None,
        frontend_kind: str = "",
        tree_bl_idname: str = "",
    ) -> bool:
        if cls._undo_recovery_active_for_scene(scene):
            return False
        resolved_kind = str(frontend_kind or "")
        resolved_tree_bl_idname = str(tree_bl_idname or getattr(tree, "bl_idname", "") or "")
        if resolved_kind == FRONTEND_KIND_SHADER or resolved_tree_bl_idname == "ShaderNodeTree":
            return not cls._unsafe_for_shader_tree_mutations(scene)
        return True

    @classmethod
    def _instance_for_scene_name(cls, scene_name: str):
        token = str(scene_name or "").strip()
        if not token:
            return None
        for inst in cls._instances.values():
            if str(getattr(inst, "scene_name", "") or "") == token:
                return inst
        return None

    @staticmethod
    def _prune_timestamp_map(mapping: Dict[str, float], max_entries: int):
        if max_entries <= 0 or len(mapping) <= max_entries:
            return
        overflow = len(mapping) - max_entries
        if overflow <= 0:
            return
        for key, _ts in sorted(mapping.items(), key=lambda kv: float(kv[1]))[:overflow]:
            mapping.pop(key, None)

    @classmethod
    def get_instance(cls, scene=None):
        """Return the existing engine for *scene*, or None if none exists."""
        key = cls._scene_key(scene)
        inst = cls._instances.get(key)
        if inst:
            return inst
        if scene is not None:
            by_name = cls._instance_for_scene_name(cls._scene_name(scene))
            if by_name is not None:
                return by_name
        if scene is None:
            active_scene = getattr(getattr(bpy, "context", None), "scene", None)
            active_key = cls._scene_key(active_scene)
            inst = cls._instances.get(active_key)
            if inst:
                return inst
            if cls._instances:
                return next(iter(cls._instances.values()))
        return None

    @classmethod
    def ensure_instance(cls, scene=None):
        """Return the engine for *scene*, creating one if it does not yet exist."""
        resolved_scene = scene if scene is not None else getattr(getattr(bpy, "context", None), "scene", None)
        key = cls._scene_key(resolved_scene)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls._instance_for_scene_name(cls._scene_name(resolved_scene))
        if inst is None:
            inst = cls(key)
            cls._instances[key] = inst
        if resolved_scene is not None and not getattr(inst, "scene_name", ""):
            inst.scene_name = cls._scene_name(resolved_scene)
        return inst

    @classmethod
    def iter_instances(cls):
        """Return a snapshot list of all active engine instances."""
        return list(cls._instances.values())

    @classmethod
    def shutdown_all_instances(cls):
        """Shut down every engine instance, the preview monitor, and background workers."""
        for inst in list(cls._instances.values()):
            with utils.logged_errors("engine.ACEAudioEngine.shutdown_all_instances"):
                inst._shutdown_graphs(stop_parallel_pool=True, clear_bake_cache=True)
        cls._instances.clear()
        with utils.logged_errors("engine.ACEAudioEngine.shutdown_all_instances#2"):
            PreviewRouter._preview_monitor.close()
        with utils.logged_errors("engine.ACEAudioEngine.shutdown_all_instances#3"):
            SourceSpecsTask.shutdown_background_worker()
        with utils.logged_errors("engine.ACEAudioEngine.shutdown_all_instances#4"):
            _clear_file_open_locks()
        with utils.logged_errors("engine.ACEAudioEngine.shutdown_all_instances#5"):
            reset_biquad_fallback_warned()

    @staticmethod
    def _engine_settings(scene_or_engine_settings=None):
        if scene_or_engine_settings is None:
            scene = getattr(getattr(bpy, "context", None), "scene", None)
            return getattr(getattr(scene, "audiocore", None), "engine_settings", None)
        # Accept direct engine settings object
        if hasattr(scene_or_engine_settings, "ace_scheduler_mode"):
            return scene_or_engine_settings
        scene = scene_or_engine_settings
        return getattr(getattr(scene, "audiocore", None), "engine_settings", None)

    def _shutdown_graphs(self, stop_parallel_pool: bool = False, clear_bake_cache: bool = False):
        self.preview._stop_preview_worker()
        if stop_parallel_pool:
            self.parallel.shutdown_pool(wait=True)
        else:
            self.parallel.reset_generation()
        for graph in self.graphs.values():
            for task in graph.tasks.values():
                close_fn = getattr(task, "close", None)
                if callable(close_fn):
                    with utils.logged_errors("engine.ACEAudioEngine._shutdown_graphs"):
                        close_fn()
        self.graphs.clear()
        self.frame._block_size_state.clear()
        self._clear_blender_runtime_refs()
        self.diagnostics._pending_geometry_events.clear()
        self.probes._clear_probe_runtime_state()
        self.compiler._last_tree_signature = None
        self._attr_schema_dirty.clear()
        self._last_attr_sync_frame.clear()
        self.frame._source_specs_task_registry.clear()
        self.writer = None
        self.frame._preview_last_chunk = None
        self.frame._preview_last_eval_frame = -1
        self.frame._preview_next_eval_ts = 0.0
        if clear_bake_cache:
            self.bake._clear_bake_cache()
        self.bake._bake_jobs.clear()
        try:
            while True:
                self.bake._bake_result_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("engine.ACEAudioEngine._shutdown_graphs#2", exc)

    def invalidate_runtime_caches(self):
        writer = self.writer
        if writer is None:
            return
        with utils.logged_errors("engine.ACEAudioEngine.invalidate_runtime_caches"):
            writer.clear_runtime_caches()

    def begin_undo_recovery(self, scene=None, cooldown_s: float = 0.75) -> None:
        self.undo_generation = int(getattr(self, "undo_generation", 0) or 0) + 1
        self.undo_recovery_until = time.monotonic() + max(float(cooldown_s), 0.0)
        with utils.logged_errors("engine.ACEAudioEngine.begin_undo_recovery"):
            self.invalidate_runtime_caches()
        with utils.logged_errors("engine.ACEAudioEngine.begin_undo_recovery#2"):
            SourceSpecsTask.shutdown_background_worker()
        self._shutdown_graphs(clear_bake_cache=False)
        now = time.time()
        self.frontends._pending_scene_rebuild = True
        self.frontends._last_dirty_time = now
        self.frontends._first_dirty_time = now
        self.frontends._last_request_time = now
        if scene is not None:
            self.scene_key = self._scene_key(scene)
            self.scene_name = self._scene_name(scene)

    @staticmethod
    def _is_rendering_job() -> bool:
        try:
            checker = getattr(bpy.app, "is_job_running", None)
            return bool(checker("RENDER")) if callable(checker) else False
        except Exception as exc:
            utils.note_swallowed("engine.ACEAudioEngine._is_rendering_job", exc)
            return False

    @staticmethod
    def _unsafe_for_handler_writes(scene) -> bool:
        is_rendering = ACEAudioEngine._is_rendering_job()
        if not is_rendering:
            return False
        try:
            return not bool(getattr(getattr(scene, "render", None), "use_lock_interface", False))
        except Exception as exc:
            utils.note_swallowed("engine.ACEAudioEngine._unsafe_for_handler_writes", exc)
            return True

    @staticmethod
    def _is_shader_viewport_preview_active() -> bool:
        """Return True when a 3D viewport is compiling live shader previews."""
        try:
            wm = getattr(getattr(bpy, "context", None), "window_manager", None)
            windows = getattr(wm, "windows", []) if wm else []
            for window in windows:
                screen = getattr(window, "screen", None)
                if screen is None:
                    continue
                for area in getattr(screen, "areas", []) or []:
                    if str(getattr(area, "type", "") or "") != "VIEW_3D":
                        continue
                    spaces = getattr(area, "spaces", None)
                    active_space = getattr(spaces, "active", None) if spaces is not None else None
                    if active_space is None:
                        try:
                            space_iter = list(spaces or [])
                        except Exception as exc:
                            utils.note_swallowed("engine.ACEAudioEngine._is_shader_viewport_preview_active", exc)
                            space_iter = []
                    else:
                        space_iter = [active_space]
                    for space in space_iter:
                        if str(getattr(space, "type", "") or "") != "VIEW_3D":
                            continue
                        shading = getattr(space, "shading", None)
                        shading_type = str(getattr(shading, "type", "") or "")
                        if shading_type in {"RENDERED", "MATERIAL"}:
                            return True
        except Exception as exc:
            utils.note_swallowed("engine.ACEAudioEngine._is_shader_viewport_preview_active#2", exc)
            return False
        return False

    @classmethod
    def _unsafe_for_shader_tree_mutations(cls, scene) -> bool:
        if cls._unsafe_for_handler_writes(scene):
            return True
        return cls._is_shader_viewport_preview_active()

    @staticmethod
    def _is_animation_playing() -> bool:
        with utils.logged_errors("engine.ACEAudioEngine._is_animation_playing"):
            wm = getattr(getattr(bpy, "context", None), "window_manager", None)
            windows = getattr(wm, "windows", []) if wm else []
            for win in windows:
                screen = getattr(win, "screen", None)
                if screen and bool(getattr(screen, "is_animation_playing", False)):
                    return True
        return False

    @staticmethod
    def _tree_has_ace_nodes(tree: bpy.types.NodeTree | None):
        return FrontendTracker._tree_has_ace_nodes(tree)

    @classmethod
    def _tree_is_ace_managed(cls, tree):
        return FrontendTracker._tree_is_ace_managed(tree)

    def bake_source_to_cache(self, source, tree=None, scene=None):
        return self.bake.bake_source_to_cache(source, tree, scene)

    def compile(self, abstract_tree: ACEAbstractTree, blender_scene):
        return self.compiler.compile(abstract_tree, blender_scene)

    def compile_from_blender_tree(self, bl_tree: bpy.types.NodeTree, scene):
        return self.compiler.compile_from_blender_tree(bl_tree, scene)

    def compile_scene_frontends(self, scene, preferred_tree=None):
        return self.compiler.compile_scene_frontends(scene, preferred_tree)

    def discover_scene_frontends(self, scene):
        return self.frontends.discover_scene_frontends(scene)

    def execute_next_frame(self, scene, force: bool=False):
        return self.frame.execute_next_frame(scene, force)

    def flush_bake_results(self):
        return self.bake.flush_bake_results()

    def flush_source_tempo_detection_updates(self, sources):
        return self.probes.flush_source_tempo_detection_updates(sources)

    def get_dsp_node_profile_snapshot(self, reset: bool=False):
        return self.diagnostics.get_dsp_node_profile_snapshot(reset)

    def get_perf_current(self):
        return self.diagnostics.get_perf_current()

    def get_perf_snapshot(self, reset: bool=False):
        return self.diagnostics.get_perf_snapshot(reset)

    def get_runtime_diagnostics(self, reset: bool=False):
        return self.diagnostics.get_runtime_diagnostics(reset)

    def get_runtime_diagnostics_current(self):
        return self.diagnostics.get_runtime_diagnostics_current()

    @classmethod
    def preview_monitor_available(cls):
        return PreviewRouter.preview_monitor_available()

    @classmethod
    def preview_monitor_consecutive_errors(cls):
        return PreviewRouter.preview_monitor_consecutive_errors()

    def request_recompile(self, bl_tree: bpy.types.NodeTree | None):
        return self.frontends.request_recompile(bl_tree)

    def request_source_tempo_detection(self, source: Optional[ACEAudioSource]):
        return self.probes.request_source_tempo_detection(source)
