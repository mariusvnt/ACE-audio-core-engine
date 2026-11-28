import sys
import threading
import types
import unittest

from module_loader import PKG_NAME, load_module, load_tasks

import fake_bpy

try:
    import numpy  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    numpy = None


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


def _install_fake_bpy():
    return fake_bpy.install()


def _install_properties_stub():
    properties_mod = types.ModuleType(f"{PKG_NAME}.properties")

    class ACEAudioSource:  # pragma: no cover - import placeholder
        pass

    properties_mod.ACEAudioSource = ACEAudioSource
    sys.modules[f"{PKG_NAME}.properties"] = properties_mod


class _EngineStub(types.SimpleNamespace):
    """Engine stand-in for handler tests: component names resolve to the stub itself."""

    _COMPONENTS = frozenset({"diagnostics", "probes", "preview", "bake", "parallel", "frontends", "sinks", "compiler", "frame"})

    def __getattr__(self, name):
        if name in self._COMPONENTS:
            return self
        raise AttributeError(name)


class _FakeTree(fake_bpy.TruthyDict):
    def __init__(self, name: str, bl_idname: str, ptr: int, ace_nodes: bool = False):
        super().__init__()
        self.name = name
        self.bl_idname = bl_idname
        self.nodes = [_FakeAceNode("Loudness")] if ace_nodes else []
        self._ptr = int(ptr)

    def as_pointer(self):
        return self._ptr


class _FakeModifier:
    def __init__(self, name: str, node_group):
        self.name = name
        self.type = "NODES"
        self.node_group = node_group


class _FakeMaterial:
    def __init__(self, name: str, node_tree):
        self.name = name
        self.node_tree = node_tree


class _FakeMaterialSlot:
    def __init__(self, material):
        self.material = material


class _FakeObject:
    def __init__(self, name: str, modifiers=None, material_slots=None):
        self.name = name
        self.modifiers = list(modifiers or [])
        self.material_slots = list(material_slots or [])


class _FakeScene:
    def __init__(self, objects=None):
        self.objects = list(objects or [])


class _FakePropertyCtx(fake_bpy.TruthyDict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - test convenience fallback
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeSocket:
    def __init__(self, name: str, socket_idx: int = 0, is_output: bool = False, data_type: str = "NodeSocketFloat"):
        self.name = name
        self.socket_idx = int(socket_idx)
        self.identifier = name
        self.is_linked = False
        self.links = []
        self.default_value = None
        self.index = int(socket_idx)
        self.bl_socket_idname = data_type
        self.type = data_type
        self.is_output = bool(is_output)


class _FakeAceGroup(fake_bpy.TruthyDict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class _FakeAceNode(fake_bpy.TruthyDict):
    """A node using an ACE group: what makes a tree an ACE frontend."""

    def __init__(self, name: str, group_name: str = ".ACE_audio_loudness"):
        super().__init__()
        self.type = "GROUP"
        self.name = name
        self.label = name
        self.node_tree = _FakeAceGroup(group_name)
        self.inputs = []
        self.outputs = []


class TestMultiFrontendCompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if numpy is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        _purge_testpkg_modules()
        _install_fake_bpy()
        _install_properties_stub()
        cls.abs_mod = load_module("abstract_tree")
        cls.idm = load_module("id_management")
        cls.tasks_mod = load_tasks()
        cls.dummy_tree = load_module("nodes.geometry.reader")
        cls.shader_tree_mod = load_module("nodes.shader.factory")
        cls.engine_mod = load_module("core.engine.engine")
        cls.handlers_mod = load_module("core.engine.handlers")
        cls.utils_mod = load_module("utils")

    def _engine_shell(self):
        engine = self.engine_mod.ACEAudioEngine(0)
        engine.writer = types.SimpleNamespace(output_defs={})
        engine.graphs = {}
        engine.current_frame = 0
        engine.last_scene_frame = -1
        engine.abstract_tree_ref = None
        engine.frontends._pending_tree = None
        engine.frontends._pending_tree_ptr = 0
        engine.frontends._pending_tree_bl_idname = ""
        engine.frontends._pending_scene_rebuild = False
        engine.frontends._last_request_time = 0.0
        engine.compiler._last_compile_time = 0.0
        engine.frontends._dirty_trees = set()
        engine.frontends._dirty_tree_ptrs = set()
        engine.frontends._dirty_tree_bl_idnames = {}
        engine.frontends._dirty_frontend_uids = set()
        engine.compiler._compiled_tree_ref = None
        engine.compiler._compiled_tree_ptr = 0
        engine.compiler._compiled_tree_frontend_uid = ""
        engine.compiler._compiled_tree_bl_idname = ""
        engine.compiler._compiled_frontend_refs = {}
        engine.compiler._compiled_frontend_snapshots = {}
        engine.compiler._compiled_frontend_ptrs = {}
        engine.compiler._compiled_frontend_uids = ()
        engine.compiler._last_frontend_membership_signature = ()
        engine.frontends._is_compiling = False
        engine.probes._meta_cache = {}
        engine.probes._meta_cache_ttl = 30.0
        engine.compiler._last_tree_signature = None
        engine._last_attr_sync_frame = {}
        engine.frame._block_size_state = {}
        engine._bootstrap_retry_at = 0.0
        engine._bootstrap_fail_count = 0
        engine.bake._bake_cache = {}
        engine.bake._bake_jobs = {}
        import queue as _queue
        engine.bake._bake_result_queue = _queue.Queue()
        engine._current_delta_frames = 1.0
        engine.diagnostics._pending_geometry_events = []
        engine.parallel.last_submit_frame = -1
        engine.frame._preview_last_chunk = None
        engine.frame._preview_last_eval_frame = -1
        engine.frame._preview_next_eval_ts = 0.0
        engine.preview._preview_worker_stop = False
        engine.preview._preview_worker_cond = threading.Condition()
        engine.preview._preview_worker_queue = []
        engine.preview._preview_worker_thread = None
        engine.probes._probe_pending_updates = {}
        engine.probes._probe_in_flight = set()
        engine.probes._probe_thread_lock = threading.Lock()
        engine._attr_schema_dirty = set()
        engine._last_attr_sync_frame = {}
        engine.scene_key = 0
        engine.scene_name = ""
        engine.undo_generation = 0
        engine.undo_recovery_until = 0.0
        return engine

    def test_merge_from_rejects_duplicate_frontend_and_node_ids(self):
        tree_a = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_a": self.abs_mod.ACEAbstractNode(
                    id="node_a",
                    name="A",
                    category="audio",
                    type="rms",
                )
            },
            frontends={
                "frontend_a": self.abs_mod.ACEFrontendRef(
                    uid="frontend_a",
                    kind="GEOMETRY_NODES",
                    tree_name="TreeA",
                    tree_bl_idname="GeometryNodeTree",
                )
            },
            consumer_intents=[
                self.abs_mod.ACEConsumerIntent(
                    frontend_uid="frontend_a",
                    node_id="node_a",
                    socket_idx=0,
                    sink_kind="PROXY_ATTRIBUTE",
                )
            ],
        )
        merged = self.abs_mod.ACEAbstractTree()
        merged.merge_from(tree_a)
        self.assertIn("frontend_a", merged.frontends)
        self.assertIn("node_a", merged.nodes)
        self.assertEqual(len(merged.consumer_intents), 1)

        dup_frontend = self.abs_mod.ACEAbstractTree(
            frontends={
                "frontend_a": self.abs_mod.ACEFrontendRef(
                    uid="frontend_a",
                    kind="SHADER_NODES",
                    tree_name="TreeB",
                    tree_bl_idname="ShaderNodeTree",
                )
            }
        )
        with self.assertRaises(ValueError):
            merged.merge_from(dup_frontend)

        dup_node = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_a": self.abs_mod.ACEAbstractNode(
                    id="node_a",
                    name="Dup",
                    category="audio",
                    type="rms",
                )
            }
        )
        with self.assertRaises(ValueError):
            merged.merge_from(dup_node)

    def test_scene_signature_is_stable_across_frontend_and_intent_order(self):
        node_a = self.abs_mod.ACEAbstractNode(
            id="node_a",
            name="RMS",
            category="audio",
            type="rms",
            frontend_uid="frontend_a",
            frontend_kind="GEOMETRY_NODES",
        )
        node_b = self.abs_mod.ACEAbstractNode(
            id="node_b",
            name="Shader RMS",
            category="audio",
            type="rms",
            frontend_uid="frontend_b",
            frontend_kind="SHADER_NODES",
        )
        frontend_a = self.abs_mod.ACEFrontendRef(
            uid="frontend_a",
            kind="GEOMETRY_NODES",
            tree_name="GN",
            tree_bl_idname="GeometryNodeTree",
            owner_keys=("Cube:ACE GN",),
            tree_ptr=11,
        )
        frontend_b = self.abs_mod.ACEFrontendRef(
            uid="frontend_b",
            kind="SHADER_NODES",
            tree_name="Shader",
            tree_bl_idname="ShaderNodeTree",
            owner_keys=("Cube",),
            material_name="ACE Mat",
            tree_ptr=22,
        )
        intent_a = self.abs_mod.ACEConsumerIntent(
            frontend_uid="frontend_a",
            node_id="node_a",
            socket_idx=0,
            sink_kind="PROXY_ATTRIBUTE",
            source_token="src_a",
            output_name="RMS",
        )
        intent_b = self.abs_mod.ACEConsumerIntent(
            frontend_uid="frontend_b",
            node_id="node_b",
            socket_idx=0,
            sink_kind="SCENE_PROPERTY",
            source_token="src_a",
            consumer_keys=("Cube",),
            output_name="RMS",
        )

        tree_a = self.abs_mod.ACEAbstractTree(
            nodes={"node_a": node_a, "node_b": node_b},
            frontends={"frontend_a": frontend_a, "frontend_b": frontend_b},
            consumer_intents=[intent_a, intent_b],
        )
        tree_b = self.abs_mod.ACEAbstractTree(
            nodes={"node_b": node_b, "node_a": node_a},
            frontends={"frontend_b": frontend_b, "frontend_a": frontend_a},
            consumer_intents=[intent_b, intent_a],
        )
        tree_c = self.abs_mod.ACEAbstractTree(
            nodes={"node_a": node_a, "node_b": node_b},
            frontends={"frontend_a": frontend_a, "frontend_b": frontend_b},
            consumer_intents=[
                intent_a,
                self.abs_mod.ACEConsumerIntent(
                    frontend_uid="frontend_b",
                    node_id="node_b",
                    socket_idx=0,
                    sink_kind="SCENE_PROPERTY",
                    source_token="src_b",
                    consumer_keys=("Cube",),
                    output_name="RMS",
                ),
            ],
        )

        sig_a = self.engine_mod.FrontendTracker._compute_tree_signature(tree_a)
        sig_b = self.engine_mod.FrontendTracker._compute_tree_signature(tree_b)
        sig_c = self.engine_mod.FrontendTracker._compute_tree_signature(tree_c)

        self.assertEqual(sig_a, sig_b)
        self.assertNotEqual(sig_a, sig_c)

    def test_scene_and_membership_signatures_ignore_frontend_discovery_index(self):
        frontend_a = self.abs_mod.ACEFrontendRef(
            uid="frontend_a",
            kind="GEOMETRY_NODES",
            tree_name="GN",
            tree_bl_idname="GeometryNodeTree",
            owner_keys=("Cube:ACE GN",),
            tree_ptr=11,
            discovery_index=0,
        )
        frontend_b = self.abs_mod.ACEFrontendRef(
            uid="frontend_a",
            kind="GEOMETRY_NODES",
            tree_name="GN",
            tree_bl_idname="GeometryNodeTree",
            owner_keys=("Cube:ACE GN",),
            tree_ptr=11,
            discovery_index=99,
        )
        tree_a = self.abs_mod.ACEAbstractTree(frontends={"frontend_a": frontend_a})
        tree_b = self.abs_mod.ACEAbstractTree(frontends={"frontend_a": frontend_b})
        discovery_a = load_module("core.engine.common")._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_a)
        discovery_b = load_module("core.engine.common")._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_b)

        tree_sig_a = self.engine_mod.FrontendTracker._compute_tree_signature(tree_a)
        tree_sig_b = self.engine_mod.FrontendTracker._compute_tree_signature(tree_b)
        membership_sig_a = self.engine_mod.FrontendTracker._frontend_membership_signature([discovery_a])
        membership_sig_b = self.engine_mod.FrontendTracker._frontend_membership_signature([discovery_b])

        self.assertEqual(tree_sig_a, tree_sig_b)
        self.assertEqual(membership_sig_a, membership_sig_b)

    def test_global_manual_preview_winner_uses_frontend_discovery_and_node_order(self):
        engine = self._engine_shell()
        abstract_tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "preview_old": self.abs_mod.ACEAbstractNode(
                    id="preview_old",
                    name="Preview Old",
                    category="output",
                    type="preview_sink",
                    frontend_uid="frontend_a",
                    parameters={"frontend_node_index": 2},
                ),
                "preview_new": self.abs_mod.ACEAbstractNode(
                    id="preview_new",
                    name="Preview New",
                    category="output",
                    type="preview_sink",
                    frontend_uid="frontend_b",
                    parameters={"frontend_node_index": 1},
                ),
            },
            frontends={
                "frontend_a": self.abs_mod.ACEFrontendRef(
                    uid="frontend_a",
                    kind="GEOMETRY_NODES",
                    tree_name="GN A",
                    tree_bl_idname="GeometryNodeTree",
                    discovery_index=0,
                ),
                "frontend_b": self.abs_mod.ACEFrontendRef(
                    uid="frontend_b",
                    kind="SHADER_NODES",
                    tree_name="Shader B",
                    tree_bl_idname="ShaderNodeTree",
                    discovery_index=1,
                ),
            },
        )

        winner, ordered = engine.preview._resolve_global_manual_preview_winner(abstract_tree)

        self.assertEqual(winner, "preview_new")
        self.assertEqual(ordered, ["preview_old", "preview_new"])

    def test_membership_signature_gate_skips_scene_rebuild_without_membership_change(self):
        engine = self._engine_shell()
        frontend_ref = self.abs_mod.ACEFrontendRef(
            uid="frontend_a",
            kind="GEOMETRY_NODES",
            tree_name="GN",
            tree_bl_idname="GeometryNodeTree",
            tree_ptr=11,
        )
        discovery = load_module("core.engine.common")._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_ref)
        stable_sig = engine.frontends._frontend_membership_signature([discovery])
        engine.compiler._last_frontend_membership_signature = stable_sig
        engine.frontends._dirty_frontend_uids = set()
        engine.frontends._pending_scene_rebuild = True
        engine.frontends._last_dirty_time = 0.0
        engine.compiler._last_compile_time = 0.0
        engine.frontends._first_dirty_time = 0.0
        engine.frontends._is_compiling = False
        engine.compiler._compiled_frontend_refs = {}
        engine.frontends._pending_tree = None
        engine.frontends._dirty_trees = set()
        engine.frontends.discover_scene_frontends = lambda _scene: [discovery]
        engine.compiler.compile_scene_frontends = lambda *_args, **_kwargs: self.fail("compile_scene_frontends should be skipped")

        compiled = engine.frontends._maybe_compile_dirty(_FakeScene())

        self.assertFalse(compiled)
        self.assertFalse(engine.frontends._pending_scene_rebuild)

    def test_object_property_dirty_recompile_is_absorbed_by_membership_gate(self):
        engine = self._engine_shell()
        frontend_ref = self.abs_mod.ACEFrontendRef(
            uid="frontend_shader",
            kind="SHADER_NODES",
            tree_name="Shader",
            tree_bl_idname="ShaderNodeTree",
            owner_keys=("Cube",),
            material_name="ACE Mat",
            tree_ptr=22,
        )
        discovery = load_module("core.engine.common")._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_ref, consumer_object_names=("Cube",))
        stable_sig = engine.frontends._frontend_membership_signature([discovery])
        engine.compiler._last_frontend_membership_signature = stable_sig
        engine.frontends._dirty_frontend_uids = set()
        engine.frontends._pending_scene_rebuild = False
        engine.frontends._last_dirty_time = 0.0
        engine.compiler._last_compile_time = 0.0
        engine.frontends._first_dirty_time = 0.0
        engine.frontends._last_request_time = 0.0
        engine.frontends._is_compiling = False
        engine.compiler._compiled_frontend_refs = {}
        engine.frontends._pending_tree = None
        engine.frontends._dirty_trees = set()
        engine.frontends.discover_scene_frontends = lambda _scene: [discovery]
        engine.compiler.compile_scene_frontends = lambda *_args, **_kwargs: self.fail("compile_scene_frontends should be skipped")

        engine.frontends.request_recompile(None)
        engine.frontends._last_dirty_time = 0.0
        engine.frontends._first_dirty_time = 0.0
        compiled = engine.frontends._maybe_compile_dirty(_FakeScene())

        self.assertFalse(compiled)
        self.assertFalse(engine.frontends._pending_scene_rebuild)

    def test_sink_bindings_follow_the_sink_names_of_the_tree(self):
        """GN values go to the proxy, shader values to one scene property, both under the node's own name."""
        engine = self._engine_shell()
        engine.scene_name = "Scene"
        engine.writer.output_defs = {
            "gn_attr": {"node_id": "node_gn", "socket_idx": 0, "type": "FLOAT", "attr_id": "ACE_rms_aaaaaa"},
            "shader_attr": {"node_id": "node_shader", "socket_idx": 0, "type": "FLOAT", "attr_id": "ACE_rms_bbbbbb"},
            "unnamed": {"node_id": "node_gn", "socket_idx": 1, "type": "FLOAT", "attr_id": ""},
        }
        graph = types.SimpleNamespace(output_attr_keys=["gn_attr", "shader_attr", "unnamed"])
        tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_gn": self.abs_mod.ACEAbstractNode(
                    id="node_gn", name="GN RMS", category="audio", type="loudness",
                    frontend_uid="frontend_gn", frontend_kind="GEOMETRY_NODES",
                ),
                "node_shader": self.abs_mod.ACEAbstractNode(
                    id="node_shader", name="Shader RMS", category="audio", type="loudness",
                    frontend_uid="frontend_shader", frontend_kind="SHADER_NODES",
                ),
            },
        )

        bindings = engine.sinks._resolve_sink_bindings_for_graph(
            graph, tree, source_uid="src_a", proxy_name="ACE_PROXY_src_a", single_source_mode=False,
        )

        (gn,) = bindings["gn_attr"]
        self.assertEqual((gn.sink_kind, gn.target_object_name, gn.target_data_name),
                         ("PROXY_ATTRIBUTE", "ACE_PROXY_src_a", "ACE_rms_aaaaaa"))
        (shader,) = bindings["shader_attr"]
        self.assertEqual((shader.sink_kind, shader.target_object_name, shader.target_data_name),
                         ("SCENE_PROPERTY", "Scene", "ACE_rms_bbbbbb"))
        self.assertEqual(bindings["unnamed"], [])

    def test_depsgraph_handler_marks_updated_trees_and_never_object_updates(self):
        engine = _EngineStub(
            requested=[],
            _undo_recovery_active=lambda: False,
            _consume_geometry_eval_updates=lambda *_args, **_kwargs: None,
            request_recompile=lambda tree, **_kw: engine.requested.append(tree),
        )
        settings = types.SimpleNamespace(auto_compile=True)
        scene = types.SimpleNamespace(audiocore=types.SimpleNamespace(engine_settings=settings, audio_sources=[]))

        geometry_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 606, ace_nodes=True)
        shader_tree = _FakeTree("ACE_Shader", "ShaderNodeTree", 707, ace_nodes=True)
        material = _FakeMaterial("ACE Mat", shader_tree)
        # As in Blender: ID types carry their type in bl_rna, not bl_idname.
        material.bl_rna = types.SimpleNamespace(identifier="Material")
        obj = _FakeObject(
            "Cube",
            modifiers=[_FakeModifier("ACE GN", geometry_tree)],
            material_slots=[_FakeMaterialSlot(material)],
        )
        obj.bl_rna = types.SimpleNamespace(identifier="Object")
        # Depsgraph updates hand over evaluated copies; .original is the datablock.
        evaluated_obj = types.SimpleNamespace(
            original=obj,
            bl_rna=obj.bl_rna,
            name=obj.name,
            modifiers=[],
            material_slots=[],
        )

        orig_ensure = self.engine_mod.ACEAudioEngine.ensure_instance
        orig_get = self.engine_mod.ACEAudioEngine.get_instance
        try:
            self.engine_mod.ACEAudioEngine.ensure_instance = classmethod(lambda cls, _scene=None: engine)
            self.engine_mod.ACEAudioEngine.get_instance = classmethod(lambda cls, _scene=None: engine)

            self.handlers_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=geometry_tree)]),
            )
            self.handlers_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=shader_tree)]),
            )
            self.handlers_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=obj)]),
            )
            self.handlers_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=material)]),
            )
            self.handlers_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=evaluated_obj)]),
            )
        finally:
            self.engine_mod.ACEAudioEngine.ensure_instance = orig_ensure
            self.engine_mod.ACEAudioEngine.get_instance = orig_get

        # Trees and the material's embedded tree are marked; objects (evaluated or not) mark
        # nothing: which trees an object uses is checked by the compile poll's membership.
        self.assertEqual(engine.requested, [geometry_tree, shader_tree, shader_tree])

    def test_frame_change_handler_skips_compile_and_sink_work_when_render_writes_are_unsafe(self):
        calls = []
        engine = _EngineStub(
            graphs={},
            _undo_recovery_active=lambda: False,
            _is_rendering_job=lambda: True,
            _unsafe_for_handler_writes=lambda _scene: True,
            execute_next_frame=lambda *_args, **_kwargs: calls.append(("execute",)),
            discover_scene_frontends=lambda *_args, **_kwargs: calls.append(("discover",)) or [],
            compile_scene_frontends=lambda *_args, **_kwargs: calls.append(("compile",)),
        )
        scene = types.SimpleNamespace()

        orig_ensure = self.engine_mod.ACEAudioEngine.ensure_instance
        orig_log = self.utils_mod.log_throttled
        try:
            self.engine_mod.ACEAudioEngine.ensure_instance = classmethod(lambda cls, _scene=None: engine)
            self.utils_mod.log_throttled = lambda *args, **kwargs: calls.append(("log", args, kwargs))
            self.handlers_mod.frame_change_handler(scene)
        finally:
            self.engine_mod.ACEAudioEngine.ensure_instance = orig_ensure
            self.utils_mod.log_throttled = orig_log

        self.assertTrue(any(call[0] == "log" for call in calls))
        self.assertFalse(any(call[0] == "execute" for call in calls))
        self.assertFalse(any(call[0] == "discover" for call in calls))
        self.assertFalse(any(call[0] == "compile" for call in calls))

    def test_shader_tree_mutations_are_unsafe_in_rendered_viewport_preview(self):
        bpy_mod = sys.modules["bpy"]
        original_windows = list(getattr(bpy_mod.context.window_manager, "windows", []) or [])
        area = types.SimpleNamespace(
            type="VIEW_3D",
            spaces=types.SimpleNamespace(
                active=types.SimpleNamespace(
                    type="VIEW_3D",
                    shading=types.SimpleNamespace(type="RENDERED"),
                )
            ),
        )
        bpy_mod.context.window_manager.windows = [types.SimpleNamespace(screen=types.SimpleNamespace(areas=[area]))]
        try:
            scene = types.SimpleNamespace(render=types.SimpleNamespace(use_lock_interface=False))
            self.assertTrue(self.engine_mod.ACEAudioEngine._unsafe_for_shader_tree_mutations(scene))
        finally:
            bpy_mod.context.window_manager.windows = original_windows

    def test_begin_undo_recovery_clears_cached_frontend_refs_and_sets_recovery_window(self):
        engine = self._engine_shell()
        engine.abstract_tree_ref = object()
        engine.compiler._compiled_tree_ref = object()
        engine.compiler._compiled_tree_ptr = 123
        engine.compiler._compiled_tree_frontend_uid = "frontend_a"
        engine.compiler._compiled_frontend_refs = {"frontend_a": object()}
        engine.compiler._compiled_frontend_snapshots = {"frontend_a": object()}
        engine.compiler._compiled_frontend_ptrs = {"frontend_a": 123}
        engine.compiler._compiled_frontend_uids = ("frontend_a",)
        engine.frontends._pending_tree_ptr = 456
        engine.frontends._pending_tree_bl_idname = "ShaderNodeTree"
        engine.frontends._dirty_tree_ptrs = {456}
        engine.frontends._dirty_tree_bl_idnames = {456: "ShaderNodeTree"}
        engine.frontends._dirty_frontend_uids = {"frontend_a"}
        engine.frontends._pending_scene_rebuild = False
        engine.probes._probe_pending_updates = {"file.wav": {"undo_generation": 0}}
        engine.probes._probe_in_flight = {"file.wav"}
        engine.diagnostics._pending_geometry_events = [{"frame": 1}]
        engine.preview._preview_worker_thread = None
        engine.parallel.pool.executor = None
        engine.invalidate_runtime_caches = lambda: None
        scene = types.SimpleNamespace(name="Scene")
        original_monotonic = self.engine_mod.time.monotonic
        original_shutdown_bg = self.engine_mod.TempoService.shutdown
        try:
            self.engine_mod.time.monotonic = lambda: 10.0
            self.engine_mod.TempoService.shutdown = classmethod(lambda cls: None)
            engine.begin_undo_recovery(scene)
        finally:
            self.engine_mod.time.monotonic = original_monotonic
            self.engine_mod.TempoService.shutdown = original_shutdown_bg

        self.assertEqual(engine.undo_generation, 1)
        self.assertEqual(engine.undo_recovery_until, 10.75)
        self.assertIsNone(engine.abstract_tree_ref)
        self.assertIsNone(engine.compiler._compiled_tree_ref)
        self.assertEqual(engine.compiler._compiled_tree_ptr, 0)
        self.assertEqual(engine.compiler._compiled_tree_frontend_uid, "")
        self.assertEqual(engine.compiler._compiled_frontend_snapshots, {})
        self.assertEqual(engine.compiler._compiled_frontend_ptrs, {})
        self.assertEqual(engine.frontends._pending_tree_ptr, 0)
        self.assertEqual(engine.frontends._dirty_tree_ptrs, set())
        self.assertEqual(engine.probes._probe_pending_updates, {})
        self.assertEqual(engine.probes._probe_in_flight, set())
        self.assertEqual(engine.diagnostics._pending_geometry_events, [])
        self.assertTrue(engine.frontends._pending_scene_rebuild)

    def test_depsgraph_handler_during_undo_recovery_marks_scene_rebuild_only(self):
        engine = _EngineStub(
            requested=[],
            _undo_recovery_active=lambda: True,
            request_recompile=lambda tree, **_kw: engine.requested.append(tree),
        )
        settings = types.SimpleNamespace(auto_compile=True)
        scene = types.SimpleNamespace(audiocore=types.SimpleNamespace(engine_settings=settings, audio_sources=[]))
        geometry_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 808, ace_nodes=True)

        orig_get = self.engine_mod.ACEAudioEngine.get_instance
        try:
            self.engine_mod.ACEAudioEngine.get_instance = classmethod(lambda cls, _scene=None: engine)
            self.handlers_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=geometry_tree)]),
            )
        finally:
            self.engine_mod.ACEAudioEngine.get_instance = orig_get

        self.assertEqual(engine.requested, [None])

    def test_auto_compile_timer_skips_compile_during_undo_recovery(self):
        scene = types.SimpleNamespace(audiocore=types.SimpleNamespace(engine_settings=types.SimpleNamespace(auto_compile=True)))
        engine = _EngineStub(
            _undo_recovery_active=lambda: True,
            _unsafe_for_handler_writes=lambda _scene: False,
            _maybe_compile_dirty=lambda _scene: self.fail("_maybe_compile_dirty should be skipped"),
        )
        bpy_mod = sys.modules["bpy"]
        original_scene = bpy_mod.context.scene
        orig_ensure = self.engine_mod.ACEAudioEngine.ensure_instance
        try:
            bpy_mod.context.scene = scene
            self.engine_mod.ACEAudioEngine.ensure_instance = classmethod(lambda cls, _scene=None: engine)
            self.assertEqual(self.handlers_mod.auto_compile_timer(), 0.25)
        finally:
            bpy_mod.context.scene = original_scene
            self.engine_mod.ACEAudioEngine.ensure_instance = orig_ensure

    def test_source_task_flush_pending_ctx_writes_ignores_stale_generation(self):
        task = self.tasks_mod.SourceTask("src", {}, 64)
        task.source_context = _FakePropertyCtx()
        task.engine_ref = types.SimpleNamespace(undo_generation=4)
        task.undo_generation = 3
        task._pending_ctx_writes["sample_rate"] = 48000

        task._flush_pending_ctx_writes()

        self.assertNotIn("sample_rate", task.source_context)
        self.assertEqual(task._pending_ctx_writes, {})

    def test_flush_probe_updates_ignores_results_from_stale_undo_generation(self):
        engine = self._engine_shell()
        engine.undo_generation = 5
        engine.probes._probe_pending_updates = {
            "/tmp/a.wav": {
                "sample_rate": 48000,
                "native_sample_rate": 48000,
                "duration": 1.0,
                "undo_generation": 4,
            }
        }
        source = _FakePropertyCtx(source_type="FILE", filepath="/tmp/a.wav", sample_rate=0, native_sample_rate=0, duration=0.0)
        engine.frontends.discover_scene_frontends = lambda _scene: []
        engine.frontends.request_recompile = lambda *_args, **_kwargs: self.fail("request_recompile should be skipped for stale probes")

        updated = engine.probes._flush_probe_updates([source], scene=types.SimpleNamespace())

        self.assertFalse(updated)
        self.assertEqual(source["native_sample_rate"], 0)
        self.assertEqual(source["duration"], 0.0)

    def test_parallel_usable_values_accepts_current_frame_payload(self):
        engine = self._engine_shell()
        engine.parallel.results = {"graph_a": {24: {"node:0": 0.0}}}
        engine.parallel.last_valid = {}
        engine.parallel.last_valid_frame = {}
        engine._current_scene_fps = 30.0

        self.assertTrue(engine.parallel.has_usable_values("graph_a", 24))

    def test_parallel_usable_values_accepts_recent_last_valid_cache(self):
        engine = self._engine_shell()
        engine.parallel.results = {"graph_a": {}}
        engine.parallel.last_valid = {"graph_a": {"node:0": 0.42}}
        engine.parallel.last_valid_frame = {"graph_a": 100}
        engine._current_scene_fps = 30.0

        self.assertTrue(engine.parallel.has_usable_values("graph_a", 102))

    def test_parallel_usable_values_rejects_stale_or_empty_cache(self):
        engine = self._engine_shell()
        engine.parallel.results = {"graph_a": {}}
        engine.parallel.last_valid = {"graph_a": {"node:0": 0.42}}
        engine.parallel.last_valid_frame = {"graph_a": 100}
        engine._current_scene_fps = 30.0

        self.assertFalse(engine.parallel.has_usable_values("graph_a", 104))
        self.assertFalse(engine.parallel.has_usable_values("graph_missing", 104))

    def test_compile_requires_main_thread(self):
        errors = []
        engine = self.engine_mod.ACEAudioEngine(0)
        abstract_tree = self.abs_mod.ACEAbstractTree()

        self.utils_mod.set_debug_strict(True)
        try:
            def _run():
                try:
                    engine.compiler.compile(abstract_tree, types.SimpleNamespace(audiocore=None))
                except Exception as exc:  # pragma: no cover - exercised in worker thread
                    errors.append(exc)

            worker = threading.Thread(target=_run, name="ACECompileWorker")
            worker.start()
            worker.join(timeout=2.0)
        finally:
            self.utils_mod.set_debug_strict(False)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("compile", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
