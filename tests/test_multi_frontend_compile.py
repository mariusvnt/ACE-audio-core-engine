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
    def __init__(self, name: str, bl_idname: str, ptr: int, ace_frontend_uid: str = ""):
        super().__init__()
        self.name = name
        self.bl_idname = bl_idname
        self.nodes = []
        if ace_frontend_uid:
            self["ace_frontend_uid"] = ace_frontend_uid
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


class _FakeGroupNode(fake_bpy.TruthyDict):
    def __init__(self, name: str, node_tree, ace_source_id: str):
        super().__init__()
        self.type = "GROUP"
        self.name = name
        self.label = name
        self.node_tree = node_tree
        self.inputs = []
        self.outputs = [_FakeSocket("Audio", socket_idx=0, is_output=True, data_type="NodeSocketBundle")]
        self["ace_source_id"] = ace_source_id

    def as_pointer(self):
        return id(self)


class _FakeAttrNode(fake_bpy.TruthyDict):
    def __init__(self, ace_id: str, input_names=None, output_names=None):
        super().__init__()
        self.type = "GROUP"
        self.name = ace_id
        self.label = ace_id
        self.id_data = None
        self.inputs = [
            _FakeSocket(str(name), socket_idx=idx, is_output=False, data_type="NodeSocketString")
            for idx, name in enumerate(input_names or [])
        ]
        self.outputs = [
            _FakeSocket(str(name), socket_idx=idx, is_output=True, data_type="NodeSocketFloat")
            for idx, name in enumerate(output_names or [])
        ]
        self["ace_id"] = ace_id

    def as_pointer(self):
        return id(self)


class _FakeLayoutNode(fake_bpy.TruthyDict):
    def __init__(self, bl_idname: str, node_type: str, ace_id: str = "", output_names=None):
        super().__init__()
        self.bl_idname = bl_idname
        self.type = node_type
        self.name = bl_idname
        self.label = bl_idname
        self.location = (0.0, 0.0)
        self.dimensions = (0.0, 0.0)
        self.parent = None
        self.hide = False
        self.width_hidden = 0.0
        self.shrink = False
        self.inputs = []
        self.outputs = [
            _FakeSocket(str(name), socket_idx=idx, is_output=True, data_type="NodeSocketFloat")
            for idx, name in enumerate(output_names or [])
        ]
        if ace_id:
            self["ace_id"] = ace_id


class _FakeLayoutNodes(list):
    def __init__(self, tree):
        super().__init__()
        self._tree = tree

    def new(self, bl_idname: str):
        if bl_idname == "NodeFrame":
            node = _FakeLayoutNode("NodeFrame", "FRAME")
        elif bl_idname == "ShaderNodeAttribute":
            node = _FakeLayoutNode(
                "ShaderNodeAttribute",
                "ATTRIBUTE",
                output_names=["Factor", "Vector", "Color"],
            )
            node.attribute_type = "GEOMETRY"
            node.attribute_name = ""
        else:  # pragma: no cover - only the helper layout node types are used here
            raise ValueError(f"Unsupported fake node type: {bl_idname}")
        node.id_data = self._tree
        self.append(node)
        return node

    def remove(self, node):
        super().remove(node)


class _FakeLayoutTree(_FakeTree):
    def __init__(self, name: str, bl_idname: str, ptr: int):
        super().__init__(name, bl_idname, ptr)
        self.nodes = _FakeLayoutNodes(self)


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
        cls.dummy_tree = load_module("dummy_tree")
        cls.shader_tree_mod = load_module("shader_tree")
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
        engine.frame._source_specs_task_registry = {}
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
                    category="stream",
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
                    category="stream",
                    type="rms",
                )
            }
        )
        with self.assertRaises(ValueError):
            merged.merge_from(dup_node)

    def test_frontend_uid_persistence_and_duplicate_repair_are_deterministic(self):
        engine = self._engine_shell()
        gn_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 101, ace_frontend_uid="ACE_FRONTEND_dup")
        sh_tree = _FakeTree("ACE_Shader", "ShaderNodeTree", 202, ace_frontend_uid="ACE_FRONTEND_dup")
        scene = _FakeScene(
            [
                _FakeObject("Speaker", modifiers=[_FakeModifier("ACE GN", gn_tree)]),
                _FakeObject(
                    "Receiver",
                    material_slots=[_FakeMaterialSlot(_FakeMaterial("ACE Mat", sh_tree))],
                ),
            ]
        )

        frontends_a = engine.frontends.discover_scene_frontends(scene)
        frontends_b = engine.frontends.discover_scene_frontends(scene)

        self.assertEqual(len(frontends_a), 2)
        self.assertEqual([rec.frontend_ref.uid for rec in frontends_a], [rec.frontend_ref.uid for rec in frontends_b])
        self.assertEqual(gn_tree["ace_frontend_uid"], "ACE_FRONTEND_dup")
        self.assertNotEqual(sh_tree["ace_frontend_uid"], "ACE_FRONTEND_dup")
        self.assertTrue(sh_tree["ace_frontend_uid"].startswith("ACE_FRONTEND_"))

    def test_frontend_uid_duplicate_repair_is_canonical_across_scene_order(self):
        engine = self._engine_shell()

        def _build_scene(reverse: bool):
            gn_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 101, ace_frontend_uid="ACE_FRONTEND_dup")
            sh_tree = _FakeTree("ACE_Shader", "ShaderNodeTree", 202, ace_frontend_uid="ACE_FRONTEND_dup")
            objects = [
                _FakeObject("Speaker", modifiers=[_FakeModifier("ACE GN", gn_tree)]),
                _FakeObject(
                    "Receiver",
                    material_slots=[_FakeMaterialSlot(_FakeMaterial("ACE Mat", sh_tree))],
                ),
            ]
            if reverse:
                objects.reverse()
            return _FakeScene(objects), gn_tree, sh_tree

        scene_a, gn_tree_a, sh_tree_a = _build_scene(reverse=False)
        scene_b, gn_tree_b, sh_tree_b = _build_scene(reverse=True)

        frontends_a = engine.frontends.discover_scene_frontends(scene_a)
        frontends_b = engine.frontends.discover_scene_frontends(scene_b)

        mapping_a = {rec.frontend_ref.tree_name: rec.frontend_ref.uid for rec in frontends_a}
        mapping_b = {rec.frontend_ref.tree_name: rec.frontend_ref.uid for rec in frontends_b}

        self.assertEqual(mapping_a, mapping_b)
        self.assertEqual(gn_tree_a["ace_frontend_uid"], gn_tree_b["ace_frontend_uid"])
        self.assertEqual(sh_tree_a["ace_frontend_uid"], sh_tree_b["ace_frontend_uid"])
        self.assertEqual(len(set(mapping_a.values())), 2)
        self.assertEqual(mapping_a["ACE_GN"], "ACE_FRONTEND_dup")
        self.assertNotEqual(mapping_a["ACE_Shader"], "ACE_FRONTEND_dup")

    def test_shared_shader_frontend_is_deduped_before_uid_repair(self):
        engine = self._engine_shell()
        shared_tree = _FakeTree("ACE_Shader_Shared", "ShaderNodeTree", 303, ace_frontend_uid="ACE_FRONTEND_shared")
        shared_mat = _FakeMaterial("ACE Shared Mat", shared_tree)
        scene = _FakeScene(
            [
                _FakeObject("Cube", material_slots=[_FakeMaterialSlot(shared_mat)]),
                _FakeObject("Sphere", material_slots=[_FakeMaterialSlot(shared_mat)]),
            ]
        )

        frontends_a = engine.frontends.discover_scene_frontends(scene)
        tree_uid_after_a = shared_tree.get("ace_frontend_uid")
        frontends_b = engine.frontends.discover_scene_frontends(scene)

        self.assertEqual(len(frontends_a), 1)
        self.assertEqual(len(frontends_b), 1)
        self.assertEqual(frontends_a[0].frontend_ref.uid, "ACE_FRONTEND_shared")
        self.assertEqual(frontends_b[0].frontend_ref.uid, "ACE_FRONTEND_shared")
        self.assertEqual(tree_uid_after_a, "ACE_FRONTEND_shared")
        self.assertEqual(shared_tree.get("ace_frontend_uid"), "ACE_FRONTEND_shared")
        self.assertEqual(frontends_a[0].frontend_ref.owner_keys, ("Cube", "Sphere"))
        self.assertEqual(frontends_a[0].consumer_object_names, ("Cube", "Sphere"))

    def test_source_node_identity_is_separated_from_source_binding_id(self):
        source_id = self.idm.ACEIdManager.generate_node_id("source_mngmt", "src")
        source_group = _FakeTree(".ACE_source_mngmt_src", "GeometryNodeTree", 303)
        source_node = _FakeGroupNode("Managed Source", source_group, source_id)
        tree = _FakeTree("ACE Scene", "GeometryNodeTree", 404)
        tree.nodes = [source_node]
        reader = self.dummy_tree.ACEDummyTreeReader(tree, frontend_uid="frontend_scene")

        ace_id = reader._ensure_ace_id(source_node, root_id=source_id)
        abstract_node = reader._parse_node(source_node, root_id=source_id)

        self.assertTrue(ace_id)
        self.assertNotEqual(ace_id, source_id)
        self.assertEqual(source_node.get("ace_id"), ace_id)
        self.assertEqual(abstract_node.id, ace_id)
        self.assertEqual(abstract_node.source_binding_id, source_id)
        self.assertEqual(abstract_node.frontend_uid, "frontend_scene")

    def test_scene_signature_is_stable_across_frontend_and_intent_order(self):
        node_a = self.abs_mod.ACEAbstractNode(
            id="node_a",
            name="RMS",
            category="stream",
            type="rms",
            frontend_uid="frontend_a",
            frontend_kind="GEOMETRY_NODES",
        )
        node_b = self.abs_mod.ACEAbstractNode(
            id="node_b",
            name="Shader RMS",
            category="stream",
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
            sink_kind="OBJECT_PROPERTY",
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
                    sink_kind="OBJECT_PROPERTY",
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
                    category="utilities",
                    type="preview_sink",
                    frontend_uid="frontend_a",
                    parameters={"frontend_node_index": 2},
                ),
                "preview_new": self.abs_mod.ACEAbstractNode(
                    id="preview_new",
                    name="Preview New",
                    category="utilities",
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

    def test_sink_binding_resolution_supports_mixed_geometry_and_shader_consumers(self):
        engine = self._engine_shell()
        engine.writer.output_defs = {
            "gn_attr": {"node_id": "node_gn", "socket_idx": 0, "type": "FLOAT", "output_name": "RMS"},
            "shader_attr": {"node_id": "node_shader", "socket_idx": 0, "type": "FLOAT", "output_name": "RMS"},
        }
        graph = types.SimpleNamespace(output_attr_keys=["gn_attr", "shader_attr"])
        tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_gn": self.abs_mod.ACEAbstractNode(
                    id="node_gn",
                    name="GN RMS",
                    category="stream",
                    type="rms",
                    frontend_uid="frontend_gn",
                    frontend_kind="GEOMETRY_NODES",
                ),
                "node_shader": self.abs_mod.ACEAbstractNode(
                    id="node_shader",
                    name="Shader RMS",
                    category="stream",
                    type="rms",
                    frontend_uid="frontend_shader",
                    frontend_kind="SHADER_NODES",
                ),
            },
            frontends={
                "frontend_gn": self.abs_mod.ACEFrontendRef(
                    uid="frontend_gn",
                    kind="GEOMETRY_NODES",
                    tree_name="GN",
                    tree_bl_idname="GeometryNodeTree",
                    owner_keys=("Cube:ACE GN",),
                    tree_ptr=11,
                ),
                "frontend_shader": self.abs_mod.ACEFrontendRef(
                    uid="frontend_shader",
                    kind="SHADER_NODES",
                    tree_name="Shader",
                    tree_bl_idname="ShaderNodeTree",
                    owner_keys=("Cube", "Sphere"),
                    material_name="ACE Mat",
                    tree_ptr=22,
                ),
            },
            consumer_intents=[
                self.abs_mod.ACEConsumerIntent(
                    frontend_uid="frontend_gn",
                    node_id="node_gn",
                    socket_idx=0,
                    sink_kind="PROXY_ATTRIBUTE",
                    source_token="src_a",
                    output_name="RMS",
                ),
                self.abs_mod.ACEConsumerIntent(
                    frontend_uid="frontend_shader",
                    node_id="node_shader",
                    socket_idx=0,
                    sink_kind="OBJECT_PROPERTY",
                    source_token="src_a",
                    consumer_keys=("Cube", "Sphere"),
                    output_name="RMS",
                ),
            ],
        )

        bindings = engine.sinks._resolve_sink_bindings_for_graph(
            graph,
            tree,
            source_uid="src_a",
            proxy_name="ACE_PROXY_src_a",
            single_source_mode=False,
        )

        self.assertEqual(len(bindings["gn_attr"]), 1)
        self.assertEqual(bindings["gn_attr"][0].sink_kind, "PROXY_ATTRIBUTE")
        self.assertEqual(bindings["gn_attr"][0].target_object_name, "ACE_PROXY_src_a")
        self.assertEqual(len(bindings["shader_attr"]), 2)
        shader_targets = {b.target_object_name for b in bindings["shader_attr"]}
        self.assertEqual(shader_targets, {"Cube", "Sphere"})
        self.assertTrue(
            all(binding.sink_kind == "OBJECT_PROPERTY" for binding in bindings["shader_attr"])
        )

    def test_shader_sink_bindings_resolve_to_consumer_objects_without_proxy(self):
        engine = self._engine_shell()
        engine.writer.output_defs = {
            "shader_attr": {"node_id": "node_shader", "socket_idx": 0, "type": "FLOAT", "output_name": "RMS"},
        }
        graph = types.SimpleNamespace(output_attr_keys=["shader_attr"])
        tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_shader": self.abs_mod.ACEAbstractNode(
                    id="node_shader",
                    name="Shader RMS",
                    category="stream",
                    type="rms",
                    frontend_uid="frontend_shader",
                    frontend_kind="SHADER_NODES",
                ),
            },
            frontends={
                "frontend_shader": self.abs_mod.ACEFrontendRef(
                    uid="frontend_shader",
                    kind="SHADER_NODES",
                    tree_name="Shader",
                    tree_bl_idname="ShaderNodeTree",
                    owner_keys=("Cube", "Sphere"),
                    material_name="ACE Mat",
                    tree_ptr=22,
                ),
            },
            consumer_intents=[
                self.abs_mod.ACEConsumerIntent(
                    frontend_uid="frontend_shader",
                    node_id="node_shader",
                    socket_idx=0,
                    sink_kind="OBJECT_PROPERTY",
                    source_token="src_a",
                    consumer_keys=("Cube", "Sphere"),
                    output_name="RMS",
                ),
            ],
        )

        bindings = engine.sinks._resolve_sink_bindings_for_graph(
            graph,
            tree,
            source_uid="src_a",
            proxy_name="",
            single_source_mode=False,
        )

        self.assertEqual(len(bindings["shader_attr"]), 2)
        shader_targets = {b.target_object_name for b in bindings["shader_attr"]}
        self.assertEqual(shader_targets, {"Cube", "Sphere"})

    def test_shader_output_ids_use_source_roots_in_multi_source_scenes(self):
        engine = self._engine_shell()
        shader_node = _FakeAttrNode(
            "node_shader",
            input_names=["Attribute ID"],
            output_names=["RMS"],
        )
        shader_tree = _FakeTree("ACE Shader", "ShaderNodeTree", 505)
        shader_tree.nodes = [shader_node]
        shader_node.id_data = shader_tree
        source_a = types.SimpleNamespace(stable_id="src_a", proxy_object_name="ACE_PROXY_a")
        source_b = types.SimpleNamespace(stable_id="src_b", proxy_object_name="ACE_PROXY_b")
        scene = types.SimpleNamespace(
            audiocore=types.SimpleNamespace(audio_sources=[source_a, source_b])
        )
        abstract_tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_shader": self.abs_mod.ACEAbstractNode(
                    id="node_shader",
                    name="Shader RMS",
                    category="stream",
                    type="rms",
                    root_sources={"src_b"},
                    outputs=[
                        self.abs_mod.ACEAbstractSocket(
                            id="node_shader_out_0",
                            name="RMS",
                            socket_idx=0,
                            is_output=True,
                            data_type="NodeSocketFloat",
                        )
                    ],
                )
            }
        )

        original_sync = self.shader_tree_mod.ACEShaderNodeFactory.sync_node_instance_group_bindings
        self.shader_tree_mod.ACEShaderNodeFactory.sync_node_instance_group_bindings = classmethod(
            lambda cls, node, proxy_name="", scene=None, attr_names_by_idx=None: True
        )
        try:
            changed, changed_node_ids = engine.sinks._sync_shader_output_ids(shader_tree, abstract_tree, scene)
        finally:
            self.shader_tree_mod.ACEShaderNodeFactory.sync_node_instance_group_bindings = original_sync

        expected_topology = "[\"src_b\"]|[]"
        # Shader sinks are one custom property per node instance, so their
        # names leave out the topology signature (it is still stored).
        expected_name = engine.sinks._build_auto_attr_name(
            "ACE_PROXY_b",
            "node_shader",
            "RMS",
            0,
            topology_sig="",
        )
        self.assertTrue(changed)
        self.assertEqual(changed_node_ids, {"node_shader"})
        self.assertEqual(shader_node.inputs[0].default_value, expected_name)
        self.assertEqual(shader_node.get("ace_auto_attr_topology_json"), expected_topology)

    def test_shader_instance_group_sync_uses_existing_attr_ids_when_proxy_is_unresolved(self):
        engine = self._engine_shell()
        shader_node = _FakeAttrNode(
            "node_shader",
            input_names=["Attribute ID"],
            output_names=["RMS"],
        )
        shader_node.inputs[0].default_value = "ACE_manual_attr"
        shader_tree = _FakeTree("ACE Shader", "ShaderNodeTree", 506)
        shader_tree.nodes = [shader_node]
        shader_node.id_data = shader_tree
        source_a = types.SimpleNamespace(stable_id="src_a", proxy_object_name="ACE_PROXY_a")
        source_b = types.SimpleNamespace(stable_id="src_b", proxy_object_name="ACE_PROXY_b")
        scene = types.SimpleNamespace(
            audiocore=types.SimpleNamespace(audio_sources=[source_a, source_b])
        )
        abstract_tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "node_shader": self.abs_mod.ACEAbstractNode(
                    id="node_shader",
                    name="Shader RMS",
                    category="stream",
                    type="rms",
                    root_sources=set(),
                    outputs=[
                        self.abs_mod.ACEAbstractSocket(
                            id="node_shader_out_0",
                            name="RMS",
                            socket_idx=0,
                            is_output=True,
                            data_type="NodeSocketFloat",
                        )
                    ],
                )
            }
        )

        calls = []
        original_sync = self.shader_tree_mod.ACEShaderNodeFactory.sync_node_instance_group_bindings

        def _fake_sync(cls, node, proxy_name="", scene=None, attr_names_by_idx=None):
            calls.append(
                {
                    "node": node,
                    "proxy_name": proxy_name,
                    "scene": scene,
                    "attr_names_by_idx": attr_names_by_idx,
                }
            )
            return True

        self.shader_tree_mod.ACEShaderNodeFactory.sync_node_instance_group_bindings = classmethod(_fake_sync)
        try:
            changed, changed_node_ids = engine.sinks._sync_shader_output_ids(shader_tree, abstract_tree, scene)
        finally:
            self.shader_tree_mod.ACEShaderNodeFactory.sync_node_instance_group_bindings = original_sync

        self.assertTrue(changed)
        self.assertEqual(changed_node_ids, {"node_shader"})
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["node"], shader_node)
        self.assertEqual(calls[0]["proxy_name"], "")
        self.assertIs(calls[0]["scene"], scene)
        self.assertIsNone(calls[0]["attr_names_by_idx"])
        self.assertEqual(shader_node.inputs[0].default_value, "ACE_manual_attr")

    def test_shader_helper_layout_groups_helper_under_owner_in_frame(self):
        ace_id = self.idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=[])
        tree = _FakeLayoutTree("ACE Shader", "ShaderNodeTree", 1501)
        owner = _FakeLayoutNode("ShaderNodeGroup", "GROUP", ace_id=ace_id)
        owner.location = (140.0, 320.0)
        tree.nodes.append(owner)
        helper = tree.nodes.new("ShaderNodeAttribute")

        changed = self.shader_tree_mod.ACEShaderNodeFactory._sync_helper_node_layout(
            tree,
            owner,
            helper,
            ace_id,
            0,
        )

        self.assertTrue(changed)
        self.assertIsNotNone(owner.parent)
        self.assertIs(owner.parent, helper.parent)
        self.assertEqual(owner.parent.bl_idname, "NodeFrame")
        self.assertEqual(owner.parent.get("ace_shader_helper_frame_owner"), ace_id)
        self.assertTrue(owner.parent.shrink)
        self.assertEqual(owner.location, (0.0, 0.0))
        self.assertEqual(helper.location, (0.0, -146.0))
        self.assertTrue(helper.hide)
        self.assertEqual(helper.width_hidden, 150.0)

    def test_cleanup_tree_helper_nodes_removes_empty_managed_helper_frame(self):
        ace_id = self.idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=[])
        tree = _FakeLayoutTree("ACE Shader", "ShaderNodeTree", 1502)
        owner = _FakeLayoutNode("ShaderNodeGroup", "GROUP", ace_id=ace_id)
        tree.nodes.append(owner)
        helper_frame = tree.nodes.new("NodeFrame")
        helper_frame["ace_shader_helper_frame_owner"] = ace_id

        changed = self.shader_tree_mod.ACEShaderNodeFactory.cleanup_tree_helper_nodes(tree)

        self.assertTrue(changed)
        self.assertIn(owner, tree.nodes)
        self.assertNotIn(helper_frame, tree.nodes)

    def test_depsgraph_handler_tracks_runtime_geometry_shader_and_object_updates(self):
        engine = _EngineStub(
            requested=[],
            _undo_recovery_active=lambda: False,
            _consume_geometry_eval_updates=lambda *_args, **_kwargs: None,
            request_recompile=lambda tree: engine.requested.append(tree),
        )
        settings = types.SimpleNamespace(auto_compile=True)
        scene = types.SimpleNamespace(audiocore=types.SimpleNamespace(engine_settings=settings, audio_sources=[]))

        geometry_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 606, ace_frontend_uid="ACE_FRONTEND_GN")
        shader_tree = _FakeTree("ACE_Shader", "ShaderNodeTree", 707, ace_frontend_uid="ACE_FRONTEND_SHADER")
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

        self.assertEqual(engine.requested[0], geometry_tree)
        self.assertEqual(engine.requested[1], shader_tree)
        self.assertEqual(engine.requested[2], geometry_tree)
        self.assertEqual(engine.requested[3], shader_tree)
        self.assertEqual(engine.requested[4], shader_tree)
        # The evaluated copy resolves to the original object's trees.
        self.assertEqual(engine.requested[5], geometry_tree)
        self.assertEqual(engine.requested[6], shader_tree)
        self.assertEqual(len(engine.requested), 7)

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

    def test_maybe_compile_dirty_defers_shader_rebuild_during_viewport_preview(self):
        engine = self._engine_shell()
        shader_tree = _FakeTree("ACE Shader", "ShaderNodeTree", 909, ace_frontend_uid="ACE_FRONTEND_SHADER")
        frontend_ref = self.abs_mod.ACEFrontendRef(
            uid="ACE_FRONTEND_SHADER",
            kind="SHADER_NODES",
            tree_name="ACE Shader",
            tree_bl_idname="ShaderNodeTree",
            tree_ptr=shader_tree.as_pointer(),
        )
        compile_calls = []
        log_calls = []
        engine.frontends.discover_scene_frontends = lambda _scene: [
            types.SimpleNamespace(
                tree=shader_tree,
                frontend_ref=frontend_ref,
                consumer_object_names=(),
            )
        ]
        engine.compiler._last_frontend_membership_signature = ()
        engine.frontends._dirty_frontend_uids = {"ACE_FRONTEND_SHADER"}
        engine.frontends._pending_scene_rebuild = False
        engine.frontends._dirty_trees = [shader_tree]
        engine.frontends._pending_tree = shader_tree
        engine.frontends._last_dirty_time = 0.0
        engine.compiler._last_compile_time = 0.0
        engine.frontends._is_compiling = False
        engine.frontends._first_dirty_time = 0.0
        engine.frontends._dirty_tree_ptrs = {shader_tree.as_pointer()}
        engine.frontends._dirty_tree_bl_idnames = {shader_tree.as_pointer(): "ShaderNodeTree"}
        engine.frontends._pending_tree_ptr = shader_tree.as_pointer()
        engine.frontends._pending_tree_bl_idname = "ShaderNodeTree"
        engine._structural_tree_writes_allowed = lambda *_args, **_kwargs: False
        engine.compiler.compile_scene_frontends = lambda *_args, **_kwargs: compile_calls.append(True)
        orig_log = self.utils_mod.log_throttled
        self.utils_mod.log_throttled = lambda *args, **kwargs: log_calls.append((args, kwargs))
        try:
            compiled = engine.frontends._maybe_compile_dirty(types.SimpleNamespace())
        finally:
            self.utils_mod.log_throttled = orig_log

        self.assertFalse(compiled)
        self.assertFalse(compile_calls)
        self.assertEqual(engine.frontends._dirty_frontend_uids, {"ACE_FRONTEND_SHADER"})
        self.assertEqual(engine.frontends._dirty_tree_ptrs, {shader_tree.as_pointer()})
        self.assertTrue(log_calls)

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
        original_shutdown_bg = self.tasks_mod.SourceSpecsTask.shutdown_background_worker
        try:
            self.engine_mod.time.monotonic = lambda: 10.0
            self.tasks_mod.SourceSpecsTask.shutdown_background_worker = classmethod(lambda cls: None)
            engine.begin_undo_recovery(scene)
        finally:
            self.engine_mod.time.monotonic = original_monotonic
            self.tasks_mod.SourceSpecsTask.shutdown_background_worker = original_shutdown_bg

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
            request_recompile=lambda tree: engine.requested.append(tree),
        )
        settings = types.SimpleNamespace(auto_compile=True)
        scene = types.SimpleNamespace(audiocore=types.SimpleNamespace(engine_settings=settings, audio_sources=[]))
        geometry_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 808, ace_frontend_uid="ACE_FRONTEND_GN")

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

    def test_source_specs_flush_pending_ctx_writes_ignores_stale_generation(self):
        self.tasks_mod.SourceSpecsTask.clear_pending_ctx_writes_queue()
        ctx = _FakePropertyCtx(bpm=120.0, beats_per_bar=4)
        task = types.SimpleNamespace(
            source_context=ctx,
            _auto_bpm_key="ace_bpm_auto",
            _auto_meter_key="ace_meter_auto",
            undo_generation=3,
            engine_ref=types.SimpleNamespace(undo_generation=4),
        )
        self.tasks_mod.SourceSpecsTask._PENDING_CTX_WRITES.put_nowait((("file.wav", 1.0), 128.0, 7, 3))

        self.tasks_mod.SourceSpecsTask.flush_pending_ctx_writes({("file.wav", 1.0): task})

        self.assertEqual(ctx["bpm"], 120.0)
        self.assertEqual(ctx["beats_per_bar"], 4)
        self.tasks_mod.SourceSpecsTask.clear_pending_ctx_writes_queue()

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
