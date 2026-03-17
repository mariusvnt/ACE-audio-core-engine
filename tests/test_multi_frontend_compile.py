import sys
import threading
import types
import unittest

from module_loader import PKG_NAME, load_module

try:
    import numpy  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    numpy = None


def _purge_testpkg_modules():
    for name in list(sys.modules):
        if name == PKG_NAME or name.startswith(f"{PKG_NAME}."):
            sys.modules.pop(name, None)


def _install_fake_bpy():
    if "bpy" in sys.modules:
        return

    bpy_mod = types.ModuleType("bpy")
    handlers_mod = types.ModuleType("bpy.app.handlers")
    handlers_mod.persistent = lambda fn: fn
    for name in (
        "load_post",
        "undo_post",
        "redo_post",
        "frame_change_pre",
        "frame_change_post",
        "render_pre",
        "render_post",
        "render_cancel",
        "depsgraph_update_post",
    ):
        setattr(handlers_mod, name, [])

    timers_ns = types.SimpleNamespace(
        is_registered=lambda _fn: False,
        register=lambda *_args, **_kwargs: None,
        unregister=lambda *_args, **_kwargs: None,
    )
    app_ns = types.SimpleNamespace(
        handlers=handlers_mod,
        timers=timers_ns,
        is_job_running=lambda _name: False,
    )
    app_mod = types.ModuleType("bpy.app")
    app_mod.handlers = handlers_mod
    app_mod.timers = timers_ns
    app_mod.is_job_running = app_ns.is_job_running

    bpy_mod.app = app_ns
    bpy_mod.context = types.SimpleNamespace(scene=None, window_manager=types.SimpleNamespace(windows=[]))
    bpy_mod.data = types.SimpleNamespace(
        objects={},
        node_groups={},
        collections={},
        scenes=[],
        materials={},
    )
    bpy_mod.types = types.SimpleNamespace(
        NodeTree=type("NodeTree", (), {}),
        Object=type("Object", (), {}),
        Scene=type("Scene", (), {}),
        Collection=type("Collection", (), {}),
        Node=type("Node", (), {}),
    )

    sys.modules["bpy"] = bpy_mod
    sys.modules["bpy.app"] = app_mod
    sys.modules["bpy.app.handlers"] = handlers_mod


def _install_properties_stub():
    properties_mod = types.ModuleType(f"{PKG_NAME}.properties")

    class ACEAudioSource:  # pragma: no cover - import placeholder
        pass

    properties_mod.ACEAudioSource = ACEAudioSource
    sys.modules[f"{PKG_NAME}.properties"] = properties_mod


class _FakeTree(dict):
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


class _FakeGroupNode(dict):
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


class _FakeAttrNode(dict):
    def __init__(self, ace_id: str, input_names=None, output_names=None):
        super().__init__()
        self.type = "GROUP"
        self.name = ace_id
        self.label = ace_id
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
        cls.tasks_mod = load_module("audio_tasks_registry")
        cls.dummy_tree = load_module("dummy_tree")
        cls.shader_tree_mod = load_module("shader_tree")
        cls.engine_mod = load_module("audio_engine")
        cls.utils_mod = load_module("utils")

    def _engine_shell(self):
        engine = object.__new__(self.engine_mod.ACEAudioEngine)
        engine.writer = types.SimpleNamespace(output_defs={})
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

        frontends_a = engine.discover_scene_frontends(scene)
        frontends_b = engine.discover_scene_frontends(scene)

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

        frontends_a = engine.discover_scene_frontends(scene_a)
        frontends_b = engine.discover_scene_frontends(scene_b)

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

        frontends_a = engine.discover_scene_frontends(scene)
        tree_uid_after_a = shared_tree.get("ace_frontend_uid")
        frontends_b = engine.discover_scene_frontends(scene)

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

        sig_a = self.engine_mod.ACEAudioEngine._compute_tree_signature(tree_a)
        sig_b = self.engine_mod.ACEAudioEngine._compute_tree_signature(tree_b)
        sig_c = self.engine_mod.ACEAudioEngine._compute_tree_signature(tree_c)

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
        discovery_a = self.engine_mod._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_a)
        discovery_b = self.engine_mod._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_b)

        tree_sig_a = self.engine_mod.ACEAudioEngine._compute_tree_signature(tree_a)
        tree_sig_b = self.engine_mod.ACEAudioEngine._compute_tree_signature(tree_b)
        membership_sig_a = self.engine_mod.ACEAudioEngine._frontend_membership_signature([discovery_a])
        membership_sig_b = self.engine_mod.ACEAudioEngine._frontend_membership_signature([discovery_b])

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

        winner, ordered = engine._resolve_global_manual_preview_winner(abstract_tree)

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
        discovery = self.engine_mod._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_ref)
        stable_sig = engine._frontend_membership_signature([discovery])
        engine._last_frontend_membership_signature = stable_sig
        engine._dirty_frontend_uids = set()
        engine._pending_scene_rebuild = True
        engine._last_dirty_time = 0.0
        engine._last_compile_time = 0.0
        engine._first_dirty_time = 0.0
        engine._is_compiling = False
        engine._compiled_frontend_refs = {}
        engine._pending_tree = None
        engine._dirty_trees = set()
        engine.discover_scene_frontends = lambda _scene: [discovery]
        engine.compile_scene_frontends = lambda *_args, **_kwargs: self.fail("compile_scene_frontends should be skipped")

        compiled = engine._maybe_compile_dirty(_FakeScene())

        self.assertFalse(compiled)
        self.assertFalse(engine._pending_scene_rebuild)

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
        discovery = self.engine_mod._ACEFrontendDiscovery(tree=None, frontend_ref=frontend_ref, consumer_object_names=("Cube",))
        stable_sig = engine._frontend_membership_signature([discovery])
        engine._last_frontend_membership_signature = stable_sig
        engine._dirty_frontend_uids = set()
        engine._pending_scene_rebuild = False
        engine._last_dirty_time = 0.0
        engine._last_compile_time = 0.0
        engine._first_dirty_time = 0.0
        engine._last_request_time = 0.0
        engine._is_compiling = False
        engine._compiled_frontend_refs = {}
        engine._pending_tree = None
        engine._dirty_trees = set()
        engine.discover_scene_frontends = lambda _scene: [discovery]
        engine.compile_scene_frontends = lambda *_args, **_kwargs: self.fail("compile_scene_frontends should be skipped")

        engine.request_recompile(None)
        engine._last_dirty_time = 0.0
        engine._first_dirty_time = 0.0
        compiled = engine._maybe_compile_dirty(_FakeScene())

        self.assertFalse(compiled)
        self.assertFalse(engine._pending_scene_rebuild)

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

        bindings = engine._resolve_sink_bindings_for_graph(
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

        bindings = engine._resolve_sink_bindings_for_graph(
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

        changed, changed_node_ids = engine._sync_shader_output_ids(shader_tree, abstract_tree, scene)

        expected_topology = "[\"src_b\"]|[]"
        expected_name = engine._build_auto_attr_name(
            "ACE_PROXY_b",
            "node_shader",
            "RMS",
            0,
            topology_sig=expected_topology,
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
            changed, changed_node_ids = engine._sync_shader_output_ids(shader_tree, abstract_tree, scene)
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

    def test_depsgraph_handler_tracks_runtime_geometry_shader_and_object_updates(self):
        engine = types.SimpleNamespace(
            requested=[],
            _consume_geometry_eval_updates=lambda *_args, **_kwargs: None,
            request_recompile=lambda tree: engine.requested.append(tree),
        )
        settings = types.SimpleNamespace(auto_compile=True)
        scene = types.SimpleNamespace(audiocore=types.SimpleNamespace(engine_settings=settings, audio_sources=[]))

        geometry_tree = _FakeTree("ACE_GN", "GeometryNodeTree", 606, ace_frontend_uid="ACE_FRONTEND_GN")
        shader_tree = _FakeTree("ACE_Shader", "ShaderNodeTree", 707, ace_frontend_uid="ACE_FRONTEND_SHADER")
        material = _FakeMaterial("ACE Mat", shader_tree)
        material.bl_idname = "Material"
        obj = _FakeObject(
            "Cube",
            modifiers=[_FakeModifier("ACE GN", geometry_tree)],
            material_slots=[_FakeMaterialSlot(material)],
        )
        obj.bl_idname = "Object"

        orig_ensure = self.engine_mod.ACEAudioEngine.ensure_instance
        orig_get = self.engine_mod.ACEAudioEngine.get_instance
        try:
            self.engine_mod.ACEAudioEngine.ensure_instance = classmethod(lambda cls, _scene=None: engine)
            self.engine_mod.ACEAudioEngine.get_instance = classmethod(lambda cls, _scene=None: engine)

            self.engine_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=geometry_tree)]),
            )
            self.engine_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=shader_tree)]),
            )
            self.engine_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=obj)]),
            )
            self.engine_mod.depsgraph_update_handler(
                scene,
                types.SimpleNamespace(updates=[types.SimpleNamespace(id=material)]),
            )
        finally:
            self.engine_mod.ACEAudioEngine.ensure_instance = orig_ensure
            self.engine_mod.ACEAudioEngine.get_instance = orig_get

        self.assertEqual(engine.requested[0], geometry_tree)
        self.assertEqual(engine.requested[1], shader_tree)
        self.assertEqual(engine.requested[2], geometry_tree)
        self.assertEqual(engine.requested[3], shader_tree)
        self.assertEqual(engine.requested[4], shader_tree)
        self.assertEqual(len(engine.requested), 5)

    def test_frame_change_handler_skips_compile_and_sink_work_when_render_writes_are_unsafe(self):
        calls = []
        engine = types.SimpleNamespace(
            graphs={},
            _is_rendering_job=lambda: True,
            _unsafe_for_handler_writes=lambda _scene: True,
            _log_runtime=lambda *args, **kwargs: calls.append(("log", args, kwargs)),
            execute_next_frame=lambda *_args, **_kwargs: calls.append(("execute",)),
            discover_scene_frontends=lambda *_args, **_kwargs: calls.append(("discover",)) or [],
            compile_scene_frontends=lambda *_args, **_kwargs: calls.append(("compile",)),
        )
        scene = types.SimpleNamespace()

        orig_ensure = self.engine_mod.ACEAudioEngine.ensure_instance
        try:
            self.engine_mod.ACEAudioEngine.ensure_instance = classmethod(lambda cls, _scene=None: engine)
            self.engine_mod.frame_change_handler(scene)
        finally:
            self.engine_mod.ACEAudioEngine.ensure_instance = orig_ensure

        self.assertTrue(any(call[0] == "log" for call in calls))
        self.assertFalse(any(call[0] == "execute" for call in calls))
        self.assertFalse(any(call[0] == "discover" for call in calls))
        self.assertFalse(any(call[0] == "compile" for call in calls))

    def test_parallel_usable_values_accepts_current_frame_payload(self):
        engine = self._engine_shell()
        engine._parallel_results = {"graph_a": {24: {"node:0": 0.0}}}
        engine._parallel_last_valid = {}
        engine._parallel_last_valid_frame = {}
        engine._current_scene_fps = 30.0

        self.assertTrue(engine._parallel_has_usable_values_for_graph_frame("graph_a", 24))

    def test_parallel_usable_values_accepts_recent_last_valid_cache(self):
        engine = self._engine_shell()
        engine._parallel_results = {"graph_a": {}}
        engine._parallel_last_valid = {"graph_a": {"node:0": 0.42}}
        engine._parallel_last_valid_frame = {"graph_a": 100}
        engine._current_scene_fps = 30.0

        self.assertTrue(engine._parallel_has_usable_values_for_graph_frame("graph_a", 102))

    def test_parallel_usable_values_rejects_stale_or_empty_cache(self):
        engine = self._engine_shell()
        engine._parallel_results = {"graph_a": {}}
        engine._parallel_last_valid = {"graph_a": {"node:0": 0.42}}
        engine._parallel_last_valid_frame = {"graph_a": 100}
        engine._current_scene_fps = 30.0

        self.assertFalse(engine._parallel_has_usable_values_for_graph_frame("graph_a", 104))
        self.assertFalse(engine._parallel_has_usable_values_for_graph_frame("graph_missing", 104))

    def test_compile_requires_main_thread(self):
        errors = []
        engine = object.__new__(self.engine_mod.ACEAudioEngine)
        abstract_tree = self.abs_mod.ACEAbstractTree()

        self.utils_mod.set_debug_strict(True)
        try:
            def _run():
                try:
                    engine.compile(abstract_tree, types.SimpleNamespace(audiocore=None))
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
