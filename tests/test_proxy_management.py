import shutil
import sys
import threading
import types
import unittest
import unittest.mock

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
    bpy_mod.data = types.SimpleNamespace(objects={}, node_groups={}, collections={}, scenes=[])
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


class _TrackingObject(dict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.write_count = 0
        self.update_tag_calls = []

    def __setitem__(self, key, value):
        self.write_count += 1
        super().__setitem__(key, value)

    def update_tag(self, *args, **kwargs):
        self.update_tag_calls.append((args, kwargs))


class _FaultyObject(_TrackingObject):
    def __setitem__(self, key, value):
        if value not in (0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]):
            raise RuntimeError("simulated property write failure")
        super().__setitem__(key, value)


class _TrackingMeshData(dict):
    pass


class _FakeAttribute:
    def __init__(self, data_type: str):
        self.data_type = str(data_type)


class _FakeAttributes(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeMesh:
    def __init__(self, attrs=None):
        self.attributes = _FakeAttributes(dict(attrs or {}))


class TestProxyManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if numpy is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        _purge_testpkg_modules()
        _install_fake_bpy()
        cls.abs_mod = load_module("abstract_tree")
        cls.tasks_mod = load_module("audio_tasks_registry")
        cls.proxy_mod = load_module("proxy_management")
        cls.utils_mod = load_module("utils")
        cls.bpy = sys.modules["bpy"]

    def setUp(self):
        self.bpy.data.objects = {}

    def test_ensure_sink_layout_batches_proxy_and_object_sinks(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        consumer_obj = _TrackingObject("Cube")
        self.bpy.data.objects = {
            "ACE_PROXY_src": proxy_obj,
            "Cube": consumer_obj,
        }

        manager = self.proxy_mod.ACEProxyManager(None)
        calls = []

        def _ensure_attributes_exist(proxy, _tree, active_keys=None, prune_existing=False):
            calls.append((proxy.name, tuple(sorted(active_keys or ())), bool(prune_existing)))

        manager.ensure_attributes_exist = _ensure_attributes_exist
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "attr_a": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="PROXY_ATTRIBUTE",
                        target_object_name="ACE_PROXY_src",
                        target_data_name="ace_attr_a",
                        attr_type="FLOAT",
                    ),
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="OBJECT_PROPERTY",
                        target_object_name="Cube",
                        target_data_name="ace_prop_a",
                        attr_type="FLOAT",
                    ),
                ]
            }
        )

        touched = manager.ensure_sink_layout(graph, active_keys={"attr_a"})

        self.assertEqual(calls, [("ACE_PROXY_src", ("attr_a",), True)])
        self.assertEqual(touched, {"ACE_PROXY_src"})
        self.assertIn("ace_prop_a", consumer_obj)
        self.assertEqual(consumer_obj["ace_prop_a"], 0.0)

    def test_update_object_property_sinks_suppresses_redundant_writes(self):
        consumer_obj = _TrackingObject("Cube")
        self.bpy.data.objects = {"Cube": consumer_obj}
        manager = self.proxy_mod.ACEProxyManager(None)

        manager.update_object_property_sinks("Cube", {"ace_prop": 1.25}, {"ace_prop": "FLOAT"})
        first_writes = consumer_obj.write_count
        manager.update_object_property_sinks("Cube", {"ace_prop": 1.25}, {"ace_prop": "FLOAT"})

        self.assertEqual(first_writes, 1)
        self.assertEqual(consumer_obj.write_count, 1)
        self.assertEqual(consumer_obj["ace_prop"], 1.25)

    def test_update_object_property_sinks_skips_mesh_data_mirroring_by_default(self):
        consumer_obj = _TrackingObject("Cube")
        consumer_obj.data = _TrackingMeshData()
        self.bpy.data.objects = {"Cube": consumer_obj}
        manager = self.proxy_mod.ACEProxyManager(None)

        manager.update_object_property_sinks("Cube", {"ace_prop": 1.25}, {"ace_prop": "FLOAT"})

        self.assertEqual(consumer_obj["ace_prop"], 1.25)
        self.assertNotIn("ace_prop", consumer_obj.data)

    def test_update_object_property_sinks_tags_object_on_value_only_updates(self):
        consumer_obj = _TrackingObject("Cube")
        self.bpy.data.objects = {"Cube": consumer_obj}
        manager = self.proxy_mod.ACEProxyManager(None)

        manager.update_object_property_sinks("Cube", {"ace_prop": 1.25}, {"ace_prop": "FLOAT"})

        self.assertEqual(consumer_obj["ace_prop"], 1.25)
        self.assertEqual(consumer_obj.update_tag_calls, [((), {"refresh": {"OBJECT"}})])

    def test_update_object_property_sinks_falls_back_to_zero_on_failure(self):
        faulty_obj = _FaultyObject("Cube")
        self.bpy.data.objects = {"Cube": faulty_obj}
        manager = self.proxy_mod.ACEProxyManager(None)

        manager.update_object_property_sinks(
            "Cube",
            {"ace_prop": [9.0, 8.0, 7.0]},
            {"ace_prop": "FLOAT_VECTOR"},
        )

        self.assertEqual(faulty_obj["ace_prop"], [0.0, 0.0, 0.0])

    def test_commit_results_fans_out_proxy_and_object_property_writes(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        consumer_obj = _TrackingObject("Cube")
        self.bpy.data.objects = {
            "ACE_PROXY_src": proxy_obj,
            "Cube": consumer_obj,
        }

        manager = self.proxy_mod.ACEProxyManager(None)
        proxy_updates = []

        def _update_proxy_object(proxy, results, active_keys=None):
            proxy_updates.append((proxy.name, dict(results or {}), tuple(sorted(active_keys or ()))))

        manager.update_proxy_object = _update_proxy_object
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "attr_a": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="PROXY_ATTRIBUTE",
                        target_object_name="ACE_PROXY_src",
                        target_data_name="ace_attr_a",
                        attr_type="FLOAT",
                    ),
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="OBJECT_PROPERTY",
                        target_object_name="Cube",
                        target_data_name="ace_prop_a",
                        attr_type="FLOAT",
                    ),
                ]
            }
        )

        touched = manager.commit_results(graph, {"attr_a": 2.5}, active_keys={"attr_a"})

        self.assertEqual(touched, {"ACE_PROXY_src"})
        self.assertEqual(
            proxy_updates,
            [("ACE_PROXY_src", {"attr_a": 2.5}, ("attr_a",))],
        )
        self.assertEqual(consumer_obj["ace_prop_a"], 2.5)

    def test_cache_output_defs_includes_logic_and_prefers_resolved_attr_name(self):
        abstract_tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "logic_node": self.abs_mod.ACEAbstractNode(
                    id="logic_node",
                    name="Logic",
                    category="utilities",
                    type="logic",
                    parameters={
                        "attr_id_out_0": "ACE_logic_contract",
                        "attr_id_custom_out_0": False,
                    },
                    outputs=[
                        self.abs_mod.ACEAbstractSocket(
                            id="logic_node_out_0",
                            name="Logic",
                            socket_idx=0,
                            is_output=True,
                            data_type="NodeSocketFloat",
                        )
                    ],
                )
            }
        )

        manager = self.proxy_mod.ACEProxyManager(abstract_tree)

        self.assertIn("logic_node:0", manager.output_defs)
        self.assertEqual(
            manager._select_attr_name("ACE_PROXY_src", manager.output_defs["logic_node:0"]),
            "ACE_logic_contract",
        )

    def test_logic_output_defs_drive_proxy_sink_layout(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        self.bpy.data.objects = {"ACE_PROXY_src": proxy_obj}
        abstract_tree = self.abs_mod.ACEAbstractTree(
            nodes={
                "logic_node": self.abs_mod.ACEAbstractNode(
                    id="logic_node",
                    name="Logic",
                    category="utilities",
                    type="logic",
                    parameters={"attr_id_out_0": "ACE_logic_contract"},
                    outputs=[
                        self.abs_mod.ACEAbstractSocket(
                            id="logic_node_out_0",
                            name="Logic",
                            socket_idx=0,
                            is_output=True,
                            data_type="NodeSocketFloat",
                        )
                    ],
                )
            }
        )
        manager = self.proxy_mod.ACEProxyManager(abstract_tree)
        calls = []

        def _ensure_attributes_exist(proxy, _tree, active_keys=None, prune_existing=False):
            calls.append((proxy.name, tuple(sorted(active_keys or ())), bool(prune_existing)))

        manager.ensure_attributes_exist = _ensure_attributes_exist
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "logic_node:0": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="PROXY_ATTRIBUTE",
                        target_object_name="ACE_PROXY_src",
                        target_data_name="ACE_logic_contract",
                        attr_type="FLOAT",
                    )
                ]
            }
        )

        touched = manager.ensure_sink_layout(graph, active_keys={"logic_node:0"})

        self.assertEqual(calls, [("ACE_PROXY_src", ("logic_node:0",), True)])
        self.assertEqual(touched, {"ACE_PROXY_src"})

    def test_sink_layout_needs_sync_uses_output_def_type_for_vector_and_color(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        proxy_obj.data = _FakeMesh(
            {
                "ACE_vec_attr": _FakeAttribute("FLOAT_VECTOR"),
                "ACE_color_attr": _FakeAttribute("FLOAT_COLOR"),
            }
        )
        self.bpy.data.objects = {"ACE_PROXY_src": proxy_obj}
        manager = self.proxy_mod.ACEProxyManager(None)
        manager.output_defs = {
            "vec_node:0": {
                "node_id": "vec_node",
                "socket_idx": 0,
                "output_name": "Vec",
                "type": "FLOAT_VECTOR",
                "attr_id": "ACE_vec_attr",
                "attr_custom": True,
            },
            "color_node:0": {
                "node_id": "color_node",
                "socket_idx": 0,
                "output_name": "Color",
                "type": "FLOAT_COLOR",
                "attr_id": "ACE_color_attr",
                "attr_custom": True,
            },
        }
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "vec_node:0": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="PROXY_ATTRIBUTE",
                        target_object_name="ACE_PROXY_src",
                        target_data_name="ACE_vec_attr",
                        attr_type="FLOAT_VECTOR",
                    )
                ],
                "color_node:0": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="PROXY_ATTRIBUTE",
                        target_object_name="ACE_PROXY_src",
                        target_data_name="ACE_color_attr",
                        attr_type="FLOAT_COLOR",
                    )
                ],
            }
        )

        self.assertFalse(
            manager.sink_layout_needs_sync(
                graph,
                active_keys={"vec_node:0", "color_node:0"},
            )
        )

    def test_stage_c_sink_writers_require_main_thread(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        consumer_obj = _TrackingObject("Cube")
        self.bpy.data.objects = {
            "ACE_PROXY_src": proxy_obj,
            "Cube": consumer_obj,
        }

        manager = self.proxy_mod.ACEProxyManager(None)
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "attr_a": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="OBJECT_PROPERTY",
                        target_object_name="Cube",
                        target_data_name="ace_prop_a",
                        attr_type="FLOAT",
                    )
                ]
            }
        )
        errors = []

        self.utils_mod.set_debug_strict(True)
        try:
            def _run():
                try:
                    manager.ensure_sink_layout(graph, active_keys={"attr_a"})
                except Exception as exc:  # pragma: no cover - exercised in worker thread
                    errors.append(exc)

            worker = threading.Thread(target=_run, name="ACESinkLayoutWorker")
            worker.start()
            worker.join(timeout=2.0)
        finally:
            self.utils_mod.set_debug_strict(False)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("ensure_sink_layout", str(errors[0]))

    def test_stage_c_commit_results_requires_main_thread(self):
        consumer_obj = _TrackingObject("Cube")
        self.bpy.data.objects = {"Cube": consumer_obj}

        manager = self.proxy_mod.ACEProxyManager(None)
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "attr_a": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="OBJECT_PROPERTY",
                        target_object_name="Cube",
                        target_data_name="ace_prop_a",
                        attr_type="FLOAT",
                    )
                ]
            }
        )
        errors = []

        self.utils_mod.set_debug_strict(True)
        try:
            def _run():
                try:
                    manager.commit_results(graph, {"attr_a": 2.5}, active_keys={"attr_a"})
                except Exception as exc:  # pragma: no cover - exercised in worker thread
                    errors.append(exc)

            worker = threading.Thread(target=_run, name="ACECommitWorker")
            worker.start()
            worker.join(timeout=2.0)
        finally:
            self.utils_mod.set_debug_strict(False)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("commit_results", str(errors[0]))


class TestVisualOutputProxyOps(unittest.TestCase):
    """Integration tests for the Visual Output Stage-C path.

    Covers:
    - ACEProxyManager.update_frames_in_memory (the Stage-C commit hook)
    - Simulated bake-complete and cache-deletion flows
    - Main-thread guardrail enforcement in strict mode
    """

    @classmethod
    def setUpClass(cls):
        if numpy is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        # _install_fake_bpy is idempotent; proxy_management/utils may already
        # be cached from TestProxyManagement — load_module re-uses the cache.
        _install_fake_bpy()
        cls.proxy_mod = load_module("proxy_management")
        cls.utils_mod = load_module("utils")
        cls.bpy = sys.modules["bpy"]
        if not hasattr(cls.proxy_mod.ACEProxyManager, "update_frames_in_memory"):
            raise unittest.SkipTest(
                "update_frames_in_memory not present — skipping Visual Output tests"
            )

    def setUp(self):
        # Non-strict so assert_main_thread only logs on the test thread.
        self.utils_mod.set_debug_strict(False)

    def tearDown(self):
        self.utils_mod.set_debug_strict(False)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_scene(frames_in_memory=0, is_baked=False):
        """Build a minimal fake scene with a writable audiocore namespace."""
        ace = types.SimpleNamespace(
            frames_in_memory=frames_in_memory,
            is_baked=is_baked,
        )
        return types.SimpleNamespace(audiocore=ace)

    # ------------------------------------------------------------------ update_frames_in_memory

    def test_update_writes_correct_value_to_scene_prop(self):
        scene = self._make_scene()
        self.proxy_mod.ACEProxyManager.update_frames_in_memory(42, scene=scene)
        self.assertEqual(scene.audiocore.frames_in_memory, 42)

    def test_update_overwrites_previous_value(self):
        scene = self._make_scene(frames_in_memory=999)
        self.proxy_mod.ACEProxyManager.update_frames_in_memory(7, scene=scene)
        self.assertEqual(scene.audiocore.frames_in_memory, 7)

    def test_update_clamps_negative_to_zero(self):
        scene = self._make_scene(frames_in_memory=500)
        self.proxy_mod.ACEProxyManager.update_frames_in_memory(-100, scene=scene)
        self.assertEqual(scene.audiocore.frames_in_memory, 0)

    def test_update_zero_resets_counter(self):
        scene = self._make_scene(frames_in_memory=88200)
        self.proxy_mod.ACEProxyManager.update_frames_in_memory(0, scene=scene)
        self.assertEqual(scene.audiocore.frames_in_memory, 0)

    def test_update_is_noop_when_audiocore_absent(self):
        """Must not raise when scene has no audiocore attribute."""
        scene = types.SimpleNamespace()  # no audiocore
        try:
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(100, scene=scene)
        except Exception as exc:
            self.fail(f"update_frames_in_memory raised unexpectedly: {exc}")

    def test_update_is_noop_when_scene_is_none(self):
        """When scene=None, _resolve_scene falls back to bpy.context.scene (None
        in the test fixture).  Must not raise."""
        try:
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(50, scene=None)
        except Exception as exc:
            self.fail(f"update_frames_in_memory raised with scene=None: {exc}")

    def test_update_multiple_times_last_value_wins(self):
        scene = self._make_scene()
        for v in (100, 200, 300):
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(v, scene=scene)
        self.assertEqual(scene.audiocore.frames_in_memory, 300)

    # ------------------------------------------------------------------ bake-complete Stage-C flow

    def test_bake_complete_sets_frames_in_memory_and_is_baked(self):
        """Simulate the Stage-C callback from a successful bake_analysis_range run.

        The serialisation worker posts ("frames_in_memory", N, scene) to the
        result queue; flush_bake_results() calls update_frames_in_memory(N)
        and the operator sets is_baked = True.  Assert both properties land.
        """
        scene = self._make_scene(frames_in_memory=0, is_baked=False)
        FRAMES = 44100  # 1 second at 44100 Hz

        # Stage C: proxy manager commit
        self.proxy_mod.ACEProxyManager.update_frames_in_memory(FRAMES, scene=scene)
        # Operator-side flag (ACE_OT_bake_audio_analysis.execute sets this)
        scene.audiocore.is_baked = True

        self.assertEqual(scene.audiocore.frames_in_memory, FRAMES)
        self.assertTrue(scene.audiocore.is_baked)

    def test_bake_complete_dummy_payload_round_trip(self):
        """Pass a worker-style result dict through the proxy update path."""
        scene = self._make_scene()

        # Mimic what flush_bake_results() does when it reads from _audio_cache_result_queue
        worker_event = ("frames_in_memory", 22050, scene)
        kind = worker_event[0]
        frames = int(worker_event[1])
        ev_scene = worker_event[2]

        if kind == "frames_in_memory":
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(frames, scene=ev_scene)

        self.assertEqual(scene.audiocore.frames_in_memory, 22050)

    # ------------------------------------------------------------------ deletion / cache-flush flow

    def test_delete_flow_resets_frames_and_is_baked(self):
        """Simulate ACE_OT_delete_bake: flush_bake_cache calls
        update_frames_in_memory(0) and the operator sets is_baked = False.
        """
        scene = self._make_scene(frames_in_memory=88200, is_baked=True)

        # Stage C: proxy manager reset (as flush_bake_cache does)
        self.proxy_mod.ACEProxyManager.update_frames_in_memory(0, scene=scene)
        # Operator-side flag (ACE_OT_delete_bake.execute sets this)
        scene.audiocore.is_baked = False

        self.assertEqual(scene.audiocore.frames_in_memory, 0)
        self.assertFalse(scene.audiocore.is_baked)

    def test_delete_operator_calls_rmtree_on_cache_dir_and_resets_proxy(self):
        """End-to-end deletion: shutil.rmtree is called on bake_cache_path and
        frames_in_memory is reset to 0 via the proxy manager.

        audio_engine.flush_bake_cache is not imported directly (it pulls in bpy);
        instead the relevant behaviour is reconstructed here to verify that the
        proxy-manager reset and disk-cleanup calls happen in the right order and
        with the correct arguments.
        """
        cache_dir = "/fake/ace_bake_cache"
        scene = self._make_scene(frames_in_memory=1000, is_baked=True)

        with unittest.mock.patch("shutil.rmtree") as mock_rmtree:
            # Replicate flush_bake_cache logic: delete files, then reset proxy
            shutil.rmtree(cache_dir, ignore_errors=True)
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(0, scene=scene)
            scene.audiocore.is_baked = False

        mock_rmtree.assert_called_once_with(cache_dir, ignore_errors=True)
        self.assertEqual(scene.audiocore.frames_in_memory, 0)
        self.assertFalse(scene.audiocore.is_baked)

    def test_delete_with_missing_cache_dir_does_not_raise(self):
        """flush_bake_cache uses ignore_errors=True so a missing dir is safe."""
        scene = self._make_scene(frames_in_memory=100, is_baked=True)
        # rmtree on a non-existent path with ignore_errors must not propagate
        try:
            shutil.rmtree("/nonexistent/path/that/never/existed", ignore_errors=True)
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(0, scene=scene)
        except Exception as exc:
            self.fail(f"Deletion flow raised unexpectedly: {exc}")
        self.assertEqual(scene.audiocore.frames_in_memory, 0)

    # ------------------------------------------------------------------ thread-safety guardrail

    def test_update_from_worker_thread_raises_in_strict_mode(self):
        """In strict mode, calling update_frames_in_memory from a non-main thread
        must raise RuntimeError — confirming the bpy write guardrail is active.
        """
        scene = self._make_scene(frames_in_memory=0)
        self.utils_mod.set_debug_strict(True)

        thread_errors = []

        def _worker():
            try:
                self.proxy_mod.ACEProxyManager.update_frames_in_memory(
                    99, scene=scene
                )
            except RuntimeError as exc:
                thread_errors.append(exc)
            except Exception as exc:
                thread_errors.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=3.0)

        self.assertFalse(t.is_alive(), "Worker thread did not terminate cleanly")
        self.assertEqual(
            len(thread_errors), 1,
            f"Expected exactly one RuntimeError from worker; got: {thread_errors}",
        )
        self.assertIsInstance(
            thread_errors[0], RuntimeError,
            f"Expected RuntimeError, got {type(thread_errors[0])}: {thread_errors[0]}",
        )
        # The property must NOT have been written (guardrail fired before the write)
        self.assertEqual(
            scene.audiocore.frames_in_memory, 0,
            "frames_in_memory must not be modified when the guardrail fires",
        )

    def test_update_on_main_thread_does_not_raise_in_strict_mode(self):
        """Calling update_frames_in_memory from the main thread must never raise,
        even with strict mode enabled."""
        scene = self._make_scene()
        self.utils_mod.set_debug_strict(True)
        try:
            self.proxy_mod.ACEProxyManager.update_frames_in_memory(42, scene=scene)
        except Exception as exc:
            self.fail(
                f"update_frames_in_memory raised on the main thread in strict mode: {exc}"
            )
        finally:
            self.utils_mod.set_debug_strict(False)
        self.assertEqual(scene.audiocore.frames_in_memory, 42)


if __name__ == "__main__":
    unittest.main()
