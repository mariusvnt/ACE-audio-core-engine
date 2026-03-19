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

    def __setitem__(self, key, value):
        self.write_count += 1
        super().__setitem__(key, value)


class _FaultyObject(_TrackingObject):
    def __setitem__(self, key, value):
        if value not in (0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]):
            raise RuntimeError("simulated property write failure")
        super().__setitem__(key, value)


class _TrackingMeshData(dict):
    pass


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


if __name__ == "__main__":
    unittest.main()
