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


class _TrackingObject(fake_bpy.TruthyDict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.write_count = 0

    def __setitem__(self, key, value):
        self.write_count += 1
        super().__setitem__(key, value)

    def update_tag(self):
        self.tags = getattr(self, "tags", 0) + 1


class TestProxyManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if numpy is None:
            raise unittest.SkipTest("numpy unavailable in this environment")
        _purge_testpkg_modules()
        _install_fake_bpy()
        cls.abs_mod = load_module("abstract_tree")
        cls.tasks_mod = load_tasks()
        cls.proxy_mod = load_module("core.proxy.manager")
        cls.utils_mod = load_module("utils")
        cls.bpy = sys.modules["bpy"]

    def setUp(self):
        self.bpy.data.objects = {}

    def test_ensure_sink_layout_batches_proxy_and_scene_sinks(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        consumer_obj = _TrackingObject("Scene")
        self.bpy.data.objects = {"ACE_PROXY_src": proxy_obj}
        self.bpy.data.scenes = {"Scene": consumer_obj}

        manager = self.proxy_mod.ACEProxyManager(None)
        calls = []

        def _ensure_attributes_exist(proxy, required, prune_existing=False):
            calls.append((proxy.name, dict(required), bool(prune_existing)))

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
                        sink_kind="SCENE_PROPERTY",
                        target_object_name="Scene",
                        target_data_name="ace_prop_a",
                        attr_type="FLOAT",
                    ),
                ]
            }
        )

        touched = manager.ensure_sink_layout(graph, active_keys={"attr_a"})

        self.assertEqual(calls, [("ACE_PROXY_src", {"ace_attr_a": "FLOAT"}, True)])
        self.assertEqual(touched, {"ACE_PROXY_src"})
        self.assertIn("ace_prop_a", consumer_obj)
        self.assertEqual(consumer_obj["ace_prop_a"], 0.0)

    def test_update_scene_property_sinks_suppresses_redundant_writes(self):
        consumer_obj = _TrackingObject("Scene")
        self.bpy.data.scenes = {"Scene": consumer_obj}
        manager = self.proxy_mod.ACEProxyManager(None)

        manager.update_scene_property_sinks("Scene", {"ace_prop": 1.25}, {"ace_prop": "FLOAT"})
        first_writes = consumer_obj.write_count
        manager.update_scene_property_sinks("Scene", {"ace_prop": 1.25}, {"ace_prop": "FLOAT"})

        self.assertEqual(first_writes, 1)
        self.assertEqual(consumer_obj.write_count, 1)
        self.assertEqual(consumer_obj["ace_prop"], 1.25)

    def test_commit_results_fans_out_proxy_and_scene_property_writes(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        consumer_obj = _TrackingObject("Scene")
        self.bpy.data.objects = {"ACE_PROXY_src": proxy_obj}
        self.bpy.data.scenes = {"Scene": consumer_obj}

        manager = self.proxy_mod.ACEProxyManager(None)
        proxy_updates = []

        def _update_proxy_object(proxy, values, pulses=frozenset()):
            proxy_updates.append((proxy.name, dict(values or {}), tuple(sorted(pulses))))

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
                        sink_kind="SCENE_PROPERTY",
                        target_object_name="Scene",
                        target_data_name="ace_prop_a",
                        attr_type="FLOAT",
                    ),
                    # A second use of the same signal, under a name of its own.
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="PROXY_ATTRIBUTE",
                        target_object_name="ACE_PROXY_src",
                        target_data_name="ace_attr_b",
                        attr_type="FLOAT",
                        pulse=True,
                    ),
                ]
            }
        )

        touched = manager.commit_results(graph, {"attr_a": 2.5}, active_keys={"attr_a"})

        self.assertEqual(touched, {"ACE_PROXY_src"})
        self.assertEqual(
            proxy_updates,
            [("ACE_PROXY_src", {"ace_attr_a": 2.5, "ace_attr_b": 2.5}, ("ace_attr_b",))],
        )
        self.assertEqual(consumer_obj["ace_prop_a"], 2.5)

    def test_stage_c_sink_writers_require_main_thread(self):
        proxy_obj = _TrackingObject("ACE_PROXY_src")
        consumer_obj = _TrackingObject("Scene")
        self.bpy.data.objects = {"ACE_PROXY_src": proxy_obj}
        self.bpy.data.scenes = {"Scene": consumer_obj}

        manager = self.proxy_mod.ACEProxyManager(None)
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "attr_a": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="SCENE_PROPERTY",
                        target_object_name="Scene",
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
        consumer_obj = _TrackingObject("Scene")
        self.bpy.data.scenes = {"Scene": consumer_obj}

        manager = self.proxy_mod.ACEProxyManager(None)
        graph = types.SimpleNamespace(
            sink_bindings_by_attr={
                "attr_a": [
                    self.tasks_mod.ACESinkBinding(
                        sink_kind="SCENE_PROPERTY",
                        target_object_name="Scene",
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
