import unittest

from module_loader import load_module


idm = load_module("id_management")


class TestIdManagement(unittest.TestCase):
    class _FakeSource:
        def __init__(self, stable_id: str = "", name: str = "Source"):
            self.stable_id = stable_id
            self.name = name
            self.source_type = "FILE"
            self.filepath = "/tmp/source.wav"
            self.flow_identifier = ""

    def test_generate_and_parse_node_id(self):
        node_id = idm.ACEIdManager.generate_node_id("stream", "rms")
        parsed = idm.ACEIdManager.parse_node_id(node_id)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["category"], "stream")
        self.assertEqual(parsed["type"], "rms")
        self.assertEqual(len(parsed["uuid"]), 6)
        int(parsed["uuid"], 16)

    def test_generate_node_id_is_deterministic_for_same_occupancy(self):
        existing = ["ACE_stream_rms_123abc", "ACE_stream_rms_def456"]
        a = idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=existing)
        b = idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=list(existing))
        self.assertEqual(a, b)

    def test_generate_node_id_advances_deterministically_on_collision(self):
        first = idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=[])
        second = idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=[first])
        second_again = idm.ACEIdManager.generate_node_id("stream", "rms", existing_ids=[first])
        self.assertNotEqual(first, second)
        self.assertEqual(second, second_again)

    def test_proxy_name_is_deterministic_with_seed(self):
        name_a = idm.ACEIdManager.generate_source_proxy_name("Kick.wav", stable_seed="seed-1")
        name_b = idm.ACEIdManager.generate_source_proxy_name("Kick.wav", stable_seed="seed-1")
        name_c = idm.ACEIdManager.generate_source_proxy_name("Kick.wav", stable_seed="seed-2")
        self.assertEqual(name_a, name_b)
        self.assertNotEqual(name_a, name_c)

    def test_compact_attr_name_is_stable(self):
        attr_a = idm.ACEIdManager.generate_compact_attr_name("ACE_Source_ab12cd", "ACE_stream_rms_123456", "RMS", 0)
        attr_b = idm.ACEIdManager.generate_compact_attr_name("ACE_Source_ab12cd", "ACE_stream_rms_123456", "RMS", 0)
        self.assertEqual(attr_a, attr_b)
        self.assertTrue(attr_a.startswith(idm.ACEIdManager.PREFIX))
        self.assertLessEqual(len(attr_a), 63)

    def test_parse_node_id_rejects_invalid_strings(self):
        self.assertIsNone(idm.ACEIdManager.parse_node_id("not_ace"))
        self.assertIsNone(idm.ACEIdManager.parse_node_id("ACE_invalid"))

    def test_ensure_source_uid_preserves_valid_source_id(self):
        valid = idm.ACEIdManager.generate_node_id("source_mngmt", "src")
        src = self._FakeSource(stable_id=valid)

        resolved = idm.ACEIdManager.ensure_source_uid(src, existing_ids=[valid])

        self.assertEqual(resolved, valid)
        self.assertEqual(src.stable_id, valid)

    def test_ensure_source_uid_migrates_legacy_id_to_valid_source_id(self):
        src = self._FakeSource(stable_id="legacy_source_uuid")

        resolved = idm.ACEIdManager.ensure_source_uid(src, existing_ids=["legacy_source_uuid"])
        parsed = idm.ACEIdManager.parse_node_id(resolved)

        self.assertNotEqual(resolved, "legacy_source_uuid")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["category"], "source_mngmt")
        self.assertEqual(parsed["type"], "src")
        self.assertEqual(src.stable_id, resolved)


if __name__ == "__main__":
    unittest.main()
