import sys
import types
import unittest

from module_loader import load_module


def _load_audio_influx_with_fake_numpy():
    # Keep this test hermetic: do not leak fake NumPy or fake-imported modules.
    sys.modules.pop("ace_addon_testpkg.audio_influx", None)
    had_numpy = "numpy" in sys.modules
    original_numpy = sys.modules.get("numpy")
    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = object
    sys.modules["numpy"] = fake_np
    try:
        module = load_module("audio_influx")
    finally:
        if had_numpy:
            sys.modules["numpy"] = original_numpy  # type: ignore[assignment]
        else:
            sys.modules.pop("numpy", None)
    return module


class TestAudioInfluxBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ai = _load_audio_influx_with_fake_numpy()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("ace_addon_testpkg.audio_influx", None)

    def test_udp_flow_parser(self):
        cfg = self.ai.AudioReception._parse_udp_flow_identifier("udp:9000:2:48000")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["port"], 9000)
        self.assertEqual(cfg["channels"], 2)
        self.assertEqual(cfg["sample_rate"], 48000)

    def test_udp_flow_parser_rejects_invalid_port(self):
        cfg = self.ai.AudioReception._parse_udp_flow_identifier("udp:notaport")
        self.assertIsNone(cfg)

    def test_backend_parser_supports_sounddevice_default(self):
        rec = self.ai.AudioReception("LIVE", flow_id="3")
        backend, cfg = rec._parse_live_backend()
        self.assertEqual(backend, "sounddevice")
        self.assertEqual(cfg["device"], "3")

    def test_backend_parser_routes_daw_prefix(self):
        rec = self.ai.AudioReception("LIVE", flow_id="daw:9001:2:48000")
        backend, cfg = rec._parse_live_backend()
        self.assertEqual(backend, "daw_udp")
        self.assertEqual(cfg["port"], 9001)
        self.assertEqual(cfg["channels"], 2)
        self.assertEqual(cfg["sample_rate"], 48000)

    def test_backend_parser_routes_instrument_prefix(self):
        rec = self.ai.AudioReception("LIVE", flow_id="instrument:sd:2@1")
        backend, cfg = rec._parse_live_backend()
        self.assertEqual(backend, "instrument_sounddevice")
        self.assertEqual(cfg["device"], "2")
        self.assertEqual(cfg["channels"], 1)

    def test_backend_parser_marks_unsupported_host_api(self):
        rec = self.ai.AudioReception("LIVE", flow_id="asio:my-device")
        backend, cfg = rec._parse_live_backend()
        self.assertEqual(backend, "unsupported")
        self.assertEqual(cfg.get("backend"), "asio")

    def test_udp_flow_parser_clamps_channels_and_sample_rate(self):
        cfg = self.ai.AudioReception._parse_udp_flow_identifier("udp:7000:99:999999")
        self.assertIsNotNone(cfg)
        self.assertEqual(int(cfg["channels"]), 8)
        self.assertEqual(int(cfg["sample_rate"]), 192000)

    def test_open_live_sounddevice_clamps_channels_to_device_capability(self):
        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)
                self.samplerate = int(kwargs.get("samplerate", 48000))
                self.channels = int(kwargs.get("channels", 0))

            def start(self):
                return None

            def stop(self):
                return None

            def close(self):
                return None

        class FakeSoundDevice:
            def __init__(self):
                self.calls = []

            def query_devices(self, device=None, kind=None):  # noqa: ARG002
                return {
                    "max_input_channels": 1,
                    "default_samplerate": 48000.0,
                }

            def InputStream(self, **kwargs):
                self.calls.append(dict(kwargs))
                return FakeStream(**kwargs)

        fake_sd = FakeSoundDevice()
        original_available = self.ai.SOUNDDEVICE_AVAILABLE
        original_sd = getattr(self.ai, "sd", None)
        self.ai.SOUNDDEVICE_AVAILABLE = True
        self.ai.sd = fake_sd
        try:
            rec = self.ai.AudioReception("LIVE", flow_id="sd:")
            opened = rec._open_live_sounddevice({"device": None, "channels": 2})
            self.assertTrue(opened)
            self.assertEqual(len(fake_sd.calls), 1)
            self.assertEqual(int(fake_sd.calls[0]["channels"]), 1)
            self.assertEqual(int(rec.channels), 1)
            rec.close()
        finally:
            self.ai.SOUNDDEVICE_AVAILABLE = original_available
            if original_sd is not None:
                self.ai.sd = original_sd
            else:
                delattr(self.ai, "sd")

    def test_open_live_sounddevice_reports_backend_error(self):
        class FailingSoundDevice:
            def query_devices(self, device=None, kind=None):  # noqa: ARG002
                return {
                    "max_input_channels": 2,
                    "default_samplerate": 44100.0,
                }

            def InputStream(self, **kwargs):  # noqa: ARG002
                raise RuntimeError("device busy")

        original_available = self.ai.SOUNDDEVICE_AVAILABLE
        original_sd = getattr(self.ai, "sd", None)
        self.ai.SOUNDDEVICE_AVAILABLE = True
        self.ai.sd = FailingSoundDevice()
        try:
            rec = self.ai.AudioReception("LIVE", flow_id="sd:")
            opened = rec._open_live_sounddevice({"device": None, "channels": 2})
            self.assertFalse(opened)
            self.assertIn("device busy", rec.last_error)
        finally:
            self.ai.SOUNDDEVICE_AVAILABLE = original_available
            if original_sd is not None:
                self.ai.sd = original_sd
            else:
                delattr(self.ai, "sd")


if __name__ == "__main__":
    unittest.main()
