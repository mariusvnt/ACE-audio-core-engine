"""Live sessions (core/live.py): what a session changes, that it all comes back, and its readout."""

import types
import unittest

from module_loader import load_module

import fake_bpy


def _scene():
    sim = types.SimpleNamespace(
        name="Sim",
        use_simulation_cache=True,
        modifiers=[types.SimpleNamespace(type="NODES", bakes=[
            types.SimpleNamespace(node=types.SimpleNamespace(bl_idname="GeometryNodeSimulationOutput"))])],
    )
    plain = types.SimpleNamespace(name="Plain", use_simulation_cache=True, modifiers=[])
    scene = types.SimpleNamespace(
        name="Scene", render=types.SimpleNamespace(fps=24, fps_base=1.0), sync_mode="AUDIO_SYNC",
        frame_start=1, frame_end=250, frame_current=40, use_custom_simulation_range=True,
        simulation_frame_end=200, objects=[sim, plain],
    )
    scene.frame_set = lambda frame: setattr(scene, "frame_current", int(frame))
    return scene, sim, plain


class LiveSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.live = load_module("core.live")

    def test_a_session_plays_every_frame_over_the_longest_range_and_restores_it_all(self):
        scene, sim, plain = _scene()
        engine = types.SimpleNamespace(live_session=None)
        session = self.live.start(engine, scene, from_start=True, disable_simulation_cache=True, play=False)
        self.assertIs(engine.live_session, session)
        self.assertEqual(scene.sync_mode, "NONE")
        self.assertEqual(scene.frame_end, self.live.MAX_FRAME)
        self.assertEqual(scene.simulation_frame_end, self.live.MAX_FRAME, "a custom simulation range must not end it")
        self.assertFalse(sim.use_simulation_cache)
        self.assertTrue(plain.use_simulation_cache, "an object that simulates nothing is left alone")
        self.assertEqual(scene.frame_current, 1)
        self.live.stop(engine, scene, stop_playback=False)
        self.assertIsNone(engine.live_session)
        self.assertEqual((scene.sync_mode, scene.frame_end, scene.simulation_frame_end), ("AUDIO_SYNC", 250, 200))
        self.assertTrue(sim.use_simulation_cache)

    def test_a_session_can_continue_from_the_current_frame(self):
        scene, _sim, _plain = _scene()
        engine = types.SimpleNamespace(live_session=None)
        session = self.live.start(engine, scene, from_start=False, play=False)
        self.assertEqual(session.start_frame, 40)
        self.live.stop(engine, scene, stop_playback=False)

    def test_the_readout_counts_late_frames_and_the_drift_behind_real_time(self):
        session = self.live.LiveSession(scene_name="Scene", fps=24.0)
        frame_s = 1.0 / 24.0
        session.on_frame(1, now=0.0)
        session.on_frame(2, now=frame_s)
        session.on_frame(3, now=frame_s + 0.2)  # a frame 0.2 s late
        self.assertEqual(session.frames, 3)
        self.assertEqual(session.late_frames, 1)
        self.assertAlmostEqual(session.drift_s(), (frame_s + 0.2) - 2 * frame_s, places=9)


if __name__ == "__main__":
    unittest.main()
