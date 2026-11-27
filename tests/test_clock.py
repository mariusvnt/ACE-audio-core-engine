"""The rule table of core/clock.py: how each source gets its value under each clock (FUTURE.md 1.1)."""

import types
import unittest

from module_loader import load_module

import fake_bpy


class ClockRuleTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_bpy.install()
        cls.clock = load_module("core.clock")

    def test_a_file_source_reads_its_bake_when_valid_and_streams_otherwise_under_every_clock(self):
        c = self.clock
        for clock in c.Clock:
            self.assertIs(c.method_for("FILE", clock, baked=True), c.Method.BAKE, clock)
            self.assertIs(c.method_for("FILE", clock, baked=False), c.Method.STREAM, clock)

    def test_a_live_input_streams_except_in_a_render_where_it_is_held(self):
        c = self.clock
        self.assertIs(c.method_for("LIVE", c.Clock.TIMELINE, baked=False), c.Method.STREAM)
        self.assertIs(c.method_for("LIVE", c.Clock.LIVE, baked=False), c.Method.STREAM)
        self.assertIs(c.method_for("LIVE", c.Clock.RENDER, baked=False), c.Method.HOLD)
        self.assertIs(c.method_for("live", c.Clock.RENDER, baked=True), c.Method.HOLD, "a live input has no bake")

    def test_the_current_clock(self):
        c = self.clock
        bpy = __import__("bpy")
        original = getattr(bpy.app, "is_job_running", None)
        try:
            bpy.app.is_job_running = lambda job: job == "RENDER"
            self.assertIs(c.current_clock(types.SimpleNamespace(live_session=object())), c.Clock.RENDER)
            bpy.app.is_job_running = lambda job: False
            self.assertIs(c.current_clock(types.SimpleNamespace(live_session=object())), c.Clock.LIVE)
            self.assertIs(c.current_clock(types.SimpleNamespace(live_session=None)), c.Clock.TIMELINE)
            self.assertIs(c.current_clock(None), c.Clock.TIMELINE)
            # A render in background mode runs in place, not as a job: the handlers mark it.
            self.assertIs(c.current_clock(types.SimpleNamespace(render_active=True, live_session=None)), c.Clock.RENDER)
        finally:
            if original is None:
                del bpy.app.is_job_running
            else:
                bpy.app.is_job_running = original


if __name__ == "__main__":
    unittest.main()
