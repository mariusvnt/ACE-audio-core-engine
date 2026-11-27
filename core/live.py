"""Live sessions: Blender's own playback, set up so frames are never dropped or wrapped (FUTURE.md 1.5).

- Play Every Frame (``sync_mode = 'NONE'``): frames are always consecutive. When the scene can't
  keep up, playback runs late (time stretches) instead of dropping frames; the readout shows how
  far the show has slipped behind real time.
- The longest range: ``frame_end`` is raised to Blender's last frame, and so is the simulation
  range when the scene uses a custom one, so playback never wraps (a wrap would rewind every
  stateful node and simulation).
- Optionally, simulation caching is turned off on the objects that simulate (Blender keeps every
  simulated frame in memory otherwise), and the 3D views drop to solid shading.

Everything changed is saved on entry and restored on exit. While a session runs the clock is
Clock.LIVE (core/clock.py): live inputs read their latest block, file sources stream by frame.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import utils

MAX_FRAME = 1048574  # Blender's last frame (Scene.frame_set accepts [-1048574, 1048574])


def _simulates(obj) -> bool:
    """True when one of the object's Geometry Nodes modifiers holds a simulation zone."""
    for mod in getattr(obj, "modifiers", []) or []:
        if getattr(mod, "type", "") != "NODES":
            continue
        for bake in getattr(mod, "bakes", []) or []:
            node = getattr(bake, "node", None)
            if node is not None and getattr(node, "bl_idname", "") == "GeometryNodeSimulationOutput":
                return True
    return False


@dataclass
class LiveSession:
    """One live session: what it changed, and how playback is keeping up."""

    scene_name: str
    fps: float
    saved_scene: Dict[str, Any] = field(default_factory=dict)
    saved_sim_cache: Dict[str, bool] = field(default_factory=dict)  # object name -> use_simulation_cache
    saved_shading: List[Tuple[Any, str]] = field(default_factory=list)  # (3D view space, shading type)
    start_frame: int = 0
    started_at: float = 0.0
    last_frame: Optional[int] = None
    last_frame_at: float = 0.0
    frames: int = 0
    late_frames: int = 0
    measured_fps: float = 0.0

    def on_frame(self, frame: int, now: Optional[float] = None) -> None:
        """Count a frame shown: how fast playback runs, and the frames that came late."""
        now = time.perf_counter() if now is None else float(now)
        if self.last_frame is None:
            self.started_at = now
        else:
            interval = now - self.last_frame_at
            if interval > 1.5 / max(self.fps, 1.0):
                self.late_frames += 1
            if interval > 0.0:
                instant = 1.0 / interval
                self.measured_fps = instant if self.measured_fps <= 0.0 else 0.9 * self.measured_fps + 0.1 * instant
        self.last_frame = int(frame)
        self.last_frame_at = now
        self.frames += 1

    def drift_s(self) -> float:
        """How far the show was behind real time at its last frame, in seconds (0 when it keeps up)."""
        if self.last_frame is None:
            return 0.0
        shown = (self.frames - 1) / max(self.fps, 1.0)
        return max(0.0, (self.last_frame_at - self.started_at) - shown)


def start(engine, scene, from_start: bool = True, disable_simulation_cache: bool = False,
          solid_viewports: bool = False, play: bool = True) -> LiveSession:
    """Set the scene up for a live session and start playback (when there is a window to play in)."""
    utils.assert_main_thread("live.start")
    if getattr(engine, "live_session", None) is not None:
        return engine.live_session
    fps = float(scene.render.fps) / max(float(scene.render.fps_base or 1.0), 1e-6)
    session = LiveSession(scene_name=scene.name, fps=fps)
    for attr in ("sync_mode", "frame_end", "simulation_frame_end"):
        session.saved_scene[attr] = getattr(scene, attr)
    scene.sync_mode = "NONE"
    scene.frame_end = MAX_FRAME
    if scene.use_custom_simulation_range:
        scene.simulation_frame_end = MAX_FRAME
    if disable_simulation_cache:
        for obj in scene.objects:
            if _simulates(obj) and obj.use_simulation_cache:
                session.saved_sim_cache[obj.name] = True
                obj.use_simulation_cache = False
    if solid_viewports:
        for space in _view3d_spaces():
            session.saved_shading.append((space, space.shading.type))
            space.shading.type = "SOLID"
    if from_start:
        scene.frame_set(scene.frame_start)
    session.start_frame = int(scene.frame_current)
    engine.live_session = session
    if play:
        _play(True)
    utils.log(f"Live session started at frame {session.start_frame} ({fps:g} fps, Play Every Frame).", "INFO")
    return session


def stop(engine, scene=None, stop_playback: bool = True) -> None:
    """End the live session: stop playback and restore everything it changed."""
    session = getattr(engine, "live_session", None)
    if session is None:
        return
    engine.live_session = None
    if stop_playback:
        _play(False)
    import bpy  # type: ignore

    scene = scene or bpy.data.scenes.get(session.scene_name)
    if scene is not None:
        with utils.logged_errors("live.stop#scene"):
            scene.sync_mode = session.saved_scene["sync_mode"]
            scene.frame_end = session.saved_scene["frame_end"]
            scene.simulation_frame_end = session.saved_scene["simulation_frame_end"]
    objects = {obj.name: obj for obj in getattr(scene, "objects", [])} if scene is not None else {}
    for name, value in session.saved_sim_cache.items():
        obj = objects.get(name)
        if obj is not None:
            obj.use_simulation_cache = value
    for space, shading in session.saved_shading:
        with utils.logged_errors("live.stop#shading"):
            space.shading.type = shading
    utils.log(f"Live session ended after {session.frames} frames ({session.late_frames} late).", "INFO")


def _view3d_spaces():
    import bpy  # type: ignore

    wm = getattr(bpy.context, "window_manager", None)
    for window in getattr(wm, "windows", []) or []:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        yield space


def _play(on: bool) -> None:
    """Start or stop the timeline's playback from the first window that has a screen."""
    import bpy  # type: ignore

    wm = getattr(bpy.context, "window_manager", None)
    window = next(iter(getattr(wm, "windows", []) or []), None)
    if window is None:
        return  # background mode: nothing plays, frames are set by whoever drives them
    playing = bool(getattr(window.screen, "is_animation_playing", False))
    if playing == on:
        return
    with bpy.context.temp_override(window=window, screen=window.screen):
        if on:
            bpy.ops.screen.animation_play()
        else:
            bpy.ops.screen.animation_cancel(restore_frame=False)
