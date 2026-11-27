"""Which clock decides the frame, and how each source gets its value under it (FUTURE.md 1.1).

This module is the only place that decides how a value is obtained:

| Clock \\ Source | File                      | Live input                  |
| --------------- | ------------------------- | --------------------------- |
| Timeline        | bake if valid, else stream | stream (the latest block)   |
| Render          | bake if valid, else stream | held at 0, and reported     |
| Live            | bake if valid, else stream | stream (the latest block)   |

A file source streams by frame under every clock, so its stream, its bake and its render read
the same block for a frame. A live input has nothing to compute a render from until it is
recorded as a take, which is a file source.
"""

import enum


class Clock(enum.Enum):
    TIMELINE = "timeline"  # playback and scrubbing
    RENDER = "render"  # Blender's render job
    LIVE = "live"  # a live session: Blender's playback in Play Every Frame mode


class Method(enum.Enum):
    BAKE = "bake"  # the stored curve
    STREAM = "stream"  # compute the frame
    HOLD = "hold"  # nothing to compute from: the outputs are held at 0


def method_for(source_type: str, clock: Clock, baked: bool) -> Method:
    """How a source of *source_type* ("FILE" or "LIVE") gets its values under *clock*."""
    if str(source_type or "").upper() == "LIVE":
        return Method.HOLD if clock is Clock.RENDER else Method.STREAM
    return Method.BAKE if baked else Method.STREAM


def is_rendering() -> bool:
    """True while Blender's render job runs (a viewport in rendered shading is not one)."""
    import bpy  # type: ignore

    checker = getattr(bpy.app, "is_job_running", None)
    return bool(checker("RENDER")) if callable(checker) else False


def current_clock(engine=None) -> Clock:
    """The clock now: a render, else a live session of *engine*, else the timeline.

    A render is known from its handlers (render_init to render_complete or render_cancel):
    a render from a script in background mode runs in place, not as a job.
    """
    if getattr(engine, "render_active", False) or is_rendering():
        return Clock.RENDER
    if getattr(engine, "live_session", None) is not None:
        return Clock.LIVE
    return Clock.TIMELINE
