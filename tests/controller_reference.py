"""The controllers' maths in Python: what their Geometry Nodes groups compute (nodes/geometry/controllers.py).

Each controller is a step function over (state, input, step duration in seconds). The node
groups are checked against these in the Blender smoke test; the unit tests check the maths
itself, such as fps not changing the result for an input defined in seconds.
"""

import math
from dataclasses import dataclass

EPSILON_S = 1e-6
MIN_SPEED_HZ = 0.01
MAX_DAMPING = 0.9999


def _one_pole(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-dt / max(tau, EPSILON_S))


@dataclass
class EnvelopeState:
    follow: float
    hold: float
    spring: float
    velocity: float


def envelope_start(x: float) -> EnvelopeState:
    """At rest on the first input: no ramp from 0."""
    return EnvelopeState(follow=x, hold=0.0, spring=x, velocity=0.0)


def envelope_step(state: EnvelopeState, x: float, dt: float, attack: float, hold: float, release: float,
                  speed: float, bounce: float) -> EnvelopeState:
    """Both modes advance on every step; the Mode menu only picks which one is output."""
    dt = max(dt, 0.0)
    y = state.follow
    if x >= y:
        follow, held = y + (x - y) * _one_pole(dt, attack), hold
    else:
        follow = y + (x - y) * _one_pole(max(dt - state.hold, 0.0), release)
        held = max(state.hold - dt, 0.0)

    w = 2.0 * math.pi * max(speed, MIN_SPEED_HZ)
    zeta = min(1.0 - 0.95 * min(max(bounce, 0.0), 1.0), MAX_DAMPING)
    a = zeta * w
    wd = w * math.sqrt(1.0 - zeta * zeta)
    e, v = state.spring - x, state.velocity
    decay = math.exp(-a * dt)
    c, s = math.cos(wd * dt), math.sin(wd * dt)
    e_next = decay * (e * c + (v + a * e) / wd * s)
    v_next = decay * (v * c - (w * w * e + a * v) / wd * s)
    return EnvelopeState(follow=follow, hold=held, spring=x + e_next, velocity=v_next)


def envelope_output(state: EnvelopeState, mode: str) -> float:
    return state.spring if mode == "Spring" else state.follow


def run_envelope(inputs, times, mode="Follow", attack=0.01, hold=0.0, release=0.3, speed=2.0, bounce=0.3):
    """The Envelope output on each frame, for inputs sampled at *times* (seconds)."""
    out = []
    state = None
    last_t = None
    for x, t in zip(inputs, times):
        if state is None:
            state = envelope_start(x)
            dt = 0.0
        else:
            dt = max(t - last_t, 0.0)
        state = envelope_step(state, x, dt, attack, hold, release, speed, bounce)
        last_t = t
        out.append(envelope_output(state, mode))
    return out
