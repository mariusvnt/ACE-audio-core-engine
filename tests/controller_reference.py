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


def running_total_step(total: float, x: float, dt: float, leak: float, reset: bool) -> float:
    """dT/dt = x - T/leak (no leak at 0), solved over the step; Reset gives 0."""
    if reset:
        return 0.0
    dt = max(dt, 0.0)
    if leak <= 0.0:
        return total + x * dt
    kept = math.exp(-dt / max(leak, EPSILON_S))
    return total * kept + x * leak * (1.0 - kept)


NEVER_S = 1e9


@dataclass
class GateState:
    open: bool = False
    armed: bool = True
    hit: bool = False
    since: float = NEVER_S
    count: float = 0.0


def gate_step(state: GateState, x: float, dt: float, threshold: float, rearm: float, min_interval: float,
              reset: bool) -> GateState:
    since = state.since + max(dt, 0.0)
    falls = x < min(rearm, threshold)
    crossing = state.armed and x >= threshold
    hit = crossing and since >= min_interval
    armed = True if falls else (False if crossing else state.armed)
    is_open = True if hit else (False if falls else state.open)
    count = (0.0 if reset else state.count) + (1.0 if hit else 0.0)
    return GateState(open=is_open, armed=armed, hit=hit, since=0.0 if hit else since, count=count)


def _frames(times):
    last = None
    for t in times:
        yield (0.0 if last is None else max(t - last, 0.0)), t
        last = t


def run_running_total(inputs, times, leak=0.0, resets=None):
    out, total = [], 0.0
    resets = resets or [False] * len(inputs)
    for (dt, _t), x, reset in zip(_frames(times), inputs, resets):
        total = running_total_step(total, x, dt, leak, reset)
        out.append(total)
    return out


def run_gate(inputs, times, threshold=0.5, rearm=0.4, min_interval=0.1, resets=None):
    """(open, hit, count) on each frame."""
    out, state = [], GateState()
    resets = resets or [False] * len(inputs)
    for (dt, _t), x, reset in zip(_frames(times), inputs, resets):
        state = gate_step(state, x, dt, threshold, rearm, min_interval, reset)
        out.append((state.open, state.hit, int(state.count)))
    return out


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
