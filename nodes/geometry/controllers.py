"""Controllers: native Geometry Nodes groups that shape a value over time.

A controller's state lives in a Simulation Zone inside its group, so each use of the group (each
modifier, each group path) has its own, and Blender's simulation cache and Bake record it. Any
node can feed it: what reaches its inputs is what Blender computes.

Each step lasts the Scene Time that passed since the last step, which the zone keeps as its
"Time" state. The zone's own Delta Time is one frame long even when playback skipped frames or
the timeline jumped ahead (probed on 5.2), so it would undercount. Each step is the exact
solution of its equation over the step, for an input held during it: results don't depend on
fps. ``tests/controller_reference.py`` is the same maths in Python.
"""

import math

TIME_STATE = "Time"
EPSILON_S = 1e-6  # a time constant of 0 is this: the value jumps to its input
MIN_SPEED_HZ = 0.01
MAX_DAMPING = 0.9999  # a Bounce of 0 is critically damped, as near as the formula allows


class _Builder:
    """Nodes and links of one group, from expressions: a float is a constant, a socket a link."""

    def __init__(self, ng):
        self.ng = ng
        self.column = 0

    def node(self, bl_idname: str, **props):
        node = self.ng.nodes.new(bl_idname)
        for key, value in props.items():
            setattr(node, key, value)
        return node  # placed by nodes/layout.py once the group is built

    @staticmethod
    def named(sock, label: str):
        """Label the node computing *sock*: what the step means, read in the editor."""
        sock.node.label = label
        return sock

    def feed(self, sock, value) -> None:
        if hasattr(value, "is_output"):
            self.ng.links.new(value, sock)
        else:
            sock.default_value = value

    def math(self, operation: str, *args):
        node = self.node("ShaderNodeMath", operation=operation)
        for sock, value in zip(node.inputs, args):
            self.feed(sock, value)
        return node.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b):
        return self.math("MULTIPLY", a, b)

    def div(self, a, b):
        return self.math("DIVIDE", a, b)

    def max(self, a, b):
        return self.math("MAXIMUM", a, b)

    def min(self, a, b):
        return self.math("MINIMUM", a, b)

    def exp(self, a):
        return self.math("EXPONENT", a)

    def one_pole(self, dt, tau):
        """1 - e^(-dt/tau): the share of the way to the input a one-pole filter goes in *dt*."""
        return self.sub(1.0, self.exp(self.div(self.mul(dt, -1.0), self.max(tau, EPSILON_S))))

    def compare(self, operation: str, a, b):
        node = self.node("FunctionNodeCompare", data_type="FLOAT", operation=operation)
        self.feed(node.inputs[0], a)
        self.feed(node.inputs[1], b)
        return node.outputs[0]

    def greater_equal(self, a, b):
        return self.compare("GREATER_EQUAL", a, b)

    def logic(self, operation: str, a, b=None):
        node = self.node("FunctionNodeBooleanMath", operation=operation)
        self.feed(node.inputs[0], a)
        if b is not None:
            self.feed(node.inputs[1], b)
        return node.outputs[0]

    def switch(self, condition, if_false, if_true, input_type: str = "FLOAT"):
        node = self.node("GeometryNodeSwitch", input_type=input_type)
        self.feed(node.inputs["Switch"], condition)
        self.feed(node.inputs["False"], if_false)
        self.feed(node.inputs["True"], if_true)
        return node.outputs[0]


def _zone(b: _Builder, states: dict, initial: dict):
    """A Simulation Zone holding *states* ({name: type}), set to *initial* ({name: value}) on its first frame.

    Returns (zone input node, zone output node, seconds now, step duration).
    """
    sim_in = b.node("GeometryNodeSimulationInput")
    sim_out = b.node("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    for item in list(sim_out.state_items):
        sim_out.state_items.remove(item)  # no geometry: a controller holds only values
    for name, socket_type in ({TIME_STATE: "FLOAT", **states}).items():
        sim_out.state_items.new(socket_type, name)
    seconds = b.node("GeometryNodeInputSceneTime").outputs["Seconds"]
    b.feed(sim_in.inputs[TIME_STATE], seconds)
    for name, value in initial.items():
        b.feed(sim_in.inputs[name], value)
    dt = b.named(b.max(b.sub(seconds, sim_in.outputs[TIME_STATE]), 0.0), "Step (s)")
    b.feed(sim_out.inputs[TIME_STATE], seconds)
    return sim_in, sim_out, seconds, dt


def _build_envelope(b: _Builder, gi, go, menus) -> None:
    x = gi.outputs["Value"]
    sim_in, sim_out, _seconds, dt = _zone(
        b,
        {"Follow": "FLOAT", "Hold": "FLOAT", "Spring": "FLOAT", "Velocity": "FLOAT"},
        {"Follow": x, "Hold": 0.0, "Spring": x, "Velocity": 0.0},
    )

    # Follow: a new peak moves toward the input with Attack and restarts Hold; then, once Hold
    # has run out, toward the input with Release (only for the part of the step past Hold).
    y, hold = sim_in.outputs["Follow"], sim_in.outputs["Hold"]
    rising = b.named(b.greater_equal(x, y), "Follow: Rising")
    toward = b.named(b.sub(x, y), "Follow: To Go")
    y_rise = b.named(b.add(y, b.mul(toward, b.one_pole(dt, gi.outputs["Attack"]))), "Follow: Rise (Attack)")
    after_hold = b.named(b.max(b.sub(dt, hold), 0.0), "Follow: Step Past Hold")
    y_fall = b.named(b.add(y, b.mul(toward, b.one_pole(after_hold, gi.outputs["Release"]))), "Follow: Fall (Release)")
    b.feed(sim_out.inputs["Follow"], b.named(b.switch(rising, y_fall, y_rise), "Follow: Next"))
    b.feed(sim_out.inputs["Hold"], b.named(
        b.switch(rising, b.max(b.sub(hold, dt), 0.0), gi.outputs["Hold"]), "Follow: Hold Left"))

    # Spring: y'' + 2 zeta w y' + w^2 y = w^2 x, solved exactly over the step (x held), for
    # the offset e = y - x and the velocity v.
    w = b.named(b.mul(b.max(gi.outputs["Speed"], MIN_SPEED_HZ), 2.0 * math.pi), "Spring: ω (2π Speed)")
    bounce = b.min(b.max(gi.outputs["Bounce"], 0.0), 1.0)
    zeta = b.named(b.min(b.sub(1.0, b.mul(bounce, 0.95)), MAX_DAMPING), "Spring: Damping ζ")
    a = b.named(b.mul(zeta, w), "Spring: Decay Rate")
    wd = b.named(b.mul(w, b.math("SQRT", b.sub(1.0, b.mul(zeta, zeta)))), "Spring: Damped ω")
    e = b.named(b.sub(sim_in.outputs["Spring"], x), "Spring: Offset")
    v = sim_in.outputs["Velocity"]
    decay = b.named(b.exp(b.mul(b.mul(a, dt), -1.0)), "Spring: Decay")
    phase = b.named(b.mul(wd, dt), "Spring: Phase")
    c, s = b.math("COSINE", phase), b.math("SINE", phase)
    e_next = b.mul(decay, b.add(b.mul(e, c), b.mul(b.div(b.add(v, b.mul(a, e)), wd), s)))
    v_next = b.mul(decay, b.sub(b.mul(v, c), b.mul(b.div(b.add(b.mul(b.mul(w, w), e), b.mul(a, v)), wd), s)))
    b.feed(sim_out.inputs["Spring"], b.named(b.add(x, b.named(e_next, "Spring: Next Offset")), "Spring: Next"))
    b.feed(sim_out.inputs["Velocity"], b.named(v_next, "Spring: Next Velocity"))

    mode = b.node("GeometryNodeMenuSwitch", data_type="FLOAT", label="Mode")
    items = list(menus.get("Mode", ["Follow", "Spring"]))
    for item, name in zip(mode.enum_items, items):
        item.name = name
    b.feed(mode.inputs["Menu"], gi.outputs["Mode"])
    b.feed(mode.inputs[1], sim_out.outputs["Follow"])  # the items' sockets, in item order
    b.feed(mode.inputs[2], sim_out.outputs["Spring"])
    b.feed(go.inputs["Envelope"], mode.outputs[0])


def _build_running_total(b: _Builder, gi, go, _menus) -> None:
    x, leak = gi.outputs["Value"], gi.outputs["Leak Time"]
    sim_in, sim_out, _seconds, dt = _zone(b, {"Total": "FLOAT"}, {"Total": 0.0})
    total = sim_in.outputs["Total"]
    plain = b.named(b.add(total, b.mul(x, dt)), "Total + Value x Step")
    # Leak Time L: dT/dt = x - T/L, solved over the step: T a + x L (1 - a), a = e^(-dt/L).
    kept = b.named(b.exp(b.div(b.mul(dt, -1.0), b.max(leak, EPSILON_S))), "Share Kept (e^-Step/Leak)")
    leaky = b.named(b.add(b.mul(total, kept), b.mul(b.mul(x, leak), b.sub(1.0, kept))), "Leaky Total")
    stepped = b.named(b.switch(b.named(b.compare("GREATER_THAN", leak, 0.0), "Leaks?"), plain, leaky), "Next")
    b.feed(sim_out.inputs["Total"], b.named(b.switch(gi.outputs["Reset"], stepped, 0.0), "Reset?"))
    b.feed(go.inputs["Total"], sim_out.outputs["Total"])


NEVER_S = 1e9  # the time since the last Hit before any Hit: the first one is never too soon


def _build_gate(b: _Builder, gi, go, _menus) -> None:
    x = gi.outputs["Value"]
    sim_in, sim_out, _seconds, dt = _zone(
        b,
        {"Open": "BOOLEAN", "Armed": "BOOLEAN", "Hit": "BOOLEAN", "Since": "FLOAT", "Count": "FLOAT"},
        {"Open": False, "Armed": True, "Hit": False, "Since": NEVER_S, "Count": 0.0},
    )
    armed, was_open = sim_in.outputs["Armed"], sim_in.outputs["Open"]
    since = b.named(b.add(sim_in.outputs["Since"], dt), "Since Last Hit")
    rearm = b.named(b.min(gi.outputs["Re-arm Level"], gi.outputs["Threshold"]), "Re-arm Below")
    falls = b.named(b.compare("LESS_THAN", x, rearm), "Fell Below Re-arm")
    crossing = b.named(b.logic("AND", armed, b.compare("GREATER_EQUAL", x, gi.outputs["Threshold"])), "Armed Crossing")
    # A crossing within Min Interval is ignored: the gate disarms without opening, and waits for
    # the input to fall below Re-arm Level, so it never gives a late Hit off the beat.
    hit = b.named(b.logic("AND", crossing, b.compare("GREATER_EQUAL", since, gi.outputs["Min Interval"])), "Hit")
    b.feed(sim_out.inputs["Armed"], b.switch(falls, b.switch(crossing, armed, False, "BOOLEAN"), True, "BOOLEAN"))
    b.feed(sim_out.inputs["Open"], b.switch(hit, b.switch(falls, was_open, False, "BOOLEAN"), True, "BOOLEAN"))
    b.feed(sim_out.inputs["Hit"], hit)
    b.feed(sim_out.inputs["Since"], b.switch(hit, since, 0.0))
    count = b.switch(gi.outputs["Reset"], sim_in.outputs["Count"], 0.0)  # Reset comes before the frame's Hit
    b.feed(sim_out.inputs["Count"], b.add(count, b.switch(hit, 0.0, 1.0)))
    b.feed(go.inputs["Open"], sim_out.outputs["Open"])
    b.feed(go.inputs["Hit"], sim_out.outputs["Hit"])
    b.feed(go.inputs["Count"], sim_out.outputs["Count"])  # a whole number, as a float state


_BUILDERS = {
    "envelope": _build_envelope,
    "running_total": _build_running_total,
    "gate": _build_gate,
}


def has_builder(node_type: str) -> bool:
    return str(node_type or "") in _BUILDERS


def build_internals(ng, node_def: dict, node_type: str) -> None:
    """Fill a controller group: its Simulation Zone and maths, between its Group Input and Output."""
    b = _Builder(ng)
    gi = b.node("NodeGroupInput")
    go = b.node("NodeGroupOutput")
    menus = dict(node_def.get("menus", {}) or {})
    _BUILDERS[str(node_type)](b, gi, go, menus)
    # A menu socket gets its items from the Menu Switch it feeds: its default is set after the link.
    for name, items in menus.items():
        item = next((i for i in ng.interface.items_tree if getattr(i, "name", "") == name), None)
        if item is not None and items:
            item.default_value = items[0]


def controller_bakes(scene):
    """(object, modifier, bake) for each Simulation Zone of a controller that the scene's objects run.

    A modifier lists one bake per zone it runs, nested groups included; the zone's Simulation
    Output node is in the controller's group.
    """
    from ..identity import group_kind
    from ...vars import node_is_native

    found = []
    for obj in getattr(scene, "objects", []) or []:
        for modifier in getattr(obj, "modifiers", []) or []:
            if getattr(modifier, "type", "") != "NODES":
                continue
            for bake in getattr(modifier, "bakes", []) or []:
                node = getattr(bake, "node", None)
                kind = group_kind(getattr(node, "id_data", None)) if node is not None else ("", "")
                if kind[0] and node_is_native(*kind):
                    found.append((obj, modifier, bake))
    return found


def bake_controllers(scene) -> int:
    """Bake every controller's Simulation Zone in *scene*, over its frame range; how many were baked.

    A render reads the viewport's simulation, so a controller renders exactly only on frames
    that are cached or baked. Each zone is baked on its own: the user's other simulations are left
    as they are.
    """
    import bpy  # type: ignore

    from ... import utils

    baked = 0
    for obj, modifier, bake in controller_bakes(scene):
        with utils.logged_errors("controllers.bake_controllers"):
            with bpy.context.temp_override(scene=scene, object=obj, active_object=obj):
                bpy.ops.object.geometry_node_bake_single(
                    session_uid=obj.session_uid, modifier_name=modifier.name, bake_id=bake.bake_id,
                )
            baked += 1
    return baked


def prepare_interface(ng, node_def: dict) -> None:
    """Controllers take single values: a field linked to one is drawn as an invalid link by Blender."""
    for item in ng.interface.items_tree:
        if getattr(item, "in_out", None) != "INPUT" or getattr(item, "socket_type", "") == "NodeSocketMenu":
            continue
        if hasattr(item, "structure_type"):
            item.structure_type = "SINGLE"
