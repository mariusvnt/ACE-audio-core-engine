"""Values that reach an operator's input through a link.

Blender gives Python no access to the value travelling through a link, so it is computed here
from the nodes feeding it: audio operator outputs (their latest value), Value, Integer, Math,
Clamp, Map Range, Mix, Switch, Float Curve, Float to Integer, Scene Time, reroutes, node groups,
muted nodes and the modifier's own inputs. Anything else is unknown.
"""

import math
import bpy  # type: ignore
from .. import utils
from .identity import instance_node_id, is_ace_node, node_id, node_kind

_MAX_DEPTH = 64


class _Unknown(Exception):
    """The chain holds something that cannot be computed here."""


class WalkContext:
    """Where a node is used: the group nodes entered from the root, the root's modifier, the root's key.

    A link from a group's Group Input then reads what that use of the group receives.
    """

    def __init__(self, chain=(), modifier=None, root_key: str = ""):
        self.chain = tuple(chain or ())
        self.modifier = modifier
        self.root_key = str(root_key or "")

    def ref(self):
        """Plain data to find the context again on a later frame."""
        modifier = None
        if self.modifier is not None:
            modifier = (str(self.modifier.id_data.name), str(self.modifier.name))
        return ([node_ref(node) for node in self.chain], modifier, self.root_key)

    @classmethod
    def from_ref(cls, ref):
        if not ref:
            return None
        chain_refs, modifier_ref, root_key = ref
        chain = [node_from_ref(r) for r in chain_refs or ()]
        if any(node is None for node in chain):
            return None
        modifier = None
        if modifier_ref:
            obj = bpy.data.objects.get(modifier_ref[0])
            modifier = obj.modifiers.get(modifier_ref[1]) if obj is not None else None
        return cls(chain, modifier, root_key)


class _Walk:
    """Where the computation is: the scene, and the node groups entered on the way up."""

    def __init__(self, scene, context: WalkContext | None = None):
        self.scene = scene
        self.groups: list = list(context.chain) if context is not None else []
        self.modifier = context.modifier if context is not None else None
        self.root_key = context.root_key if context is not None else ""


def linked_value(socket, scene=None, context: WalkContext | None = None) -> float | None:
    """The value reaching input *socket* (its own value when unlinked), or None if it cannot be computed."""
    try:
        value = _input_value(socket, _Walk(scene, context), 0)
    except _Unknown:
        return None
    except Exception as exc:  # noqa: BLE001 boundary: a node layout we do not know is just unknown
        utils.note_swallowed("driven_values.linked_value", exc)
        return None
    return value if math.isfinite(value) else None


_EMBEDDED_OWNERS = (("material", "materials"), ("world", "worlds"), ("light", "lights"))


def node_ref(node) -> tuple[str, str, str] | None:
    """(owner kind, owner name, node name): finds *node* again without holding a reference to it."""
    tree = node.id_data
    if bpy.data.node_groups.get(tree.name) == tree:
        return ("group", tree.name, node.name)
    for kind, collection in _EMBEDDED_OWNERS:
        for owner in getattr(bpy.data, collection):
            if getattr(owner, "node_tree", None) == tree:
                return (kind, owner.name, node.name)
    return None


def node_from_ref(ref):
    if not ref:
        return None
    kind, owner_name, node_name = ref
    if kind == "group":
        tree = bpy.data.node_groups.get(owner_name)
    else:
        collection = dict(_EMBEDDED_OWNERS).get(kind, "materials")
        owner = getattr(bpy.data, collection).get(owner_name)
        tree = getattr(owner, "node_tree", None) if owner is not None else None
    return tree.nodes.get(node_name) if tree is not None else None


def _input_value(socket, walk: _Walk, depth: int) -> float:
    if depth > _MAX_DEPTH:
        raise _Unknown
    link = next((link for link in socket.links if not link.is_muted), None)
    if link is None:
        return float(socket.default_value)
    return _output_value(link.from_node, link.from_socket, walk, depth + 1)


def _socket(sockets, identifier: str):
    found = next((s for s in sockets if s.identifier == identifier), None)
    if found is None:
        raise _Unknown
    return found


def _output_value(node, output, walk: _Walk, depth: int) -> float:
    if is_ace_node(node):
        return _audio_value(node, output, walk)
    if node.mute:
        through = next((l for l in node.internal_links if l.to_socket == output), None)
        if through is None:
            raise _Unknown
        return _input_value(through.from_socket, walk, depth)
    kind = node.bl_idname

    def arg(which) -> float:
        socket = node.inputs[which] if isinstance(which, int) else _socket(node.inputs, which)
        return _input_value(socket, walk, depth)

    if kind == "NodeReroute":
        if not node.inputs[0].is_linked:
            raise _Unknown
        return arg(0)
    if kind == "ShaderNodeValue":
        return float(node.outputs[0].default_value)
    if kind == "FunctionNodeInputInt":
        return float(node.integer)
    if kind == "ShaderNodeMath":
        arity, function = _MATH[node.operation]
        result = function(*[arg(i) for i in range(arity)])
        return min(max(result, 0.0), 1.0) if node.use_clamp else result
    if kind == "ShaderNodeClamp":
        value, low, high = arg(0), arg(1), arg(2)
        if node.clamp_type == "RANGE" and low > high:
            low, high = high, low
        return min(max(value, low), high)
    if kind == "ShaderNodeMapRange":
        return _map_range(node, arg)
    if kind == "ShaderNodeMix":
        if node.data_type != "FLOAT":
            raise _Unknown
        factor = arg("Factor_Float")
        if node.clamp_factor:
            factor = min(max(factor, 0.0), 1.0)
        a = arg("A_Float")
        return a + factor * (arg("B_Float") - a)
    if kind == "GeometryNodeSwitch":
        if node.input_type not in {"FLOAT", "INT"}:
            raise _Unknown
        return arg("True" if arg("Switch") else "False")
    if kind == "ShaderNodeFloatCurve":
        factor, value = arg("Factor"), arg("Value")
        mapping = node.mapping
        mapping.initialize()
        curved = mapping.evaluate(mapping.curves[0], value)
        if mapping.use_clip:
            curved = min(max(curved, mapping.clip_min_y), mapping.clip_max_y)
        return value + factor * (curved - value)
    if kind == "FunctionNodeFloatToInt":
        value = arg(0)
        rounding = {
            "ROUND": lambda v: math.floor(abs(v) + 0.5) * (1.0 if v >= 0.0 else -1.0),
            "FLOOR": math.floor, "CEILING": math.ceil, "TRUNCATE": math.trunc,
        }
        return float(rounding[node.rounding_mode](value))
    if kind == "GeometryNodeInputSceneTime":
        scene = walk.scene or bpy.context.scene
        frame = scene.frame_current + scene.frame_subframe
        if output.identifier == "Frame":
            return float(frame)
        return float(frame / (scene.render.fps / scene.render.fps_base))
    if kind in {"GeometryNodeGroup", "ShaderNodeGroup"}:
        inner = node.node_tree
        outputs = [n for n in inner.nodes if n.bl_idname == "NodeGroupOutput"] if inner else []
        if not outputs:
            raise _Unknown
        group_output = next((n for n in outputs if n.is_active_output), outputs[0])
        walk.groups.append(node)
        try:
            return _input_value(_socket(group_output.inputs, output.identifier), walk, depth)
        finally:
            walk.groups.pop()
    if kind == "NodeGroupInput":
        if not walk.groups:
            return _modifier_input(node.id_data, output.identifier, walk.scene, walk.modifier)
        outer = walk.groups.pop()  # the value comes from outside the group entered above
        try:
            return _input_value(_socket(outer.inputs, output.identifier), walk, depth)
        finally:
            walk.groups.append(outer)
    raise _Unknown


def _audio_value(node, output, walk: _Walk) -> float:
    """The latest value an audio operator produced on *output*, in the use of the group the walk is in."""
    from ..core.engine.engine import ACEAudioEngine

    engine = ACEAudioEngine.get_instance(walk.scene or bpy.context.scene)
    index = next((i for i, s in enumerate(node.outputs) if s.identifier == output.identifier), -1)
    if walk.root_key:
        category, node_type = node_kind(node)
        used = instance_node_id(category, node_type, walk.root_key, [g.name for g in walk.groups], node.name)
    else:
        used = node_id(node)
    # Values are kept per signal: the task computing this use may compute others too.
    signal = engine.signal_id(used) if engine is not None else ""
    value = getattr(engine, "latest_values", {}).get(f"{signal or used}:{index}")
    if value is None:
        raise _Unknown  # not computed yet
    return float(value)


def _modifier_input(tree, identifier: str, scene, modifier=None) -> float:
    """The value a modifier gives the tree's input: *modifier*'s when known, else the active object's wins."""
    if modifier is not None and modifier.node_group == tree:
        entry = getattr(modifier.properties.inputs, identifier, None)
        if entry is None or getattr(entry, "type", "VALUE") != "VALUE":
            raise _Unknown
        return float(entry.value)
    objects = list(scene.objects if scene is not None else bpy.data.objects)
    active = getattr(getattr(bpy.context, "view_layer", None), "objects", None)
    active = getattr(active, "active", None)
    if active is not None:
        objects.insert(0, active)
    for obj in objects:
        for modifier in obj.modifiers:
            if modifier.type != "NODES" or modifier.node_group != tree:
                continue
            entry = getattr(modifier.properties.inputs, identifier, None)
            if entry is None or getattr(entry, "type", "VALUE") != "VALUE":
                raise _Unknown
            return float(entry.value)
    raise _Unknown  # a nested group or a shader tree: its inputs have no single value


def _divide(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _power(a: float, b: float) -> float:
    if a < 0.0 and b != math.floor(b):
        return 0.0
    if a == 0.0 and b < 0.0:
        return 0.0
    return math.pow(a, b)


def _log(a: float, base: float) -> float:
    if a <= 0.0 or base <= 0.0:
        return 0.0
    return _divide(math.log(a), math.log(base))


def _wrap(value: float, high: float, low: float) -> float:
    span = high - low
    return value - span * math.floor((value - low) / span) if span != 0.0 else low


def _pingpong(a: float, scale: float) -> float:
    if scale == 0.0:
        return 0.0
    turn = (a - scale) / (scale * 2.0)
    return abs((turn - math.floor(turn)) * scale * 2.0 - scale)


def _smooth_min(a: float, b: float, distance: float) -> float:
    if distance == 0.0:
        return min(a, b)
    h = max(distance - abs(a - b), 0.0) / distance
    return min(a, b) - h * h * h * distance / 6.0


# Blender's Math node: operation -> (inputs used, function).
_MATH = {
    "ADD": (2, lambda a, b: a + b),
    "SUBTRACT": (2, lambda a, b: a - b),
    "MULTIPLY": (2, lambda a, b: a * b),
    "DIVIDE": (2, _divide),
    "MULTIPLY_ADD": (3, lambda a, b, c: a * b + c),
    "POWER": (2, _power),
    "LOGARITHM": (2, _log),
    "SQRT": (1, lambda a: math.sqrt(a) if a > 0.0 else 0.0),
    "INVERSE_SQRT": (1, lambda a: 1.0 / math.sqrt(a) if a > 0.0 else 0.0),
    "ABSOLUTE": (1, abs),
    "EXPONENT": (1, math.exp),
    "MINIMUM": (2, min),
    "MAXIMUM": (2, max),
    "LESS_THAN": (2, lambda a, b: float(a < b)),
    "GREATER_THAN": (2, lambda a, b: float(a > b)),
    "SIGN": (1, lambda a: float((a > 0.0) - (a < 0.0))),
    "COMPARE": (3, lambda a, b, c: float(abs(a - b) <= max(c, 1.1920929e-07))),
    "SMOOTH_MIN": (3, _smooth_min),
    "SMOOTH_MAX": (3, lambda a, b, c: -_smooth_min(-a, -b, c)),
    "ROUND": (1, lambda a: float(math.floor(a + 0.5))),
    "FLOOR": (1, lambda a: float(math.floor(a))),
    "CEIL": (1, lambda a: float(math.ceil(a))),
    "TRUNC": (1, lambda a: float(math.trunc(a))),
    "FRACT": (1, lambda a: a - math.floor(a)),
    "MODULO": (2, lambda a, b: math.fmod(a, b) if b != 0.0 else 0.0),
    "FLOORED_MODULO": (2, lambda a, b: a - math.floor(a / b) * b if b != 0.0 else 0.0),
    "WRAP": (3, _wrap),
    "SNAP": (2, lambda a, b: math.floor(_divide(a, b)) * b),
    "PINGPONG": (2, _pingpong),
    "SINE": (1, math.sin),
    "COSINE": (1, math.cos),
    "TANGENT": (1, math.tan),
    "ARCSINE": (1, lambda a: math.asin(min(max(a, -1.0), 1.0))),
    "ARCCOSINE": (1, lambda a: math.acos(min(max(a, -1.0), 1.0))),
    "ARCTANGENT": (1, math.atan),
    "ARCTAN2": (2, math.atan2),
    "SINH": (1, math.sinh),
    "COSH": (1, math.cosh),
    "TANH": (1, math.tanh),
    "RADIANS": (1, math.radians),
    "DEGREES": (1, math.degrees),
}


def _map_range(node, arg) -> float:
    if node.data_type != "FLOAT":
        raise _Unknown
    value, from_min, from_max, to_min, to_max = (arg(i) for i in range(5))
    if from_max == from_min:
        return 0.0
    factor = (value - from_min) / (from_max - from_min)
    mode = node.interpolation_type
    if mode == "STEPPED":
        steps = arg(5)
        factor = _divide(math.floor(factor * (steps + 1.0)), steps)
    elif mode in {"SMOOTHSTEP", "SMOOTHERSTEP"}:
        t = min(max(factor, 0.0), 1.0)
        factor = t * t * (3.0 - 2.0 * t) if mode == "SMOOTHSTEP" else t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    result = to_min + factor * (to_max - to_min)
    if node.clamp and mode in {"LINEAR", "STEPPED"}:
        result = min(max(result, min(to_min, to_max)), max(to_min, to_max))
    return result
