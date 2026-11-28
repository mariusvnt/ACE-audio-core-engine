"""vars.py include all the hard-coded information of the add-on, especially the registry of all available audio nodes."""
from typing import TypedDict

# Global Parameters
ACE_MAIN_COLLECTION_NAME = "ACE_Navigation"
ACE_MAIN_COLLECTION_COLOR = 'COLOR_05'  # blue
ACE_OBJECT_COLOR = (0.047, 0.2, 0.6, 1.0)
ACE_COLLECTION_NAME = "ACE_Sources_&_Streams"
ACE_PROXY_PREFIX = "ACE_"
ACE_BLUE = (79 / 255.0, 146 / 255.0, 254 / 255.0, 1.0)
ACE_LOGO_FILENAME = "ACE_Logo.png"

# Hierarchy of the available audio nodes
class Meta(TypedDict):
    """format of a category: a submenu of the add menu"""
    label: str
    icon: str
    description: str

class NodeType(TypedDict):
    """format of a node type"""
    label: str
    icon: str
    description: str
    role: str
    inputs: list[tuple[str, str]]
    outputs: list[tuple[str, str]]
    # Optional keys: "modes", "default_mode", "mode_inputs" ({input name: the mode that uses
    # it}), "native" (True: the node computes in Blender and has no task), and "stateful" (True,
    # or the list of modes that are): a stateful node's value depends on the frames before, so
    # it runs on the main thread and settles after a seek.

class Category(TypedDict):
    """category container"""
    meta: Meta
    nodes: dict[str, NodeType]


# What a node is, whatever its category: the code asks for roles, never for categories.
ROLE_SOURCE = "source"  # audio enters the tree
ROLE_SPLITTER = "splitter"  # audio to audio: the bands of its input
ROLE_MEASURE = "measure"  # audio to values, written where the source's proxy carries them
ROLE_CONTROLLER = "controller"  # a value to a value
ROLE_PREVIEW = "preview"  # audio leaves the tree, to be heard

# The categories are the submenus of the add menu, in this order.
ACE_NODES: dict[str, Category] = {
    "input": {
        "meta": {
            "label": "Input",
            "icon": "",
            "description": "Where audio enters the tree."
        },
        "nodes": {
            "src": {
            "label": "Source",
            "icon": "",
            "description": "Reference to an audio source, with its tempo.",
            "role": ROLE_SOURCE,
            "inputs": [],
            "outputs": [('NodeSocketBundle', 'Audio'), ('NodeSocketFloat', 'BPM'), ('NodeSocketInt', 'Beats per Bar')],
            },
        },
    },

    "output": {
        "meta": {
            "label": "Output",
            "icon": "",
            "description": "Where audio leaves the tree, to be heard."
        },
        "nodes": {
            "preview_sink": {
            "label": "Audio Preview",
            "icon": "PLAY_SOUND",
            "description": "Branch audio monitor sink used for in-editor preview playback.",
            "role": ROLE_PREVIEW,
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [],
            },
        },
    },

    "audio": {
        "meta": {
            "label": "Audio",
            "icon": "GRAPH",
            "description": "Nodes with an Audio input: its bands, and measurements of it."
        },
        "nodes": {
            "frequency_splitter": {
            "label": "Frequency Splitter",
            "icon": "",
            "description": "Splits the incoming audio into multiple audios, each corresponding to a band of frequencies.",
            "role": ROLE_SPLITTER,
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'Audio'), ('NodeSocketBundle', 'Audio')], # template sockets, runtime variants are dynamic
            },
            "loudness": {
            "label": "Loudness",
            "icon": "DRIVER_DISTANCE",
            "description": (
                "Loudness in 0..1: 1 at 0 dB, 0 at -70 dB (silence) and below, linear in dB "
                "in between. Method: RMS (signal energy), LUFS (perceived loudness, "
                "averaged over Window) or Peak (highest sample)."
            ),
            "role": ROLE_MEASURE,
            "inputs": [
                ('NodeSocketBundle', 'Audio'),
                ('NodeSocketFloat', 'Window (s)'),
            ],
            "outputs": [('NodeSocketFloat', 'Loudness')],
            "modes": [
                ("rms", "RMS", "Root-mean-square energy of the frame, on the mono signal."),
                ("lufs", "LUFS", "K-weighted perceived loudness averaged over Window (s)."),
                ("peak", "Peak", "Highest K-weighted sample of the frame."),
            ],
            "default_mode": "rms",
            "mode_inputs": {"Window (s)": "lufs"},
            "stateful": ["lufs", "peak"],
            },
            "stereo": {
            "label": "Stereo",
            "icon": "ORIENTATION_GIMBAL",
            "description": (
                "Where the sound is, whatever its volume. Balance: -1 left, 0 centre, 1 right. "
                "Width: 0 mono, 1 side only. Behind a Frequency Splitter, within the band."
            ),
            "role": ROLE_MEASURE,
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Balance'), ('NodeSocketFloat', 'Width')],
            },
            "brightness": {
            "label": "Brightness",
            "icon": "SEQ_HISTOGRAM",
            "description": (
                "Where the energy of the frame sits, from dark to bright (its spectral centroid), in 0..1: "
                "0 at 20 Hz and below, 1 at 20 kHz, on a log scale. Behind a Frequency Splitter, within the band."
            ),
            "role": ROLE_MEASURE,
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Brightness')],
            },
            "pitch": {
            "label": "Pitch",
            "icon": "",
            "description": (
                "The pitch of the frame's root note, in 0..1: 0 at 20 Hz or with no pitch, 1 at 5 kHz, "
                "on a log scale, so equal musical intervals are equal steps."
            ),
            "role": ROLE_MEASURE,
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Pitch')],
            },
            "transient": {
            "label": "Transient",
            "icon": "",
            "description": "Sharp attacks in the audio (0 = smooth, 1 = sharp transient).",
            "role": ROLE_MEASURE,
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Transient')],
            "stateful": True,
            },
        },
    },

    "controllers": {
        "meta": {
            "label": "Controllers",
            "icon": "DRIVER",
            "description": "Nodes that take a value and turn it into motion or events."
        },
        "nodes": {
            "util_rolling_avg": {
            "label": "Rolling Average",
            "icon": "",
            "description": "Smooths a scalar attribute stream over a time window.",
            "role": ROLE_CONTROLLER,
            "inputs": [('NodeSocketFloat', 'Value'), ('NodeSocketFloat', 'Window')],
            "outputs": [('NodeSocketFloat', 'Rolling Avg')],
            "stateful": True,
            },
            "stream_accumulator": {
            "label": "Stream Accumulator",
            "icon": "",
            "description": "Maintains a running sum of the input stream values.",
            "role": ROLE_CONTROLLER,
            "inputs": [('NodeSocketFloat', 'Value')],
            "outputs": [('NodeSocketFloat', 'Accumulated')],
            "stateful": True,
            },
        },
    },
}

# The add menu draws a separator after this submenu.
MENU_SEPARATOR_AFTER = "output"


def node_def(category: str, node_type: str) -> dict:
    """The registry entry of a node kind, {} for an unknown one."""
    return ACE_NODES.get(str(category or ""), {}).get("nodes", {}).get(str(node_type or ""), {})  # type: ignore[return-value]


def node_role(category: str, node_type: str) -> str:
    """The role of a node kind (one of the ROLE_* values), "" for an unknown one."""
    return str(node_def(category, node_type).get("role", "") or "")


def node_is_native(category: str, node_type: str) -> bool:
    """True for a node that computes in Blender: it has no task, no sinks and no signal."""
    return bool(node_def(category, node_type).get("native", False))


def kinds_with_role(role: str) -> tuple[tuple[str, str], ...]:
    """(category, type) of every node kind with this role, in registry order."""
    return tuple(
        (category, node_type)
        for category, cat_def in ACE_NODES.items()
        for node_type, nd in cat_def.get("nodes", {}).items()
        if nd.get("role") == role
    )


def kind_of_role(role: str) -> tuple[str, str]:
    """The one node kind with this role: the source, the splitter or the preview."""
    kinds = kinds_with_role(role)
    if len(kinds) != 1:
        raise LookupError(f"{len(kinds)} node kinds have the role {role!r}")
    return kinds[0]


def writes_values(category: str, node_type: str) -> bool:
    """True for a node whose values ACE computes and writes for Blender to read."""
    return node_role(category, node_type) in (ROLE_MEASURE, ROLE_CONTROLLER) and not node_is_native(category, node_type)


def reads_values_object(category: str, node_type: str) -> bool:
    """True for a node fed a value, not audio: no bundle carries a proxy to it, so its group reads ACE_Values."""
    return node_role(category, node_type) == ROLE_CONTROLLER and not node_is_native(category, node_type)


def value_outputs(category: str, node_type: str) -> list[tuple[int, str]]:
    """(output index, output name) of each value a node writes."""
    if not writes_values(category, node_type):
        return []
    outputs = node_def(category, node_type).get("outputs", [])
    return [(idx, str(name)) for idx, (sock_type, name) in enumerate(outputs) if "Bundle" not in str(sock_type)]


def mode_inputs(category: str, node_type: str) -> dict[str, str]:
    """{input name: the mode that uses it}: such an input is hidden in the other modes."""
    return dict(node_def(category, node_type).get("mode_inputs", {}) or {})


SOURCE_KIND = kind_of_role(ROLE_SOURCE)
SPLITTER_KIND = kind_of_role(ROLE_SPLITTER)
PREVIEW_KIND = kind_of_role(ROLE_PREVIEW)


# Silence floor of every Loudness method: 0 at this level and below. Not user-editable.
LOUDNESS_FLOOR_DB = -70.0


def node_modes(category: str, node_type: str) -> list[tuple[str, str, str]]:
    """Return the (id, label, description) modes of a node type, or [] if it has none."""
    return [tuple(m) for m in (node_def(category, node_type).get("modes") or [])]  # type: ignore[misc]


def resolve_node_mode(category: str, node_type: str, raw_mode=None) -> str:
    """Return a valid mode id for a node type: *raw_mode* if known, else the default, else ""."""
    modes = node_modes(category, node_type)
    if not modes:
        return ""
    ids = [m[0] for m in modes]
    mode = str(raw_mode or "").strip().lower()
    if mode in ids:
        return mode
    default = str(node_def(category, node_type).get("default_mode", "") or "")
    return default if default in ids else ids[0]


def node_is_stateful(category: str, node_type: str, raw_mode=None) -> bool:
    """True when the node's value depends on earlier frames (its "stateful" entry, for its mode)."""
    flag = node_def(category, node_type).get("stateful", False)
    if isinstance(flag, (list, tuple)):
        return resolve_node_mode(category, node_type, raw_mode) in flag
    return bool(flag)


def is_offload_candidate(category: str, node_type: str, raw_mode=None) -> bool:
    """True for a node whose values can be computed in worker processes.

    Only measurements are, and only stateless ones: each worker keeps its own copy of a task,
    so a stateful task's state would be split across them.
    """
    return node_role(category, node_type) == ROLE_MEASURE and not node_is_stateful(category, node_type, raw_mode)


def node_mode_label(category: str, node_type: str, mode: str) -> str:
    for mode_id, label, _desc in node_modes(category, node_type):
        if mode_id == mode:
            return label
    return ""


# Interface defaults and ranges for numeric controls: (default, min, max).
SOCKET_RANGES: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {
    ("audio", "loudness"): {
        "Window (s)": (0.4, 0.05, 3.0),
    },
    ("controllers", "util_rolling_avg"): {
        "Window": (0.25, 0.001, 60.0),
    },
}

# Sockets whose field shows a unit: a time in seconds reads "250 ms".
SOCKET_SUBTYPES: dict[tuple[str, str], dict[str, str]] = {
    ("controllers", "util_rolling_avg"): {"Window": "TIME_ABSOLUTE"},
}


def apply_socket_ranges(ng, category: str, node_type: str) -> None:
    """Set defaults and ranges on a group interface from SOCKET_RANGES."""
    key = (str(category or ""), str(node_type or ""))
    subtypes = SOCKET_SUBTYPES.get(key, {})
    for item in list(getattr(getattr(ng, "interface", None), "items_tree", [])):
        wanted = subtypes.get(str(getattr(item, "name", "") or ""))
        if wanted and getattr(item, "in_out", None) == "INPUT" and getattr(item, "subtype", wanted) != wanted:
            item.subtype = wanted
    ranges = SOCKET_RANGES.get(key)
    if not ranges:
        return
    for item in getattr(getattr(ng, "interface", None), "items_tree", []):
        if getattr(item, "in_out", None) != "INPUT":
            continue
        spec = ranges.get(str(getattr(item, "name", "") or ""))
        if spec is None:
            continue
        default, lo, hi = spec
        for attr, value in (("min_value", lo), ("max_value", hi), ("default_value", default)):
            if hasattr(item, attr):
                setattr(item, attr, value)
