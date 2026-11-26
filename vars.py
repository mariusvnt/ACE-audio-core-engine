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
    """format of a category"""
    label: str
    icon: str
    description: str

class NodeType(TypedDict):
    """format of a node type"""
    label: str
    icon: str
    description: str
    inputs: list[tuple[str, str]]
    outputs: list[tuple[str, str]]

class Category(TypedDict):
    """category container"""
    meta: Meta
    nodes: dict[str, NodeType]

ACE_NODES: dict[str, Category] = {
    "source_mngmt": {
        "meta": {
            "label": "Source Management",
            "icon": "",
            "description": "Particular nodes"
        },
        "nodes": {
            "src": {
            "label": "Source",
            "icon": "",
            "description": "Reference to an audio source.",
            "inputs": [],
            "outputs": [('NodeSocketBundle', 'Audio')],
            },
        },
    },

    "delta": {
        "meta": {
            "label": "Delta",
            "icon": "",
            "description": "Nodes performing audio to audio operations."
        },
        "nodes": {
            "frequency_splitter": {
            "label": "Frequency Splitter",
            "icon": "",
            "description": "Splits the incoming audio into multiple audios, each corresponding to a band of frequencies.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'Audio'), ('NodeSocketBundle', 'Audio')], # template sockets, runtime variants are dynamic
            },
        },
    },

    "stream": {
        "meta": {
            "label": "Stream",
            "icon": "",
            "description": "Nodes performing audio to float/vector operations."
        },
        "nodes": {
            "loudness": {
            "label": "Loudness",
            "icon": "DRIVER_DISTANCE",
            "description": (
                "Loudness in 0..1: 1 at 0 dB, 0 at -70 dB (silence) and below, linear in dB "
                "in between. Method: RMS (signal energy), LUFS (perceived loudness, "
                "averaged over Window) or Peak (highest sample)."
            ),
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
            "default_mode": "lufs",
            },
            "stereo": {
            "label": "Stereo",
            "icon": "ORIENTATION_GIMBAL",
            "description": "Stereo image. Width: side-to-mid energy (0 = mono, 1 = very wide). Balance: -1 = left, 0 = center, 1 = right.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Stereo')],
            "modes": [
                ("width", "Width", "Side-to-mid energy ratio (0 = mono, 1 = very wide)."),
                ("balance", "Balance", "Left/right energy balance (-1 = left, 0 = center, 1 = right)."),
            ],
            "default_mode": "width",
            },
            "stereo_vector": {
            "label": "Stereo Vector",
            "icon": "ORIENTATION_GIMBAL",
            "description": "Time-domain stereo descriptor packed as vector (L RMS, R RMS, Balance).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketVector', 'Stereo Vector')],
            },
            "brightness": {
            "label": "Brightness",
            "icon": "SEQ_HISTOGRAM",
            "description": "Spectral centroid in Hz: where the energy of the frame sits, from dark to bright. Behind a Frequency Splitter, within the band.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Brightness')],
            },
            "pitch": {
            "label": "Pitch",
            "icon": "",
            "description": "Detected root note pitch in Hz.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Pitch')],
            },
            "onset": {
            "label": "Onset",
            "icon": "",
            "description": "Sharp transient witness (0 = smooth, 1 = sharp transient).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Onset')],
            },
        },
    },

    "utilities": {
        "meta": {
            "label": "Utilities",
            "icon": "",
            "description": "Helper nodes for graph monitoring and utility operations."
        },
        "nodes": {
            "src_specs": {
            "label": "Source Specs",
            "icon": "",
            "description": "Long-term specs and passing through the audio.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [
                ('NodeSocketFloat', 'Frames per Beat'),
                ('NodeSocketFloat', 'Beat Phase'),
                ('NodeSocketBundle', 'Audio')
            ],
            },
            "preview_sink": {
            "label": "Audio Preview",
            "icon": "PLAY_SOUND",
            "description": "Branch audio monitor sink used for in-editor preview playback.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [],
            },
            "util_rolling_avg": {
            "label": "Rolling Average",
            "icon": "",
            "description": "Smooths a scalar attribute stream over a time window.",
            "inputs": [('NodeSocketFloat', 'Value'), ('NodeSocketFloat', 'Window')],
            "outputs": [('NodeSocketFloat', 'Rolling Avg')],
            },
            "stream_accumulator": {
            "label": "Stream Accumulator",
            "icon": "",
            "description": "Maintains a running sum of the input stream values.",
            "inputs": [('NodeSocketFloat', 'Value')],
            "outputs": [('NodeSocketFloat', 'Accumulated')],
            },
        },
    },
}


# Stream node types folded into mode nodes: old type -> (node type, mode).
# Files saved with the old types are migrated on load; the engine also accepts them.
LEGACY_STREAM_TYPES: dict[str, tuple[str, str]] = {
    "rms": ("loudness", "rms"),
    "lufs": ("loudness", "lufs"),
    "peak": ("loudness", "peak"),
    "stereo_width": ("stereo", "width"),
    "stereo_balance": ("stereo", "balance"),
}

# Silence floor of every Loudness method: 0 at this level and below. Not user-editable.
LOUDNESS_FLOOR_DB = -70.0


def node_modes(category: str, node_type: str) -> list[tuple[str, str, str]]:
    """Return the (id, label, description) modes of a node type, or [] if it has none."""
    node_def = ACE_NODES.get(str(category or ""), {}).get("nodes", {}).get(str(node_type or ""), {})
    return [tuple(m) for m in (node_def.get("modes") or [])]  # type: ignore[misc]


def resolve_node_mode(category: str, node_type: str, raw_mode=None) -> str:
    """Return a valid mode id for a node type: *raw_mode* if known, else the default, else ""."""
    typ = str(node_type or "")
    legacy = LEGACY_STREAM_TYPES.get(typ) if str(category or "") == "stream" else None
    if legacy is not None:
        return legacy[1]
    modes = node_modes(category, typ)
    if not modes:
        return ""
    ids = [m[0] for m in modes]
    mode = str(raw_mode or "").strip().lower()
    if mode in ids:
        return mode
    node_def = ACE_NODES.get(str(category or ""), {}).get("nodes", {}).get(typ, {})
    default = str(node_def.get("default_mode", "") or "")
    return default if default in ids else ids[0]


def node_mode_label(category: str, node_type: str, mode: str) -> str:
    for mode_id, label, _desc in node_modes(category, node_type):
        if mode_id == mode:
            return label
    return ""


# Interface defaults and ranges for numeric controls: (default, min, max).
SOCKET_RANGES: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {
    ("stream", "loudness"): {
        "Window (s)": (0.4, 0.05, 3.0),
    },
    ("utilities", "util_rolling_avg"): {
        "Window": (0.25, 0.001, 60.0),
    },
}

# Sockets whose field shows a unit: a time in seconds reads "250 ms".
SOCKET_SUBTYPES: dict[tuple[str, str], dict[str, str]] = {
    ("utilities", "util_rolling_avg"): {"Window": "TIME_ABSOLUTE"},
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
