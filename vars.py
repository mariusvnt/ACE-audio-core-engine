"""
vars.py include all the hard-coded information of the add-on, especially the registry of all available audio nodes.
"""
from typing import TypedDict

# Global Parameters
ACE_MAIN_COLLECTION_NAME = "ACE_Navigation"
ACE_MAIN_COLLECTION_COLOR = 'COLOR_05'  # blue
ACE_OBJECT_COLOR = (0.047, 0.2, 0.6, 1.0)
ACE_COLLECTION_NAME = "ACE_Sources_&_Streams"
ACE_PROXY_PREFIX = "ACE_"
ACE_BLUE = (79 / 255.0, 146 / 255.0, 254 / 255.0, 1.0)
ACE_LOGO_FILENAME = "ACE_Logo.png"

# Required external libraries
REQUIRED_LIBRARIES = {
    "numpy": {
        "name": "NumPy",
        "default": True,
        "desc": "Core numeric processing (bundled with Blender)"
    },
    "audioflux": {
        "name": "AudioFlux",
        "default": False,
        "desc": "High-level audio analysis engine (spectrum, pitch, onset)"
    },
    "numba": {
        "name": "NumBa (Optional)",
        "default": False,
        "desc": "Optional turbo-acceleration for DSP (envelope/filters).",
    },
    "soundfile": {
        "name": "SoundFile",
        "default": False,
        "desc": "Robust audio file reader (WAV, FLAC, etc.)"
    },
    "sounddevice": {
        "name": "SoundDevice",
        "default": False,
        "desc": "Real-time audio capture and preview output"
    }
}

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
            "icon": "SOUND",
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
            "icon": "MOD_WAVE",
            "description": "Splits the incoming audio into multiple audios, each corresponding to a band of frequencies.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'Audio'), ('NodeSocketBundle', 'Audio')], # template sockets, runtime variants are dynamic
            },
            "midside_splitter": {
            "label": "Mid/Side Splitter",
            "icon": "NODE",
            "description": "Splits the incoming audio between the middle and side part of the stereo image.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'Mid'), ('NodeSocketBundle', 'Side')],
            },
            "attackrelease_splitter": {
            "label": "Attack/Release Splitter",
            "icon": "NODE",
            "description": "Splits the incoming audio between the rising and falling parts of the signal.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'Attack'), ('NodeSocketBundle', 'Release')],
            },
            "chromagram": {
            "label": "Chromagram",
            "icon": "NODE",
            "description": "Splits the incoming audio into 12 audios, each corresponing to the level of a particular note.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'C'), ('NodeSocketBundle', 'C#'), ('NodeSocketBundle', 'D'), ('NodeSocketBundle', 'D#'), ('NodeSocketBundle', 'E'), ('NodeSocketBundle', 'F'), ('NodeSocketBundle', 'F#'), ('NodeSocketBundle', 'G'), ('NodeSocketBundle', 'G#'), ('NodeSocketBundle', 'A'), ('NodeSocketBundle', 'A#'), ('NodeSocketBundle', 'B')],
            },
            "auto_component_splitter": {
            "label": "Automatic Component Splitter",
            "icon": "NODE",
            "description": "Automatically splits the incoming audio between its percussive and harmonics components (residual undefined elements are also outputted separately).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketBundle', 'Percussive'), ('NodeSocketBundle', 'Harmonics'), ('NodeSocketBundle', 'Residual')],
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
            "rms": {
            "label": "RMS",
            "icon": "FORCE_HARMONIC",
            "description": "Short-term average loudness measured on the mono-converted signal.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'RMS')],
            },
            "lufs": {
            "label": "Loudness (LUFS)",
            "icon": "SPEAKER",
            "description": (
                "Normalized loudness output in 0..1 "
                "(0 = silence / -inf LUFS, 1 = 0 LUFS). "
                "Window sets averaging time: larger values are smoother/slower, "
                "smaller values react faster. "
                "Overlap Ratio (0..1) controls update density: "
                "0 = stepped window updates, 1 = maximally smooth updates."
            ),
            "inputs": [('NodeSocketBundle', 'Audio'), ('NodeSocketFloat', 'Window (s)'), ('NodeSocketFloat', 'Overlap Ratio')],
            "outputs": [('NodeSocketFloat', 'Loudness')],
            },
            "peak": {
            "label": "Peak",
            "icon": "PMARKER_ACT",
            "description": "Normalized K-weighted peak level in 0..1 (0 = silence, 1 = 0 dB full scale).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Peak')],
            },
            "spectral_centroid": {
            "label": "Spectral Centroid",
            "icon": "CURVE_NCIRCLE",
            "description": "Short-term Weighted mean frequency, perceived atmosphere in Hz from dark to bright.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Centroid')],
            },
            "spectral_flux": {
            "label": "Spectral Flux",
            "icon": "SEQ_HISTOGRAM",
            "description": "Spectral change rate between the precedent and current frame.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Flux')],
            },
            "stereo_width": {
            "label": "Stereo Width",
            "icon": "ORIENTATION_GIMBAL",
            "description": "Side-to-mid energy ratio (0 = mono, 1 = very wide).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Stereo Width')],
            },
            "stereo_balance": {
            "label": "Stereo Balance",
            "icon": "VIEW_PAN",
            "description": "Stereo balance (-1 = left, 0 = center, 1 = right).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Stereo Balance')],
            },
            "pitch": {
            "label": "Pitch",
            "icon": "IPO_SINE",
            "description": "Detected root note pitch in Hz.",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Pitch')],
            },
            "onset": {
            "label": "Onset",
            "icon": "RADIOBUT_ON",
            "description": "Sharp transient witness (0 = smooth, 1 = sharp transient).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'Onset')],
            },
            "loudness": {
            "label": "Loudness",
            "icon": "SPEAKER",
            "description": (
                "Unified loudness operator. Measure method selectable in the N panel: "
                "RMS (short-term average), LUFS (K-weighted integrated), or Peak (K-weighted instantaneous)."
            ),
            "inputs": [('NodeSocketBundle', 'Audio'), ('NodeSocketFloat', 'Window (s)'), ('NodeSocketFloat', 'Overlap Ratio')],
            "outputs": [('NodeSocketFloat', 'Loudness')],
            },
            "zero_crossing_rate": {
            "label": "Zero Crossing Rate",
            "icon": "IPO_BACK",
            "description": "Measure of frequency activity (signal noisiness) (0 = no active frequency, 1 = every frequency is active).",
            "inputs": [('NodeSocketBundle', 'Audio')],
            "outputs": [('NodeSocketFloat', 'ZCR')],
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
            "icon": "INFO",
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
            "audio_math": {
            "label": "Audio Math",
            "icon": "NODETREE",
            "description": "Audio operations: add, subtract, multiply.",
            "inputs": [('NodeSocketBundle', 'Audio')], # dynamic
            "outputs": [('NodeSocketBundle', 'Audio')], # dynamic
            },
            "logic": {
            "label": "Logic",
            "icon": "FAKE_USER_ON",
            "description": "Turn a float stream into a binary 0 or 1 output based on a threshold.",
            "inputs": [('NodeSocketFloat', 'Value'), ('NodeSocketFloat', 'Threshold')],
            "outputs": [('NodeSocketFloat', 'Logic')],
            },
            "util_smooth_operator": {
            "label": "Smooth Operator",
            "icon": "SMOOTHCURVE",
            "description": "Per-sample attack/release envelope follower. Attack controls how fast the output rises, Release controls how slowly it falls.",
            "inputs": [('NodeSocketFloat', 'Value'), ('NodeSocketFloat', 'Attack (frames)'), ('NodeSocketFloat', 'Release (frames)')],
            "outputs": [('NodeSocketFloat', 'Smoothed Value')],
            },
            "stream_accumulator": {
            "label": "Stream Accumulator",
            "icon": "STICKY_UVS_LOC",
            "description": "Maintains a running sum of the input stream values.",
            "inputs": [('NodeSocketFloat', 'Value')],
            "outputs": [('NodeSocketFloat', 'Accumulated')],
            },
        },
    },
}
