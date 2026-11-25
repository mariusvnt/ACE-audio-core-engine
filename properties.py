"""Blender property-group definitions for the AudioCore Engine add-on."""
import bpy                                                                                                                          # type: ignore
import os
import re
import time
from . import utils
from .utils import compose_live_flow_identifier as _compose_live_flow_identifier


# Update callbacks — tempo/meter override tracking

def _mark_user_bpm(self, context):  # noqa: ARG001 - Blender passes context
    with utils.logged_errors("properties._mark_user_bpm"):
        self["ace_bpm_user_override"] = True


def _mark_user_meter(self, context):  # noqa: ARG001
    with utils.logged_errors("properties._mark_user_meter"):
        self["ace_meter_user_override"] = True


def compose_live_flow_identifier(source_prop) -> str:
    """Compatibility wrapper around the shared live-route token builder."""
    return _compose_live_flow_identifier(source_prop)


def _sync_live_flow_identifier(source_prop):
    with utils.logged_errors("properties._sync_live_flow_identifier"):
        source_prop.flow_identifier = compose_live_flow_identifier(source_prop)


def _clear_tempo_detection_state(source_prop):
    stale_keys = (
        "ace_tempo_detect_status",
        "ace_tempo_detect_message",
        "ace_tempo_detect_filepath",
        "ace_tempo_detect_mtime",
        "ace_bpm_auto",
        "ace_meter_auto",
    )
    for key in stale_keys:
        with utils.logged_errors("properties._clear_tempo_detection_state"):
            if hasattr(source_prop, "keys") and key in source_prop.keys():
                del source_prop[key]


# Names the add-on gives a source before it points at anything.
_AUTOMATIC_SOURCE_NAME = re.compile(r"(New Source|Audio Source( \d+)?)?(\.\d{3})?")


def _adopt_source_name(source, wanted: str, scene) -> None:
    """Give a still unnamed *source* the name *wanted*; a name the user chose is kept."""
    wanted = str(wanted or "").strip()
    old = str(source.name)
    project = getattr(scene, "audiocore", None)
    if not wanted or project is None or not _AUTOMATIC_SOURCE_NAME.fullmatch(old):
        return
    taken = {other.name for other in project.audio_sources if other.stable_id != source.stable_id}
    name, number = wanted, 2
    while name in taken:
        name, number = f"{wanted} {number}", number + 1
    source.name = name
    # The Source nodes already placed carry the source's name as their title.
    trees = list(bpy.data.node_groups) + [m.node_tree for m in bpy.data.materials if m.node_tree is not None]
    for tree in trees:
        for node in tree.nodes:
            if source.stable_id and node.get("ace_source_id") == source.stable_id and node.label == old:
                node.label = node.name = name


def _adopt_live_name(source, scene) -> None:
    if str(source.source_type) != "LIVE":
        return
    if str(source.live_transport) == "UDP":
        _adopt_source_name(source, f"UDP {int(source.live_udp_port)}", scene)
        return
    device = str(source.live_device_selector or "")
    _adopt_source_name(source, _LIVE_DEVICE_NAMES.get(device, ""), scene)


def _on_live_routing_change(self, context):
    _sync_live_flow_identifier(self)
    with utils.logged_errors("properties._on_live_routing_change"):
        _adopt_live_name(self, getattr(context, "scene", None))


def _on_filepath_change(self, context):
    """Validate the selected filepath immediately and write ace_source_error."""
    fp = str(getattr(self, "filepath", "") or "").strip()
    error_msg = ""
    if fp and str(getattr(self, "source_type", "FILE")) == "FILE":
        with utils.logged_errors("properties._on_filepath_change"):
            from .core.input.reception import AudioReception
            if not os.path.isfile(fp):
                error_msg = "File not found"
            elif not AudioReception.is_supported_file_path(fp):
                error_msg = "Unsupported or unreadable audio format"
    with utils.logged_errors("properties._on_filepath_change#2"):
        self.ace_source_error = error_msg
    _clear_tempo_detection_state(self)
    if fp and not error_msg and str(self.source_type) == "FILE":
        with utils.logged_errors("properties._on_filepath_change#3"):
            stem = os.path.splitext(os.path.basename(fp.replace("\\", "/")))[0]
            _adopt_source_name(self, stem, getattr(context, "scene", None))
    _on_engine_runtime_setting_change(self, context)


def _on_source_type_change(self, context):
    if str(getattr(self, "source_type", "") or "") == "LIVE":
        _sync_live_flow_identifier(self)
    else:
        # Switching back to FILE: clear the live flow identifier so stale "sd:"
        # or UDP tokens don't persist and confuse the source routing logic.
        with utils.logged_errors("properties._on_source_type_change"):
            self.flow_identifier = ""
    _clear_tempo_detection_state(self)
    with utils.logged_errors("properties._on_source_type_change#2"):
        _adopt_live_name(self, getattr(context, "scene", None))
    # Invalidate engine caches and force a recompile so the new source type
    # takes effect immediately (clears probe cache, stale parallel futures, etc.)
    _on_engine_runtime_setting_change(self, context)


def _on_engine_runtime_setting_change(self, context):  # noqa: ARG001
    scene = getattr(context, "scene", None) if context else None
    if scene is None:
        return
    with utils.logged_errors("properties._on_engine_runtime_setting_change"):
        from .core.engine.engine import ACEAudioEngine
        engine = ACEAudioEngine.ensure_instance(scene)
        engine.request_recompile(None)


def _on_active_source_index_change(self, context):
    with utils.logged_errors("properties._on_active_source_index_change"):
        from .ui.panel.sync import _sync_ui_dummy_tree, follow_active_source
        follow_active_source(self)
        _sync_ui_dummy_tree(context, force=True)


# Module-level cache for live-device enumeration.
_LIVE_DEVICE_CACHE: list = []
_LIVE_DEVICE_NAMES: dict[str, str] = {}  # selector value -> bare device name
_LIVE_DEVICE_CACHE_TS: float = 0.0
_LIVE_DEVICE_CACHE_INTERVAL: float = 2.0  # seconds between PortAudio rescans


class _NoLiveBackend(Exception):
    """sounddevice is unavailable; only the Auto entry is listed."""


def _refresh_live_device_cache() -> None:
    """Enumerate live input devices and update the module-level cache."""
    global _LIVE_DEVICE_CACHE, _LIVE_DEVICE_CACHE_TS
    items = [
        ("__AUTO__", "Auto (Default Input)", "Use the system default input device"),
    ]
    try:
        import sounddevice as sd  # type: ignore
    except Exception:  # noqa: BLE001 boundary: not installed (or its PortAudio DLL fails) means no live devices
        sd = None
    try:
        if sd is None:
            raise _NoLiveBackend()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for idx, dev in enumerate(devices):
            in_ch = int(dev.get("max_input_channels", 0) or 0)
            if in_ch <= 0:
                continue
            name = str(dev.get("name", f"Device {idx}") or f"Device {idx}")
            _LIVE_DEVICE_NAMES[str(idx)] = name
            host_name = ""
            try:
                host_idx = int(dev.get("hostapi", -1))
                if 0 <= host_idx < len(hostapis):
                    host_name = str(hostapis[host_idx].get("name", "") or "")
            except Exception as exc:
                utils.note_swallowed("properties._refresh_live_device_cache", exc)
                host_name = ""
            label = f"{idx}: {name}"
            if host_name:
                label = f"{label} ({host_name})"
            items.append((str(idx), label, f"{in_ch} input channel(s)"))

        # Expose output-only devices as WASAPI loopback sources (Windows only).
        if hasattr(sd, "WasapiSettings"):
            for idx, dev in enumerate(devices):
                in_ch = int(dev.get("max_input_channels", 0) or 0)
                out_ch = int(dev.get("max_output_channels", 0) or 0)
                if in_ch > 0 or out_ch <= 0:
                    continue
                name = str(dev.get("name", f"Device {idx}") or f"Device {idx}")
                _LIVE_DEVICE_NAMES[f"loopback:{idx}"] = name
                host_name = ""
                try:
                    host_idx = int(dev.get("hostapi", -1))
                    if 0 <= host_idx < len(hostapis):
                        host_name = str(hostapis[host_idx].get("name", "") or "")
                except Exception as exc:
                    utils.note_swallowed("properties._refresh_live_device_cache#2", exc)
                    host_name = ""
                label = f"{idx}: {name} [Loopback]"
                if host_name:
                    label = f"{label} ({host_name})"
                items.append((f"loopback:{idx}", label, f"WASAPI loopback — capture {out_ch} ch from this output"))
    except Exception as exc:
        if not isinstance(exc, _NoLiveBackend):
            utils.note_swallowed("properties._refresh_live_device_cache#3", exc)
        items.append(
            (
                "__MISSING__",
                "SoundDevice Unavailable",
                "Install SoundDevice in add-on preferences to enumerate live devices",
            )
        )
    _LIVE_DEVICE_CACHE = items
    _LIVE_DEVICE_CACHE_TS = time.monotonic()


def _live_device_cache_timer() -> float:
    _refresh_live_device_cache()
    return _LIVE_DEVICE_CACHE_INTERVAL


def _live_device_items(self, context):  # noqa: ARG001
    if not _LIVE_DEVICE_CACHE:
        _refresh_live_device_cache()  # first draw only
    return _LIVE_DEVICE_CACHE


class ACEUISegment(bpy.types.PropertyGroup):
    """One clickable name in a row of the Active Operators tree."""
    node_id: bpy.props.StringProperty(name="Node ID", default="")                                                                    # type: ignore
    label: bpy.props.StringProperty(name="Label", default="")                                                                        # type: ignore
    kind: bpy.props.StringProperty(name="Kind", default="node")                                                                      # type: ignore


class ACEUINode(bpy.types.PropertyGroup):
    """One row of the Active Operators tree: a chain of operators without branching."""
    name: bpy.props.StringProperty(name="Name", default="Node")                                                                      # type: ignore
    node_id: bpy.props.StringProperty(name="Node ID", default="")                                                                    # type: ignore
    node_type: bpy.props.StringProperty(name="Type", default="")                                                                     # type: ignore
    current_value: bpy.props.FloatProperty(name="Value", default=0.0)                                                                # type: ignore
    description: bpy.props.StringProperty(name="Description", default="")                                                            # type: ignore
    segments: bpy.props.CollectionProperty(type=ACEUISegment)                                                                        # type: ignore
    depth: bpy.props.IntProperty(default=0)                                                                                          # type: ignore
    has_children: bpy.props.BoolProperty(default=False)                                                                              # type: ignore
    branch_key: bpy.props.StringProperty(default="")                                                                                 # type: ignore
    parent_row: bpy.props.IntProperty(default=-1)                                                                                    # type: ignore


def _on_active_operator_row(self, context):
    from .ui.panel.selection import select_row_operator
    select_row_operator(self, context)


class ACEAudioSource(bpy.types.PropertyGroup):
    """Represents a single audio source in the project."""
    name: bpy.props.StringProperty(name="Name", default="New Source")# type: ignore
    
    source_type: bpy.props.EnumProperty(                                                                                            # type: ignore
        name="Type",
        items=[
            ('FILE', "File", "Audio file from disk"),
            ('LIVE', "Live Flow", "Real-time audio stream identifier (e.g. Loopback, Microphone)"),
        ],
        default='FILE',
        update=_on_source_type_change,
    )
    
    filepath: bpy.props.StringProperty(                                                                                             # type: ignore
        name="File Path",
        subtype='FILE_PATH',
        description="Path to the audio file",
        update=_on_filepath_change,
    )
    
    flow_identifier: bpy.props.StringProperty(                                                                                      # type: ignore
        name="Flow ID", 
        description=(
            "Internal live routing token generated from transport settings."
        ),
        default="sd:"
    )
    live_transport: bpy.props.EnumProperty(                                                                                         # type: ignore
        name="Transport",
        description="Live transport backend used by ACE",
        items=[
            ('SOUNDDEVICE', "SoundDevice", "Capture audio from a local input device"),
            ('UDP', "UDP", "Capture UDP float stream (port/channels/sample rate)"),
        ],
        default='SOUNDDEVICE',
        update=_on_live_routing_change,
    )
    live_device_selector: bpy.props.EnumProperty(                                                                                   # type: ignore
        name="Capture Device",
        description=(
            "Select the live input device used by ACE. "
            "Selecting a device updates the internal route token automatically."
        ),
        items=_live_device_items,
        update=_on_live_routing_change,
    )
    live_udp_port: bpy.props.IntProperty(                                                                                           # type: ignore
        name="UDP Port",
        description="UDP listening port for live float stream capture",
        default=7001,
        min=1,
        max=65535,
        update=_on_live_routing_change,
    )
    live_udp_channels: bpy.props.IntProperty(                                                                                       # type: ignore
        name="UDP Channels",
        description="Channel count expected from UDP stream payload",
        default=2,
        min=1,
        max=8,
        update=_on_live_routing_change,
    )
    live_udp_sample_rate: bpy.props.EnumProperty(                                                                                   # type: ignore
        name="UDP Sample Rate",
        description="Sample rate expected from UDP stream payload",
        items=[
            ("24000", "24 kHz", "Low-cost UDP stream"),
            ("48000", "48 kHz", "Standard UDP stream sample rate"),
            ("96000", "96 kHz", "High-fidelity UDP stream"),
        ],
        default="48000",
        update=_on_live_routing_change,
    )
    stable_id: bpy.props.StringProperty(                                                                                            # type: ignore
        name="Stable ID",
        description="Internal identifier used to keep node sockets stable across renames.",
        default=""
    )
    
    proxy_object_name: bpy.props.StringProperty(                                                                                    # type: ignore
        name="Proxy Object",
        description="Name of the Blender Object acting as the data container"
    )
    
    duration: bpy.props.FloatProperty(                                                                                              # type: ignore
        name="Duration", 
        default=0.0, 
        unit='TIME',
        description="Detected duration in seconds"
    )
    sample_rate: bpy.props.IntProperty(                                                                                             # type: ignore
        name="Sample Rate", 
        default=44100, 
        description="Detected sample rate in Hz"
    )
    native_sample_rate: bpy.props.IntProperty(                                                                                      # type: ignore
        name="Original Sample Rate",
        default=0,
        description="Original detected sample rate in Hz before any engine working-rate cap"
    )
    
    live_latency_ms: bpy.props.IntProperty(                                                                                         # type: ignore
        name="Live Latency",
        default=10,
        min=1, max=1000,
        description="Desired latency buffer for live capture (ms)"
    )

    # These are fed into the Engine to pre-process the source signal
    enable_preprocessing: bpy.props.BoolProperty(                                                                                   # type: ignore
        name="Envelope",
        description="Apply source-side HPF/LPF filtering and signed attack/release smoothing before analysis",
        default=False,
    )
    envelope_attack: bpy.props.FloatProperty(                                                                                       # type: ignore
        name="Attack", 
        description="Smoothing time for rising signal magnitude (ms)", 
        default=10.0, min=0.0
    )
    envelope_release: bpy.props.FloatProperty(                                                                                      # type: ignore
        name="Release", 
        description="Smoothing time for falling signal magnitude (ms)", 
        default=100.0, min=0.0
    )
    freq_min: bpy.props.FloatProperty(                                                                                              # type: ignore
        name="Min Freq", 
        description="Lowest relevant frequency (Hz). Signals below this are filtered out.", 
        default=20.0, min=0.0, soft_max=20000.0
    )
    freq_max: bpy.props.FloatProperty(                                                                                              # type: ignore
        name="Max Freq", 
        description="Highest relevant frequency (Hz).", 
        default=20000.0, min=0.0, soft_max=20000.0
    )
    bpm: bpy.props.FloatProperty(                                                                                                   # type: ignore
        name="BPM",
        description="Tempo used for beat-synced calculations.",
        default=120.0, min=1.0, soft_max=240.0,
        update=_mark_user_bpm
    )
    beats_per_bar: bpy.props.IntProperty(                                                                                           # type: ignore
        name="Beats/Bar",
        description="Beats in one measure for beat position reporting.",
        default=4, min=1, max=32,
        update=_mark_user_meter
    )
    live_status: bpy.props.StringProperty(                                                                                          # type: ignore
        name="Live Status",
        description="Status message for live inputs (warning when unavailable).",
        default=""
    )

    ace_source_error: bpy.props.StringProperty(                                                                                     # type: ignore
        name="Source Error",
        description=(
            "Non-empty when the selected file cannot be opened by the engine "
            "(unsupported format, missing file, or corrupt data)."
        ),
        default="",
    )

    ui_nodes: bpy.props.CollectionProperty(type=ACEUINode)                                                                          # type: ignore
    active_ui_node_index: bpy.props.IntProperty(default=0, update=_on_active_operator_row)                                         # type: ignore
    # The selected operator (ace_id): rows hold several operators, so the selection is a node, not a row.
    active_ui_node_id: bpy.props.StringProperty(default="")                                                                         # type: ignore


class ACEEngineSettings(bpy.types.PropertyGroup):
    """Global configuration for the Audio Engine."""

    splitter_default_bands: bpy.props.IntProperty(                                                                                 # type: ignore
        name="Default Splitter Bands",
        description="Initial band count when adding a Frequency Splitter node",
        default=2,
        min=2,
        max=8,
    )
    audio_math_default_inputs: bpy.props.IntProperty(                                                                              # type: ignore
        name="Default Audio Math Inputs",
        description="Initial input count when adding an Audio Math node",
        default=2,
        min=2,
        max=8,
    )
    preview_numba_turbo: bpy.props.BoolProperty(                                                                                   # type: ignore
        name="High quality audio preview",
        description="Enable Numba acceleration for high quality time-domain preview filtering.",
        default=True,
    )
    ace_preview_block_size: bpy.props.EnumProperty(                                                                                # type: ignore
        name="Preview Block",
        description="Low-latency block size for Layer 3 preview processing.",
        items=[
            ("128", "128", "Highest responsiveness, highest CPU load"),
            ("256", "256", "Balanced latency and CPU load"),
            ("512", "512", "Most stable under heavy graphs"),
        ],
        default="256",
    )
    ace_file_working_sample_rate: bpy.props.EnumProperty(                                                                          # type: ignore
        name="Engine Sample Rate",
        description=(
            "Cap file sources to this working sample rate for analysis, preview, and prefetch proxy generation"
        ),
        items=[
            ("22050", "22 kHz", "Minimum viable sample rate for voice analysis"),
            ("44100", "44.1 kHz", "CD-quality standard for music"),
            ("48000", "48 kHz", "Standard balance for preview and analysis"),
            ("96000", "96 kHz", "Higher fidelity with higher cost"),
        ],
        default="48000",
        update=_on_engine_runtime_setting_change,
    )
    ace_perf_enabled: bpy.props.BoolProperty(                                                                                      # type: ignore
        name="Performance Metrics",
        description=(
            "Collect lightweight per-frame timings for audio stages, Blender geometry updates, and render timing"
        ),
        default=True,
    )
    ace_scheduler_mode: bpy.props.EnumProperty(                                                                                    # type: ignore
        name="Scheduler Mode",
        description="Runtime scheduling policy for source evaluation",
        items=[
            ("legacy", "Legacy", "Always tick source nodes each frame"),
            ("demand", "Demand-Driven", "Evaluate sources only through reachable terminal outputs"),
        ],
        default="demand",
    )
    ace_timeslice_mode: bpy.props.EnumProperty(                                                                                    # type: ignore
        name="Timeslice Mode",
        description="Block-size policy across frame deltas",
        items=[
            ("fixed", "Fixed", "Use a fixed frame-locked block size"),
            ("adaptive", "Adaptive", "Adapt block size to frame delta (timeslice style)"),
        ],
        default="fixed",
    )
    ace_parallel_mode: bpy.props.EnumProperty(                                                                                     # type: ignore
        name="Parallel Mode",
        description=(
            "Execution mode for ACE runtime scheduling. "
            "Processes-DSP offloads selected heavy stream terminals to subprocess workers."
        ),
        items=[
            ("threads_io", "Threads-I/O", "Use threads for I/O buffering and async helpers only"),
            ("processes_dsp", "Processes-DSP", "Use subprocess workers for selected per-frame DSP terminals"),
            ("off", "Off", "Disable parallel offload and keep full DSP on main thread"),
        ],
        default="processes_dsp",
    )


class ACEUISourceRow(bpy.types.PropertyGroup):
    """One row of the audio source list: a source, or an object it drives."""
    name: bpy.props.StringProperty(default="")                                                                                       # type: ignore
    # "source" or "instance".
    kind: bpy.props.StringProperty(default="instance")                                                                               # type: ignore
    # Source key (stable_id) for a source row, object name for an instance row.
    target: bpy.props.StringProperty(default="")                                                                                     # type: ignore
    detail: bpy.props.StringProperty(default="")                                                                                     # type: ignore
    icon: bpy.props.StringProperty(default="NONE")                                                                                   # type: ignore
    detail_icon: bpy.props.StringProperty(default="NONE")                                                                            # type: ignore
    depth: bpy.props.IntProperty(default=0)                                                                                          # type: ignore
    has_children: bpy.props.BoolProperty(default=False)                                                                              # type: ignore
    branch_key: bpy.props.StringProperty(default="")                                                                                 # type: ignore
    parent_row: bpy.props.IntProperty(default=-1)                                                                                    # type: ignore


def _on_active_source_row(self, context):
    from .ui.panel.selection import select_source_row
    select_source_row(self, context)


class ACEProjectSettings(bpy.types.PropertyGroup):
    """Root Settings container for the Add-on."""
    
    # Collections
    audio_sources: bpy.props.CollectionProperty(type=ACEAudioSource)                                                                # type: ignore
    active_audio_source_index: bpy.props.IntProperty(                                                                               # type: ignore
        default=0,
        update=_on_active_source_index_change,
    )
    
    # The audio source list: sources and, folded under each, the objects it drives.
    source_rows: bpy.props.CollectionProperty(type=ACEUISourceRow)                                                                  # type: ignore
    active_source_row: bpy.props.IntProperty(default=0, update=_on_active_source_row)                                              # type: ignore

    # Sub-Properties
    engine_settings: bpy.props.PointerProperty(type=ACEEngineSettings)                                                              # type: ignore
    gn_target_object: bpy.props.PointerProperty(type=bpy.types.Object, name="GN Target Object")                                     # type: ignore
    
    # Unused; kept so files saved with a project-level mode still load.
    mode: bpy.props.EnumProperty(                                                                                                   # type: ignore
        name="Mode",
        items=[
            ('FILE', "File Playback", "Read from disk (Animation Sync)"),
            ('LIVE', "Live Stream", "Real-time audio capture"),
        ],
        default='FILE'
    )


classes = (
    ACEUISegment,
    ACEUINode,
    ACEAudioSource,
    ACEEngineSettings,
    ACEUISourceRow,
    ACEProjectSettings,
)

def _get_rename_name(self):
    from .ui.panel.selection import rename_box_name
    return rename_box_name(self)


def _set_rename_name(self, value):
    from .ui.panel.selection import set_rename_box_name
    set_rename_box_name(self, value)


def _write_split_freq(node, index: int, hz: float) -> None:
    from .nodes.split_freqs import set_split_freq
    if set_split_freq(node, index, hz):
        from .ui.n_panel_operators import sort_split_freqs_when_released
        sort_split_freqs_when_released()


def _split_freq_property(index: int):
    def get(node):
        from .nodes.split_freqs import get_split_freq
        return get_split_freq(node, index)

    def set_(node, value):
        _write_split_freq(node, index, value)

    return bpy.props.FloatProperty(
        name=f"Split Freq {index + 1}",
        description="Frequency (Hz) where one band ends and the next begins. Sorted into ascending order when released",
        min=0.0, max=20000.0, precision=0, step=100, get=get, set=set_,
    )


def _split_position_property(index: int):
    def get(node):
        from .nodes.split_freqs import freq_to_position, get_split_freq
        return freq_to_position(get_split_freq(node, index))

    def set_(node, value):
        from .nodes.split_freqs import position_to_freq
        _write_split_freq(node, index, position_to_freq(value))

    return bpy.props.FloatProperty(
        name=f"Split Freq {index + 1}",
        description="Logarithmic slider from 20 Hz (0) to 20 kHz (1)",
        min=0.0, max=1.0, precision=2, get=get, set=set_,
    )


# UI-only node properties: each splitter socket's slider and Hz field (7 sockets = 8 bands).
_NODE_PROPS = {f"ace_split_freq_{i + 1}": _split_freq_property(i) for i in range(7)}
_NODE_PROPS.update({f"ace_split_pos_{i + 1}": _split_position_property(i) for i in range(7)})


def _get_band_fill(_self):
    from .ui.panel.selection import band_fill_color
    return band_fill_color()


# UI-only window-manager properties: the rename box and the band strip's colour.
_WM_PROPS = {
    "ace_rename_name": bpy.props.StringProperty(
        name="Name",
        description="Name of the operator, shown in the Active Operators list and as its node's title. Empty restores the automatic name",
        get=_get_rename_name,
        set=_set_rename_name,
    ),
    "ace_band_fill": bpy.props.FloatVectorProperty(
        name="Band", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        get=_get_band_fill, set=lambda _self, _value: None,
    ),
}


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Attach to Scene
    bpy.types.Scene.audiocore = bpy.props.PointerProperty(type=ACEProjectSettings)
    for name, prop in _WM_PROPS.items():
        setattr(bpy.types.WindowManager, name, prop)
    for name, prop in _NODE_PROPS.items():
        setattr(bpy.types.Node, name, prop)


def unregister():
    for name in _NODE_PROPS:
        if hasattr(bpy.types.Node, name):
            delattr(bpy.types.Node, name)
    for name in _WM_PROPS:
        if hasattr(bpy.types.WindowManager, name):
            delattr(bpy.types.WindowManager, name)
    if hasattr(bpy.types.Scene, "audiocore"):
        del bpy.types.Scene.audiocore

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
