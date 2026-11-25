"""
Blender property-group definitions for the AudioCore Engine add-on.

Defines ACEUINode, ACEAudioSource, ACEEngineSettings, and ACEProjectSettings
along with the update callbacks and dynamic enum helpers that wire them to
the engine and the live device list.
"""
import bpy                                                                                                                          # type: ignore
import time
from .utils import as_int as _as_int
from .utils import compose_live_flow_identifier as _compose_live_flow_identifier


# ---------------------------------------------------------------------------
# Update callbacks — tempo/meter override tracking
# ---------------------------------------------------------------------------

def _mark_user_bpm(self, context):  # noqa: ARG001 - Blender passes context
    """Set ace_bpm_user_override so auto-detection does not overwrite the user's BPM."""
    try:
        self["ace_bpm_user_override"] = True
    except Exception:
        pass


def _mark_user_meter(self, context):  # noqa: ARG001
    """Set ace_meter_user_override so auto-detection does not overwrite the user's meter."""
    try:
        self["ace_meter_user_override"] = True
    except Exception:
        pass


def compose_live_flow_identifier(source_prop) -> str:
    """Compatibility wrapper around the shared live-route token builder."""
    return _compose_live_flow_identifier(source_prop)


def _sync_live_flow_identifier(source_prop):
    """Write the current live route token into source_prop.flow_identifier."""
    try:
        source_prop.flow_identifier = compose_live_flow_identifier(source_prop)
    except Exception:
        pass


def _source_has_tempo_meter_selection(source_prop) -> bool:
    """Return True when the source has a selected file or a valid live route."""
    src_type = str(getattr(source_prop, "source_type", "") or "").upper()
    if src_type == "FILE":
        return bool(str(getattr(source_prop, "filepath", "") or "").strip())
    if src_type != "LIVE":
        return False

    transport = str(getattr(source_prop, "live_transport", "SOUNDDEVICE") or "SOUNDDEVICE").upper()
    if transport == "UDP":
        try:
            return int(getattr(source_prop, "live_udp_port", 0) or 0) > 0
        except Exception:
            return False

    device = str(getattr(source_prop, "live_device_selector", "") or "").strip()
    return bool(device) and device != "__MISSING__"


def _sync_tempo_meter_panel_expansion(source_prop):
    """Keep Tempo & Meter foldout state aligned with source selection/config."""
    should_open = _source_has_tempo_meter_selection(source_prop)
    try:
        if bool(getattr(source_prop, "tempo_meter_panel_open", False)) != bool(should_open):
            source_prop.tempo_meter_panel_open = bool(should_open)
    except Exception:
        pass


def _clear_tempo_detection_state(source_prop):
    """Drop cached tempo-detection idprops when the source path/type changes."""
    stale_keys = (
        "ace_tempo_detect_status",
        "ace_tempo_detect_message",
        "ace_tempo_detect_filepath",
        "ace_tempo_detect_mtime",
        "ace_bpm_auto",
        "ace_meter_auto",
    )
    for key in stale_keys:
        try:
            if hasattr(source_prop, "keys") and key in source_prop.keys():
                del source_prop[key]
        except Exception:
            pass


def _on_live_routing_change(self, context):  # noqa: ARG001
    """Blender update callback: rebuild the live flow token when any routing property changes."""
    _sync_live_flow_identifier(self)
    _sync_tempo_meter_panel_expansion(self)


def _on_filepath_change(self, context):
    """Validate the selected filepath immediately and write ace_source_error.

    Runs on the main thread (Blender update callback).  Uses the pure-Python
    ``AudioReception.is_supported_file_path()`` probe — no bpy access inside.
    Clears the error for non-FILE sources and empty paths.
    """
    fp = str(getattr(self, "filepath", "") or "").strip()
    error_msg = ""
    if fp and str(getattr(self, "source_type", "FILE")) == "FILE":
        try:
            import os
            from .core.audio_influx import AudioReception
            if not os.path.isfile(fp):
                error_msg = "File not found"
            elif not AudioReception.is_supported_file_path(fp):
                error_msg = "Unsupported or unreadable audio format"
        except Exception:
            pass
    try:
        self.ace_source_error = error_msg
    except Exception:
        pass
    _clear_tempo_detection_state(self)
    _sync_tempo_meter_panel_expansion(self)
    _on_engine_runtime_setting_change(self, context)


def _on_source_type_change(self, context):
    """Blender update callback: sync live routing or clear stale tokens on source-type switch.

    When switching to LIVE, rebuilds the flow_identifier from the current
    transport settings.  When switching back to FILE, clears the identifier so
    stale "sd:" or UDP tokens do not persist and confuse source routing.
    Also triggers a full engine recompile via _on_engine_runtime_setting_change.
    """
    if str(getattr(self, "source_type", "") or "") == "LIVE":
        _sync_live_flow_identifier(self)
    else:
        # Switching back to FILE: clear the live flow identifier so stale "sd:"
        # or UDP tokens don't persist and confuse the source routing logic.
        try:
            self.flow_identifier = ""
        except Exception:
            pass
    _clear_tempo_detection_state(self)
    _sync_tempo_meter_panel_expansion(self)
    # Invalidate engine caches and force a recompile so the new source type
    # takes effect immediately (clears probe cache, stale parallel futures, etc.)
    _on_engine_runtime_setting_change(self, context)


def _on_engine_runtime_setting_change(self, context):  # noqa: ARG001
    """Blender update callback: request an engine recompile on any runtime setting change."""
    scene = getattr(context, "scene", None) if context else None
    if scene is None:
        return
    try:
        from .core.audio_engine import ACEAudioEngine
        engine = ACEAudioEngine.ensure_instance(scene)
        engine.request_recompile(None)
    except Exception:
        pass


def _on_active_source_index_change(self, context):  # noqa: ARG001
    """Auto-open Tempo & Meter when selecting a source with configured input."""
    sources = getattr(self, "audio_sources", None) or []
    try:
        idx = _as_int(getattr(self, "active_audio_source_index", None), -1)
    except Exception:
        idx = -1
    if 0 <= idx < len(sources):
        _sync_tempo_meter_panel_expansion(sources[idx])


# Module-level cache for live-device enumeration.
_LIVE_DEVICE_CACHE: list = []
_LIVE_DEVICE_CACHE_TS: float = 0.0
_LIVE_DEVICE_CACHE_INTERVAL: float = 2.0  # seconds between PortAudio rescans


def _refresh_live_device_cache() -> None:
    """Enumerate live input devices and update the module-level cache."""
    global _LIVE_DEVICE_CACHE, _LIVE_DEVICE_CACHE_TS
    items = [
        ("__AUTO__", "Auto (Default Input)", "Use the system default input device"),
    ]
    try:
        import sounddevice as sd  # type: ignore
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for idx, dev in enumerate(devices):
            in_ch = int(dev.get("max_input_channels", 0) or 0)
            if in_ch <= 0:
                continue
            name = str(dev.get("name", f"Device {idx}") or f"Device {idx}")
            host_name = ""
            try:
                host_idx = int(dev.get("hostapi", -1))
                if 0 <= host_idx < len(hostapis):
                    host_name = str(hostapis[host_idx].get("name", "") or "")
            except Exception:
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
                host_name = ""
                try:
                    host_idx = int(dev.get("hostapi", -1))
                    if 0 <= host_idx < len(hostapis):
                        host_name = str(hostapis[host_idx].get("name", "") or "")
                except Exception:
                    host_name = ""
                label = f"{idx}: {name} [Loopback]"
                if host_name:
                    label = f"{label} ({host_name})"
                items.append((f"loopback:{idx}", label, f"WASAPI loopback — capture {out_ch} ch from this output"))
    except Exception:
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
    """bpy.app.timers callback: refresh device list every 2 seconds."""
    _refresh_live_device_cache()
    return _LIVE_DEVICE_CACHE_INTERVAL


def _live_device_items(self, context):  # noqa: ARG001
    """Blender dynamic-enum callback: return cached live capture devices."""
    if not _LIVE_DEVICE_CACHE:
        _refresh_live_device_cache()  # first draw only
    return _LIVE_DEVICE_CACHE

# ==============================================================================
# 0. UI NODE DATA
# ==============================================================================

class ACEUINode(bpy.types.PropertyGroup):
    """Lightweight UI representation of a node linked to an audio source."""
    name: bpy.props.StringProperty(name="Name", default="Node")                                                                      # type: ignore
    node_id: bpy.props.StringProperty(name="Node ID", default="")                                                                    # type: ignore
    node_type: bpy.props.StringProperty(name="Type", default="")                                                                     # type: ignore
    current_value: bpy.props.FloatProperty(name="Value", default=0.0)                                                                # type: ignore
    description: bpy.props.StringProperty(name="Description", default="")                                                            # type: ignore

# ==============================================================================
# 1. AUDIO SOURCE DATA
# ==============================================================================

class ACEAudioSource(bpy.types.PropertyGroup):
    """Represents a single audio source in the project."""
    name: bpy.props.StringProperty(name="Name", default="New Source")# type: ignore
    
    # --- CONFIGURATION ---
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
    
    # --- PROXY LINKING ---
    proxy_object_name: bpy.props.StringProperty(                                                                                    # type: ignore
        name="Proxy Object",
        description="Name of the Blender Object acting as the data container"
    )
    
    # --- ANALYSIS METADATA (Read-Only usually) ---
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
    
    # --- LIVE SETTINGS ---
    live_latency_ms: bpy.props.IntProperty(                                                                                         # type: ignore
        name="Live Latency",
        default=10,
        min=1, max=1000,
        description="Desired latency buffer for live capture (ms)"
    )

    # --- PREPROCESSING SMOOTHING SETTINGS ---
    # These are fed into the Engine to pre-process the source signal
    enable_preprocessing: bpy.props.BoolProperty(                                                                                   # type: ignore
        name="Enable Preprocessing",
        description="Apply source-side HPF/LPF filtering and signed attack/release smoothing before analysis",
        default=False,
    )
    envelope_attack: bpy.props.FloatProperty(                                                                                       # type: ignore
        name="Attack", 
        description="Smoothing time for rising signal magnitude (ms)", 
        default=10.0, min=0.0, unit='TIME'
    )
    envelope_release: bpy.props.FloatProperty(                                                                                      # type: ignore
        name="Release", 
        description="Smoothing time for falling signal magnitude (ms)", 
        default=100.0, min=0.0, unit='TIME'
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
    tempo_meter_panel_open: bpy.props.BoolProperty(                                                                                 # type: ignore
        name="Tempo & Meter Expanded",
        description="UI foldout state for Tempo & Meter settings.",
        default=False,
        options={'SKIP_SAVE'},
    )

    # --- LIVE STATUS ---
    live_status: bpy.props.StringProperty(                                                                                          # type: ignore
        name="Live Status",
        description="Status message for live inputs (warning when unavailable).",
        default=""
    )

    # --- FILE ERROR ---
    ace_source_error: bpy.props.StringProperty(                                                                                     # type: ignore
        name="Source Error",
        description=(
            "Non-empty when the selected file cannot be opened by the engine "
            "(unsupported format, missing file, or corrupt data)."
        ),
        default="",
    )

    # --- UI NODE LIST ---
    ui_nodes: bpy.props.CollectionProperty(type=ACEUINode)                                                                          # type: ignore
    active_ui_node_index: bpy.props.IntProperty(default=0)                                                                          # type: ignore

# ==============================================================================
# 2. ENGINE SETTINGS
# ==============================================================================

class ACEEngineSettings(bpy.types.PropertyGroup):
    """Global configuration for the Audio Engine."""

    auto_compile: bpy.props.BoolProperty(                                                                                           # type: ignore
        name="Auto-Compile",
        default=True,
        description="Engine adapts in real-time to modifications made in the n-panel or in the GN editor."
    )
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

# ==============================================================================
# 3. PROJECT SETTINGS
# ==============================================================================

class ACEProjectSettings(bpy.types.PropertyGroup):
    """Root Settings container for the Add-on."""
    
    # Collections
    audio_sources: bpy.props.CollectionProperty(type=ACEAudioSource)                                                                # type: ignore
    active_audio_source_index: bpy.props.IntProperty(                                                                               # type: ignore
        default=0,
        update=_on_active_source_index_change,
    )
    
    # Sub-Properties
    engine_settings: bpy.props.PointerProperty(type=ACEEngineSettings)                                                              # type: ignore
    gn_target_object: bpy.props.PointerProperty(type=bpy.types.Object, name="GN Target Object")                                     # type: ignore
    
    # Legacy project-level mode retained for backward compatibility with
    # older .blend data. Source type is now chosen per source.
    mode: bpy.props.EnumProperty(                                                                                                   # type: ignore
        name="Mode",
        items=[
            ('FILE', "File Playback", "Read from disk (Animation Sync)"),
            ('LIVE', "Live Stream", "Real-time audio capture"),
        ],
        default='FILE'
    )

# ==============================================================================
# 4. REGISTRATION
# ==============================================================================

classes = (
    ACEUINode,
    ACEAudioSource,
    ACEEngineSettings,
    ACEProjectSettings,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Attach to Scene
    bpy.types.Scene.audiocore = bpy.props.PointerProperty(type=ACEProjectSettings)

def unregister():
    if hasattr(bpy.types.Scene, "audiocore"):
        del bpy.types.Scene.audiocore
        
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
