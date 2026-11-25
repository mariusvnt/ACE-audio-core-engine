# AudioCore Engine Architecture

## 1. Document Control
Document title: AudioCore Engine Architecture  
Document type: Technical architecture specification  
Status: Active  
Primary language: Python (Blender add-on)  
Target host: Blender 5.0+  

## 2. Purpose and Scope
AudioCore Engine (ACE) is a Blender add-on that enables audio-reactive procedural animation by converting file or live audio streams into per-frame data outputs consumed by Geometry Nodes and object-bound shader materials.

This document specifies:
the runtime architecture
the data and control flow
the subsystem responsibilities
the extension and operational constraints

This document does not cover:
UI styling details
packaging/release process
external DAW integration protocol specs beyond current parser behavior

## 3. Architectural Overview
ACE uses a dual-graph model:
scene frontends (top-level Geometry Nodes modifier trees and object-bound Shader node trees) act as the user authoring surface
the backend task graph performs real DSP/analysis and writes results to proxy attributes or object custom properties

Core principle:
frontend node groups are primarily interface contracts; backend tasks execute analysis.

High-level flow:
1. User configures sources and ACE nodes.
2. ACE discovers all managed scene frontends and parses them into a merged abstract graph.
3. ACE compiles one runtime task graph per source.
4. Each frame, ACE reads audio blocks, computes outputs, and writes named attributes to source proxies or object properties.
5. Geometry Nodes and shader materials read those values and drive scene behavior.

## 4. System Context
External dependencies:
NumPy (core numeric path)
AudioFlux (primary spectral feature engine)
SoundFile (file decoding for non-WAV formats)
SoundDevice (live input and preview output)
Numba (optional acceleration for selected DSP paths)

Blender integration points:
`Scene.audiocore` pointer properties
`frame_change_pre`, `depsgraph_update_post`, `load_post` handlers
persistent timer-based auto-compile and UI synchronization
Geometry Nodes add menu and 3D View N-panel
object-bound material Shader node trees

## 5. Component Architecture
### 5.1 Bootstrap and Module Orchestration
Responsibilities:
module import/reload order
registration rollback on failure
dependency path bootstrap

Primary modules:
`__init__.py`
`utils.py`
`vars.py`

### 5.2 Configuration and Scene State
Responsibilities:
persistent scene-level settings
source definitions
engine behavior toggles

Primary modules:
`properties.py`

Key objects:
`ACEProjectSettings`
`ACEAudioSource`
`ACEEngineSettings`
`ACEUINode`

### 5.3 Identity and Naming
Responsibilities:
stable node/source IDs
deterministic proxy naming
compact attribute key generation under Blender limits

Primary modules:
`id_management.py`

### 5.4 Frontend Node Group Factories and Readers
Responsibilities:
build ACE Geometry Nodes and Shader node group datablocks
maintain dynamic group variants
translate Blender scene frontends to abstract graph fragments

Primary modules:
`dummy_tree.py`
`shader_tree.py`
`dynamic_audio_operators.py`
`abstract_tree.py`
`gn_menu.py`

### 5.5 Runtime Engine and Compilation
Responsibilities:
scene-scoped engine lifecycle
abstract graph compilation to task graphs
frame-by-frame execution
preview routing and push
bake cache generation and replay

Primary modules:
`audio_engine.py`: `ACEAudioEngine`, one per scene. It owns the runtime state
(graphs, sink writer, frame clock, undo state) and one component per concern;
the public methods the UI and handlers call are one-line delegates.
`engine_compiler.py` (`engine.compiler`): Stage A, frontends to per-source graphs.
`engine_frame.py` (`engine.frame`): Stage C, per-frame evaluation and sink commits.
`engine_parallel.py` (`engine.parallel`): Stage B, worker-process DSP offload.
`engine_frontends.py` (`engine.frontends`): frontend discovery and dirty tracking.
`engine_sinks.py` (`engine.sinks`): sink bindings and auto attribute names.
`engine_preview.py` (`engine.preview`): preview routing and the audio monitor.
`engine_bake.py` (`engine.bake`), `engine_probes.py` (`engine.probes`),
`engine_diagnostics.py` (`engine.diagnostics`).
`engine_handlers.py`: Blender handlers and timers. `engine_common.py`: shared imports and types.
`audio_tasks_registry.py`: the DSP tasks.

Components reach the runtime and each other through `self.engine`, never by
name strings, and never store themselves where the engine belongs
(`tests/test_engine_structure.py`).

### 5.6 Audio Input Adapters
Responsibilities:
file and live source ingestion
backend parsing and fallback behavior
block-oriented read API

Primary modules:
`audio_influx.py`

### 5.7 Proxy and Attribute Writer
Responsibilities:
proxy object lifecycle
sink schema synchronization
proxy attribute and object property write path
ACE_Sonar management

Primary modules:
`proxy_management.py`

### 5.8 UI and Operators
Responsibilities:
source and engine controls
graph-node controls
capture/preview tooling
dependency management UI

Primary modules:
`n_panel_ui.py`
`n_panel_operators.py`
`prefs_panel_ui.py`
`prefs_library_manager.py`

## 6. Data Model
### 6.1 Stable IDs
Every ACE node has an `ace_id` with category/type semantics and UUID suffix.
Every source has `stable_id`.
Source placeholder nodes keep `source_binding_id` / `ace_source_id` separate from their frontend-scoped `ace_id`.
Source proxy names are deterministic hashes of source identity.

### 6.2 Abstract Graph
`ACEAbstractTree` contains:
nodes (`ACEAbstractNode`)
links (`ACEAbstractLink`)
group-output audio links (`ACEAbstractOutputLink`)
frontends (`ACEFrontendRef`)
consumer intents (`ACEConsumerIntent`)

`root_sources` propagation on abstract nodes is used to partition runtime graphs per source.

### 6.3 Runtime Audio Unit
`AudioChunk` is the canonical branch payload:
mono/stereo samples
sample rate
latency metadata
lineage/spectrum token
band constraints (`band_indices`, `band_hz_bounds`, `band_index_fft_size`)
analysis operation metadata (`analysis_ops`)

### 6.4 Sink Contract
Outputs are written to concrete sink bindings. In V1 those are either proxy mesh point-domain attributes or object custom properties:
FLOAT
FLOAT_VECTOR
FLOAT_COLOR

Naming:
compact deterministic IDs by source + node + output socket
optional user override via Attribute ID sockets for supported nodes

## 7. Control Flow and Lifecycle
### 7.1 Registration
1. Submodules are imported/reloaded in deterministic order.
2. Classes/operators/panels are registered.
3. engine handlers and timers are attached.
4. default node groups and dynamic variants are ensured.

### 7.2 Source Lifecycle
1. Source created/imported.
2. Stable source ID assigned.
3. Single-vertex proxy mesh created or reused.
4. ACE_Sonar object/tree synchronized with managed source placeholders.
5. Engine recompilation requested/triggered.

### 7.3 Compilation Lifecycle
1. Managed scene frontends are discovered from Geometry Nodes modifiers and object-bound shader materials.
2. Geometry frontends synchronize managed source placeholders.
3. Each frontend is parsed into a partial abstract graph and local sink IDs are synchronized.
4. Partial frontend graphs are merged into one scene-scoped `ACEAbstractTree`.
5. One `ACEAudioGraph` is compiled per source with only required upstream tasks.
6. Each graph resolves concrete sink bindings for proxy attributes and object properties.

### 7.4 Frame Execution Lifecycle
1. Frame handler invokes engine execution.
2. Source tasks fetch the next block (file/live).
3. Downstream task graph computes outputs.
4. Output values are fanned out to proxy attributes and/or object properties on the main thread.
5. Preview chunks mixed and optionally emitted to output monitor during timeline playback.

### 7.5 Shutdown Lifecycle
1. Handlers and timers are removed.
2. Runtime graphs and readers are closed.
3. Preview monitor is closed.
4. scene-scoped engine instances are cleared.

## 8. Processing Model
### 8.1 Block Sizing
Block size is derived from source sample rate and scene FPS, clamped to safe bounds.
This keeps processing frame-locked for animation sync.

### 8.2 Source Preprocessing
Optional source-side processing:
high-pass and low-pass filtering
signed attack/release smoothing

### 8.3 Feature Extraction
Primary spectral/temporal features are computed through AudioFlux analyzers with cache reuse.
Fallbacks exist for selected cases when direct analyzer output is unavailable.

### 8.4 Delta Nodes
Delta tasks usually propagate analysis constraints instead of reconstructing audio branches:
frequency bands
mid/side modes
attack/release modes
chroma masks
auto-component masks

This design keeps branch expansion lightweight.

### 8.5 Stream Nodes
Stream tasks terminate branches with float/vector values (RMS, centroid, flux, onset, pitch, stereo descriptors).

### 8.6 Utility Nodes
`src_specs` exposes frames-per-beat and beat-phase metadata plus passthrough audio.
`audio_math` combines multiple branch audios.
`preview_sink` reconstructs preview audio for monitoring and supports optional accelerated paths.

## 9. Blender UI and UX Integration
### 9.1 GN Add Menu
ACE nodes are added via dedicated categories.
Internal source-management nodes are intentionally hidden from manual creation.

### 9.2 N-Panel
Provides:
project mode and engine controls
source CRUD and metadata
tempo/envelope controls
active graph node listing
capture controls for splitter/audio math/preview
bake controls
GN target selection

### 9.3 Preferences
Provides dependency status, install/uninstall operators, and debug logging toggle.

## 10. Auto-Compile and Synchronization Strategy
Auto-compile is driven by depsgraph updates + debounce windows.
Dirty trees are compiled after a quiet interval to avoid excessive rebuild churn.
UI sync uses timer-based refresh with throttling and source-scoped filtering.

## 11. Preview Audio Strategy
Preview output is:
derived from manual preview sinks, or auto-derived from Group Output Audio links when manual sinks are absent
mixed across contributing branches
emitted only while timeline playback is active
protected by limiter and queue freshness policy to avoid drift and clipping

## 12. Baking Strategy
File sources can be baked over scene frame range.
Baked values are cached in memory with invalidation on:
tree signature changes
block-size changes
source file path/mtime changes

At runtime, baked values override live computation when valid.

## 13. Reliability and Safety Mechanisms
scene-scoped engine instances to avoid cross-scene contamination
registration rollback on partial failure
render lock-interface safety guard for handler write operations
rate-limited runtime logging to avoid console flooding
graceful fallback paths for missing optional dependencies

### 13.1 Error handling
No error is absorbed silently.
Boundaries keep one broad catch and log it, marked `# noqa: BLE001 boundary: <reason>`:
Blender handlers and timers, panel `draw()`, operator `execute()`/`modal()`, thread loops,
the worker job entry, registration rollback, and imports of optional native packages.
Inside a boundary, a fallback that is the right outcome records the error:
`with utils.logged_errors("module.function"):` to carry on, or
`utils.note_swallowed("module.function", exc)` before a fallback value.
Recorded errors are counted in `utils.swallowed_errors()` and printed (throttled) with debug logging on.
Prefer a specific exception type, or `utils.as_int` / `utils.as_float` for value coercion, over a broad catch.
`tests/test_error_handling.py` fails on any broad handler that neither records, logs nor re-raises,
and the headless Blender run (`tests/run_in_blender.py`) fails if a normal session records any error.

## 14. Performance Design Notes
proxy attributes are cached and only updated when needed
task outputs are frame-cached per node
long-lived analyzer objects are pooled/reused
splitter/chroma/component tasks prioritize metadata propagation over full signal duplication
optional Numba acceleration is gated by capability and user settings

### 14.1 Threads vs worker processes (measured 2026-11-25)
Worker-process DSP (`processes_dsp`) is kept. Blender 5.0, one 24 fps file
source feeding N stream nodes, 240 frames at real-time pace, optional
libraries installed (`tests/blender_perf.py`):

| Nodes | p95 frame ms, threads | p95 frame ms, processes | Fresh worker values |
| --- | --- | --- | --- |
| 24 | 3.0 | 1.8 | 99.6% |
| 36 | 3.7 | 2.1 | 99.6% |
| 50 | 7.4 | 2.4 | 99.6% |

The rule set before measuring was: keep processes only if their p95 frame
time is at most 0.8x threads and at least 95% of frames get worker values.
Both modes stay far under the 41.7 ms frame budget. Costs of processes:
a 3.3-3.6 s cold start after the pool is created (local DSP covers it), and
with two job shards per frame (a frequency splitter feeding a stream node)
about half the jobs still miss the 66 ms stale-job budget; that cause is
not identified yet.

## 15. Testing Architecture
Test coverage includes:
ID generation and parsing invariants
audio input parser/backends behavior
task semantics and regression checks for splitters/preview/math/rolling-average/specs
Blender integration smoke tests for handlers/source lifecycle/ACE_Sonar synchronization

Tests outside Blender load the add-on modules against one shared fake `bpy`
(`tests/fake_bpy.py`); every test file also passes when run on its own.
Tests check behaviour, never source text.

Primary test modules:
`tests/test_invariants.py` (the README guardrails: bpy-free worker, time-domain preview, persistent handlers, render guard)
`tests/test_audio_tasks_registry.py`
`tests/test_audio_influx.py`
`tests/test_audio_influx_backends.py`
`tests/test_id_management.py`
`tests/test_multi_frontend_compile.py`
`tests/test_proxy_management.py`
`tests/run_in_blender.py` (headless Blender smoke run: `blender -b --factory-startup --python tests/run_in_blender.py`)

## 16. Extension Guidelines
When adding a new node type:
1. Register schema in `ACE_NODES` (`vars.py`).
2. Provide dummy group interface/internals (`dummy_tree.py` or dynamic variant manager).
3. Add parser mapping in abstract conversion path if needed.
4. Implement runtime task in `audio_tasks_registry.py`.
5. Register task creation mapping in engine (`audio_engine.py`).
6. Add UI controls/operators if node has runtime-editable parameters.
7. Add regression tests for output correctness and topology sync behavior.

## 17. Current Architectural Constraints
Audio processing is frame-block based, optimized for reactivity over exact offline fidelity.
Bake cache is in-memory and not persisted as external cache assets.
Some advanced spectral branches rely on AudioFlux availability for full fidelity.
Live backend robustness depends on source transport quality and optional dependencies.

## 18. Summary
AudioCore Engine is architected as a deterministic, source-partitioned backend analysis engine driven by a user-authored GN facade. The design prioritizes:
stable IDs and repeatable mapping
low-overhead branch metadata propagation
frame-synchronous attribute writes into proxy geometry
tight Blender integration with guarded runtime behavior

This architecture enables responsive audio-reactive Geometry Nodes workflows while maintaining extensibility through the abstract graph and task registry model.
