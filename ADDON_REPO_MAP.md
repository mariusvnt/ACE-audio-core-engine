# AudioCore Engine Repo Map

This report summarizes the meaningful tracked repository files that were explored with dedicated sub-agents. Generated cache files under `__pycache__/` were intentionally excluded because they do not describe addon behavior.

## High-Level Function

AudioCore Engine is a Blender 5.0+ add-on that converts audio sources into per-frame numeric outputs for Geometry Nodes and object-bound shader materials.

The design is split into three runtime layers:

1. Layer 1: Blender-facing frontends.
   Geometry Node groups and Shader node groups act as interface contracts. They declare routing and output intent, but they do not perform DSP.
2. Layer 2: Backend analysis engine.
   The add-on compiles frontends into abstract graphs, then into per-source runtime task graphs that read audio, derive spectral or temporal features, and produce scalar or vector outputs.
3. Layer 3: Optional preview playback.
   When a preview sink exists, ACE builds a separate time-domain preview path for audible monitoring. It is deliberately isolated from the spectral-analysis pipeline.

Runtime flow:

1. Blender loads `__init__.py`, which registers properties, core modules, node factories, UI, handlers, and timers.
2. `properties.py` stores scene-level sources and engine settings under `Scene.audiocore`.
3. `nodes/dummy_tree.py` and `nodes/shader_tree.py` build ACE node groups and read user-authored frontends into `ACEAbstractTree`.
4. `core/audio_engine.py` compiles the abstract tree into runtime audio graphs.
5. `core/audio_influx.py` reads file or live audio.
6. `core/audio_tasks_registry.py` defines the DSP tasks executed per graph.
7. `core/parallel_dsp.py` evaluates eligible work in bpy-free worker processes.
8. `core/proxy_management.py` writes computed results back to Blender attributes and object custom properties.
9. Geometry Nodes and shader materials consume those values passively.

## File Map

### Tooling, Docs, Asset

- `.claude/settings.json`: Claude Code permission config for repo-local inspection and test commands.
- `.claude/settings.local.json`: Local Claude permission overrides for targeted test and shell actions.
- `.github/workflows/ci.yml`: GitHub Actions workflow running the repo test suite in CI.
- `.vscode/settings.json`: VS Code Python test discovery settings for this workspace.
- `ACE_Logo.png`: UI branding asset displayed in the addon sidebar.
- `AI_Code_Review.md`: Internal review prompt/spec for architecture and quality audits.
- `ARCHITECTURE.md`: Formal technical architecture document describing runtime model and subsystem boundaries.
- `README.md`: Developer onboarding brief explaining the 3-layer model and execution pipeline.

### Bootstrap and Shared Definitions

- `__init__.py`: Blender addon entrypoint; bootstraps dependency paths, imports modules in stable order, and registers/unregisters the addon.
- `vars.py`: Central registry for node definitions, dependency metadata, collection names, proxy prefixes, and other constants.
- `utils.py`: Shared helpers for logging, main-thread assertions, vendor-path setup, live-flow ID composition, and logo/icon handling.
- `properties.py`: Defines `Scene.audiocore` property groups for sources, engine settings, and UI state plus update callbacks that trigger recompiles.

### Core Backend

- `core/__init__.py`: Package marker for backend modules.
- `core/abstract_tree.py`: Defines the abstract IR used between frontend tree readers and the runtime compiler.
- `core/audio_engine.py`: Main orchestrator for compile, frame execution, worker offload, preview monitoring, handlers, and scene-scoped engine lifetime.
- `core/audio_influx.py`: Unified audio input boundary for file playback and live capture routes.
- `core/audio_tasks_registry.py`: DSP registry containing `AudioChunk`, task classes, runtime graph logic, and preview-sink processing.
- `core/id_management.py`: Deterministic naming and ID generation for nodes, sources, proxies, and output attributes.
- `core/parallel_dsp.py`: bpy-free worker-side execution of serialized graph IR and shared-memory payloads.
- `core/proxy_management.py`: Stage-C commit bridge from DSP outputs to Blender proxy attributes and object properties.

### Node Frontends

- `nodes/__init__.py`: Package marker for node frontend modules.
- `nodes/dummy_tree.py`: Geometry Nodes frontend factory and reader; builds hidden `.ACE_*` node groups and parses GN trees into abstract graphs.
- `nodes/dynamic_audio_operators.py`: Maintains dynamic splitter and audio-math node-group variants for Geometry and Shader trees.
- `nodes/shader_tree.py`: Shader frontend factory and reader; builds hidden `.ACE_SHADER_*` groups and maps shader outputs to property readers.

### UI Layer

- `ui/__init__.py`: Package marker for UI modules.
- `ui/gn_menu.py`: Blender Node Editor add-menu integration and node insertion operator.
- `ui/n_panel_operators.py`: Sidebar operator actions for sources, preview, splitter/audio-math controls, and engine-affecting commands.
- `ui/n_panel_ui.py`: Main 3D View sidebar UI, active-node presentation, metadata display, and panel layout.
- `ui/prefs_library_manager.py`: Install/uninstall operators for optional Python dependencies in the addon vendor path.
- `ui/prefs_panel_ui.py`: Addon preferences panel for dependency state and debug toggles.

### Test Artifacts and Reports

- `tests/artifacts/engine_perf_current.json`: Generated current engine performance report used by the quality gate.
- `tests/artifacts/perf_baseline_current.json`: Generated current synthetic perf baseline artifact.
- `tests/artifacts/quality_report.json`: Generated machine-readable summary of test, perf, and integration gate status.
- `tests/artifacts/quality_report.md`: Generated human-readable quality report.
- `tests/engine_perf_report.json`: Checked-in engine performance evidence file and comparison baseline.
- `tests/perf_baseline_report.json`: Checked-in synthetic performance baseline and contract fixture.

### Test Harnesses and Shared Test Infrastructure

- `tests/blender_integration_smoke.py`: Real-Blender smoke script that verifies addon enablement, handlers, operators, and frontend discovery.
- `tests/engine_perf_harness.py`: Full engine benchmark harness comparing threaded IO against process-based DSP offload.
- `tests/module_loader.py`: Synthetic package loader that lets tests import addon modules outside Blender.
- `tests/perf_harness.py`: Lightweight synthetic task-stack performance harness.
- `tests/run_quality_gate.py`: Test orchestration and report generator that combines integrity, performance, and integration results.

### Targeted Tests

- `tests/test_ambiguous_compile_pruning.py`: Guards mixed-source graph partitioning and compile pruning contracts.
- `tests/test_audio_influx.py`: Low-level byte-decoding tests for `AudioReception`.
- `tests/test_audio_influx_backends.py`: Live-route parsing and backend-dispatch tests for audio input.
- `tests/test_audio_tasks_registry.py`: Main behavior suite for DSP tasks, metadata propagation, utilities, and preview processing.
- `tests/test_audio_tasks_registry_smoke.py`: Minimal import-path smoke test for the task registry with a fake NumPy surface.
- `tests/test_blender_integration_smoke.py`: Thin unittest wrapper that invokes the Blender smoke script when `bpy` is available.
- `tests/test_ci_env_requirements.py`: CI-only sanity check that NumPy is installed when CI is enabled.
- `tests/test_engine_perf_gate.py`: Validates the engine performance report structure and strict thresholds.
- `tests/test_id_management.py`: Verifies deterministic node/source/proxy/attribute naming behavior.
- `tests/test_multi_frontend_compile.py`: Large regression suite for multi-frontend compile behavior, signatures, sink bindings, and event handling.
- `tests/test_parallel_dsp.py`: Contract tests for worker-side graph execution and payload handling.
- `tests/test_parallel_offload_policy.py`: Guards which nodes are eligible for worker offload.
- `tests/test_perf_gate.py`: Validates the lightweight perf baseline report and strict perf thresholds.
- `tests/test_preferences_panel_ui.py`: Protects addon preference lookup and `bl_idname` binding.
- `tests/test_proxy_management.py`: Tests proxy sink layout, redundant write suppression, zero fallback, and main-thread guardrails.
- `tests/test_regression_findings.py`: Broad source-level regression suite covering many previously fixed structural behaviors.
- `tests/test_source_sample_rate_metadata.py`: Verifies compile planning prefers stored native sample-rate metadata.

### Miscellaneous

- `tmp_check.py`: Ad hoc troubleshooting script for import-path behavior and proxy-manager symbol checks; not part of runtime or normal tests.

## Detailed Addon Functioning

### 1. Addon Registration and Scene State

Blender loads `__init__.py`, which ensures vendored dependency paths are importable, then registers the addon in a deliberate order:

- constants and helpers
- scene properties
- backend modules
- node factories/readers
- UI

`properties.py` then installs `Scene.audiocore`, which becomes the shared state container for:

- audio source definitions
- live/file routing choices
- BPM and meter metadata
- engine options and UI state

Any relevant property update can request a recompile through `ACEAudioEngine`.

### 2. Frontend Authoring Model

ACE’s visible nodes are not the execution engine.

Instead:

- `vars.py` declares the node taxonomy and socket contracts.
- `nodes/dummy_tree.py` builds hidden Geometry Node group datablocks from that registry.
- `nodes/shader_tree.py` builds equivalent hidden Shader node groups.
- `nodes/dynamic_audio_operators.py` expands dynamic node variants for splitters and audio-math operators.
- `ui/gn_menu.py` inserts those groups into Blender node trees.

When ACE needs to compile the scene, the frontend readers walk top-level GN and Shader trees and emit a merged `ACEAbstractTree` that describes:

- nodes and their ACE IDs
- links
- frontend ownership
- output intent
- consumer bindings

This is the critical architectural split: node trees declare analysis intent, but analysis happens elsewhere.

### 3. Compilation Pipeline

`core/audio_engine.py` performs compilation in Stage A on the main thread.

Key compile responsibilities:

- discover all ACE-managed frontends in the scene
- repair or assign stable frontend IDs
- parse frontends into abstract graphs
- merge per-frontend fragments into one scene-scoped IR
- partition per-source and mixed-source graph islands
- convert the abstract graph into executable runtime task graphs
- serialize eligible work into compact IR for process offload

`core/abstract_tree.py` is the contract that lets this happen cleanly across modules.

### 4. Audio Ingestion

`core/audio_influx.py` abstracts away where audio comes from.

It supports:

- file sources
- sounddevice live capture
- UDP routes
- DAW/instrument route tokens

It also owns format probing, PCM decode helpers, and resampling support used during source preparation and some proxy/bake flows.

### 5. DSP Runtime

`core/audio_tasks_registry.py` is the backend task vocabulary.

It defines:

- source tasks
- metadata/spec tasks
- delta/splitter tasks
- stream/output tasks
- utility tasks
- preview sink tasks
- runtime graph/container primitives

Important behavior:

- delta nodes usually propagate metadata constraints rather than duplicating audio buffers
- stream nodes terminate branches and emit values, not audio
- preview nodes are isolated into a time-domain monitor path
- optional acceleration comes from `audioflux`, `numba`, and `scipy` when available

### 6. Parallelism and Blender Safety

ACE keeps Blender-facing work on the main thread and pushes eligible DSP work to worker processes.

- `core/parallel_dsp.py` evaluates serialized graph IR in bpy-free worker contexts
- `core/audio_engine.py` manages dispatch, future polling, stale-result reuse, and fallback behavior
- `utils.assert_main_thread()` and related tests guard the rule that Blender data access remains main-thread only

This gives the addon a clean concurrency boundary:

- main thread for scene discovery, handler integration, and commits
- subprocess workers for DSP
- daemon threads for preview audio IO and some support tasks

### 7. Output Commit Path

Computed outputs must become Blender-readable values.

`core/proxy_management.py` handles that by:

- creating or finding hidden proxy objects/meshes
- defining output sinks and schemas
- writing point-domain attributes for Geometry Nodes consumers
- writing deterministic object properties for shader consumers

`core/id_management.py` guarantees naming stability so the same logical output maps to the same compact identifier across sessions and reloads.

### 8. UI and User Control Surface

The sidebar UI is implemented by:

- `ui/n_panel_ui.py` for presentation and status views
- `ui/n_panel_operators.py` for actions
- `ui/prefs_panel_ui.py` and `ui/prefs_library_manager.py` for dependency management and preferences

The UI does not own the engine logic. It mainly:

- edits `Scene.audiocore`
- triggers operators
- shows active source/node status
- reflects runtime or metadata state

### 9. Repository Testing Strategy

The test suite mirrors the same architecture:

- ingestion tests for `audio_influx`
- DSP and task tests for `audio_tasks_registry`
- worker tests for `parallel_dsp`
- compile/front-end tests for `audio_engine`, `dummy_tree`, and `shader_tree`
- commit-path tests for `proxy_management`
- UI/package tests for preferences and menu/sidebar wiring
- performance harnesses and gate tests
- large source-level regression guards in `test_regression_findings.py`

That structure reinforces the intended separation of concerns in the product code.

## Structural Summary

The repository is organized around a clear pipeline:

1. Registry and settings define what nodes and sources exist.
2. Node factories expose that vocabulary in Blender.
3. Frontend readers translate user-authored node trees into a neutral graph.
4. The engine compiles that graph into runtime tasks.
5. Audio input feeds the tasks.
6. Workers optionally execute heavy DSP.
7. Proxy/property writers commit results back into Blender.
8. Geometry Nodes and shader materials consume those outputs passively.
9. Optional preview audio runs beside, not inside, the analysis pipeline.

That is the addon’s central design idea: Blender frontends express intent, the backend engine computes, and the write bridge publishes results.
