AudioCore Engine (ACE)
======================

**AudioCore Engine (ACE)** is a Blender 5.0+ add-on designed to convert continuous audio streams---whether from file playback or live microphone/device input---into per-frame numeric outputs. These outputs are consumed by Geometry Nodes and object-bound shader materials, enabling fully procedural, audio-reactive animation without the need for keyframes.

🏛️ Architectural Philosophy: The 3-Layer Model
-----------------------------------------------

ACE is built on a strict philosophy of separation of concerns. The architecture is divided into three independent runtime layers:

#### Layer 1: The Facade (UI & Routing)

Blender-facing frontends (Geometry Node groups and Shader node groups) act purely as **interface contracts**. They declare routing, analysis intent, and output destinations, but they **do not** perform any DSP or mathematical calculation.

### Layer 2: The Data Engine (Backend DSP & Scene Compile)

The analysis core of the add-on. The engine compiles the frontends into abstract graphs, then into per-source runtime task graphs. It executes spectral and temporal analysis (using tools like `audioflux`, `numba`, and `scipy`) entirely isolated from the Blender API.

### Layer 3: The Acoustic Monitor (Time-Domain Preview)

An optional, lazily-evaluated preview playback system. When an audio preview sink exists, ACE builds a completely separate time-domain filter chain for audible monitoring. It is deliberately isolated from the Layer 2 spectral-analysis pipeline---**no frequency-domain reconstruction (iFFT/iBFT) is ever performed here.**

⚙️ Runtime Execution Flow
-------------------------

1. **Initialization:** Blender loads `__init__.py`, which registers properties, core modules, node factories, UI, handlers, and timers. Settings are stored in `Scene.audiocore`.

2. **Authoring:** Users place ACE nodes in Geometry or Shader trees. `nodes/dummy_tree.py` and `nodes/shader_tree.py` parse these into an `ACEAbstractTree`.

3. **Compilation (Stage A):** `core/audio_engine.py` (Main Thread) compiles the abstract tree into executable runtime audio graphs.

4. **Ingestion:** `core/audio_influx.py` captures and routes file or live audio data.

5. **Execution (Stage B):** `core/parallel_dsp.py` offloads heavy DSP tasks to `bpy`-free worker processes to prevent UI stalling.

6. **Commit (Stage C):** `core/proxy_management.py` (Main Thread) writes computed scalar/vector results back to Blender proxy attributes and object custom properties.

7. **Consumption:** Geometry Nodes and shader materials passively consume the updated numeric values to drive animations.

📂 Repository Structure
-----------------------

### Core Backend (`core/`)

- **`audio_engine.py`**: Main orchestrator for compiling, frame execution, worker offload, and scene-scoped engine lifetime.

- **`abstract_tree.py`**: Defines the abstract IR used between frontend readers and the runtime compiler.

- **`audio_tasks_registry.py`**: DSP registry defining tasks, delta/splitter behavior, stream outputs, and preview processing.

- **`parallel_dsp.py`**: `bpy`-free worker-side execution of serialized graph IR.

- **`proxy_management.py`**: Stage-C commit bridge writing DSP outputs to Blender proxy attributes and object properties.

- **`audio_influx.py`**: Unified audio input boundary (files, sounddevice live capture, UDP routes).

- **`id_management.py`**: Deterministic 24-bit `blake2b` naming generation for nodes, sources, and attributes.

### Node Frontends (`nodes/`)

- **`dummy_tree.py`**: Geometry Nodes frontend factory and tree reader.

- **`shader_tree.py`**: Shader frontend factory mapping outputs to property readers.

- **`dynamic_audio_operators.py`**: Maintains dynamic splitter and audio-math node-group variants.

### User Interface (`ui/`)

- **`n_panel_ui.py` & `n_panel_operators.py`**: Main 3D View sidebar UI and operator actions.

- **`gn_menu.py`**: Node Editor add-menu integration.

- **`prefs_panel_ui.py` & `prefs_library_manager.py`**: Preferences and vendor dependency management.

### Tests (`tests/`)

ACE maintains a rigorous testing structure mirroring the product architecture:

- **Unit & DSP Tests**: `test_audio_tasks_registry.py`, `test_audio_influx.py`

- **Concurrency & Execution Guards**: `test_parallel_dsp.py`, `test_parallel_offload_policy.py`

- **Integration & Regression**: `test_multi_frontend_compile.py`, `test_blender_integration_smoke.py`, `test_proxy_management.py`

- **Performance Harnesses**: Evaluates task-stack efficiency and threaded IO vs. process-based DSP.

🚧 Strict Development Guardrails
--------------------------------

If you are contributing to ACE, the following non-negotiable architectural rules apply:

1. **`bpy` Is Main-Thread Only:** `bpy.data`, `bpy.context`, node trees, and handler registrations are strictly forbidden outside the main thread. Worker processes in `parallel_dsp.py` must remain completely `bpy`-free.

2. **Render Safety:** Blender's `use_lock_interface` must be strictly respected during Stage C commits.

3. **No Layer Contamination:** Time-domain preview logic (Layer 3) must never attempt inverse BFT/FFT reconstruction. It relies strictly on time-domain biquad filters mapped from Layer 2 metadata.

4. **Deterministic Signatures:** Proxy attribute names and tree signatures must use `hashlib.blake2b` hashing. `ace_id` is frontend-scoped and unique per node instance.
