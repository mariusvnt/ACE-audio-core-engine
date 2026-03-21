ACE Architecture & Developer Onboarding Brief

1. Executive Summary & Philosophy

Mission: AudioCore Engine (ACE) converts continuous audio streams — file playback or live microphone/device input — into per-frame numeric outputs consumed by Blender 5.0+ Geometry Nodes and object-bound shader materials, enabling fully procedural, audio-reactive animation without any keyframes.

Core philosophy: Complete decoupling across three independent concerns.

The UI (Layer 1) expresses intent only. No GN node performs a calculation. The visible node tree is a configuration contract, not an execution graph.

The DSP backend (Layer 2) performs all spectral and temporal analysis in isolation from the Blender API, writing scalar/vector results into proxy mesh attributes and object custom properties that Blender frontends read passively.

The audio preview (Layer 3) is a strictly time-domain, lazily-evaluated playback path that only activates when the user explicitly routes audio to a Preview or Group Output socket. It exists independently of Layer 2 and shares no frequency-domain machinery with it.

The design consequence: the three layers can be reasoned about, tested, and extended independently. A bug in the IIR preview filter does not affect RMS output. A missing audioflux feature does not silence the audio preview. This separation is not a preference — it is enforced at the code boundary level and verified by the test suite.

2. The 3-Layer Architectural Model

Layer 1 — The Facade (UI & Routing)

Scene frontends are pure interface contracts. In V1 that means top-level Geometry Nodes modifier trees plus top-level object-bound material Shader node trees. Every ACE node the user places belongs to a category defined in ACE_NODES (vars.py) and maps to a pre-built frontend group datablock managed by ACEDummyNodeFactory in dummy_tree.py or ACEShaderNodeFactory in shader_tree.py.

Nodes have zero internal calculation logic. Their GN group internals are stubs — sockets pass data through without modification.

The sole purpose of a Layer 1 node is to declare what analysis should happen and which audio branch carries the signal, by virtue of its type, its position in the tree, and the sockets it exposes.

Each frontend reader walks one top-level Blender node tree, reads frontend-scoped ace_id properties on nodes, resolves their category/type, and emits a partial ACEAbstractTree. Stage A then merges all discovered frontends into one scene-scoped ACEAbstractTree, including frontend metadata and consumer intents.

Geometry Nodes read back proxy mesh attributes written by Layer 2. Shader frontends read object custom properties via Blender's Attribute node in Object mode. Neither frontend participates in computing the values.

Practical implication for developers: You will never find feature extraction code, signal processing, or numpy operations inside anything managed by dummy_tree.py. If you do, that is a Layer violation.



Layer 2 — The Data Engine (Backend DSP & Attributes)

Layer 2 is ACE's analysis core. It is the only layer with direct access to audio data as a numeric signal and is the only layer allowed to perform frequency-domain analysis.

audio_engine.py compiles the ACEAbstractTree into one ACEAudioGraph per source. Each graph is a sorted task DAG (ACEAudioTask subclasses from audio_tasks_registry.py).

AudioFluxAnalyzer wraps the audioflux BFT engine. It maintains a cross-instance _GLOBAL_SPECTRUM_CACHE keyed on (chunk_token, feature_name, frame_index) so that multiple downstream tasks sharing an audio branch pay the BFT cost only once per frame.

Each AudioChunk carries not just audio samples but branch metadata: band constraints (band_indices, band_hz_bounds), lineage tokens, and analysis_ops flags. Delta nodes propagate these metadata fields rather than duplicating the audio signal — keeping branch expansion O(metadata) rather than O(samples).

Terminal Stream tasks (RMSTask, PitchTask, SpectralCentroidTask, OnsetTask, etc.) consume the chunk's metadata and the cached spectrum to emit a single float or vector.

Results are committed in Stage C by proxy_management.py, which writes FLOAT / FLOAT_VECTOR / FLOAT_COLOR outputs to either point-domain attributes on single-vertex proxy meshes or deterministic object custom properties. Attribute and property names are generated through the same compact hashing path in id_management.py.

Practical implication for developers: Layer 2 only cares about extracting numbers. There is no audio output here. The final AudioChunk at a Stream node's output has no audio consumer.

Layer 3 — The Acoustic Monitor (Time-Domain Preview Only)

Layer 3 activates conditionally and lazily: only when the compiled abstract graph contains an audio link that terminates at a Preview node or the Group Output audio socket.

The critical constraint — no frequency-domain reconstruction:

Layer 3 must never attempt an inverse BFT or iFFT to recover audio from spectral data. The reasons are latency (iFFT adds 1–2 block windows of delay at minimum), phase artefacts, and the inherent mismatch between BFT analysis windows and real-time block sizing.

Instead, ACE performs a symbolic translation of the abstract tree into a time-domain filter chain:

A frequency_splitter (BFT bin mask in Layer 2) becomes a biquad bandpass IIR filter in Layer 3.

A midside_splitter becomes M/S matrix operations on the stereo signal.

Source-level HP/LP filters (Layer 2 preprocessing) become their biquad equivalents.

The canonical IIR formulation implemented in Layer 3 is:

$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{j=1}^{N} a_j y[n-j]$$

where $b_k$ are feedforward (FIR) coefficients and $a_j$ are feedback (IIR) coefficients. In practice ACE uses second-order sections (biquads, $N=M=2$) chained via _PreviewSinkTask._filter_section_channel().

Three-tier acceleration for biquad execution (_run_biquad() in audio_tasks_registry.py):

Tier 1: Numba JIT _biquad_filter_block — active when Numba is available and the "High quality audio preview" toggle is on.

Tier 2: scipy.signal.lfilter — active when Numba is absent or disabled.

Tier 3: Pure Python loop — last resort, emits a one-time module-level WARNING via _BIQUAD_PYTHON_FALLBACK_WARNED.

_ACEPreviewAudioMonitor runs in a daemon thread (multithreading, I/O category), buffers processed blocks, and emits them to sounddevice. It is the only component in ACE that writes audio to an output device.

Optimization: When an audio branch connects directly to the Group Output with no intermediate Delta or Utility nodes, PreviewSinkTask detects the passthrough path and routes raw source samples without any DSP.

3. Node Taxonomy & Graph Logic

ACE defines four node categories. The table below summarizes how each translates between layers.

CategoryIdentifierLayer 2 BehaviorLayer 3 BehaviorSourcesource_mngmtRoot of every task graph; provides AudioChunk from AudioReceptionRaw audio stream origin for preview chainDeltadeltaPropagates metadata constraints (band, mode, mask) — no signal copyBecomes a time-domain filter (IIR bandpass, M/S matrix, A/R envelope)StreamstreamTerminates branch; emits float/vector attributeNot applicable — no audio outputUtilitiesutilitiesSpecs, latency, cross-source mixing, preview routingpreview_sink is the sole Layer 3 sink; Audio Math merges branches

Source nodes are automatically generated — one per ACEAudioSource entry. Users do not place them manually. They are hidden from the GN Add menu.

Delta nodes are the key to ACE's efficiency. A FrequencySplitter does not split the audio signal into N copied buffers. It emits N AudioChunk objects that differ only in their band_indices field. Downstream Stream tasks read that field when querying the spectrum cache, extracting the slice of the BFT result that corresponds to the assigned band — zero extra FFTs.

Stream nodes are terminal. RMSTask, ZeroCrossTask, PitchTask, SpectralCentroidTask, SpectralFluxTask, OnsetTask, StereoWidthTask, StereoBalanceTask all inherit ACEAudioTask and implement process(chunk) -> float | np.ndarray. The return value is written to the proxy attribute. None of these return an AudioChunk.

Utility nodes span both layers. src_specs (SourceSpecsTask) exposes frames-per-beat and beat-phase attributes while passing the audio chunk downstream. audio_math (AudioMathTask) is the one node that explicitly crosses source-branch boundaries — it synchronizes two distinct root-source branches and must handle mismatched block sizes.

4. The Parallel Execution Pipeline & Concurrency Strategy

ACE uses a strict three-stage model to keep Blender's main thread unblocked during DSP.



Stage A — Main-Thread Graph Setup

Owner: audio_engine.py — compile_scene_frontends(), invoked from main thread only (enforced by utils.assert_main_thread()).

Discovers all managed scene frontends (Geometry Nodes modifiers and object-bound shader materials).

Discovers user-authored ACE source groups from visible editors and scene frontends without creating placeholder nodes.

Reads each frontend via ACEDummyTreeReader or ACEShaderTreeReader.

Synchronizes auto-generated sink IDs per frontend and refreshes abstract parameters.

Merges partial frontend IR into one scene-scoped ACEAbstractTree.

Compiles source-scoped audio graphs plus mixed-source subgraphs when a branch explicitly merges multiple roots.

Serializes each graph into a compact parallel IR (integer/float dictionaries, no Python objects) via _prepare_parallel_graph_ir().

Resolves concrete sink bindings so each computed attr_key knows whether it writes to proxy attributes, object properties, or both.

Stage A only runs when the scene frontend signature changes. Between compiles it is a no-op. The signature is a blake2b digest of node topology, parameter values, frontend metadata, and consumer intents (_compute_tree_signature()).



Stage B — Worker Compute (DSP)

Owner: parallel_dsp.py worker processes — launched via multiprocessing spawn context.

Receives compact IR + shared-memory audio block references via IPC.

Executes the full task DAG: feature extraction, spectral analysis, output computation.

Zero bpy calls are permitted. Workers are validated to be bpy-free at the module level. Any attempt to import bpy inside a worker process is a critical violation.

Results are returned as flat numeric arrays via shared memory (SharedMemory).

In-flight count per source is capped at _PARALLEL_MAX_INFLIGHT_PER_SOURCE = 2 (enforced by _parallel_inflight_limit()). Stale jobs older than _PARALLEL_STALE_REUSE_MS = 66 ms are dropped and the last valid value is reused.

_parallel_poll_futures() drains completed futures. A fallback 2-second drain timer (_shm_drain_timer) fires unconditionally to prevent SHM leaks if no frame events arrive (e.g., during render or pause).

Stage C — Main-Thread Commit (Sink Dispatch)

Owner: audio_engine.py / proxy_management.py — execute_next_frame(), main thread only.

Reads completed future results from Stage B.

Calls proxy_management to fan out float/vector/color values to the named point-domain attributes on source proxy meshes and to deterministic object custom properties for shader consumers.

Stage C checks ``scene.render.use_lock_interface`` before any sink-layout or commit write. During a render job with Lock Interface disabled, ACE still runs Stage A/B evaluation but defers Stage C writes until the render path is safe again.

_attr_schema_dirty dirty flag triggers attribute schema resync when node topology has changed since last commit (replaces the old 120-frame polling approach).

Quick Reference — Threading/Processing Environment

StageThread Environmentbpy AccessTriggerStage A — Graph CompileMain threadAllowed (required)Tree signature changeStage B — Worker DSPMultiprocessing (spawn)ForbiddenPer-frame, async dispatchStage C — Attribute WriteMain threadAllowed (required)Future completion in execute_next_frame()Audio I/O (influx)Multithreading (daemon)Forbiddensounddevice callbackPreview MonitorMultithreading (daemon)ForbiddenTimeline playbackProbe (metadata)Multithreading (daemon)ForbiddenCache miss on file sourceAuto-compile timerMain thread (Blender timer)AllowedRegistered via bpy.app.timers5. Strict Guardrails & Non-Negotiables

Rule 1 — bpy is Main-Thread Only

bpy.data, bpy.context, node trees, mesh attributes, object properties, and all handler registration are forbidden outside the main thread. This is not a Blender recommendation — it is a hard crash boundary; accessing bpy from a worker process will either segfault or corrupt Blender's internal state.

utils.assert_main_thread() is called at the entry of every Stage A/C function. In debug strict mode (utils.set_debug_strict(True)), violations raise RuntimeError. In normal mode they emit a rate-limited ERROR log.

Blender's use_lock_interface is enforced as a Stage C gate: commit/layout writes are skipped or deferred when a render job is active and Lock Interface is disabled.

Worker processes in parallel_dsp.py are spawn-context children — they do not inherit the parent's bpy module state.

Rule 2 — Layer 3 Isolation & Conditional Activation

Layer 3 only activates when _compile_preview_graph() (in audio_engine.py) detects at least one audio link terminating at a Preview node or Group Output audio socket in the compiled abstract graph.

PreviewSinkTask.process() sets _LAYER3_THREAD_LOCAL.in_preview = True via try/finally before delegating to _process_impl().

_masked_mono_signal() (a Layer 2 utility) checks this thread-local flag and returns silence with an ERROR log if called from within the preview path, preventing Layer 2 internals from leaking into Layer 3 execution.

_ACEPreviewAudioMonitor runs in a daemon thread entirely separate from the Stage B worker pool. It has no access to the task DAG and cannot trigger a compile.

Rule 3 — Time-Domain Previews Only (No iFFT)

Any code that attempts to recover audio from BFT/FFT output (inverse transform, overlap-add reconstruction, phase vocoder, etc.) inside PreviewSinkTask or _ACEPreviewAudioMonitor is an architectural violation.

The module-level _run_biquad() function is the sanctioned DSP primitive for Layer 3 filtering. Its three-tier fallback chain (Numba → SciPy → Python loop) provides graceful degradation without any frequency-domain intermediate.

reset_biquad_fallback_warned() is called in shutdown_all_instances() to clear the one-time warning flag between engine sessions.

The _apply_preview_gain() function enforces a channels-first (2, N) stereo layout and uses np.minimum.accumulate for the instant-attack limiter, avoiding any block-by-block Python loop in the audio callback hot path.

Rule 4 — Realtime Scheduling & Render Safety

ACE must not stall Blender's main thread or corrupt a render pass.

Stage B job queues use maxlen=2 (deque). When a new block arrives and the queue is full, the oldest entry is discarded. This ensures the system never accumulates backpressure that would freeze playback.

Stale results older than _PARALLEL_STALE_REUSE_MS = 66 ms are reused from cache rather than blocking on a slow worker. The threshold is FPS-invariant (milliseconds, not frames).

_parallel_restart_count persists for the engine's lifetime — it is never reset at compile time. Restarts are limited to _PARALLEL_RESTART_LIMIT = 3 with exponential backoff (0.5, 1.0, 2.0) seconds via _parallel_restart_backoff_until.

During F12 renders, Blender's use_lock_interface is active. Stage C checks this flag before writing attributes and skips or defers writes accordingly.

Rule 5 — Cross-Source Branch Synchronization

AudioMathTask is the sole node permitted to merge audio branches that originate from different root sources.

Single-root branches are still compiled per source, but any multi-root island is compiled into an explicit mixed graph that contains the ``audio_math`` node and all of its downstream dependents.

Block sizes may differ between sources (different sample rates or GPT-derived sizes). AudioMathTask.process() is responsible for resampling or zero-padding the shorter block to match.

The ``root_sources`` set propagated through the abstract graph is the mechanism that identifies cross-source dependencies at compile time and partitions mixed islands away from the per-source graphs.

LIVE sources have no auto-BPM detection. _refresh_specs_ir_bpm() patches scene_bpm, scene_beats_per_bar, and scene_fps into the parallel IR each frame so that user BPM changes propagate without requiring a full recompile.

Rule 6 — Deterministic IDs

Attribute names written to proxy meshes must be identical across Blender sessions, Reload Scripts cycles, and file saves/loads.

generate_compact_attr_name() in id_management.py produces a 24-bit (6 hex char) blake2b hash of node_id:socket_index. This namespace is large enough to make collisions negligible at typical graph sizes while remaining within Blender's 64-character attribute name limit.

_reserve_unique_ace_id() in dummy_tree.py writes bl_node["ace_id"] = new_id on collision, ensuring persistent deduplication across reload cycles.

All internal hashing (tree signature, content hash, probe cache) uses hashlib.blake2b with digest_size=4. hashlib.md5 is absent from the codebase.

Adding a New Node Type — Developer Checklist

Register schema in ACE_NODES (vars.py) — category, type key, socket definitions.

Add dummy group interface/internals in dummy_tree.py (or a dynamic variant manager).

Update _node_group_needs_refresh() in dummy_tree.py with the new type key.

Add abstract tree parser mapping in abstract_tree.py if the node requires special routing logic.

Implement a runtime task class inheriting ACEAudioTask in audio_tasks_registry.py.

Register the task creation mapping in audio_engine.py (_build_task_for_node()).

Add Layer 3 time-domain equivalent in PreviewSinkTask._build_filter_chain() if the node has a preview audio path.

Add UI controls and operators in n_panel_ui.py / n_panel_operators.py if the node exposes runtime-editable parameters.

Write regression tests covering output range, topology sync, and any cross-source behavior.
