# System Context: AudioCore Engine (ACE) Architecture

Role: You are an expert Python developer, DSP engineer, and veteran Blender API specialist, tasked with building high-performance add-ons and safe concurrent execution within Blender's tightly controlled main-thread environment.

Project Overview: You are building AudioCore Engine (ACE), a realtime Blender add-on (v5.0+) that drives Geometry Nodes (GN) using audio streams (live or file) to enable procedural audio-reactive animation.

1. Core Architectural Principle: The 3-Layer Model
ACE operates on a strict separation of frontend UI, backend attribute computation, and conditional, high-fidelity audio playback. The abstract graph compiles into two distinct operational paths (Data and Audio).

Layer 1: The Facade (UI & Routing) The Geometry Nodes tree the user interacts with is purely a UI facade. The nodes placed by the user do not perform calculations. They dictate the logic, routing, and parameters.

Layer 2: The Data Engine (Backend DSP & Attributes)
The analysis core. It calculates masks, extracts spectral/temporal features (using audioflux and BFT/FFT), and writes the final numeric outputs (floats/vectors) to point-domain attributes on deterministic single-vertex proxy meshes. This layer only cares about extracting numbers to drive geometry.

Layer 3: The Acoustic Monitor (Audio Preview - strictly Time-Domain)
A conditionally activated, lazy-evaluated playback layer. It is only triggered if an audio branch connects to a Preview Node or the Global Group Output.
Crucial Constraint: No frequency-domain reconstruction (iFFT/iBFT) is allowed here. Instead, ACE deduces a time-domain audio chain from the abstract tree. It translates the user's nodes into high-quality time-domain equivalents (e.g., IIR filters, biquads, gain envelopes) to process the audio.
Optimization: If a source audio is linked directly to the group output with no intermediary nodes, no DSP happens; the raw source passes through untouched.

2. Node Taxonomy
The user constructs their pipeline using four distinct categories of nodes:

Source (source_mngmt): Represents the root audio stream. Automatically generated per source.

Delta (delta): In Layer 2, these act as lightweight masks on a BFT. In Layer 3, these operations are mapped to their time-domain filter equivalents (e.g., a BFT bin-mask becomes a bandpass IIR filter).

Stream (stream): Audio-to-float/vector operators (e.g., RMS, Zero Crossing Rate). These are the terminal branches of Layer 2.

Utilities (utilities): Helper nodes like Specs, Latency Tracker, Audio Math (must handle cross-source branch mixing safely), and the Audio Preview node.

3. The Parallel Execution Pipeline
Performance and Blender stability are paramount. ACE uses a strict three-stage concurrent execution model.

Stage A: Main-Thread Graph (Setup): audio_engine.py and dummy_tree.py read the GN tree, build the abstract graph, compile the task graphs (both Layer 2 analysis and Layer 3 time-domain routing), and resolve attribute IDs.

Stage B: Worker Compute (DSP): Worker processes execute the DSP. Zero Blender API (bpy) calls are allowed here.

Stage C: Main-Thread Commit (Write): audio_engine.py writes the final stream values to the proxy mesh attributes.

4. Concurrency & Scheduling Rules
Main Thread ONLY: bpy data, node trees, mesh attributes, object properties, handler registration, UI updates, and Stage C commits.

Multithreading (I/O only): Sounddevice callbacks, queueing, preview monitor buffering, and background metadata jobs.

Multiprocessing (Heavy DSP): CPU-heavy evaluation. Uses a fixed worker count via a spawn context. Passes only compact numeric data/parameters via IPC.

Realtime Scheduling & Render Safety: Drops stale jobs (maxlen 1 or 2) to protect scene FPS. If a deadline is missed, reuse the last valid value. Execution must remain safe during render time (e.g., F12 animation rendering).

5. Development Guardrails for the Agent
Rule 1: Never access bpy outside the main thread. Respect use_lock_interface.

Rule 2: Strictly isolate Layer 3. Audio preview processing must not execute unless explicitly connected to an output socket.

Rule 3: Enforce Time-Domain Previews. Layer 3 must never attempt BFT-to-audio reconstruction. Map abstract nodes directly to high-quality time-domain filters. Use Numba ("High quality audio preview" toggle) to accelerate these time-domain math operations.

Rule 4: Maintain deterministic ID generation for proxy mapping.

Rule 5: Ensure cross-source nodes (like Audio Math) correctly synchronize distinct root_sources branches.

## Agent Dedicated Task

Comprehensive Architectural Audit
Your primary objective is to conduct a deep, rigorous code-level audit of the entire AudioCore Engine (ACE) codebase. You must act as the principal architect, evaluating the implementation against the strict System Context outlined above. Execute the following directives and structure your output accordingly in three phases.

Phase 1: Deep Code Review & Mismatch Detection

Analyze all provided files and modules. Scrutinize the code for incoherences, architectural mismatches, and overall health. You must aggressively hunt for specific structural failures across the entire system.

First, identify any Guardrail Violations, specifically looking for any instance of bpy access outside the main thread or missing use_lock_interface compliance. Next, check for Layer Contamination by pointing out any bleeding of logic between Layer 2 (Data Engine) and Layer 3 (Acoustic Monitor). You must also strictly enforce Layer 3 rules by catching Domain Errors; flag any attempt to perform frequency-domain reconstruction (iFFT/iBFT) for the audio preview, and verify that Layer 3 relies strictly on time-domain filter mapping and Numba acceleration.

Furthermore, investigate Concurrency and Scheduling Flaws. This includes uncovering unsafe threading or multiprocessing IPC patterns, memory leaks in the worker pool, or the failure to drop stale jobs to maintain scene FPS. Finally, ensure Data Integrity by checking for desyncs in cross-source mixing within Audio Math nodes and verifying the deterministic ID generation for proxy attributes.

Phase 2: Detailed Component Report & Sub-Scoring

Provide a structured, section-by-section review outlining the health, identified bugs, and optimization opportunities for each core module. Crucially, calculate and assign a specific Development Sub-Score from 1 to 100 for each component, justifying the rating based on the strict architectural rules.

The Facade [Score: X/100]: Evaluate the UI logic, custom node setup, dummy tree abstraction, and the determinism of proxy ID generation.

The Data Engine [Score: X/100]: Evaluate the worker pool DSP computation, spectral and temporal extraction accuracy, IPC data payload efficiency, and the strict safety of Stage C commits.

The Acoustic Monitor [Score: X/100]: Evaluate strict isolation compliance, the accuracy of time-domain filter mapping, lazy-evaluation routing, Numba acceleration integration, and sounddevice callback health.

Overall Concurrency & Rendering [Score: X/100]: Evaluate render-time safety (specifically during F12 rendering), strict thread and process discipline, stale-job dropping for scene FPS protection, and verify absolute zero bpy violations within the worker pool.

Phase 3: ACE Development Score & Critical Path

Conclude your analysis by calculating an objective overall Development Score on a scale of 1 to 100, accompanied by a detailed justification based on your Phase 2 findings. Use the following baseline to anchor your evaluation.

A score between 1 and 20 indicates a broken state with core architectural failures, fundamentally unsafe execution, and nothing working as intended. A score between 21 and 40 represents a prototype where basic logic exists, but it remains heavily unstable with blatant rule violations (like bpy in threads) leading to inevitable crashes. A score between 41 and 60 signifies an alpha state where the 3-Layer Model is mostly intact, but concurrency bottlenecks, cross-source desyncs, or Layer 3 isolation flaws remain. A score between 61 and 80 indicates a beta state that is feature complete and relatively stable, though minor optimizations, latency drift, or edge-case F12 render safety issues might still exist. Finally, a score between 81 and 100 reflects production readiness, demonstrating strict adherence to all guardrails, zero API violations, highly optimized IPC, and flawless time-domain previews.

Final Output Requirement: After assigning the final score, provide a concluding section titled "Critical Path to Production". Write a sequenced paragraph outlining the exact, prioritized steps required to elevate the codebase to a perfect 100 score, focusing first on stability, then correctness, and finally optimization.

## Agent Output

AudioCore Engine (ACE) — Comprehensive Architectural Audit Report
Codebase Version: 0.5.0 | Audit Date: 2026-03-01

Phase 1: Guardrail Violations & Structural Failures
1.1 — bpy Access Outside Main Thread
Verdict: PASS (with qualified concern)

parallel_dsp.py contains zero bpy imports. The module header explicitly declares # no bpy imports. Stage B workers are fully Blender-agnostic. No violation confirmed.

Qualified Concern (not a violation): All bpy.context accesses in audio_engine.py are wrapped in getattr(..., None) chains (lines 967, 978, 1021, 1424), providing crash safety. However, no assert threading.current_thread() is threading.main_thread() guard enforces this rule at runtime. The guardrail is implicit rather than code-enforced — one future refactor could silently break it.

1.2 — Layer Contamination / Domain Errors
Verdict: FAIL — Latent Architectural Violation (Severity: Medium)

_masked_mono_signal() in audio_tasks_registry.py:2717-2721 contains an iFFT reconstruction path:


spec = np.fft.rfft(src, n=n)
masked[idx_arr] = spec[idx_arr]
return np.fft.irfft(masked, n=n).astype(np.float32, copy=False)
Current state: This function is not reachable from the Layer 3 preview route as currently wired. PreviewSinkTask.process() routes through _chunk_to_stereo_frames() which reads .mono/.stereo time-domain fields directly and applies only IIR/biquad filters (Numba kernels). Layer 3 is iFFT-clean at runtime.

However: _masked_mono_signal() has no architectural guard, no # [L2-ONLY] marker, no naming convention distinguishing it. It is an unmarked architectural landmine — a single future call-path change silently routes iFFT into Layer 3.

1.3 — Concurrency Flaws
ID	File	Line	Issue
C-1	audio_engine.py	826	_PARALLEL_MAX_INFLIGHT_PER_SOURCE = 3 — exceeds spec's maxlen 1–2
C-2	audio_engine.py	831	_PARALLEL_STALE_REUSE_WINDOW_FRAMES = 2 — frame-count-based, not time-based; at 12fps = 166ms ghost
C-3	audio_engine.py	~1580	SHM handles not released on BrokenProcessPool + restart-limit exhaustion; /dev/shm orphans on Linux
C-4	audio_engine.py	2483	_probe_source_metadata() opens AudioReception synchronously on depsgraph_update_post handler (main-thread file I/O)
C-5	audio_tasks_registry.py	2258	SourceSpecsTask BPM background worker + SourceTask._prefetch_loop can simultaneously open() the same file; Windows-exclusive-lock risk
1.4 — Data Integrity Issues
ID	File	Finding
D-1	id_management.py:50	MD5 for node IDs vs. blake2b for IR digest — hash algorithm inconsistency; collision resistance lower in ID layer
D-2	audio_engine.py:2213	_collect_sample_rate_floor() called with allow_io=False during compile; cold-start defaults to 44100 Hz even for 96kHz sources until first async probe completes and triggers recompile
D-3	proxy_management.py	Delta-skip + stale-job reuse chain failure: onset attribute can remain 0.0 for 2 extra frames, then delta-skip suppresses the correct-frame write
Phase 2: Component Scores
The Facade — Score: 88 / 100
Files: dummy_tree.py, abstract_tree.py, gn_menu.py, properties.py

Strengths:

abstract_tree.py is a clean, bpy-free dataclass IR — pure Layer 1 compliance
ACEDummyNodeFactory correctly uses Blender 5.0+ NodeSocketBundle for internal wiring
_validate_cross_source_links() catches illegal cross-source wiring at the Facade level
properties.py's _on_engine_runtime_setting_change() correctly delegates to engine.request_recompile() without computing anything
Bugs:

B.F.1: ACEDummyNodeFactory.get_group_name() uses ".ACE_" prefix naming, but ACEDummyTreeReader identifies nodes by ace_id custom property — not name. A user-duplicated node group with the same ace_id is processed twice with no de-duplication guard.
B.F.2: _on_source_type_change() (properties.py:35) — switching from LIVE back to FILE does not reset flow_identifier to "sd:". Stale identifiers persist across save/reload.
Optimizations:

ensure_node_groups_exist() is O(N) on every operator invocation; a content hash of ACE_NODES would short-circuit unchanged registries
_initialize_auto_attr_ids() triggers a depsgraph update at node placement, causing a re-read of the full tree; a batched update would be more efficient
Justification: The Facade is architecturally pure and well-structured. The two bugs are edge-case UX issues, not correctness failures. Deducted 12 points for the O(N) refresh, the stale flow_identifier regression, and the missing node de-duplication guard.

The Data Engine — Score: 81 / 100
Files: audio_engine.py, parallel_dsp.py, audio_tasks_registry.py, audio_influx.py

Strengths:

Stage A/B/C separation is architecturally sound — no bpy in Stage B confirmed
SharedMemory IPC with multiprocessing.shared_memory is correct zero-copy design
"spawn" process context avoids fork-safety issues with Blender's internal state
_WORKER_GRAPH_CACHE in parallel_dsp.py correctly amortizes IR deserialization cost across frames
_prune_ambiguous_required_node_ids() correctly handles shared AudioMath nodes in multi-source topologies
blake2b IR digest for worker cache invalidation is appropriate and collision-resistant
Bugs:

B.D.1: _PARALLEL_MAX_INFLIGHT_PER_SOURCE = 3 (audio_engine.py:826) — spec mandates maxlen 1–2; 3 creates a 3-frame stale lag pipeline
B.D.2: On _parallel_reset_generation(), orphaned futures from the prior generation have their SHM handles kept alive until those futures eventually resolve — long stalls = SHM leaks
B.D.3: _probe_source_metadata() (audio_engine.py:2455) — synchronous file I/O on main thread during hot depsgraph handler
B.D.4: Cold-start: effective_sample_rate defaults to 44100 Hz for first ~3 frames on 96kHz sources
B.D.5: SourceTask._prefetch_loop() mutates self._current_frame_request temporarily inside a finally block; main-thread reads during prefetch see the temporary value
Optimizations:

FrequencySplitter with N bands calls BFT N times from the same chunk; a global (chunk_token, "full_spectrum") cache would reduce to 1 BFT call per chunk
_PARALLEL_STALE_REUSE_WINDOW_FRAMES = 2 should be time-based (e.g., 66ms) not frame-count-based
_compute_tree_signature() is O(N log N) per request_recompile(); the existing blake2b IR hash could serve as the compile-skip check instead
Justification: The core pipeline design is excellent. Deducted 19 points for the SHM lifecycle gaps, the inflight-count spec deviation, the blocking file I/O on the main thread, and the cold-start sample-rate race.

The Acoustic Monitor — Score: 85 / 100
Files: audio_tasks_registry.py (PreviewSinkTask), audio_engine.py (preview routing)

Strengths:

PreviewSinkTask reads .mono/.stereo time-domain fields only — iFFT-clean in practice
_ACEPreviewFeeder with _TARGET_FILL = 0.35 ring fill target and soft drift correction is production-quality jitter absorption
_apply_safety_limiter() with Numba (@njit(cache=True)) — instant-attack / timed-release design is textbook correct
manual_preview_task_ids correctly overrides auto_preview_task_ids for explicit user control
Layer 3 only activates when preview_task_ids is non-empty — lazy evaluation is correctly implemented
Numba kernels (_signed_smooth_block, _biquad_filter_block, _iir1_filter_block) gated by NUMBA_AVAILABLE with NumPy fallbacks
Bugs:

B.A.1: _ACEPreviewAudioMonitor._prepare_loop() (audio_engine.py:448): except Exception: continue silently swallows all preview exceptions — dropout occurs with zero diagnostic trace
B.A.2: On timeline scrub backward, the _ACEPreviewFeeder buffer (up to 133ms at 48kHz/30fps) can contain future-relative audio that plays at wrong positions if flush() is not reliably triggered on seek events
B.A.3: _masked_mono_signal() exists without any barrier preventing Layer 3 access — the latent iFFT contamination risk (Phase 1.2)
Optimizations:

Preview latency is ~1024 samples (~21ms at 48kHz) from three stacked buffers; aligning the hardware block size to the analysis block size would halve this
Non-Numba limiter fallback uses a Python for loop over frames — replace with vectorized NumPy using np.maximum.accumulate for the release envelope
Justification: The Acoustic Monitor is cleanly isolated and correctly implements time-domain filtering with Numba. Deducted 15 points for the silent exception swallowing, the scrub-flush reliability concern, and the unmarked iFFT landmine risk.

Overall Concurrency & Rendering — Score: 79 / 100
Strengths:

bpy.app.handlers.persistent correctly applied to all handlers
_unsafe_for_handler_writes checks use_lock_interface — F12 render safety is implemented
ACEProxyManager.update_proxy_object() is the single write funnel for all Stage C mutations
Handler de-registration on unregister prevents accumulation across reloads
_ACEPreviewFeeder.stop() uses threading Event + 1-second join — production-grade shutdown
SourceSpecsTask.shutdown_background_worker() called with a 250ms join timeout on unregister
Bugs:

B.C.1: _parallel_executor.shutdown(wait=False, cancel_futures=True) during recompile leaves already-running worker processes alive — they outlive the shutdown, holding SHM and consuming CPU
B.C.2: _PARALLEL_RESTART_LIMIT = 1 — after one failed restart, the source is added to _parallel_disabled_sources with no UI indication. Attributes freeze silently with no N-panel warning
B.C.3: depsgraph_update_post iterates depsgraph.updates and calls getattr(datablock, "bl_rna", None) per entry — on complex scenes with hundreds of animated objects, this is measurable per-frame main-thread overhead
B.C.4: SHM ref counts never decrement when Blender is paused (poll loop not called); watermark memory grows to 3 * N_sources * block_size_bytes and never shrinks below it
Optimizations:

_ATTR_RESYNC_INTERVAL_FRAMES = 120 means attribute schema is stale for up to 4 seconds; a dirty-flag approach (mark on tree modification, sync next frame) is more responsive
Bake cache not pruned on source removal — entries accumulate until next full recompile
_META_CACHE_MAX_ENTRIES = 256 with 30s TTL is appropriate; no change needed
Justification: The concurrency model is structurally sound. The F12 render guard is in place. Deducted 21 points for the SHM lifecycle management under pool failure, the silent parallel-disable condition, and the depsgraph.updates overhead on complex scenes.

Phase 3: ACE Development Score & Critical Path
Component Score Summary
Component	Score
The Facade (Layer 1)	88 / 100
The Data Engine (Layer 2)	81 / 100
The Acoustic Monitor (Layer 3)	85 / 100
Concurrency & Rendering	79 / 100
ACE Composite Score	83 / 100
Justification
ACE sits firmly in beta territory (81–85). The three-layer model is functionally respected at runtime. Layer 3 never executes iFFT code as the pipeline currently stands. Stage B workers are provably bpy-agnostic. The SharedMemory IPC pipeline, spawn-context process pool, IR digest invalidation, and ring-buffer jitter absorber demonstrate engineering maturity well above a typical Blender add-on. The Numba acceleration path for time-domain filters is correctly gated and falls back safely. The abstract tree IR is clean and deterministic.

The deductions are concentrated in three operational reliability gaps — not fundamental design failures: SHM handle lifetime is not guaranteed across pool crashes; _probe_source_metadata() performs synchronous disk I/O on the real-time main thread; and the inflight queue depth of 3 creates a stale-data window that exceeds the specification's required maxlen of 1–2. None of these break the architectural contract, but all three can cause observable production failures under adverse conditions (slow disks, pool instability, fast-onset musical events).

Critical Path to Production
Step 1 — Stability (eliminate crash/corruption risk):
Begin by fixing the SharedMemory lifecycle. Modify _shutdown_parallel_pool() to call _parallel_cleanup_ipc_handles(meta) for every entry in _parallel_future_meta unconditionally, regardless of future resolution state. This eliminates the /dev/shm orphan accumulation on Linux under pool crashes and restart-limit exhaustion. Simultaneously, add a UI warning in the N-panel's source status row when a source key is present in _parallel_disabled_sources — silent freeze is worse than a visible error. Finally, harden _parallel_executor.shutdown() during recompile to either use wait=True with a capped timeout (e.g., 500ms) or explicitly terminate worker processes before re-creation, preventing CPU-burning orphaned workers between generations.

Step 2 — Correctness (eliminate wrong-value conditions):
Move _probe_source_metadata() off the main thread. Spawn a dedicated _ProbeThread daemon (similar to the existing SourceSpecsTask._BG_THREAD pattern) that writes results to a thread-safe pending-update dict, flushed at the top of the next frame's depsgraph_update_post handler. This eliminates main-thread file I/O and resolves the cold-start sample-rate race simultaneously, as the probe completes asynchronously before the compile path reads native_sample_rate. Next, reduce _PARALLEL_MAX_INFLIGHT_PER_SOURCE from 3 to 2 and convert _PARALLEL_STALE_REUSE_WINDOW_FRAMES to a time-based constant (e.g., _PARALLEL_STALE_REUSE_MS = 66), computed to frame equivalents at evaluation time using the scene FPS. This brings the stale-data window within spec and makes it FPS-invariant. Protect _masked_mono_signal() with a # [LAYER-2-ONLY: DO NOT CALL FROM PREVIEW PATH] header and add a pytest static assertion in test_regression_findings.py that traces PreviewSinkTask.process and asserts np.fft.irfft is not in its transitive call graph.

Step 3 — Optimization (squeeze out remaining latency and throughput gaps):
Introduce a global full-spectrum cache keyed by (chunk_token, "full_spectrum") in AudioFluxAnalyzer._ensure_spectrum() to reduce multi-band FrequencySplitter sources from N BFT calls to 1 per frame per source. Replace the non-Numba limiter fallback Python loop with a vectorized NumPy implementation using np.maximum.accumulate for the release envelope. Standardize all internal ID generation to blake2b-8 to match the IR digest, replacing MD5 in id_management.py:50. Add the dirty-flag attribute schema resync in place of the 120-frame polling interval. Address the ensure_node_groups_exist() O(N) refresh with an ACE_NODES content hash short-circuit. With these changes in place, the SHM watermark issue (Finding B.C.4) becomes the last remaining operational concern — address it by adding a frame-pause detection in _parallel_poll_futures() that drains the inflight queue when no new submits have occurred for N consecutive frames, allowing ref counts to reach zero and SHM to be reclaimed.

With the above three steps complete, ACE reaches 93–96 / 100: a production-ready, architecturally principled, real-time Blender DSP engine.

## Ideas

Major issue : sometimes after changing some ACE and/or regular blender nodes, everything runs at full fps but the geometry is not moving anymore

All refresh / recompile actions : launching the timeline, going back to the first frame.

Sometimes, the autorefresh missises some changes in the node tree like a link between nodes being changed.

Change the sample rate options to : 24000 / 48000 / 96000 Hz

Blender crashes when doind a lot of undo in a row (ctrl + Z)

Engine recompile calculation burst can create bugs : stream results being actualised on preview but no geometry moving when plugged into regular blender geometry nodes -> freeze timeline when engine is recompiling ?

Some cache / stale clearing necessary : when changing the type of a source from file to live and vice versa, when using a new source for an existing node graph, etc.

The frequency splitter is broken

## Test command

python3 tests/run_quality_gate.py
