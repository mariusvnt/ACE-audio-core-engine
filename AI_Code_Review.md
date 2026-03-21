# ACE Architectural Code Review

Role & Context:
You are a Principal Software Architect and Real-Time Performance Engineer specializing in 3D Digital Content Creation (DCC) tools (specifically Blender, Houdini, and TouchDesigner) and high-performance Audio/DSP programming. You have access to the full codebase for "AudioCore Engine" (ACE), a Blender add-on, as well as a local clone of the Blender source code.

Objective:
Conduct a ruthless, deep-dive code review of the entire ACE codebase. Your goal is to evaluate its structural integrity, alignment with Blender's internal C/C++/Python architecture, and its real-time performance viability compared to industry standards.

Please structure your review strictly into the following four phases:

Phase 1: Architectural Compliance & Add-on Design Integrity

Evaluate the current code against the official ACE Architecture Brief (the 3-Layer Model: Facade, Data Engine, Acoustic Monitor).

Layer Isolation: Detect any bleed between Layer 1 (UI/Routing), Layer 2 (DSP Backend), and Layer 3 (Time-Domain Preview). Specifically, hunt for DSP/math logic inside node frontend stubs (dummy_tree.py, shader_tree.py) or frequency-domain reconstruction (iFFT/iBFT) inside Layer 3.

Concurrency Guardrails: Verify the strict threading model. Flag any instance of bpy access outside the main thread (especially in parallel_dsp.py), missing use_lock_interface implementations during Stage C commits, or unsafe Multiprocessing/IPC patterns.

Data Model & State: Confirm that the recent shift to a merged IR Data Model (incorporating ACEFrontendRef and ACEConsumerIntent) has been successfully implemented without breaking the deterministic hashing (24-bit blake2b) or the root_sources partitioning.

Phase 2: Blender Source Code Alignment
Cross-reference the ACE codebase with the local Blender source code clone to validate API usage and execution safety.

Depsgraph & Evaluation: Analyze ACE's interaction with Blender's dependency graph. Are depsgraph_update_handler triggers optimal, or are they causing redundant graph evaluations?

C/C++ API Parity: Verify that ACE's data writing (e.g., updating proxy mesh attributes and object custom properties) aligns with the most efficient pathways exposed by Blender's RNA/BKE layers.

Deprecation & Anti-Patterns: Identify any usage of Blender Python API calls that are deprecated, computationally heavy on the main thread, or fundamentally misaligned with Blender 5.0+ paradigms.

Phase 3: Cruft, Broken Logic, & Inefficiency Detection
Actively seek out code that serves no purpose or actively harms performance.

Dead Code & Orphans: Identify unused functions, outdated fallback loops, or variables left over from previous architectures (e.g., single-tree assumptions that survived the multi-frontend refactor).

Performance Bottlenecks: Pinpoint inefficiencies in memory sharing (SHM), audio chunk propagation, or the BFT caching mechanism (_GLOBAL_SPECTRUM_CACHE).

Queue/Job Leaks: Verify that stale multiprocessing jobs (>66ms) are successfully dropped and not ballooning memory or causing micro-stutters during realtime playback.

Phase 4: Industry Standard Grading & Rubric
Construct a detailed, objective grade of the current state of the add-on relative to industry standards for real-time procedural audio-reactivity (i.e., Houdini CHOPs or TouchDesigner's audio analysis pipelines).

Score the engine out of 100 points using the following weighted categories. Justify every deduction with specific file names and line numbers:

Real-Time Latency & Jitter (25 pts): How close is ACE to a strict deterministic execution budget per frame without stalling the DCC host?

Concurrency & Thread Safety (25 pts): Strict separation of bpy main-thread writes vs. headless worker DSP processing.

Algorithmic Efficiency (25 pts): Complexity of the audio graph expansion, BFT caching reuse, and biquad filter acceleration (Numba/SciPy/Python fallback).

Host API Harmony (25 pts): Efficiency of bulk-writing to Blender's geometry attributes and shader properties without triggering UI locks or crashes.

Deliverable:
Conclude with a "Critical Path to Industry Standard" section: a highly prioritized, sequential action plan detailing exactly what code needs to be deleted, rewritten, or refactored to achieve a perfect 100/100 score. Prioritize crash-risks and thread-safety first, followed by Blender API alignment, and finally micro-optimizations.

## agent output

Phase 1: Architectural Compliance & Add-on Design Integrity

PASS Layer isolation is mostly correct.
Frontend readers are IR/routing focused (no DSP task execution in stubs): nodes/dummy_tree.py (line 1076), nodes/dummy_tree.py (line 1138), nodes/shader_tree.py (line 1271).
Layer-3 has explicit iFFT guardrail and time-domain preview path: core/audio_tasks_registry.py (line 3113), core/audio_tasks_registry.py (line 4489), core/audio_tasks_registry.py (line 4836).
HIGH Concurrency guardrails exist but one core guard is soft-fail.
assert_main_thread only logs by default, does not stop unsafe writes: utils.py (line 33).
Good: render lock-interface gating is consistently enforced in frame handlers/execute/compile: core/audio_engine.py (line 5427), core/audio_engine.py (line 5846), core/audio_engine.py (line 6591), core/audio_engine.py (line 2756).
HIGH Stale job policy is not strict to a hard ~66 ms wall-clock budget.
Timeout is 3.0 s and cancellation is frame-delta based: core/audio_engine.py (line 1068), core/audio_engine.py (line 2532).
Stale-drop is frame-grace based (2 frames), not direct ms SLA: core/audio_engine.py (line 2416).
PASS with caveat Merged IR model is implemented and wired end-to-end.
Datamodel includes ACEFrontendRef and ACEConsumerIntent: core/abstract_tree.py (line 25), core/abstract_tree.py (line 39).
Emission + merge + sink resolution is present: nodes/dummy_tree.py (line 1138), core/audio_engine.py (line 5167), core/audio_engine.py (line 3990).
root_sources propagation/partition logic exists: nodes/dummy_tree.py (line 1280), core/audio_engine.py (line 3768), core/audio_engine.py (line 3780).
MEDIUM Deterministic-hash intent is partially undermined by Python hash().
Non-deterministic process-salted hash used for bake cache naming and UI topology signature: core/audio_engine.py (line 6492), ui/n_panel_ui.py (line 1251).
Phase 2: Blender Source Code Alignment

PASS bpy.app.is_job_running("RENDER") usage is API-valid.
Blender exposes is_job_running(job_type) with enum-backed job types including RENDER: bpy_app.cc (line 719), bpy_app.cc (line 728), rna_wm.cc (line 214), rna_wm.cc (line 215).
ACE usage: core/audio_engine.py (line 5405).
PASS Handler safety model aligns with Blender guidance.
Blender warns that frame-change handlers can race viewport/render threads and recommends lock interface: bpy.app.handlers.2.py (line 5).
ACE mirrors this guard: core/audio_engine.py (line 6591).
PASS ID.update_tag(refresh=...) usage is consistent with RNA.
Valid refresh flags are OBJECT, DATA, TIME: rna_ID.cc (line 2266).
update_tag validates and forwards to depsgraph tagging: rna_ID.cc (line 1046).
ACE uses update_tag(refresh={"OBJECT","DATA"}): core/proxy_management.py (line 539), core/proxy_management.py (line 590).
PASS Attribute bulk writes use the fast RNA path.
Blender foreach_set goes through raw collection setters and updates property once: bpy_rna.cc (line 5919), bpy_rna.cc (line 5976), bpy_rna.cc (line 6081), rna_access.cc (line 5856).
ACE uses foreach_set correctly for multi-element attributes: core/proxy_management.py (line 751).
HIGH Depsgraph invalidation policy is too broad and likely causes redundant recompiles.
Any updated Object with ACE-managed GN/shader linkage can trigger scene-wide request_recompile(None): core/audio_engine.py (line 6748), core/audio_engine.py (line 6763), core/audio_engine.py (line 5802).
Phase 3: Cruft, Broken Logic, & Inefficiency Detection

Dead/legacy leftovers:
attached_shm plumbing is effectively vestigial (_resolve_source_payloads always returns empty attached handles): core/parallel_dsp.py (line 679), core/parallel_dsp.py (line 807).
Stage-A class doc still describes older single-tree compile narrative: core/audio_engine.py (line 1013).
Performance bottlenecks:
Process mode is slower than thread mode in strict scenarios and fails gate thresholds: tests/artifacts/quality_report.md (line 30), tests/artifacts/quality_report.md (line 32), tests/artifacts/engine_perf_current.json (line 695), tests/artifacts/engine_perf_current.json (line 875), tests/artifacts/engine_perf_current.json (line 1055).
Queue pressure drops are high under load (up to 29.17%): tests/artifacts/engine_perf_current.json (line 1034).
Main-thread fallback recomputes offloaded keys when worker values are missing, creating duplicate work and jitter: core/audio_engine.py (line 6254).
_GLOBAL_SPECTRUM_CACHE_MAX = 16 with FIFO eviction is likely too small for high-fanout sessions: core/audio_tasks_registry.py (line 775), core/audio_tasks_registry.py (line 988).
Object-property sink writes are scalar Python loops, not bulk pathways: core/proxy_management.py (line 561).
Queue/job leaks:
Positive: stale-result drop, inflight pressure-drop, and SHM drain timer exist: core/audio_engine.py (line 2416), core/audio_engine.py (line 2524), core/audio_engine.py (line 6801).
Negative: stale job lifetime is still far above a strict 66 ms contract in low-FPS conditions due 3 s timeout and frame-based cancellation: core/audio_engine.py (line 1068), core/audio_engine.py (line 2532).
Phase 4: Industry Standard Grading & Rubric (100 pts)

Real-Time Latency & Jitter: 11/25
Deductions:
-8 strict process-vs-thread DSP ratio failures in heavy scenarios (>1.1): tests/artifacts/quality_report.md (line 32), tests/artifacts/engine_perf_current.json (line 695), tests/artifacts/engine_perf_current.json (line 875), tests/artifacts/engine_perf_current.json (line 1055).
-4 queue-drop and low fresh-result rates: tests/artifacts/quality_report.md (line 35), tests/artifacts/engine_perf_current.json (line 854), tests/artifacts/engine_perf_current.json (line 1034), tests/artifacts/engine_perf_current.json (line 1043).
-2 SLA p95 exceeds 15 ms in heavy topology: tests/artifacts/engine_perf_current.json (line 30), tests/artifacts/engine_perf_current.json (line 994), tests/artifacts/engine_perf_current.json (line 1052).
Concurrency & Thread Safety: 18/25
Deductions:
-3 main-thread guard logs instead of hard-stopping unsafe writes: utils.py (line 33).
-2 stale-job expiry policy is not strict wall-clock bounded: core/audio_engine.py (line 1068), core/audio_engine.py (line 2532).
-2 handler safety depends on user/project lock-interface state during render: bpy.app.handlers.2.py (line 5), core/audio_engine.py (line 6591).
Algorithmic Efficiency: 12/25
Deductions:
-7 process offload path regresses relative to thread path in complex scenarios: tests/artifacts/engine_perf_current.json (line 695), tests/artifacts/engine_perf_current.json (line 875), tests/artifacts/engine_perf_current.json (line 1055).
-3 worker-miss fallback recomputes on main thread (duplicate path): core/audio_engine.py (line 6254).
-3 global spectrum cache cap/eviction policy likely under-sized for large fan-out: core/audio_tasks_registry.py (line 775), core/audio_tasks_registry.py (line 988).
Host API Harmony: 16/25
Deductions:
-5 depsgraph handler invalidates too broadly (object/material updates trigger recompiles): core/audio_engine.py (line 6739), core/audio_engine.py (line 6763).
-2 request_recompile(None) forces full frontend rebuild path: core/audio_engine.py (line 5802).
-2 object-property sink writes are Python-loop bound and less RNA-optimal than attribute bulk writes: core/proxy_management.py (line 561), core/proxy_management.py (line 590).
Total Score: 57/100

Critical Path to Industry Standard (Priority Order)

Rewrite thread-safety guardrails to fail-safe, not log-only, for write paths.
Refactor utils.assert_main_thread callers in writer/compile/handler paths to early-return hard, not soft log: utils.py (line 33).
Enforce strict stale-job cancellation budget in ms.
Replace frame-delta + 3s timeout policy with submit-age cutoff (<=66 ms target), then cancel/drop immediately: core/audio_engine.py (line 1068), core/audio_engine.py (line 2532).
Remove duplicate DSP execution under process mode.
Delete/replace main-thread offload fallback branch; choose deterministic policy (worker results only with deterministic stale-reuse, or explicit mode downgrade): core/audio_engine.py (line 6254).
Refactor depsgraph invalidation to topology-aware signals only.
Rewrite depsgraph_update_handler to avoid Object broad invalidation and avoid request_recompile(None) unless frontend membership truly changed: core/audio_engine.py (line 6739), core/audio_engine.py (line 6763).
Shift sink strategy toward bulk attribute writes.
Reduce OBJECT_PROPERTY fan-out for realtime paths; keep proxy-attribute bulk writes as primary fast path: core/proxy_management.py (line 561), core/proxy_management.py (line 751).
Replace non-deterministic Python hash usage with stable blake2b.
Delete hash(...) usage for bake/ui signatures and use explicit deterministic digest: core/audio_engine.py (line 6492), ui/n_panel_ui.py (line 1251).
Increase/adapt spectral cache policy for heavy topologies.
Replace fixed-size 16 cache with topology-aware or frame-local adaptive sizing and explicit hit-rate telemetry: core/audio_tasks_registry.py (line 775).
Delete vestigial SHM-handle close loop plumbing and stale architecture comments.
Remove unused attached_shm loop and update stale Stage-A doc comments to current scene-frontends architecture: core/parallel_dsp.py (line 679), core/audio_engine.py (line 1013).
I could not rerun the test suite in this shell (no pytest available), so performance conclusions are based on existing artifacts in tests/artifacts.

## Needed modifications

Modify rolling average -> new operator that add attack / release function to a stream

Regroup all loudness stream operators into one where the measure method is selectable