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

# Ideas

Change the default live latency buffer to 5 ms.
