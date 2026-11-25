# AudioCore Engine Architecture

How ACE is built, and why. The code says *what* each part does; this document
keeps the decisions that span modules.

## 1. Model

ACE turns file or live audio into per-frame values for Geometry Nodes and
object-bound shader materials. It is built in three layers:

1. **Facade.** ACE node groups in Geometry Nodes and shader trees are interface
   contracts. They declare routing, analysis intent and outputs; they compute
   nothing. Users edit graphs in Blender's own editors, and the backend stays
   free of Blender's evaluation rules.
2. **Data engine.** The facades are read into one abstract graph per scene,
   compiled into one task graph per source, and evaluated every frame. All DSP
   happens here.
3. **Acoustic monitor.** Preview playback rebuilds audible audio with
   time-domain filters derived from Layer 2 metadata (band limits, analysis
   ops). It never inverts spectral analysis: inverse-FFT reconstruction would
   reintroduce block artefacts and couple monitoring to analysis settings.

Delta nodes (splitters, chromagram, components) mostly attach constraints to
the audio chunk (`band_indices`, `analysis_ops`, lineage) instead of producing
new signals. Downstream stream nodes apply them while analysing, so a split
costs metadata, not a copy of the audio.

## 2. Packages

| Package | Role |
| --- | --- |
| `core/tasks` | DSP tasks, the audio chunk, analyzers and the per-source runtime graph |
| `core/engine` | `ACEAudioEngine` (one per scene) and its components |
| `core/engine/compiler` | Stage A: frontends to task graphs |
| `core/engine/parallel` | Stage B: worker-process offload |
| `core/engine/frame` | Stage C: per-frame evaluation and sink commits |
| `core/engine/preview` | Preview routing, mixing and playback |
| `core/worker` | Graph evaluation inside worker processes (no `bpy`) |
| `core/input` | File, sound-device and UDP input |
| `core/proxy` | Proxy objects, attribute and property sinks, ACE_Sonar |
| `nodes/geometry`, `nodes/shader` | Node-group factories and tree readers |
| `ui` | N-panel, add menu, operators and preferences |

## 3. Frame pipeline

1. **Stage A, compile (main thread).** Managed frontends (Geometry Nodes
   modifiers and object materials) are discovered, read into partial abstract
   graphs and merged. One `ACEAudioGraph` is built per source with only the
   nodes its outputs need, and each output gets a concrete sink binding.
2. **Stage B, offload (worker processes).** Numeric outputs run in worker
   processes from a serialized graph IR. Results come back asynchronously; a
   frame uses this frame's values, else values at most 50 ms old, else local
   DSP. The 50 ms window bounds how stale an output can look at any fps.
3. **Stage C, commit (main thread).** Values are written to proxy mesh
   attributes (Geometry Nodes) or object custom properties (shaders).

Block size follows the source sample rate and scene fps, so one block is one
frame of audio and analysis stays locked to the timeline.

## 4. Engine components

`ACEAudioEngine` owns the scene runtime (graphs, sink writer, frame clock,
undo state) and one component per concern: compiler, frame, parallel,
frontends, sinks, preview, bake, probes, diagnostics. Components reach the
runtime and each other through `self.engine`, never by attribute-name strings
and never by importing each other: a misspelled `getattr(self, "name")`
silently returns its default, and cross-imports would form cycles.
`tests/test_engine_structure.py` enforces both rules.

Blender handlers and timers live in `core/engine/handlers.py`. They are
`@persistent` and are identified by `__qualname__`, so a script reload
replaces them instead of stacking a second copy.

## 5. Identity and naming

- Every ACE node has an `ace_id` scoped to its frontend; source placeholders
  keep their source binding separately.
- Proxy object names, attribute names and tree signatures come from
  `hashlib.blake2b`, so the same scene always produces the same names, and
  attribute keys fit Blender's length limits.

## 6. Worker processes

### 6.1 Rules

- Workers never import `bpy`. `bpy` (data, context, trees, handlers) is used
  on the main thread only.
- `stream_accumulator` stays on the main thread: a running sum must not jump
  when execution moves between a worker and local DSP.
- A job older than 66 ms is dropped. A late result would either stall the
  frame or contaminate a later one.
- At most two jobs per source are in flight.

### 6.2 Pool lifecycle

A spawned worker imports the DSP stack (~4 s) before it can run a job. So:

- the import runs in the process initializer, and a cold process never takes a job;
- after the pool starts, one warm-up job per needed worker (shards x in-flight
  limit) must answer before any real job is sent; local DSP covers that time;
- the pool survives recompiles: a compile starts a new job generation, and
  workers drop the graphs of older generations themselves;
- a graph's IR is sent with every job until every worker process has
  acknowledged it;
- a job dropped while still running keeps its shared-memory payload slot until
  it finishes, so the next frame cannot overwrite input it is reading;
- a crashed pool restarts at most three times, with backoff, then the affected
  sources stay local.

`spawn` re-runs `__main__` from its file in each child. Under
`blender --python script.py` that is a `bpy` script, so the file is hidden
from `__main__` while workers start.

Without these rules every compile restarted the pool, the pool grew to
`cpu_count - 1` cold processes, and every job submitted in those ~4 s missed
its budget.

### 6.3 Threads vs processes (measured 2026-11-25)

Blender 5.0, one 24 fps file source feeding N stream nodes, 240 frames at
real-time pace, optional libraries installed (`tests/blender_perf.py`):

| Nodes | p95 frame ms, threads | p95 frame ms, processes | Fresh worker values |
| --- | --- | --- | --- |
| 24 | 3.6 | 1.8 | 99.6% |
| 36 | 4.6 | 2.2 | 99.6% |
| 50 | 4.8 | 2.8 | 99.6% |

Processes are kept: the rule set before measuring was at most 0.8x the thread
p95 and at least 95% fresh worker values. Their cost is a ~2 s cold start,
covered by local DSP.

## 7. Blender integration

- **Render safety.** Writes from frame handlers stop while a render runs with
  `use_lock_interface` off; viewport renders keep updating.
- **Auto-compile.** Depsgraph updates mark trees dirty. A rebuild runs after a
  quiet window, or after 2 s of continuous edits, so link-editing does not
  rebuild on every click.
- **Undo.** The engine holds no RNA references across undo; an undo bumps the
  undo generation, drops in-flight results and rebuilds.
- **UI sync.** A timer refreshes the N-panel from the compiled graph, throttled
  and scoped to the active source.
- **Shared memory.** Completed jobs are drained on a timer only while playback
  is paused; during playback the frame handler drains them, and an extra drain
  would add pipe reads to the frame.

## 8. Preview and bake

- **Preview.** Audio comes from manual preview sinks, or from links to the
  tree's audio output when there is none. Branches are mixed, limited and
  played only while the timeline plays. A feeder thread paces delivery at the
  device block rate because Blender timers jitter.
- **Bake.** A file source can be analysed over the frame range ahead of time.
  Baked values replace live computation until the tree signature, block size,
  or the file's path or modification time changes. The cache lives in memory.

## 9. Error handling

No error is absorbed silently. Boundaries keep one broad catch and log it,
marked `# noqa: BLE001 boundary: <reason>`: Blender handlers and timers, panel
`draw()`, operator `execute()`/`modal()`, thread loops, the worker job entry,
registration rollback, and imports of optional native packages. Inside a
boundary, a fallback that is the right outcome records the error with
`utils.logged_errors("module.function")` or
`utils.note_swallowed("module.function", exc)`. Recorded errors are counted in
`utils.swallowed_errors()` and printed (throttled) with debug logging on.
A specific exception type, or `utils.as_int` / `utils.as_float`, is preferred
over a broad catch.

`tests/test_error_handling.py` fails on any broad handler that neither records,
logs nor re-raises, and the headless Blender run fails if a normal session
records an error.

## 10. Testing

Tests outside Blender load the add-on against one shared fake `bpy`
(`tests/fake_bpy.py`) and check behaviour, never source text. Every test file
also passes when run on its own. `tests/run_in_blender.py` runs a headless
Blender session (`--extra-path` adds the optional libraries), and
`tests/blender_perf.py` compares threads and worker processes at real-time pace.
The perf gates (`test_perf_gate`, `test_engine_perf_gate`) are slow and
machine-dependent, so they run only with `ACE_PERF=1`; CI runs them in a
separate job.

## 11. Adding a node type

1. Declare it in `ACE_NODES` (`vars.py`). A new method of an existing measure
   is a mode of that node (`modes` in its definition), not a new node type.
2. Give it a group interface in `nodes/geometry/factory.py` and, for shader
   use, `nodes/shader/`.
3. Implement the task in `core/tasks/` and map it in
   `core/engine/compiler/graph_build.py`; for worker offload, also in
   `core/worker/graph.py`.
4. Add UI controls if it has runtime-editable parameters.
5. Add tests for its output and its topology sync.

## 12. Constraints

- Processing is frame-block based: tuned for reactivity, not offline fidelity.
- The bake cache is in memory only.
- Some spectral features need AudioFlux for full fidelity.
- Live input quality depends on the transport and its optional library.
