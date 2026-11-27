# Audio Core Engine Architecture

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
   time-domain filters derived from Layer 2 metadata (band limits). It never
   inverts spectral analysis: inverse-FFT reconstruction would reintroduce
   block artefacts and couple monitoring to analysis settings.

The Frequency Splitter, the only delta node, attaches a constraint to the audio
chunk (`band_hz_bounds` and lineage) instead of producing new signals.
Downstream stream nodes apply it while analysing, so a split costs metadata,
not a copy of the audio.

A chunk carries only the source's audio; nothing is filtered or smoothed at
the source. The Source node also outputs the source's BPM and Beats per Bar
(§8, Tempo).

## 2. Packages

| Package | Role |
| --- | --- |
| `core/tasks` | DSP tasks, the audio chunk, tempo detection and the per-source runtime graph |
| `core/engine` | `ACEAudioEngine` (one per scene) and its components |
| `core/engine/compiler` | Stage A: frontends to task graphs |
| `core/engine/parallel` | Stage B: worker-process offload |
| `core/engine/frame` | Stage C: per-frame evaluation and sink commits |
| `core/engine/preview` | Preview routing, mixing and playback |
| `core/worker` | Graph evaluation inside worker processes (no `bpy`) |
| `core/input` | File decoding (Blender's `aud`), sound-device and UDP input |
| `core/proxy` | Proxy objects (only for sources a node reads), attribute and property sinks |
| `nodes/geometry`, `nodes/shader` | Node-group factories and tree readers |
| `ui` | N-panel, add menu, operators and preferences |
| `deps.py` | Optional libraries: which load, and the user's switches (§13) |
| `tools` | Wheel fetching for the extension build (§13) |

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

`spawn` re-runs the parent's `__main__` file in each child before it
unpickles the job, and the job imports ACE by its full name. Installed as an
extension that name is `bl_ext.<repository>.audio_core_engine`, and Blender
creates `bl_ext` and `bl_ext.<repository>` in memory only, so a plain
interpreter cannot import them; under `blender --python script.py`,
`__main__` is also a `bpy` script. While workers start, `__main__` therefore
points at `core/worker/spawn_main.py` (`common._spawn_bootstrap_main`), which
registers those parent packages with a real `__path__` and runs nothing else.

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
  rebuild on every click. It cannot be turned off; *Manual Engine Reset*
  drops every engine, worker and cache and rebuilds, as a file load does.
- **Undo.** The engine holds no RNA references across undo; an undo bumps the
  undo generation, drops in-flight results and rebuilds.
- **UI sync.** A timer refreshes the N-panel from the compiled graph, throttled
  and scoped to the active source.
- **Shared memory.** Completed jobs are drained on a timer only while playback
  is paused; during playback the frame handler drains them, and an extra drain
  would add pipe reads to the frame.

## 8. File input, analysis, preview and bake

- **File input.** A file is decoded whole into memory by Blender's `aud`
  module (Blender's FFmpeg), so every format Blender reads works with nothing
  to install (`core/input/decode.py`). Decoded files are shared: the source,
  the tempo analysis and the bake read one read-only array, each with its own
  cursor, from a cache capped by size and keyed by path, size, modification
  time and target rate. Files above the working sample rate are resampled
  there, in memory. Metadata probes read the header only. Without `aud` (it
  fails to import when another extension shadows Blender's NumPy) only WAV
  files are read, through the `wave` module; the preferences say so. `aud`
  maps 16-bit PCM by 1/32767, `wave` decoding here by 1/32768.
- **Analysis frames.** A source hands out each block with the 4096 samples
  before it: read from the decoded file (zeros before its start, so a seek
  gives the same values), or kept from the blocks already played for live
  input. Splitters and worker payloads carry them along. Analysis that needs
  more than one block takes a frame that ends with the block
  (`_analysis_frame`), never a block padded with zeros. All of it is NumPy.
  - **Band loudness** (splitter bands): the block plus a 5 ms fade at each
    end. The squares of the fades add up to 1 across consecutive blocks, so
    every sample counts once and a band level is on the same scale as the full
    level.
  - **Stereo** behind a splitter: Width, Balance and the left and right levels
    come from the in-band powers of left, right, mid and side, measured like
    band loudness. Without a splitter they are measured on the whole block.
  - **Brightness:** the centroid of the magnitude spectrum of a periodic Hann
    window, as long as the next power of two of the block. A band counts only
    its own bins, with the same bounds test as band loudness, and a band too
    narrow to hold a bin reads 0. The bands of one block share one spectrum,
    cached by source, frame, sample rate, length and a checksum of the
    samples, so another file or frame rate never gets a stale one.
  - **Pitch:** YIN over 4096 samples, comparing the newest 1024, down to
    20 Hz and inside the chunk's band; 0 when there is no pitch.
- **Tempo.** A file source's tempo is detected on its own when the source gets
  a file or the file changes on disk, unless the user typed a BPM or Beats per
  Bar; *Auto Detect* replaces typed values. One background thread
  (`TempoService` in `core/tasks/tempo.py`) reads the first 15 s, builds an
  onset envelope (10 ms steps, log-compressed spectral flux) and analyses it:
  - **BPM:** the beat period from the envelope's autocorrelation, between 40
    and 240 BPM. A preference around 120 settles half and double tempo, and a
    period must also repeat at 2 and 4 times its length (bars of 2 or 4) or at
    3 and 6 (bars of 3).
  - **Beats per Bar:** each beat's strength is the envelope peak within a sixth
    of a beat, so off-beats never count. The strengths are folded by beat
    number for bar lengths 2 to 8, and the length where one fold stands out
    most clearly wins; a shorter bar wins a near tie. Without a clear accent
    pattern the answer is 4. The first downbeat is found too, and kept for the
    Downbeat output planned in FUTURE.md.

  A file without a steady pulse gets no tempo, and the source keeps its BPM.
  Results are cached per file path and modification time and written to the
  source on the main thread. Every tick, each source writes its BPM and Beats
  per Bar on point 0 of its proxy mesh (`ACE_bpm`, `ACE_beats_per_bar`); the
  Geometry Nodes Source group reads them there, and sink pruning leaves them
  alone. The shader Source node has no tempo outputs yet.
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
also passes when run on its own. The perf gates (`test_perf_gate`,
`test_engine_perf_gate`) are slow and machine-dependent, so they run only with
`ACE_PERF=1`; CI runs them in a separate job.

`tests/run_in_blender.py` runs a script in headless Blender with ACE enabled
as an extension, in temporary user folders, so nothing else installed on the
machine takes part (installed extensions share one site-packages). By default
it links the checkout into the extensions folder (`--extra-path` adds
libraries); `--package` builds the zip for this platform and installs it as a
user does, wheels included. `tests/blender_perf.py` compares threads and
worker processes at real-time pace.

**Reference curves** pin the numbers every node outputs
(`tests/reference/`). Generated signals (sweeps, clicks, pink noise, a stereo
pan, a music loop, 44.1 and 96 kHz files) play at 24, 30 and 60 fps through
one tree holding every node and mode, and each output is compared frame by
frame with the goldens in `tests/reference/curves`, within a tolerance per
node (`spec.py`); the Source node's own case records the detected tempo.
Each pass starts from a fresh engine, as after a file load,
because a seek keeps analyzer history; it runs twice and must repeat exactly,
and the I/O-thread mode must match. A new node type fails
`test_reference_goldens` until it has curves or an exclusion reason.
`-- --update` records the goldens; `tests/reference/report.py` plots drifting
curves. `test_reference_crosscheck` compares the goldens with librosa,
pyloudnorm and the signals' known content; checks ACE fails today are listed
in `KNOWN_FAILURES` as expected failures. Linux and Windows give the same
curves.

CI (`.github/workflows/ci.yml`) runs on Python 3.13 with Blender's NumPy: unit
tests with NumPy only and with the bundled wheels, ruff, the perf gates, the
installed extension in headless Blender 5.2 (smoke test and reference curves,
after checking that the manifest lists the pinned wheels), and the
cross-checks.

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
- Files saved with earlier versions are not converted. Nodes removed or
  renamed since (Audio Math, the Mid/Side, Attack/Release, Chromagram and
  Automatic Component splitters, Source Specs, Onset, the one-method RMS, LUFS,
  Peak and Stereo nodes) no longer work in them.
- Files are decoded into memory: a 5-minute stereo file at 48 kHz takes about
  115 MB.
- Live input quality depends on the transport. On Linux, SoundDevice needs
  PortAudio from the system.

## 13. Packaging and optional libraries

ACE is a Blender extension (`blender_manifest.toml`, Blender 5.2 or later).
Its libraries ship as wheels inside the platform zip, and Blender installs
them with it: nothing is installed at runtime, and an install behaves the same
on every machine. NumPy is never bundled; Blender ships its own.

- **Wheels.** `tools/wheels.txt` pins them: Numba with llvmlite, and
  SoundDevice with cffi and pycparser. `tools/fetch_wheels.py` downloads them
  for Blender's Python (3.13) and every platform in the manifest, and rewrites
  the manifest's `wheels` list; `blender --command extension build
  --split-platforms` then gives one zip per platform. A pure-Python wheel
  needed by only some platforms is tagged for them, so no zip holds two copies
  of one package (SoundDevice bundles PortAudio on Windows and macOS only).
- **Switches.** `deps.py` decides whether ACE uses an optional library: it
  imports, and its switch is on. The switches (Numba; SoundDevice for live
  input and the preview) live in `settings.json` in the extension's user folder,
  not in the preferences datablock, so they are read before any class
  registers, and reach worker processes through environment variables. Numba's
  applies at the next start, since kernels compile at import; SoundDevice's the
  next time a live source or the preview opens. Numba caches its kernels in the
  same user folder.
- **Preferences** only report: one row per library (Blender's `aud` included)
  with its version or why it is missing, plus the two switches.
