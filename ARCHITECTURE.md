# Audio Core Engine Architecture

How ACE is built, and why. The code says *what* each part does; this document
keeps the decisions that span modules.

## 1. Model

ACE turns file or live audio into per-frame values for Geometry Nodes and
shaders (materials, lights, the World). It is built in three layers:

1. **Facade.** ACE node groups in Geometry Nodes and shader trees are interface
   contracts. They declare routing, analysis intent and outputs; they compute
   nothing. Users edit graphs in Blender's own editors, and the backend stays
   free of Blender's evaluation rules.
2. **Data engine.** The facades are read into abstract graphs, merged into
   signals, compiled into one task graph per source, and evaluated every
   frame. All DSP happens here.
3. **Acoustic monitor.** Preview playback rebuilds audible audio with
   time-domain filters derived from Layer 2 metadata (band limits). It never
   inverts spectral analysis: inverse-FFT reconstruction would reintroduce
   block artefacts and couple monitoring to analysis settings.

**Controllers are the one exception to layer 1.** Envelope, Running Total and
Gate shape values over time, and they compute in Blender: each is a Geometry
Nodes group with a Simulation Zone inside (`nodes/geometry/controllers.py`).
Audio analysis stays in the backend; shaping the values it gives is done
where the values are used. So any node can feed a controller (Math, Map
Range, another ACE output), each use has its own state, and Blender's
simulation cache and Bake record it. The engine has no controller tasks and
no sinks for them (§3, Sinks and Time).

**Every node has a role** in `ACE_NODES` (`vars.py`): `source`, `splitter`,
`measure`, `controller` or `preview`, and `native` for the controllers. The
reader, sinks, compiler, offload policy, workers and both factories ask for
roles; none checks a category. The categories are the add menu's submenus
(Input, Output, Audio, Controllers), built from the registry, and nothing
else.

The Frequency Splitter attaches a constraint to the audio chunk
(`band_hz_bounds` and lineage) instead of producing new signals. The
measurements behind it apply it while analysing, so a split costs metadata,
not a copy of the audio.

A chunk carries only the source's audio; nothing is filtered or smoothed at
the source. The Source node also outputs the source's BPM, Beats per Bar and
first Downbeat (§8, Tempo).

Every measurement is on 0..1, so outputs can be swapped and a Gate's
Threshold is always set on the same range: Loudness linear in dB from -70 dB,
Pitch (20 to 5000 Hz) and Brightness (20 Hz to 20 kHz) on a log scale, Width,
Transient. Stereo's Balance keeps -1 (left) to 1 (right).

The engine keeps three notions apart:

- **Definition:** a node tree or group as the user built it. It is parsed once
  into plain data (`nodes/definitions.py`) and kept until its content changes,
  so an edit parses only the edited tree.
- **Instance**, or use: an ACE node reached from an owner (a Geometry Nodes
  modifier, a material slot, a light, the World) through the user's group
  nodes, at any depth. A use binds what its path passes in: the modifier's
  input values, then each group node's own values. It reads the source whose
  proxy reaches its Source node, through group inputs too, so one group used
  with two sources reads both.
- **Signal:** one computation. Uses that compute the same thing share one
  signal and one task, which writes the sinks of every use
  (`core/engine/compiler/signals.py`). The same thing means the same
  operation, mode, parameters and inputs, down to the same source read at the
  same sample rate, block size and fps. The same RMS in two trees, or in a
  material and a modifier, runs once.

Uses of one node share its sink names. When they compute different values (a
group input that differs per use), two signals claim one sink: the use first
in id order keeps it, and the conflict is logged, named in the *Operators*
panel, and explained on the node when it is selected. Names per use are
FUTURE.md §3.

## 2. Packages

| Package | Role |
| --- | --- |
| `core/tasks` | DSP tasks, the audio chunk, tempo detection and the per-source runtime graph |
| `core/engine` | `ACEAudioEngine` (one per scene) and its components |
| `core/engine/compiler` | Stage A: uses to signals to task graphs |
| `core/engine/parallel` | Stage B: worker-process offload |
| `core/engine/frame` | Stage C: per-frame evaluation and sink commits |
| `core/engine/preview` | Preview routing, mixing and playback |
| `core/worker` | Graph evaluation inside worker processes (no `bpy`) |
| `core/input` | File decoding (Blender's `aud`), sound-device and UDP input |
| `core/proxy` | Proxy objects (only for sources a node reads), attribute and scene-property sinks |
| `core/clock.py` | Which clock decides the frame, and how each source gets its values under it (§3, Time) |
| `core/bake_store.py` | Bakes as Blender data, and their layout onto proxies and scene properties (§8, Bake) |
| `core/live.py`, `core/takes.py` | Live sessions, and recording them as takes (§3, Time) |
| `nodes` | Node identity, cached tree definitions, Source node bindings, linked values, the links ACE ignores |
| `nodes/geometry`, `nodes/shader` | Node-group factories, the controllers' Simulation Zones, the shader controllers' Geometry Nodes copy, and the instance walker that reads one use of a tree |
| `ui` | N-panel, add menu, operators and preferences |
| `deps.py` | Bundled libraries: which load, and why one did not (§13) |
| `tools` | Wheel fetching for the extension build (§13) |

## 3. Frame pipeline

1. **Stage A, compile (main thread).**
   - **Discovery:** the owners on the scene's objects and on objects instanced
     from collections, other scenes' included (modifiers, material slots,
     lights), and the World. A tree counts when it or a group it uses holds an
     ACE node. Discovery does not read `depsgraph.ids`: a depsgraph not
     evaluated yet lists deleted objects, and reading one crashes Blender.
   - **Reading:** each owner's tree is walked through its groups over the
     cached definitions (`nodes/geometry/reader.py`), one abstract node per
     use. Owners whose bound values are equal share one reading of the tree;
     owners whose values differ are read as separate uses.
   - **Signals:** the uses are merged into signals, and a compile is a
     difference. A task whose signal remains is kept with its state (an
     accumulator keeps its sum, an analyzer its history), new signals get
     tasks, and the tasks of signals gone are closed. A source whose signals
     are unchanged keeps its worker graph and its bake.
   - One `ACEAudioGraph` is built per source with only the signals its
     outputs need, and each output gets the sinks of every use it stands for.
2. **Stage B, offload (worker processes).** Numeric outputs run in worker
   processes from a serialized graph IR. Results come back asynchronously; a
   frame uses this frame's values, else values at most 50 ms old, else local
   DSP. The 50 ms window bounds how stale an output can look at any fps.
   Workers serve only a timeline that plays or scrubs: any other frame (a
   jump, a script's `frame_set`, a native bake stepping its frames, a render)
   is computed locally, here and now. The IR holds no sink names, so renaming
   a sink sends workers nothing.
3. **Stage C, commit (main thread).** Values are written, when they change,
   to proxy mesh attributes or to scene custom properties.

**Sinks.**

- **Geometry Nodes:** measurements read the proxy that the audio bundle
  carries from the Source node.
- **Controllers** have no sinks: Blender computes them.
- **Shaders:** one global value per signal, `scene[name]`. View Layer
  Attribute nodes read it inside a `.ACE_sink_<name>` group of the node's
  own, so nothing is added to the user's material. Inside every shader
  group, each input reaches an output, or Blender greys it out as unused: a
  value output is its scene property plus 0 x the inputs, an Audio bundle
  going through a Separate Bundle and a menu through a Menu Switch; a
  splitter's bands are Combine Bundles of the input and the split
  frequencies (Cycles and EEVEE render bundle nodes in 5.2, probed). Each change is followed
  by one `scene.update_tag()`: the evaluated scene sees a property write only
  after a tag. The shader Source node reads its source's tempo the same way
  (`ACE_bpm_<uid>`, `ACE_bpb_<uid>`, `ACE_db_<uid>`).
- **Controllers in shaders** (`nodes/shader/mirror.py`): shader trees have no
  Simulation Zone, so ACE copies the chain feeding each shader controller
  into a Geometry Nodes tree, `.ACE_Controllers_Mirror`, on the object
  `.ACE_Controllers` (in the scene's own collection: a hidden collection
  isn't evaluated). A shader measurement in the chain is read from its
  source's proxy, where the engine then writes it too; a Source node's tempo
  likewise; nodes Geometry Nodes also has (Math, Map Range, Clamp, Mix, Float
  Curve, Value...) are copied with their settings. The controllers' outputs
  are stored as attributes named by the shader node's sink names, and after
  every frame and every update `read_back` writes them to the scene
  properties the shader node reads. In a render, `frame_change_post` gets the
  render's own depsgraph, and the value reaches that same frame (probed on
  5.2). A node that can't be copied makes the link it feeds ignored (§7).

Block size follows the source sample rate and scene fps, so one block is one
frame of audio and analysis stays locked to the timeline. Every frame of a file
source reads from its own start sample, `round((f - frame_start) · sr / fps)`,
whether it is played, jumped to or baked, so all three read the same block.

A sink is written unless the float32 value it would store is unchanged. A
looser test (it was "within 1e-6") made a stored value depend on the frames
written before it, and a bake could not equal playback.

**Time.** Every value is a value at a frame; `core/clock.py` decides how a
source gets it:

| Clock \ Source | File | Live input |
| --- | --- | --- |
| Timeline (playback, scrubbing) | its bake, else computed | computed from the latest block |
| Render | its bake, else computed, with a warning | its take (§8), else held at 0, with a warning |
| Live (a live session) | its bake, else computed | computed from the latest block |

- **Render.** The render clock runs from `render_init` to `render_complete` or
  `render_cancel`: a render started from a script in background mode runs in
  place, not as a job, so `bpy.app.is_job_running("RENDER")` misses it. A
  rendered frame runs the graph once (`render_pre`, `frame_change_pre` and
  `frame_change_post` all ask for it), with local DSP only (worker values may
  be 50 ms old), and gets its values even on a compile frame. `render_init`
  lists what will not render exactly (a file source not baked, a live input
  without a take); the Baking panel shows it. Handlers cannot stop a render.
- **Seeks.** Stateless measurements are exact after a jump (their block and
  its history are read from the file). A stateful task declares how many
  frames it needs to forget the past (`settle_frames`): Loudness in LUFS and
  Peak modes (its Window and filters), Transient (64 frames). When its last
  frame is not the previous one (a jump, a rewind, frames dropped in
  playback, a task started mid-timeline), the frame runner resets it and
  replays up to 4 s before the target through the stateful tasks and what
  feeds them, writing nothing, never before `frame_start`. Live inputs are
  not replayed.
- **Controllers** follow Blender's simulation rules (probed on 5.2). A frame
  in the simulation cache or a bake is exact. A jump ahead past the cache, or
  a frame skipped in playback, takes one step from the last state: a step
  lasts the Scene Time that passed since the last one, kept in the zone's
  "Time" state, because the zone's Delta Time is one frame long even then.
  Without the cache, a jump back starts again from the first frame. A render
  reads the viewport's simulation: frames past the cache render frozen, so an
  exact render needs the controllers cached over the range or baked (§8,
  Bake); `render_init` notes the objects that have controllers.
- **Live sessions** (`core/live.py`) are Blender's own playback in *Play Every
  Frame* mode (`sync_mode = 'NONE'`): frames are consecutive, and a scene that
  can't keep up plays late instead of dropping frames. `frame_end` is raised to
  Blender's last frame (1,048,574), and so is the simulation range when the
  scene uses a custom one, so playback never wraps. Simulation caching can be
  turned off on the objects that simulate, and 3D views dropped to solid. All
  of it is restored when playback stops (`animation_playback_post`). The Live
  panel shows the measured fps, late frames and the drift behind real time.
- **Takes** (`core/takes.py`): a recorded session writes each live input to a
  WAV from the device callback (the analysis keeps only the latest block),
  after silence from `frame_start` to the frame the session started on, and
  stores the values shown on each frame in the live source's bake store
  (origin "take"). The WAV is added as a file source. With *Use Take* on, the
  live source reads its take: a render shows the show as it was seen.

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

Nothing ACE needs to tell nodes apart is stored on them.

- **Categories** are the add menu's submenus (`input`, `output`, `audio`,
  `controllers`); ids and golden keys carry them, but no code branches on one.
- **Groups** (`nodes/groups.py`). Each kind has one shared group per tree
  type, its template, named after the node ("Envelope"; "Envelope (Shader)"
  in shaders, since groups of both types share one name space). A visible
  name is what lets Blender's link-drag search and Add › Group offer it. A
  template is found by its `ace_kind` and `ace_template` properties, never by
  name, so a user's own group of that name is never taken for it; a file
  with the old hidden names (`.ACE_<category>_<type>`) has its templates
  renamed on load. They are built on each load, and by a one-shot timer when
  the add-on is enabled: `bpy.data` can't be written while it registers. The groups a placed node gets are hidden: a splitter's
  variant per band count (`.ACE_audio_frequency_splitter_<n>_bands`), a
  shader node's sink group, a shader Source's source group.
- **Placing a node** (`nodes/placement.py`). The add menu places what it
  adds; a node added by Blender's search, Add › Group or a paste is placed
  when its tree next changes (the depsgraph handler, and again at compile): a
  splitter gets its band count's variant, a Source the active source, the
  node its label, mode and preview flag. Its sink names come from the
  compile's repair. A placed node carries `ace_placed`, so a copy keeps its
  settings.
- **Copy and paste across editors** (`ui/clipboard.py`). Blender refuses to
  paste a group node into a tree of the other type, so Ctrl+C and Ctrl+V in
  the Node Editor go through ACE: the copy also notes the selected ACE nodes
  (kind, label, settings, input values, links), and a paste into the other
  tree type adds them with that type's groups and links them again, to each
  other and to the nodes Blender pasted. Ctrl+V pastes at the mouse, as
  Blender's own does. The Edit menu's Copy and Paste stay Blender's own.
- **Node ids** are computed: `ACE_<category>_<type>_<blake2b(tree key:group
  path/node name)>` (`nodes/identity.py`). The tree key is the tree's
  `session_uid`, which is stable across undo. The first use of a tree keeps the
  plain key, so a tree used once has the ids the sidebar computes from the
  tree alone; further uses with other values add a digest of those values.
  Renaming an ACE node changes its id and recompiles once.
- **Signal ids** (`S_<category>_<type>_<blake2b>`) leave out sink names,
  Attribute ID values, proxy references and the node's place in the editor.
- **Node kind** comes from the group's `ace_kind`, or from its name for a
  group without one (a splitter variant of an older file).
- **Sources** are chosen by the user on the Source node. The add menu and the
  node's source menu in the sidebar set its Proxy Object (Geometry Nodes) or
  its source group `.ACE_SHADER_source_<uid>` (shaders). A proxy carries
  `["ace_source_id"]`: ACE maps the proxy to its source and never writes the
  node.
- **Sink names** belong to the user's tree. The add menu writes
  `ACE_<metric>_<6 hex>` once. A compile writes nothing into user trees, with
  one exception: a name that two nodes claim (after Shift+D, a copied
  material, an append) or a missing name is replaced on the later node, in a
  stable order, once (`SinkResolver.repair_sink_names`). Linked trees claim
  their names first, since they cannot be written. The smoke test checks
  that a compile and playback leave every user tree as it was.
- Proxy object names and digests come from `hashlib.blake2b`, so the same
  scene always produces the same names, and keys fit Blender's length limits.

## 6. Worker processes

### 6.1 Rules

- Workers never import `bpy`. `bpy` (data, context, trees, handlers) is used
  on the main thread only.
- Only measurements are offloaded, and only stateless ones (`"stateful"` in
  `ACE_NODES`: Loudness in LUFS and Peak modes, Transient, stay on the main
  thread). Each worker keeps its own copy of a task, so a stateful task
  offloaded would split its state across copies.
- A job older than 66 ms is dropped. A late result would either stall the
  frame or contaminate a later one.
- At most two jobs per source are in flight.

### 6.2 Pool lifecycle

A spawned worker imports the DSP stack (~4 s) before it can run a job. So:

- the import runs in the process initializer, and a cold process never takes a job;
- after the pool starts, one warm-up job per needed worker (shards x in-flight
  limit) must answer before any real job is sent; local DSP covers that time;
- the pool survives recompiles, and so do the workers' graphs: a source's IR
  is sent again only when its digest changes, and values already received
  stay valid, since they are keyed by signal. Undo and engine resets start a
  new job generation, and workers drop older graphs themselves;
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

- **Render safety.** Writes from frame handlers stop while a render job runs
  with `use_lock_interface` off; viewport renders keep updating. A baked
  source needs no writes at all (§8, Bake).
- **Auto-compile.** A depsgraph update of a node tree, or of the material,
  light or World holding one, marks that tree dirty; object updates mark
  nothing. A 0.25 s poll compares which trees each object uses (modifiers,
  material slots, modifier input values) and catches the rest. Writing a
  node's custom property sends no update, so operators that set one (the mode
  menu) ask for the recompile themselves. A rebuild runs after a quiet
  window, or after 2 s of continuous edits, so link-editing does not rebuild
  on every click. It cannot be turned off; *Manual Engine Reset* drops every
  engine, task, worker and cache and rebuilds, as a file load does.
- **No feedback loop.** Proxy writes re-evaluate the trees that read them,
  and the depsgraph reports that like an edit. A marked tree whose content is
  unchanged since the compile does not recompile.
- **Ignored links** (`nodes/link_rules.py`). A link ACE can't use is drawn
  invalid (`NodeLink.is_valid = False`), and the N-panel says why, on the
  selected node: a linked Attribute ID, a Proxy
  Object not fed by a group input, audio through a node other than a
  reroute, a control (a Window) that isn't a constant, a split frequency
  through a node `driven_values` can't compute, a shader controller's chain
  through a node the Geometry Nodes copy can't hold. Its flows are dropped,
  so ACE really ignores it, and the input keeps its own value; a constant
  chain into a control is read as its value. The flag only shows (Blender
  still evaluates through the link), and Blender resets it when a tree's
  nodes or links change and on file load, so ACE marks the links again after
  every compile and tree update. It is not part of a tree's content
  signature, so marking never recompiles. Controllers take single values
  (`structure_type` `SINGLE`): Blender itself marks a field linked to one.
- **Undo.** The engine holds no RNA references across undo; an undo bumps the
  undo generation, drops in-flight results and every task, and rebuilds.
  No thread but the main one reads Blender data: an undo or a delete frees
  it at any time. The prefetch thread reads what it needs from a snapshot
  the main thread takes when it asks for the read (`SourceTask._context_view`),
  and leaves a source that changed to the main thread; the bake reads
  snapshots of its source and settings. (Blender crashed on an undo while the
  prefetch thread read a source's type.)
- **Mode inputs.** A node shows only the inputs its mode uses: Loudness's
  Window in LUFS, Envelope's Attack, Hold and Release in Follow and Speed and
  Bounce in Spring (`mode_inputs` in `ACE_NODES`; for Envelope the mode is its
  Mode menu, followed on each tree update). A linked input always shows.
  Blender's own greying can't do it: it doesn't trace a Simulation Zone's
  state.
- **Internals layout** (`nodes/layout.py`). Every group ACE builds is laid
  out along its flow of data: one column per step from Group Input to Group
  Output, nodes in a column ordered by what feeds them, spaced by their
  estimated size; the controllers' steps are labelled. The smoke test checks
  that no two nodes overlap and that every link runs left to right.
- **UI sync.** A timer refreshes the N-panel, throttled and scoped to the
  active source. The *Operators* list shows the compiled use of the tree in
  the Node Editor, so a group's operators appear under the source each use
  gives them. While the tree has edits not compiled yet, the list reads the
  tree itself, so it follows edits even with auto-compile off. Controllers
  aren't in the engine's model: they are listed from the tree, under the ACE
  node they read through any other nodes, with their inputs and description
  when selected. An ACE node with no row (it reads no source) selected in the
  Node Editor shows in the panel until a row is selected. *Audio Inputs* and a selected node's sink conflicts read
  only the compiled model (`ui/panel/instances.py`).
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
  files are read, through the `wave` module; registration logs it as an error. `aud`
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
  - **Stereo** behind a splitter: Balance and Width come from the in-band
    powers of left, right, mid and side, measured like band loudness. Without
    a splitter they are measured on the whole block. Both are the two values
    of one task's result.
  - **Brightness:** the centroid of the magnitude spectrum of a periodic Hann
    window, as long as the next power of two of the block. A band counts only
    its own bins, with the same bounds test as band loudness, and a band too
    narrow to hold a bin reads 0. The bands of one block share one spectrum,
    cached by source, frame, sample rate, length and a checksum of the
    samples, so another file or frame rate never gets a stale one. The
    centroid is mapped from 20 Hz..20 kHz to 0..1 on a log scale
    (`stream.hz_to_unit`).
  - **Pitch:** YIN over 4096 samples, comparing the newest 1024, from 20 to
    5000 Hz and inside the chunk's band, mapped to 0..1 on a log scale, so
    equal musical intervals are equal steps; 0 when there is no pitch.
- **Tempo.** A file source's tempo is detected on its own when the source gets
  a file or the file changes on disk, unless the user typed a BPM, Beats per
  Bar or Downbeat; *Detect Features* replaces typed values. One background thread
  (`TempoService` in `core/tasks/tempo.py`) builds an onset envelope (10 ms
  steps, log-compressed spectral flux) over up to 10 minutes, estimates the
  tempo and meter on its first 15 s, then fits the beat grid over all of it:
  - **BPM:** the beat period from the envelope's autocorrelation, between 40
    and 240 BPM. A preference around 120 settles half and double tempo, and a
    period must also repeat at 2 and 4 times its length (bars of 2 or 4) or at
    3 and 6 (bars of 3).
  - **Beats per Bar:** each beat's strength is the envelope peak within a sixth
    of a beat, so off-beats never count. The strengths are folded by beat
    number for bar lengths 2 to 8, and the length where one fold stands out
    most clearly wins; a shorter bar wins a near tie. Without a clear accent
    pattern the answer is 4, and there is no downbeat.
  - **Downbeat and the grid fit:** the downbeat is where the envelope summed
    over all downbeats peaks. A phase computed from a fixed BPM drifts (an
    error of 0.1 %, what 15 s give, is half a beat after 4 minutes at 120
    BPM), so the onsets near each beat of the grid are fitted with a straight
    line by weighted least squares (`_fit_grid`), over spans that double so
    the onsets never drift out of reach, and the fit gives the BPM and the
    downbeat. A fit whose onsets lie more than an eighth of a beat off the
    line, on average, is dropped and the estimate stays. The downbeat
    reported is the first of the file.

  A file without a steady pulse gets no tempo, and the source keeps its BPM.
  Results are cached per file path and modification time and written to the
  source on the main thread. Every tick, each source writes its BPM, Beats
  per Bar and Downbeat on its proxy mesh (`ACE_bpm`, `ACE_beats_per_bar`,
  `ACE_downbeat`); the Geometry Nodes Source group reads them there, and
  sink pruning leaves them alone. A source with a shader Source node also
  writes them as scene properties (§3, Sinks). The Downbeat is written in
  scene seconds, the clock of Scene Time's Seconds output: a file starts at
  `frame_start`, so it is `frame_start / fps` plus its time in the file, and a
  beat's phase is `(Seconds - Downbeat) * BPM / 60`. A live input writes 0.
- **Preview.** Audio comes from manual preview sinks, or from links to the
  tree's audio output when there is none. Branches are mixed, limited and
  played only while the timeline plays. A feeder thread paces delivery at the
  device block rate because Blender timers jitter.
- **Bake** (`core/engine/bake.py`, `core/bake_store.py`). A file source's
  outputs are computed over the frame range on a thread, on a fresh copy of
  its graph (`fresh_graph_copy`: new tasks, snapshots of the source and
  settings, no prefetch thread). The bake is therefore what a playback from
  the first frame computes, bit for bit, and playback goes on meanwhile; a
  recompile or an undo leaves a running bake alone.
  - **Store:** one hidden mesh per source, `.ACE_bake_<uid>` (fake user), one
    point per frame, one attribute per output key. An output key is a signal
    identity, so a curve stays right for its key whatever else is edited, and
    an edit leaves only the signals it changes unbaked. Its custom properties
    hold the range and the file (path, size, modification time, content
    hash). A store whose file changed is dropped; one whose file is missing is
    kept, since the bake is then all there is.
  - **Layout,** after every compile and bake: a proxy gets
    one point per frame, its baked attributes their curves, and `ACE_frame0`
    the frame of point 0. The Geometry Nodes groups sample point
    `floor(frame) - ACE_frame0` with Sample Index clamping, so a live proxy
    (one point) reads the same as before; outputs not baked are written live
    to every point. Baked scene properties are animated by F-curves converted
    to sample points, in the action `ACE_Bake` on its own NLA track of the
    scene. No Python runs to read any of it: a baked .blend renders with ACE
    disabled.
  - **Live and baked:** `source["ace_bake_use"]` off lays a source out live
    and keeps its store. A live input reads its take only with it on.
  - **Controllers:** once the sources are baked, the Simulation Zone of every
    controller the scene's objects run (the shader controllers' copy
    included) is baked with `geometry_node_bake_single`, one zone at a time,
    so the user's other simulations are left alone
    (`controllers.bake_controllers`). The modifier's bakes list each zone, nested
    groups included; a zone is a controller's when its Simulation Output is in
    a controller's group.
  - **Downstream Bake nodes.** ACE bakes nothing for a Bake node or a
    simulation zone downstream of ACE nodes. Python can't tell whether a
    native bake holds data, so holding ACE's bake for one would mean
    re-baking on every edit while the Bake node sits idle. A native bake
    steps its frames in order from its first, and ACE computes each one
    locally, so it records what a playback from that frame computes,
    stateful nodes included.

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
one tree holding every measurement, mode and splitter band, and each output
is compared frame by frame with the goldens in `tests/reference/curves`,
within a tolerance per node (`spec.py`); the Source node's own case records
the detected tempo and downbeat. The fixtures are too short (4 s) for a
downbeat or a grid fit: those are tested on synthetic onset envelopes
(`test_tempo_grid`).

**Controllers** are checked against the same maths in Python
(`tests/controller_reference.py`): the unit tests check the maths (a step or
an impulse in seconds gives the same curve at 24 and 48 fps), and the smoke
test plays the node groups at 24 and 48 fps and compares them with it, a
baked jump and a shader controller included.
Each pass starts from a fresh engine with its tasks dropped and no bake, as
after a file load; it runs twice and must repeat exactly, and the I/O-thread
mode must match. Two more passes check time: a bake of the same case must
equal the played curves bit for bit (`bake_vs_stream`), and a jump to the
middle then playing on must match within tolerance (`seek_then_play`). A new
measurement fails
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
cross-checks. That full set runs on pull requests and by hand
(`workflow_dispatch`). A push runs only ruff and the NumPy-only unit tests,
and nothing when its branch has an open pull request, which already tests it.

## 11. Adding a node type

1. Declare it in `ACE_NODES` (`vars.py`), with its role and the category of
   its submenu. A new method of an existing measure is a mode of that node
   (`modes` in its definition), not a new node type.
2. **A measurement:** implement the task in `core/tasks/` and add it to its
   family's table (`stream._STREAM_TASKS`); `core/tasks/factory.py` creates
   it for the engine and the workers. The factories build its groups from
   the registry: one Attribute ID per value output.
3. **A controller** (`"native": True`): add its builder to
   `nodes/geometry/controllers.py`, its maths to
   `tests/controller_reference.py`, and a smoke check comparing the two. It
   takes single values; a menu input needs a `menus` entry, and a menu that
   is its mode a `mode_menu` and `mode_inputs`. Label its steps
   (`_Builder.named`): the layout puts them in order, the labels say what
   they compute.
4. Add UI controls if it has runtime-editable parameters.
5. Add tests for its output and its topology sync.

## 12. Constraints

- Processing is frame-block based: tuned for reactivity, not offline fidelity.
- At a subframe (motion blur), Geometry Nodes read the frame's baked value
  (floor), while baked scene properties interpolate between frames.
- Eevee reads View Layer attributes at half precision (about 1e-4), live or
  baked; Cycles reads them exactly.
- An unbaked playback can drop frames (Blender's default *Sync to Audio*):
  stateful nodes settle after each jump, so they are close, not exact. A baked
  source, or a live session, is exact. Controllers time their steps with
  Scene Time, so a skipped frame costs them only the input of that frame.
- A controller renders exactly only on frames its simulation has cached or
  baked; Bake bakes them.
- Controllers in shaders: only those directly in a material's, the World's
  or a light's tree (not inside shader groups) are computed, and their chains
  may hold only ACE outputs and nodes Geometry Nodes also has. They need ACE
  installed to render: the copy's values reach the shader through Python.
- An ignored link is only drawn invalid: Blender still evaluates through it,
  so a linked Attribute ID or Proxy Object shows something other than what
  ACE writes until it is fixed.
- Takes record device inputs only (not UDP), and no MIDI or OSC.
- Uses of one ACE node share its sink names, so uses that compute different
  values conflict, and only one of them reaches the sink (§1).
- Shader values are one global value per signal, not one per object.
- Trees linked from a library are compiled, but ACE never writes them: their
  sink names are not repaired, and local trees give way to them.
- Files saved with earlier versions are not converted. Nodes removed or
  renamed since (Audio Math, the Mid/Side, Attack/Release, Chromagram and
  Automatic Component splitters, Source Specs, Onset, the one-method RMS, LUFS,
  Peak and Stereo nodes, Stereo Vector, Rolling Average, Stream Accumulator)
  no longer work in them, and node ids and group names changed with the
  categories (Input, Output, Audio, Controllers).
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
- **No switches.** ACE always uses Numba, and SoundDevice for live input and
  preview playback. `deps.py` only records whether each one imports (worker
  processes check for themselves); one that does not leaves ACE on its
  fallback paths: NumPy and Python filters and limiter, no live input or
  preview playback. Numba caches its kernels in the extension's user folder.
- **Failures are logged, not shown.** At registration, each library that did
  not load (Blender's `aud` included) is logged as an error with the reason and
  what is lost. The preferences hold only the extended-logs option.
