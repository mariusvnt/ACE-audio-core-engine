# ACE: Future Work

Planned changes that aren't built yet, in the order they should be done. Nothing here describes current behaviour; [ARCHITECTURE.md](ARCHITECTURE.md) does.

1. [Time model: render, bake, live](#1-time-model-render-bake-live)
2. [Core operator set](#2-core-operator-set): the node set, the add menu, motion and event nodes
3. [Live output window](#3-live-output-window)
4. [Presets as assets](#4-presets-as-assets)
5. [Per-instance parameters](#5-per-instance-parameters), only if needed

The extension packaging, the reference-curve tests, the fixes and removals, and the instance model (definitions, uses and shared signals) are built ([ARCHITECTURE.md](ARCHITECTURE.md) §13, §10, §8, §1 and §3). The reference curves of the nodes that remain must stay unchanged through the work below.

## 1. Time model: render, bake, live

### Why define time

**Today:** every value is computed as a stream, driven by the frame-change handlers. Scrubbing, rendering, baking and live input are each partly special cases of that stream, and several behaviours are undefined or wrong:

- **A render goes through the playback handlers.** `render_pre`, `frame_change_pre` and `frame_change_post` all run the graph, so one rendered frame can step stateful nodes up to three times. Worker values can be up to 50 ms old, and a compile on that frame writes nothing. Without Lock Interface nothing is written at all.
- **Streaming is not a function of the frame.** A seek reads frame f at `round((f − frame_start) · sr / fps)`. Consecutive frames instead advance a cursor by `N = round(sr / fps)`. At 44.1 kHz and 24 fps (N = 1838 against 1837.5) the blocks drift half a sample per frame, and a skip of one or two frames can leave the audio a frame behind.
- **State survives a seek.** Transient's running average, and Loudness's filters on a forward jump, carry on from before the jump. Stream Accumulator and Rolling Average reset only when time goes backwards, so a loop wrap looks like a rewind.
- **The bake depends on what was played.** It streams through the live graph's own tasks, starting from whatever state they hold, pauses live evaluation while it runs, and is lost if a recompile happens meanwhile. It lives in memory, so it's gone when the file is reloaded.

### 1.1 Clocks and the rule table

Every signal is `value(signal, frame)`. The engine's only job is to answer that question for the current frame. How it answers depends on the clock and the source.

**Clocks** (what decides the frame):

- **Timeline:** playback and scrubbing.
- **Render:** Blender's render job.
- **Live:** Blender's own playback in *Play Every Frame* mode, during a live session (1.5).

**Ways to get a value:**

- **Bake:** the stored curve, read by Blender with no Python (1.3).
- **Stream:** compute the frame. For a file source, the block of that exact frame (1.2). For a live input, the latest block captured.

**Rule table:**

| Clock \ Source | File | Live input |
| --- | --- | --- |
| Timeline | Bake if valid, else stream, settling stateful nodes after a seek | Stream: the latest block, and scrubbing doesn't apply |
| Render | Bake if valid. Otherwise stream, with a warning that the frames may not match playback | The take's bake (1.6). Otherwise the values are held at 0 and the render reports it |
| Live | Stream by frame, like a render, so a take can be replayed exactly | Stream: the latest block |

`core/clock.py` owns this table and is the only place that decides how a value is obtained.

- **Unbaked renders** keep the handler path with Lock Interface, but run the graph once per rendered frame, with local DSP only and no stale worker values.
- Blender gives handlers no way to stop a render, so a live source without a take can only be reported, in render_init and in the panel.
- Which thread runs the frame handlers during a render is to be checked first (Blender's API notes say it is not the viewport's). ACE asserts the main thread in its frame path today.

### 1.2 Exact streaming and seeks

- **The block of a frame is fixed.** Frame f of a file source always reads from sample `round((f − frame_start) · sr / fps)`, with the 4096 samples before it as history (ARCHITECTURE.md §8). Playing, seeking and baking then read the same samples. This changes the reference curves only where `sr / fps` is not an integer (44.1 kHz at 24 fps), and those are recorded again.
- **Stateless measurements are exact after a seek already.** Band loudness, RMS, Brightness, Pitch and Stereo depend on their block and its history only.
- **Stateful nodes settle after a seek.** Each task declares how many frames it needs to forget its past: Loudness in LUFS and Peak modes (K-weighting filters, and the Window), Transient (its running average, about 50 frames), Rolling Average (its window). On a seek, the engine resets them and replays that many frames before the target without writing them.
- **Running Total and Gate's Count can't settle**, since they remember indefinitely. They are exact only when baked, and the panel says so. There are no checkpoints: the bake is exact for them and simpler.
- **Stateful tasks run in one place.** Today LUFS, Peak and Transient are offloaded to worker processes with no affinity, so their state is split across copies. They get the `stateful` flag of section 2 and stay on the main thread.
- **Timeline playback drops frames** in Blender's default *Sync to Audio* mode, so an unbaked stateful node can differ from the render. A baked source is always exact.

### 1.3 Baking into Blender data

A bake writes each signal's curve over the frame range into ordinary Blender data. Consumers then read it with no Python at all, so the .blend renders anywhere, with ACE disabled or not installed.

- **Filled by streaming in order** from the range's first frame, on fresh tasks and a background thread. The bake is then exactly what a fresh playback computes, stateful nodes included, and live playback carries on while it runs. A test checks that bake and fresh stream match value for value.
- **Geometry Nodes: one baked mesh per proxy.** One point per frame, one attribute per sink, and the first frame's number in `ACE_frame0`. Stream groups sample index `floor(frame) − ACE_frame0` with *Clamp* on, instead of point 0:
  - a live proxy has one point, so the clamp keeps reading point 0;
  - a baked proxy reads its frame;
  - switching a source between live and baked swaps the proxy object's mesh. The node internals are the same in both modes, and the choice is saved in the .blend.
  - `ACE_Values` gets the same treatment.
- **Shaders: F-curves on the scene properties the shader sinks already read.** One F-curve per sink, converted to sample points (`FCurve.convert_to_samples`), in an ACE action on its own NLA track of the scene, so the scene's own action is untouched. The View Layer Attribute nodes don't change. A float image was the other option. It was dropped because Eevee loads float images at 16 bits by default while Cycles doesn't, GPU textures are limited to about 16384 pixels, and every sink group would need texture maths.
- **Live vs baked** is per source. A baked source's F-curves are active and its proxy uses the baked mesh. Switching it to live mutes the F-curves and swaps the live mesh back, and the frame handler writes only to live sources.
- **Identity.** Each curve is keyed by its signal key (ARCHITECTURE.md §1), the source file (path, size, modification time, content hash), the range and the fps. An edit invalidates only the signals it changes, and the panel marks a source's bake stale.
- **Persistence.** The bake is the Blender data, so it survives reloads. The key is stored on the source.

`core/bake_store.py` owns it.

### 1.4 Bake nodes downstream

A Geometry Nodes **Bake node or simulation zone** downstream of ACE nodes bakes the ACE analysis upstream as well, and releases it when its own bake is deleted.

- **What Blender gives:** each Geometry Nodes modifier lists its bakes (`NodesModifier.bakes`: the node, its id, mode, target and frame range), and `object.geometry_node_bake_single` bakes one. There is no handler and no job type for these bakes, and no documented flag that says whether one is baked.
- **ACE's own bake is the carrier.** A native bake steps through frames and records what the proxies hold. With the proxies on their baked meshes, every frame it steps reads exact values, with no Python involved.
- **ACE follows the native bake, it doesn't anticipate it.** Keeping ACE's bake for as long as a Bake node sits downstream would bake again on every edit of the ACE tree.
  - The compile finds, through groups, which ACE signals feed each downstream bake, and over which range.
  - The 0.25 s poll watches each downstream bake's state. When one becomes baked while ACE's upstream bake was not valid for its range, ACE bakes upstream, swaps the proxies, and bakes the native node again. One click gives an exact result, after a short second pass.
  - When the native bake is deleted, ACE releases its upstream bake, unless another downstream bake or the user's own bake still holds it, and the proxies go back to live.
  - Editing the ACE tree while the native bake exists bakes nothing, as in Blender, where a baked node ignores upstream edits until it is baked again.
- **To verify first:** whether Python can tell a baked bake from an unbaked one, for packed and disk bakes. If it can't, ACE keeps the upstream bake while a downstream bake exists and bakes again only the signals an edit changed, debounced like auto-compile, with live values meanwhile.
- **Shaders** have no Bake node. Cycles texture baking bakes one frame and reads the current value, so ACE only commits the current frame in `object_bake_pre`.

### 1.5 Live sessions

A live session is Blender's own playback, set up so that frames are never dropped or wrapped:

- **Play Every Frame** (`sync_mode = 'NONE'`): frames are always consecutive. If the scene can't keep up, playback runs late (time stretches) instead of dropping frames. The readout shows how far the show has slipped behind real time.
- **A long range.** On entry, `frame_end` is raised to Blender's maximum frame (1,048,574, about 4.8 h at 60 fps or 9.7 h at 30 fps). So is the simulation range when the scene uses a custom one. Both are restored on exit, and the session stops cleanly if it reaches the end.
- **The start point is the user's choice:** either `frame_start` (simulations reset there, as Blender does) or the current frame (simulations continue).
- **Simulation caches.** Blender keeps simulated frames in memory for scrubbing. A session can turn that off per object (`use_simulation_cache`) and restore it on exit. A prototype measures memory over a long session with the cache on and off.
- **Exit** when playback stops (`animation_playback_post`), or with Esc in the live window (section 3). Everything changed on entry is restored.

Keyframed animation past its last key behaves as its F-curves say (hold, or cycle with a Cycles modifier). That's the scene author's choice.

**Other rules while live:**

- No source resets or seeks on a frame change.
- **Live inputs read the latest block**, as today, at the capture clock minus the latency setting. Reading "the next block" would let latency grow without bound once playback runs late.
- File sources stream by frame, as backing tracks.
- The main window's 3D views can drop to solid shading to leave the GPU and main thread to the show, and are restored on exit.
- The side panel shows fps, end-to-end latency, late frames and drift.

### 1.6 Takes: record live, render later

A **take** is a recorded live session:

- the input audio saved as WAV, recorded from the device callback (the analysis blocks keep only the latest device block);
- the values computed live on each frame, written into the bake store. Live frames are consecutive, so tick n is frame `start + n`, exactly;
- MIDI and OSC events once ACE reads them. It reads neither today.

A take is a file source starting at its first frame, so the rule table applies: it can be scrubbed, baked and rendered. The use: perform live with an Eevee look, then render the same performance in Cycles at full quality.

**Rendering a take, two options:**

- **reproduce the live look:** the values recorded during the session, exactly as the audience saw them. Latency varied from frame to frame, so no single shift would reproduce it;
- **ideal sync:** a normal bake of the recorded WAV, with values exactly on the audio.

Recording writes the WAV and the values off the main thread. `core/takes.py` owns it.

### 1.7 Order

1. **Probes**, headless on Blender 5.2:
   - animated scene properties reach View Layer Attribute nodes in a render, from an NLA track;
   - Sample Index *Clamp* on a one-point and an N-point mesh, with ACE disabled;
   - the thread that runs frame handlers during a render;
   - whether a Geometry Nodes bake's state can be read, and what a bake job does with the handlers.
2. **Exact streaming and the clock layer:** fixed block placement, settling after a seek, stateful tasks on the main thread, the bake on fresh tasks, and the render rule. Rendering becomes defined, and exact for baked sources.
3. **Bake into Blender data** (1.3).
4. **Bake nodes downstream** (1.4).
5. **Live sessions** (1.5), with the simulation-cache prototype first.
6. **Takes** (1.6).

The live output window of section 3 needs live sessions.

## 2. Core operator set

### Why a core set

ACE has 11 nodes, after the removals that came first. Stereo and Stereo Vector still overlap, and none of them gives motion weight or turns a measurement into an event. The goal is a small set where each node answers a question the others can't:

| Node | Outputs |
| --- | --- |
| **Source** | Audio, BPM, Beats per Bar, Downbeat |
| **Frequency Splitter** | one Audio per band |
| **Loudness** | level, 0..1: RMS (default), LUFS or Peak |
| **Pitch** | 0..1 |
| **Brightness** | 0..1 |
| **Transient** | sharpness of the transient, 0..1 |
| **Stereo** | Balance (-1..1), Width (0..1) |
| **Envelope** | the input, smoothed |
| **Running Total** | the input summed over time |
| **Gate** | Open, Hit, Count |
| **Audio Preview** | none: plays a branch, in Geometry Nodes and shaders |

### Add menu

**Today:** the Audio Core Engine entry of the add menu has four submenus, each with an icon: Input (Source), Delta (Frequency Splitter), Stream (the measurements) and Utilities (Audio Preview, Rolling Average, Stream Accumulator).

The menus become:

```text
Input
Output
─────────
Audio
Controllers
```

| Menu | Nodes | What puts a node there |
| --- | --- | --- |
| **Input** | Source | audio enters the tree |
| **Output** | Audio Preview | audio leaves the tree, to be heard |
| **Audio** | Frequency Splitter, Loudness, Pitch, Brightness, Transient, Stereo | at least one Audio input |
| **Controllers** | Envelope, Running Total, Gate | no Audio input: they take a value and turn it into motion or events |

- Input and Output have no icon, and a separator sets them apart from Audio and Controllers. Audio and Controllers keep an icon each.
- Audio Preview has an Audio input, but it's where a branch ends, so it goes under Output.
- Until Envelope and Running Total replace them, Rolling Average and Stream Accumulator are listed under Controllers.
- A menu with no node for the current editor is hidden, as today.
- The categories in `ACE_NODES` become these four, so the menu stays registry-driven.

### Outputs on 0..1

Every measurement is a float between 0 and 1, so outputs can be swapped and a Gate's Threshold is always set on the same range. Balance is the exception: it keeps its range of -1 (left) to 1 (right), with 0 at the centre.

| Output | 0 | 1 | Scale |
| --- | --- | --- | --- |
| Loudness | -70 dB and below | 0 dB | linear in dB, as today |
| Pitch | 20 Hz, or no pitch | 5000 Hz | logarithmic, so equal musical intervals are equal steps |
| Brightness | 20 Hz and below | 20000 Hz | logarithmic |
| Transient | smooth | sharp | as today |
| Width | mono | side only | as today |

The ranges are fixed, also behind a Frequency Splitter, so values from different bands compare. Envelope keeps the scale of its input.

These are quantities rather than measurements, and keep their units: BPM, Beats per Bar and Downbeat, Running Total, and Gate's Count.

Pitch and Brightness get new reference curves.

### Stereo

Stereo and Stereo Vector become one node without modes, with two outputs: **Balance** and **Width**. Both are ratios that say where the sound is, whatever its volume. Levels are Loudness's job, so the left and right levels of Stereo Vector go.

### Motion and event nodes

ACE measures audio, but every measurement describes the current frame, so whatever it drives moves in short twitches. Giving that motion weight, and turning it into "something happens on each kick", takes smoothing, thresholds, holds and counters built from math nodes. That is hard to get right per frame, and impossible to get right across a scrub, because the state lives nowhere.

**Today:** two utilities keep state:

- **Rolling Average** is a symmetric box window. It lags on the way up as much as it blurs on the way down.
- **Stream Accumulator** adds its input once per frame, so the same scene moves twice as far at 48 fps as at 24 fps.

### Design rules

- These are new backend tasks. The frontends stay contracts with no maths, as the 3-layer model requires.
- **A node that writes values is described by `ACE_NODES` alone.**
  - Its value outputs give its attribute sockets ("Attribute ID", "Attribute ID 2", …).
  - A `stateful` flag keeps it on the main thread.
  - The reader, sinks, compiler, offload policy and both factories ask the registry instead of checking node names. A new node is then one registry entry and one task.
- **Times are in seconds**, and each step uses the duration of a frame. Results don't depend on fps.
- **Each stateful task is a pure step**: `(state, input, dt) → (state, output)`. This is what makes replay possible (see State and timeline).
- Booleans are Bool sockets in Geometry Nodes and 0/1 floats in shaders.

### Envelope, Running Total and Gate

**Envelope** replaces Rolling Average, in two modes, with Follow as the default:

- **Follow:** rises with Attack and falls with Release. Hold keeps the value at its peak for Hold seconds before it starts to fall. With Attack at 0, this is a peak hold, and fed a Gate's Hit it gives a hit-then-decay shape.
- **Spring:** the output follows the input like a weight on a spring. It speeds up toward the input, overshoots and settles. Speed (Hz) sets how fast it follows. Bounce sets how much it overshoots: 0 means not at all, 1 means it barely stops. This gives motion weight and follow-through.

Inputs used by one mode only are hidden in the other.

**Running Total** replaces Stream Accumulator. Its input is a rate and its output is the total over time (value × seconds), for a scalar or a vector:

- **Leak Time:** the total drifts back toward 0 when the input goes quiet. Energy pushes it forward, and silence lets it return. At 0 it never leaks, and sums from the start of the timeline.
- **Reset**, a Boolean input, for example from a Gate's Hit.

**Gate:** float in, three outputs:

- **Open:** true from the moment the input crosses Threshold until it falls below Re-arm Level (the hysteresis).
- **Hit:** true only on the frame it opens.
- **Count:** the number of hits so far, set back to 0 by the **Reset** input. A Floored Modulo math node turns it into "change colour every 4 kicks".

The **Min Interval** input prevents a new Hit sooner than this, so one kick gives one pulse.

### Musical time: the Source node

The Source node outputs only what musical time depends on: **BPM, Beats per Bar and Downbeat**. The Geometry Nodes one already outputs BPM and Beats per Bar. Downbeat is the time of the first downbeat in seconds from the source start; a file source starts at the scene's `frame_start`. Once these are known, every phase (beat, bar, phrase) is a function of time, so no node outputs a phase.

- **Downbeat estimate:** the tempo detection already finds the first downbeat, as the accented beat of the bar, and keeps it unused. It becomes a source property the user can override, like BPM and meter, and an output of the Source node.
- **Phases in Geometry Nodes** come from a **Beat Clock** preset (section 4). It turns Scene Time, BPM and Downbeat into the phase and index of the beat, the bar and the phrase (4, 8 or 16 bars).

### State and timeline

- **Rule:** a stateful node gives the same value on a frame whether that frame is reached by playing or by jumping.
- **File sources replay.**
  - Envelope and Gate's Open forget within a few time constants. They declare how many frames they need, and the settling after a seek (1.2) replays those frames before the target.
  - Running Total and Gate's Count remember indefinitely, so they are exact only when baked (1.3).
- **Until section 1 exists**, state resets on a rewind or a jump, and the panel says so.
- **Live sources** can't be replayed, so their state simply carries on.
- **Recompiles:** a recompile keeps every task whose signal remains, with its state (ARCHITECTURE.md §3), so these nodes keep theirs too.

### Core set order

1. Registry-driven value nodes and the `stateful` flag, then the merged Stereo node, the 0..1 outputs and the add menu.
2. Envelope and Running Total.
3. The Downbeat estimate.
4. Gate.
5. Settling after a seek for Envelope and Gate, once 1.2 exists.

Each new node gets reference curves (ARCHITECTURE.md §10), which also check that 24 and 48 fps give the same curve.

### Core set open questions

- Envelope's Spring overshoots, so on an input near either end of its range it leaves it. Proposed: let it, since clamping would flatten the bounce.
- Should Running Total and Gate outputs be baked like any other signal? This is the simplest and makes scrubbing exact.
- Should Gate's Re-arm Level be a free level, or a fraction of Threshold?

## 3. Live output window

### Why a live output window

Live visuals go to a projector or to a mixing tool (OBS, Resolume, projection mapping). Making a clean output from Blender by hand takes about ten steps that have to be redone every session: new window, rendered shading, camera view, overlays off, every panel hidden, fullscreen. One button does it.

The window is the output: the program that receives it captures the window. That adds no dependency to ACE, and ACE never copies frames from the GPU. A live session (1.5) needs a 3D view that keeps redrawing anyway, and this window is that view.

### The window

- **Go Live** opens a new window on a dedicated "ACE Live" workspace that holds a single 3D view. ACE creates that workspace once, so your own workspaces stay untouched.
- **The view:**
  - it looks through the scene camera (or one you choose), fitted to the window, in rendered shading;
  - overlays, gizmos, header, tool header, toolbar, sidebar and asset shelf are all hidden;
  - the area outside the camera frame is black (passepartout fully opaque), so a display with a different shape gets clean letterboxing.
- **Fullscreen** on the monitor where the window opens.
- **Going live:** opening the window starts a live session (1.5).
- **Leaving:** Esc ends live. A timer notices if the window was closed some other way, and cleans up either way: the live session's `frame_end`, simulation range and caches, and the main window's 3D view shading are restored.
- **The readout** (fps, end-to-end latency, late ticks) is in the main window's side panel, never in the image.

### Getting the image out

- **Directly:** the window is fullscreen on a projector or a second display.
- **Through a capture program on the same machine:** OBS Window Capture or Display Capture. The operating system's compositor captures on the GPU. From OBS the image can go to a projector, a recording, Resolume, or out over NDI through OBS's own plugin.

### Limits

- the output size is the window size: the display resolution, or the scaling done by the capture program;
- no alpha channel, so no keying downstream;
- the window must not be minimized: Windows can capture a window hidden behind others, but not a minimized one;
- the image leaves the machine only through the capture program.

### To verify with a prototype

- **Capture:** OBS captures the fullscreen Blender window reliably, with Window Capture (Windows Graphics Capture method) and with Display Capture.
- **Clean frame:** no interface pixels anywhere, with the camera frame filling the window exactly.
- **Cost:** rendered shading in this window, with the main window's 3D views dropped to solid (1.5).

### Code

`ui/live_window.py` holds the operator, the workspace, and window setup and cleanup. The clock itself stays in `core/clock.py`.

## 4. Presets as assets

Ship ready-made setups as an asset library that a user drags in from the Asset Browser:

- Geometry Nodes groups: bass pulse scale, spectrum bars, beat flash, stereo sway, Beat Clock (section 2).
- Materials: emission on level, hue on pitch.

Each preset is an ordinary user tree built from ACE nodes, so presets also exercise the instance model (ARCHITECTURE.md §1: nesting, reuse across objects). The asset library ships inside the extension and is registered when the extension loads.

## 5. Per-instance parameters

Uses of one ACE node share its sink names (ARCHITECTURE.md §1). Uses that compute different values, such as a group used twice with different band frequencies, or a Rolling Average behind two sources, are reported as conflicts in *Active ACE Instances*, and only one use reaches the sink. Different sources through the bundle already work in Geometry Nodes.

- **Geometry Nodes:** the bundle could carry a signal-key string that each ACE node extends with its operation and parameters (Value to String, Join Strings, possibly Hash Value), so Geometry Nodes computes the attribute names itself.
- **Shaders:** there are no string operations, so this would need generated copies of the groups.

Build this only if the conflict reports show a real need.

### Open questions

- Are different **parameters** per object needed (the same tree on several objects with different bands), or are different sources and different trees enough?
- Shaders: is a value per object (OBJECT attributes) ever needed, or is one global value per signal (View Layer, today) enough?
