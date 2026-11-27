# ACE: Future Work

Planned changes that aren't built yet, in the order they should be done. Nothing here describes current behaviour; [ARCHITECTURE.md](ARCHITECTURE.md) does.

1. [Time model: render, bake, live](#1-time-model-render-bake-live)
2. [Core operator set](#2-core-operator-set): the node set, motion and event nodes
3. [Live output window](#3-live-output-window)
4. [Presets as assets](#4-presets-as-assets)
5. [Per-instance parameters](#5-per-instance-parameters), only if needed

The extension packaging, the reference-curve tests, the fixes and removals, and the instance model (definitions, uses and shared signals) are built ([ARCHITECTURE.md](ARCHITECTURE.md) §13, §10, §8, §1 and §3). The reference curves of the nodes that remain must stay unchanged through the work below.

## 1. Time model: render, bake, live

### Why define time

**Today:** every value is computed as a stream, driven by the frame-change handlers. Scrubbing, rendering, baking and live input are each partly special cases of that stream, and several behaviours are undefined:

- what a render shows for a live source;
- whether a rendered frame matches what playback showed;
- what happens when the timeline loops. A wrap looks like a rewind, so Stream Accumulators reset and the preview buffers are flushed, which is wrong during a live show.

### 1.1 Signals are functions of time

Every signal is `value(signal, t)`, with `t` a position in the source audio. The engine's only job is to answer that question for the current `t`. How it answers depends on the clock and the source.

**Clocks** (what decides `t`):

- **Timeline:** playback and scrubbing. `t` comes from the current frame.
- **Render:** Blender's render job. `t` comes from the exact frame being rendered, and there's no hurry.
- **Live:** ACE's own clock (1.4), independent of timeline playback.

**Ways to get a value:**

- **Bake:** look up the stored curve (1.3).
- **Random access:** for a file source, analyse the exact sample window at `t` (1.2).
- **Stream:** read the next block from a live input. Only possible "now".

**Rule table:**

| Clock \ Source | File | Live input |
| --- | --- | --- |
| Timeline | Bake if valid, else random access | Stream: the value now, and scrubbing doesn't apply |
| Render | Bake if valid, else random access. **Never stream** | A recorded take (1.5). If there is none, the render stops with a clear message |
| Live | Real-time playback of the file as a backing track | Stream |

`core/clock.py` owns this table and is the only place that decides how a value is obtained. The render handlers stop calling the playback frame path and ask the clock layer for "value at this frame".

**Stateful nodes** (Envelope, Running Total and Gate, section 2) depend on everything before `t`. Under random access their state is rebuilt from checkpoints saved every N frames. In practice, stateful signals are baked before rendering, and the UI says so.

### 1.2 Random access for file sources

- File sources gain `read_at(t, window)` next to `read_block`.
- The block for frame f is the same one streaming would use: samples `[f·N, (f+1)·N)` with `N = sample_rate / fps`, plus the history the analyser needs (an FFT window longer than a block reads samples before it).
- **Analyser state:** Loudness in LUFS and Peak modes runs K-weighting filters (LUFS also averages over its Window), and Transient compares each frame with the previous spectrum and a running average. Random access runs a pre-roll before `t`, long enough for that state to settle, so the result matches streaming within a tolerance set by the reference curves (ARCHITECTURE.md §10). Where the tolerance is too loose, use the checkpoints instead.
- **Contract, checked by the tests:** random access = stream within tolerance, and bake = random access = render, value for value.

### 1.3 Baking into Blender data

**Today:** the bake is a cache kept in memory (ARCHITECTURE.md §12). It's lost when the file is reloaded, and rendering still needs ACE running in the frame handlers. That means:

- a render farm or another computer needs the add-on and its dependencies installed;
- final frames go through the same handler path as playback, with its render-safety guards.

A bake writes each signal's full curve over the frame range into ordinary Blender data. Consumers then read it with no Python at all:

- **Geometry Nodes:** one baked mesh per source, with one point per frame and one attribute per signal. The ACE node's internals switch from sampling point 0 of the live proxy to sampling point `frame - frame_start` of the baked mesh (Scene Time → Sample Index). The node interface is unchanged.
- **Shaders:** a float image, one row per signal and one column per frame, packed in the .blend. An ACE shader node samples it at the current frame (Scene Time → coordinate, Closest interpolation). The baked image needs to be checked against the image size limit for long ranges; past that, use several rows per signal.
- **Signal identity:** each baked curve is keyed by its signal key (ARCHITECTURE.md §1) plus the hash of the source file. A curve whose key still matches is reused after edits elsewhere; a changed key marks the bake stale in the UI.
- **Live vs baked:** each source switches between the two. With the source in baked mode, the frame handler does nothing for it.

Results:

- the .blend renders anywhere, with ACE disabled or not installed;
- scrubbing, viewport and render read exactly the same numbers;
- the bake survives reloads.

The bake uses the signal keys and is filled by random access (1.2), so it doesn't depend on playing the stream through. `core/bake_store.py` owns it.

### 1.4 Live clock

Under the live clock, ACE drives time instead of following timeline playback.

**Each tick, at the scene's fps:**

1. read the next block from the live inputs;
2. compute the signals;
3. commit the values;
4. advance the scene by exactly one frame (`frame_set`).

The frame-change handler does nothing for ACE while live, so a tick never runs twice.

**Simulation nodes keep working.** They step only from one frame to the next, so the clock guarantees:

- **Frames are always consecutive.** They are never skipped and never wrapped. If the scene can't keep up, ticks run late (time stretches) instead of frames being dropped. The readout shows how far the show has slipped behind real time.
- **A long range.** On entry, `frame_end` is raised to Blender's maximum frame (1,048,574, about 4.8 h at 60 fps or 9.7 h at 30 fps) and restored on exit. Live stops cleanly if it reaches the end.
- **The start point is the user's choice:** either `frame_start` (simulations reset there, as Blender does) or the current frame (simulations continue).
- **To verify with a prototype:** the memory used by simulation caches over a long session. Blender keeps simulated frames in memory for scrubbing. If that grows without limit, find whether it can be capped; otherwise document a maximum session length.

Keyframed animation past its last key behaves as its F-curves say (hold, or cycle with a Cycles modifier). That's the scene author's choice.

**Other rules while live:**

- Live sources never reset on a frame change.
- File sources play in real time, as backing tracks.
- The main window's 3D views drop to solid shading (or stop redrawing) to leave the GPU and main thread to the show. They're restored on exit.
- The side panel shows fps, end-to-end latency, and late ticks.

### 1.5 Takes: record live, render later

A **take** is a recorded live session:

- the input audio saved as WAV;
- the MIDI and OSC events, once ACE reads them, logged with timestamps;
- the take's first tick mapped to its scene frame. Since live frames are consecutive, tick n is frame `start + n`, exactly.

A take is a file source, so the rule table applies: it can be scrubbed, baked and rendered exactly. The use: perform live with an Eevee look, then render the same performance in Cycles at full quality.

**Rendering a take, two options:**

- **ideal sync:** values exactly on the audio;
- **reproduce the live look:** values shifted by the latency measured during the session, as the audience saw them.

Recording writes WAV and the event log off the main thread. `core/takes.py` owns it.

### 1.6 Order

1. Clock layer, random access for file sources, and the render rule. Rendering becomes defined and exact.
2. Bake into Blender data (1.3).
3. Live clock (1.4), with the simulation-cache prototype first.
4. Takes (1.5).

The live output window of section 3 needs the live clock.

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
  - Envelope and Gate's Open forget within a few time constants, so a short pre-roll before the target frame settles them. That needs random access (1.2) for the measurements upstream.
  - Running Total and Gate's Count remember indefinitely. They need checkpoints saved every N frames, or the bake (1.3).
- **Until section 1 exists**, state resets on a rewind or a jump, and the panel says so.
- **Live sources** can't be replayed, so their state simply carries on.
- **Recompiles:** a recompile keeps every task whose signal remains, with its state (ARCHITECTURE.md §3), so these nodes keep theirs too.

### Core set order

1. Registry-driven value nodes and the `stateful` flag, then the merged Stereo node and the 0..1 outputs.
2. Envelope and Running Total.
3. The Downbeat estimate.
4. Gate.
5. Replay on jump, once random access (1.2) exists.

Each new node gets reference curves (ARCHITECTURE.md §10), which also check that 24 and 48 fps give the same curve.

### Core set open questions

- Envelope's Spring overshoots, so on an input near either end of its range it leaves it. Proposed: let it, since clamping would flatten the bounce.
- Should Running Total and Gate outputs be baked like any other signal? This is the simplest and makes scrubbing exact.
- Should Gate's Re-arm Level be a free level, or a fraction of Threshold?

## 3. Live output window

### Why a live output window

Live visuals go to a projector or to a mixing tool (OBS, Resolume, projection mapping). Making a clean output from Blender by hand takes about ten steps that have to be redone every session: new window, rendered shading, camera view, overlays off, every panel hidden, fullscreen. One button does it.

The window is the output: the program that receives it captures the window. That adds no dependency to ACE, and ACE never copies frames from the GPU. The live clock (1.4) needs a 3D view that keeps redrawing anyway, and this window is that view.

### The window

- **Go Live** opens a new window on a dedicated "ACE Live" workspace that holds a single 3D view. ACE creates that workspace once, so your own workspaces stay untouched.
- **The view:**
  - it looks through the scene camera (or one you choose), fitted to the window, in rendered shading;
  - overlays, gizmos, header, tool header, toolbar, sidebar and asset shelf are all hidden;
  - the area outside the camera frame is black (passepartout fully opaque), so a display with a different shape gets clean letterboxing.
- **Fullscreen** on the monitor where the window opens.
- **Going live:** opening the window starts the live clock (1.4).
- **Leaving:** Esc ends live. A timer notices if the window was closed some other way, and cleans up either way: the live clock's `frame_end` and the main window's 3D view shading are restored.
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
- **Cost:** rendered shading in this window, with the main window's 3D views dropped to solid (1.4).

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
