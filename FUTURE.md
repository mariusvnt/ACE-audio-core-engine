# ACE: Future Work

Planned changes that aren't built yet, in the order they should be done. Nothing here describes current behaviour; [ARCHITECTURE.md](ARCHITECTURE.md) does.

1. [Infrastructure](#1-infrastructure): extension packaging, presets as assets, reference-curve tests
2. [Instance model](#2-instance-model-definitions-instances-signals): definitions, instances, signals
3. [Time model: render, bake, live](#3-time-model-render-bake-live)
4. [Event and control nodes](#4-event-and-control-nodes)
5. [Live output window](#5-live-output-window)

## 1. Infrastructure

This comes first because the later sections rewrite the compiler and the sinks. They need tests that pin down the DSP numbers and an install that behaves the same on every machine.

### 1.1 Standard extension packaging

**Today:** ACE is a legacy add-on (`bl_info` in `__init__.py`). Its optional libraries (AudioFlux, Numba, SoundFile, SoundDevice) are installed at runtime by `ui/prefs_library_manager.py`, which runs `pip install --target` into a folder of its own. On top of that there is a `_vendor` folder. Installing needs a network connection, can fail midway, and gives different results from one machine to another.

**Target:** a Blender extension (`blender_manifest.toml`, Blender ≥ 5.0):

- Each dependency ships as a wheel listed in the manifest, built per platform (`blender --command extension build --split-platforms`). NumPy is never bundled, since Blender ships its own.
- No install step at runtime: the preferences panel only reports what was found and has switches for the optional parts (Numba, live input), the way the Use SciPy switch works in the Print Eval extension.
- Each platform's wheel set has to be checked:
  - Numba brings llvmlite.
  - SoundDevice needs PortAudio. Its Windows and macOS wheels include it; on Linux it is a system library, so live input there stays optional.
- **SoundFile is dropped.** Blender's own `aud` module decodes audio files through Blender's FFmpeg: `aud.Sound.file(path).data()` returns the samples as a NumPy array. So every format Blender can read works, with nothing to install.
- **SoundDevice stays, as the optional live-input switch.** `aud` has no audio input, and it only plays finished sounds with no streaming callback. Live capture and the low-latency preview monitor still need SoundDevice.
- `prefs_library_manager.py`, `_vendor` and the dependency path setup in `utils.py` go away.
- This also allows listing on extensions.blender.org and installing offline.

### 1.2 Presets as assets

Ship ready-made setups as an asset library that a user drags in from the Asset Browser:

- Geometry Nodes groups: bass pulse scale, spectrum bars, beat flash, stereo sway.
- Materials: emission on level, hue on pitch.

Each preset is an ordinary user tree built from ACE nodes, so presets also serve as test scenes for section 2 (nesting, reuse across objects). The asset library ships inside the extension and is registered when the extension loads.

### 1.3 Reference-curve tests

**Today:** the tests check task behaviour against a fake `bpy`, and `tests/run_in_blender.py` runs a smoke test in headless Blender. Nothing pins down the actual numbers that each node outputs.

**Target:**

- A small set of reference audio files (sine sweep, click track, pink noise, a stereo-panned mix, a short music excerpt) with expected output curves stored per node and per frame rate.
- Cross-checks against independent tools where possible: librosa for spectral centroid, flux, onset and pitch, and pyloudnorm for LUFS. These tools run only on the test machine and are never shipped.
- Tolerances per node, and a report that plots any curve that drifts.
- Runs in CI inside headless Blender (extending `run_in_blender.py`). The instance-model rewrite and the bake must leave every curve unchanged.

## 2. Instance model: definitions, instances, signals

### Why

ACE already compiles several top-level trees, but one notion, the *frontend*, covers three different things: the tree that defines the nodes, the places where that tree is used, and the DSP it computes. With many trees on many objects (Geometry Nodes and shaders), this causes the following limits:

- **Discovery is limited to top-level trees on `scene.objects`** (`ACEAudioEngine.discover_scene_frontends`). It skips ACE nodes nested in user groups, World and light shaders, collection instances from other scenes, and linked-library data.
- **The source is fixed per node.** Each stream or utility node has its own `Proxy Object` socket that ACE fills in, so a group reused with different sources can't work. The Source node already puts `Proxy Geometry` into the bundle that flows downstream.
- **Parameters are read only from unlinked sockets** (`ACEDummyTreeReader._parse_node`). A value wired to a modifier input or group input is ignored, so the same tree on two objects can't differ.
- **ACE writes into user trees on every compile**: `ace_id` on nodes, Source node titles and proxy inputs (`nodes/source_nodes.py`), and helper Attribute nodes in shader trees. These writes trigger depsgraph updates that cause recompiles, they fail on linked data, and they make every added tree more expensive.
- **Every change re-parses every frontend and rebuilds every task** (`compile_scene_frontends`, `compile`). `_dirty_frontend_uids` is cleared but never narrows the work, and stateful tasks need ad hoc export/import, such as the Stream Accumulator state.
- **Identical computations aren't shared.** The same RMS of the same source in two trees runs as two tasks.
- **Shader values are copied per object.** They're written as custom properties on every object that uses the material, in a Python loop, so N objects mean N writes of the same value.

### Design

The model has three layers:

- **Definition**: a node tree or group, parsed once.
- **Instance**: one place where a definition is used, given by a path from an owner through group nodes down to an ACE node.
- **Signal**: one concrete DSP computation, identified by its content and computed once, however many instances need it.

The pipeline:

1. **Discovery from the depsgraph.** Go through `depsgraph.ids` (taking `.original`) and keep the node trees. That covers every tree actually evaluated, including nested groups, World, lights, instanced collections and linked data. Owners for the UI come from `bpy.data.user_map`. *To confirm on Blender 5.x: that node trees are listed in `depsgraph.ids`.*
2. **Definitions.** Parse each tree or group into a cached definition that depends only on its contents. Group inputs stay symbolic parameters, and ACE-free nested groups are walked through. Only changed definitions are re-parsed.
3. **Instances.** An instance is the path owner (modifier, material slot or World) → group node → … → ACE node. Instantiating binds the parameters along the path (modifier input values, group-node socket values, unlinked defaults) and the source that reaches the node through the bundle.
4. **Signals, shared by content.** Each concrete node gets the key `hash(op, params, input keys)`, using blake2b as everywhere else. Equal keys anywhere in the scene give one task. A recompile becomes a set difference between old and new keys: create what's new, destroy what's gone, keep the rest with its state. That replaces the per-task state export, keeps analyzers warm, and avoids rebuilding the worker IR.
5. **Sinks.**
   - Output names stay stable per definition node, so editing a parameter never rewrites strings in a user tree. Several sink names may share one task, and the fan-out uses bulk writes.
   - **Geometry Nodes:** stream and utility nodes read the proxy geometry from the incoming bundle. The per-node `Proxy Object` socket goes away, and a source that differs per instance then works with no extra machinery.
   - **Shaders, values:** a prototype should test the helper Attribute node in **View Layer** mode, which reads view-layer or scene custom properties. That gives one write per value for any number of materials and objects. *Still to check: live refresh in Eevee and Cycles viewports, and how much tagging it needs.*
   - **Shaders, structure:** instead of helper nodes in the user's material, each shader ACE node could point to its own generated group that holds the fixed attribute name. That gets around the fact that a shared group's internals can't be specialized, without adding nodes to the user's tree. The cost is one datablock per node.
6. **No writes into user trees.**
   - Node identity comes from the instance path plus `node.name`, not a stored `ace_id`.
   - The source is chosen on the Source node by the user (the proxy object, or a menu socket), not injected by ACE.
   - Result: no depsgraph feedback loops, and append, linking, Shift+D and undo are safe.
7. **Conflicts are reported, not silently specialized.** A definition node can produce different signals across instances: a group-input parameter that varies, or a shader group fed by different sources. In version 1 these conflicts are listed in the UI.

### UI: Active ACE Instances

The panel shows a tree: Definition → its uses (object, modifier or material, GN or shader) → the signals it produces → conflicts. Sources stay global. This also covers managing source nodes across several trees.

### Later: true per-instance parameters

This applies only to parameters, since the bundle already handles different sources in GN.

- **GN:** the bundle could carry a signal-key string that each ACE node extends with its operation and parameters (Value to String, Join Strings, possibly Hash Value), so GN computes the attribute names itself.
- **Shaders:** there are no string operations, so this would need generated copies of the groups.

Build this only if version 1's conflict reports show a real need.

### Order

1. Stop writing into user trees, and add path-based identity.
2. Depsgraph discovery and cached definitions, which gives incremental compile.
3. Shared signals plus the task diff.
4. Sinks: proxy from the bundle in GN, then the View Layer prototype for shaders.
5. Active ACE Instances with conflicts.

The latency items in [AI_Code_Review.md](AI_Code_Review.md) (stale-job budget, duplicated work on the main thread, process vs thread offload) come after this. They get simpler once tasks are shared and diffed.

### Open questions

- Are different **parameters** per object needed (the same tree on several objects with different bands), or only different sources and different trees? This decides whether version 1 can skip per-instance specialization.
- Shaders: is a per-object value (OBJECT attributes) ever needed, or is one global value per signal (View Layer) enough?
- Is linked-library support a goal? If so, step 1 is required, not optional.

## 3. Time model: render, bake, live

### Why define time

**Today:** every value is computed as a stream, driven by the frame-change handlers. Scrubbing, rendering, baking and live input are each partly special cases of that stream, and several behaviours are undefined:

- what a render shows for a live source;
- whether a rendered frame matches what playback showed;
- what happens when the timeline loops. A wrap looks like a rewind, so Stream Accumulators reset and the preview buffers are flushed, which is wrong during a live show.

### 3.1 Signals are functions of time

Every signal is `value(signal, t)`, with `t` a position in the source audio. The engine's only job is to answer that question for the current `t`. How it answers depends on the clock and the source.

**Clocks** (what decides `t`):

- **Timeline:** playback and scrubbing. `t` comes from the current frame.
- **Render:** Blender's render job. `t` comes from the exact frame being rendered, and there's no hurry.
- **Live:** ACE's own clock (3.4), independent of timeline playback.

**Ways to get a value:**

- **Bake:** look up the stored curve (3.3).
- **Random access:** for a file source, analyse the exact sample window at `t` (3.2).
- **Stream:** read the next block from a live input. Only possible "now".

**Rule table:**

| Clock \ Source | File | Live input |
| --- | --- | --- |
| Timeline | Bake if valid, else random access | Stream: the value now, and scrubbing doesn't apply |
| Render | Bake if valid, else random access. **Never stream** | A recorded take (3.5). If there is none, the render stops with a clear message |
| Live | Real-time playback of the file as a backing track | Stream |

`core/clock.py` owns this table and is the only place that decides how a value is obtained. The render handlers stop calling the playback frame path and ask the clock layer for "value at this frame".

**Stateful nodes** (Stream Accumulator, and the Envelope and Counter of section 4) depend on everything before `t`. Under random access their state is rebuilt from checkpoints saved every N frames. In practice, stateful signals are baked before rendering, and the UI says so.

### 3.2 Random access for file sources

- File sources gain `read_at(t, window)` next to `read_block`.
- The block for frame f is the same one streaming would use: samples `[f·N, (f+1)·N)` with `N = sample_rate / fps`, plus the history the analyser needs (an FFT window longer than a block reads samples before it).
- **Stateful filters:** source preprocessing (high-pass and low-pass biquads, attack/release smoothing) uses IIR filters, which carry state. Random access runs a pre-roll before `t`, long enough for them to settle, so the result matches streaming within a tolerance set by the reference tests (1.3). Where the tolerance is too loose, use the checkpoints instead.
- **Contract, checked by the tests:** random access = stream within tolerance, and bake = random access = render, value for value.

### 3.3 Baking into Blender data

**Today:** the bake is a cache kept in memory (ARCHITECTURE.md §12). It's lost when the file is reloaded, and rendering still needs ACE running in the frame handlers. That means:

- a render farm or another computer needs the add-on and its dependencies installed;
- final frames go through the same handler path as playback, with its render-safety guards.

A bake writes each signal's full curve over the frame range into ordinary Blender data. Consumers then read it with no Python at all:

- **Geometry Nodes:** one baked mesh per source, with one point per frame and one attribute per signal. The ACE node's internals switch from sampling point 0 of the live proxy to sampling point `frame - frame_start` of the baked mesh (Scene Time → Sample Index). The node interface is unchanged.
- **Shaders:** a float image, one row per signal and one column per frame, packed in the .blend. An ACE shader node samples it at the current frame (Scene Time → coordinate, Closest interpolation). The baked image needs to be checked against the image size limit for long ranges; past that, use several rows per signal.
- **Signal identity:** each baked curve is keyed by its signal key (section 2) plus the hash of the source file. A curve whose key still matches is reused after edits elsewhere; a changed key marks the bake stale in the UI.
- **Live vs baked:** each source switches between the two. With the source in baked mode, the frame handler does nothing for it.

Results:

- the .blend renders anywhere, with ACE disabled or not installed;
- scrubbing, viewport and render read exactly the same numbers;
- the bake survives reloads.

The bake uses the signal keys of section 2 and is filled by random access (3.2), so it doesn't depend on playing the stream through. `core/bake_store.py` owns it.

### 3.4 Live clock

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

### 3.5 Takes: record live, render later

A **take** is a recorded live session:

- the input audio saved as WAV;
- the MIDI and OSC events (section 4) logged with timestamps;
- the take's first tick mapped to its scene frame. Since live frames are consecutive, tick n is frame `start + n`, exactly.

A take is a file source, so the rule table applies: it can be scrubbed, baked and rendered exactly. The use: perform live with an Eevee look, then render the same performance in Cycles at full quality.

**Rendering a take, two options:**

- **ideal sync:** values exactly on the audio;
- **reproduce the live look:** values shifted by the latency measured during the session, as the audience saw them.

Recording writes WAV and the event log off the main thread. `core/takes.py` owns it.

### 3.6 Order

1. Clock layer, random access for file sources, and the render rule. Rendering becomes defined and exact.
2. Bake into Blender data (3.3).
3. Live clock (3.4), with the simulation-cache prototype first.
4. Takes (3.5).

The live output window of section 5 needs the live clock.

## 4. Event and control nodes

### Why event nodes

ACE measures audio well (level, spectrum, pitch, onset), but turning a measurement into "something happens on each kick" is left to the user in Geometry Nodes: thresholds, holds, decays and counters built from math nodes. That is hard to get right per frame, and impossible to get right across a scrub, because the state lives nowhere.

### Event design

New backend tasks. The frontends stay contracts with no maths, as the 3-layer model requires.

- **Trigger:** float in → pulse (0/1) when the input crosses a threshold. Includes a re-arm level (hysteresis) and a minimum interval, so one kick gives one pulse.
- **Envelope:** pulse in → attack / decay / sustain / release curve, times in seconds or in beats (from `src_specs` tempo).
- **Counter:** pulse in → integer that counts up, with modulo and reset. For "change colour every 4 kicks" and stepping through instances.
- **Hold / Peak hold:** keeps the maximum for a time, then falls at a set rate.
- **Beat quantize:** snaps pulses to the beat grid of the source's tempo.
- **Control sources:** MIDI (notes, velocity, CC) and OSC messages as sources of control signals, not audio, routed through the same graph. Options: `mido`/`python-rtmidi` for MIDI and a plain UDP OSC parser, both optional dependencies (1.1).

State and timeline:

- Stateful tasks (Counter, Envelope) must give the same value on any frame whether it is reached by playing or by jumping. For file sources, the task replays from a cached checkpoint before the target frame, or reads the bake (section 3).
- Live sources can't be replayed, so their state simply carries on.
- The task diff of section 2 keeps this state across recompiles.

### Event open questions

- Should Envelope and Counter outputs be baked like any other signal? This is the simplest and makes scrubbing exact.
- MIDI timing: per frame (simple), or with sub-frame timestamps folded into the envelope (more precise for fast patterns)?

## 5. Live output window

### Why a live output window

Live visuals go to a projector or to a mixing tool (OBS, Resolume, projection mapping). Making a clean output from Blender by hand takes about ten steps that have to be redone every session: new window, rendered shading, camera view, overlays off, every panel hidden, fullscreen. One button does it.

The window is the output: the program that receives it captures the window. That adds no dependency to ACE, and ACE never copies frames from the GPU. The live clock (3.4) needs a 3D view that keeps redrawing anyway, and this window is that view.

### The window

- **Go Live** opens a new window on a dedicated "ACE Live" workspace that holds a single 3D view. ACE creates that workspace once, so your own workspaces stay untouched.
- **The view:**
  - it looks through the scene camera (or one you choose), fitted to the window, in rendered shading;
  - overlays, gizmos, header, tool header, toolbar, sidebar and asset shelf are all hidden;
  - the area outside the camera frame is black (passepartout fully opaque), so a display with a different shape gets clean letterboxing.
- **Fullscreen** on the monitor where the window opens.
- **Going live:** opening the window starts the live clock (3.4).
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
- **Cost:** rendered shading in this window, with the main window's 3D views dropped to solid (3.4).

### Code

`ui/live_window.py` holds the operator, the workspace, and window setup and cleanup. The clock itself stays in `core/clock.py`.
