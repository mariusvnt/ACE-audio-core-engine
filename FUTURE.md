# ACE: Future Work

Planned changes that aren't built yet, in the order they should be done. Nothing here describes current behaviour; [ARCHITECTURE.md](ARCHITECTURE.md) does.

1. [Core operator set](#1-core-operator-set): the node set, the add menu, motion and event nodes
2. [Live output window](#2-live-output-window)
3. [Presets as assets](#3-presets-as-assets)
4. [Per-instance parameters](#4-per-instance-parameters), only if needed

The extension packaging, the reference-curve tests, the fixes and removals, the instance model (definitions, uses and shared signals) and the time model (render, bake, live sessions, takes) are built ([ARCHITECTURE.md](ARCHITECTURE.md) §13, §10, §8, §1 and §3; the time model in §3 and §8). The reference curves of the nodes that remain must stay unchanged through the work below.

Left from the time model:

- **To check by hand** (they need a window): which thread runs the frame handlers during an F12 render from the interface (background renders run them on the main thread); a live session of 30 minutes with a simulation zone, its memory with the simulation cache on and off, and that no frame is skipped.
- **Downstream Bake nodes** are kept baked for as long as they sit downstream of ACE nodes, because Python can't tell whether a native bake holds data. If Blender exposes that one day, ACE can follow the native Bake and Delete buttons instead.
- **Takes** record device inputs only: UDP inputs, and MIDI and OSC once ACE reads them, come later.

## 1. Core operator set

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
  - Its `stateful` key keeps it on the main thread (built: `vars.node_is_stateful`, used by the offload policy).
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
- **Phases in Geometry Nodes** come from a **Beat Clock** preset (section 3). It turns Scene Time, BPM and Downbeat into the phase and index of the beat, the bar and the phrase (4, 8 or 16 bars).

### State and timeline

- **Rule:** a stateful node gives the same value on a frame whether that frame is reached by playing or by jumping.
- **File sources replay.**
  - Envelope and Gate's Open forget within a few time constants. They declare how many frames they need (`settle_frames`), and the frame runner replays those frames before the target after a seek (ARCHITECTURE.md §3, Time).
  - Running Total and Gate's Count remember indefinitely (`settle_frames` is None), so they are exact only when baked (ARCHITECTURE.md §8, Bake).
- **Live sources** can't be replayed, so their state simply carries on.
- **Recompiles:** a recompile keeps every task whose signal remains, with its state (ARCHITECTURE.md §3), so these nodes keep theirs too.

### Core set order

1. Registry-driven value nodes (the `stateful` key of `ACE_NODES` exists already), then the merged Stereo node, the 0..1 outputs and the add menu.
2. Envelope and Running Total.
3. The Downbeat estimate.
4. Gate.
5. `settle_frames` and `reset_state` for Envelope and Gate.

Each new node gets reference curves (ARCHITECTURE.md §10), which also check that 24 and 48 fps give the same curve.

### Core set open questions

- Envelope's Spring overshoots, so on an input near either end of its range it leaves it. Proposed: let it, since clamping would flatten the bounce.
- Should Running Total and Gate outputs be baked like any other signal? This is the simplest and makes scrubbing exact.
- Should Gate's Re-arm Level be a free level, or a fraction of Threshold?

## 2. Live output window

### Why a live output window

Live visuals go to a projector or to a mixing tool (OBS, Resolume, projection mapping). Making a clean output from Blender by hand takes about ten steps that have to be redone every session: new window, rendered shading, camera view, overlays off, every panel hidden, fullscreen. One button does it.

The window is the output: the program that receives it captures the window. That adds no dependency to ACE, and ACE never copies frames from the GPU. A live session (ARCHITECTURE.md §3, Time) needs a 3D view that keeps redrawing anyway, and this window is that view.

### The window

- **Go Live** opens a new window on a dedicated "ACE Live" workspace that holds a single 3D view. ACE creates that workspace once, so your own workspaces stay untouched.
- **The view:**
  - it looks through the scene camera (or one you choose), fitted to the window, in rendered shading;
  - overlays, gizmos, header, tool header, toolbar, sidebar and asset shelf are all hidden;
  - the area outside the camera frame is black (passepartout fully opaque), so a display with a different shape gets clean letterboxing.
- **Fullscreen** on the monitor where the window opens.
- **Going live:** opening the window starts a live session (`core/live.py`), with the window's own Esc ending it.
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
- **Cost:** rendered shading in this window, with the main window's 3D views dropped to solid (the live session's option).

### Code

`ui/live_window.py` holds the operator, the workspace, and window setup and cleanup. The clock itself stays in `core/clock.py`.

## 3. Presets as assets

Ship ready-made setups as an asset library that a user drags in from the Asset Browser:

- Geometry Nodes groups: bass pulse scale, spectrum bars, beat flash, stereo sway, Beat Clock (section 1).
- Materials: emission on level, hue on pitch.

Each preset is an ordinary user tree built from ACE nodes, so presets also exercise the instance model (ARCHITECTURE.md §1: nesting, reuse across objects). The asset library ships inside the extension and is registered when the extension loads.

## 4. Per-instance parameters

Uses of one ACE node share its sink names (ARCHITECTURE.md §1). Uses that compute different values, such as a group used twice with different band frequencies, or a Rolling Average behind two sources, are reported as conflicts in *Active ACE Instances*, and only one use reaches the sink. Different sources through the bundle already work in Geometry Nodes.

- **Geometry Nodes:** the bundle could carry a signal-key string that each ACE node extends with its operation and parameters (Value to String, Join Strings, possibly Hash Value), so Geometry Nodes computes the attribute names itself.
- **Shaders:** there are no string operations, so this would need generated copies of the groups.

Build this only if the conflict reports show a real need.

### Open questions

- Are different **parameters** per object needed (the same tree on several objects with different bands), or are different sources and different trees enough?
- Shaders: is a value per object (OBJECT attributes) ever needed, or is one global value per signal (View Layer, today) enough?
