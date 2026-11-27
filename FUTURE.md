# ACE: Future Work

Planned changes that aren't built yet, in the order they should be done. Nothing here describes current behaviour; [ARCHITECTURE.md](ARCHITECTURE.md) does.

1. [Core operator set](#1-core-operator-set): the node set, the add menu, controllers and musical time
2. [Live output window](#2-live-output-window)
3. [Presets as assets](#3-presets-as-assets)
4. [Per-instance parameters](#4-per-instance-parameters), only if needed

The extension packaging, the reference-curve tests, the fixes and removals, the instance model (definitions, uses and shared signals) and the time model (render, bake, live sessions, takes) are built ([ARCHITECTURE.md](ARCHITECTURE.md) §13, §10, §8, §1 and §3; the time model in §3 and §8). The reference curves of the nodes that remain must stay unchanged through the work below.

Left from the time model:

- **To check by hand** (they need a window): which thread runs the frame handlers during an F12 render, and during a Bake node's bake started from its button (both run as jobs; started from a script, they run on the main thread), and that the baked values are right; a live session of 30 minutes with a simulation zone, its memory with the simulation cache on and off, and that no frame is skipped.
- **Downstream Bake nodes** record what ACE streams, and ACE bakes nothing for them: Python can't tell whether a native bake holds data. If Blender exposes that one day, ACE could bake its sources when a Bake node is baked, and release them when it is deleted.
- **Takes** record device inputs only: UDP inputs, and MIDI and OSC once ACE reads them, come later.

## 1. Core operator set

### Why a core set

ACE has 11 nodes, after the removals that came first. Stereo and Stereo Vector still overlap, and none of the nodes gives motion weight or turns a measurement into an event. The goal is a small set where each node answers a question the others can't:

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

### Add menu and categories

**Today:** the Audio Core Engine entry of the add menu has four submenus: Input (Source), Delta (Frequency Splitter), Stream (the measurements) and Utilities (Audio Preview, Rolling Average, Stream Accumulator).

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

- Input and Output have no icon, and a separator sets them apart from Audio and Controllers, which keep an icon each.
- Audio Preview has an Audio input, but it's where a branch ends, so it goes under Output.
- A menu with no node for the current editor is hidden, as today.
- **The categories of `ACE_NODES` become these four:** `input`, `output`, `audio`, `controllers`. Categories are in node ids, group names (`.ACE_<category>_<type>`), signal ids, golden keys and the worker IR, so the rename comes in the same step that removes every check on a category (see Design rules). The goldens then change only their keys. Old files are not migrated.

### Outputs on 0..1

Every measurement is a float between 0 and 1, so outputs can be swapped and a Gate's Threshold is always set on the same range. Balance is the exception: it keeps its range of -1 (left) to 1 (right), with 0 at the centre.

| Output | 0 | 1 | Scale |
| --- | --- | --- | --- |
| Loudness | -70 dB and below | 0 dB | linear in dB, as today |
| Pitch | 20 Hz, or no pitch | 5000 Hz | logarithmic, so equal musical intervals are equal steps |
| Brightness | 20 Hz and below | 20000 Hz | logarithmic |
| Transient | smooth | sharp | as today |
| Width | mono | side only | as today |

The ranges are fixed, also behind a Frequency Splitter, so values from different bands compare. 20 to 5000 Hz is also the range the pitch detector searches.

These are quantities rather than measurements, and keep their units: BPM, Beats per Bar and Downbeat, Running Total, and Gate's Count. Envelope keeps the scale of its input.

Pitch and Brightness get new reference curves.

### Stereo

Stereo and Stereo Vector become one node without modes, with two outputs: **Balance** and **Width**. Both are ratios that say where the sound is, whatever its volume. Levels are Loudness's job, so the left and right levels of Stereo Vector go. Both outputs come from one task, as the two values of one result.

### Controllers

ACE measures audio, but every measurement describes the current frame, so whatever it drives moves in short twitches. Giving that motion weight, and turning it into "something happens on each kick", takes smoothing, thresholds, holds and counters.

**Today:** two utilities keep state in the backend, and both fall short:

- **Rolling Average** is a symmetric box window. It lags on the way up as much as it blurs on the way down.
- **Stream Accumulator** adds its input once per frame, so the same scene moves twice as far at 48 fps as at 24 fps.
- Neither sees the maths in front of it. The reader walks through every node that isn't an ACE node (`nodes/definitions.py`), so Loudness → Math ×2 → Rolling Average averages the raw Loudness.

**Controllers are native Geometry Nodes groups.** ACE makes them, and they compute inside Blender, with a Simulation Zone that carries their state from frame to frame:

- **Any node can feed them**: Math, Map Range, Float Curve, other ACE outputs. What reaches the input is what Blender computes.
- **Each use has its own state**, per modifier and per group path, so two objects using one tree don't share a controller.
- **Times are in seconds**, and results don't depend on fps. A step lasts as long as the Scene Time that passed since the last step, which the zone keeps in its state. The zone's own Delta Time won't do: it is one frame long even when playback skipped frames or the timeline jumped ahead (probed on 5.2). Each step is an exact discretization (a one-pole filter uses `1 − e^(−dt/τ)`), so a step input defined in seconds gives the same curve at 24 and 48 fps.
- **Blender's simulation cache and Bake record them**, as they record any simulation.
- Their inputs take single values only (`structure_type` `SINGLE`). A field linked to one is drawn as an invalid link by Blender itself.

This is a deliberate exception to "frontends compute nothing" (ARCHITECTURE.md §1): audio analysis stays in the backend, and shaping the values it gives is done where the values are used. The backend has no controller tasks, and the values of controllers need no sinks. Rolling Average, Stream Accumulator and the `ACE_Values` mesh go.

### Envelope, Running Total and Gate

**Envelope** replaces Rolling Average. Its **Mode** is a menu on the node, with Follow as the default. Inputs used by one mode only are greyed out in the other.

- **Follow:** rises with Attack and falls with Release. Hold keeps the value at its peak for Hold seconds before it starts to fall. With Attack at 0, this is a peak hold, and fed a Gate's Hit it gives a hit-then-decay shape.
- **Spring:** the output follows the input like a weight on a spring. It speeds up toward the input, overshoots and settles. Speed (Hz) sets how fast it follows. Bounce sets how much it overshoots: 0 means not at all, 1 means it barely stops. Each step is the exact solution of the spring over the step's duration, so it is stable at any fps. The overshoot isn't clamped: clamping would flatten the bounce.
- It starts at rest on its first input, so there is no ramp up from 0.

**Running Total** replaces Stream Accumulator. Its input is a rate and its output is the total over time (value × seconds), for a scalar:

- **Leak Time:** the total drifts back toward 0 when the input goes quiet. Energy pushes it forward, and silence lets it return. At 0 it never leaks, and sums from the start of the simulation.
- **Reset**, a Boolean input, for example from a Gate's Hit.

**Gate:** float in, three outputs:

- **Open:** true from the moment the input crosses Threshold until it falls below Re-arm Level (the hysteresis). Re-arm Level is a level of its own on the input's scale. Above Threshold, it counts as Threshold.
- **Hit:** true only on the frame it opens.
- **Count:** the number of hits so far, set back to 0 by the **Reset** input. A Floored Modulo math node turns it into "change colour every 4 kicks".

The **Min Interval** input prevents a new Hit sooner than this, so one kick gives one pulse. A crossing within it is ignored, and the gate opens again only after the input has fallen below Re-arm Level, so it never gives a late Hit off the beat.

### Controllers in shaders

Shader trees have no Simulation Zone. For a controller in a material, the World or a light, ACE generates a hidden Geometry Nodes tree on a hidden object:

- it holds a copy of the chain that feeds the controller in the shader tree. ACE nodes become their Geometry Nodes twins, with the same settings and the same sink names, so they share the signals of the shader nodes and cost no extra analysis. Other nodes are copied when Geometry Nodes has the same node (Math, Map Range, Clamp, Mix, Float Curve, Value);
- the controller's result is stored as a named attribute;
- after each frame, ACE reads it from the evaluated object and writes it to the scene property the shader node reads (View Layer, as for measurements).

A node that exists only in shaders (a texture, Light Path, Fresnel) can't be copied: the link it feeds is ignored, so it is shown as invalid (see Ignored links).

Probed on 5.2: in `frame_change_post`, the evaluated object holds that frame's value, in the viewport and in a render's own depsgraph. A scene property written there, followed by `update_tag`, is in that same frame of a Cycles or Eevee render, and the tag brings no extra depsgraph update. A material on an object could also read a controller through Store Named Attribute and an Attribute node, without ACE.

### Ignored links

**Today:** some links are ignored with nothing to show for it:

- a linked Loudness Window that isn't bound through a group input;
- a linked Attribute ID;
- a Proxy Object fed by something other than a group input;
- a Split Freq fed by a chain ACE can't compute;
- an Audio input reached through a Switch.

**Every link ACE ignores becomes an invalid link**, drawn red by Blender (`NodeLink.is_valid`). The reason is in the N-panel: on the selected node's box, and in a list of the ignored links of the edited tree.

- **The flag only shows.** Probed on 5.2: Geometry Nodes and Cycles still evaluate through a link marked invalid. For most of these inputs that changes nothing, since an ACE group only declares them and ACE computes the value. For a linked Attribute ID or Proxy Object, the group does read the link, so the node shows something other than what ACE writes until the link is fixed; the reason says so.
- **The flag is reset** when the tree's nodes or links change, and when the file is reloaded (it holds across frame changes and other updates). So ACE marks the links again after each tree update, undo and file load. The flag is not part of a tree's content signature, so marking a link never recompiles.
- Marking a link is the second thing a compile writes into user trees, after the repair of sink names. It is a flag, not content.
- A link fed by an ACE output that has no value yet is not ignored: it is only waiting.

### Design rules

- **Audio analysis is in backend tasks, controllers are native groups.** The audio frontends stay contracts with no maths, as the 3-layer model requires.
- **A node is described by `ACE_NODES` alone.**
  - Its **role** (`source`, `splitter`, `measure`, `controller`, `preview`) says how it is built and read. A `native` node (the controllers) has no task.
  - Its value outputs give its attribute sockets ("Attribute ID", "Attribute ID 2", …).
  - Its `stateful` key keeps it on the main thread (built: `vars.node_is_stateful`, used by the offload policy).
  - The reader, sinks, compiler, offload policy, workers and both factories ask the registry for these traits, and never check a category. A new measurement is then one registry entry and one task.
- Booleans are Bool sockets in Geometry Nodes and 0/1 floats in shaders.

### Musical time: the Source node

The Source node outputs only what musical time depends on: **BPM, Beats per Bar and Downbeat**. The Geometry Nodes one already outputs BPM and Beats per Bar. Once these are known, every phase (beat, bar, phrase) is a function of time, so no node outputs a phase.

- **Downbeat** is the time of the first downbeat, in **scene seconds**: the clock of Scene Time's Seconds output. A file source starts at the scene's `frame_start`, so Downbeat is `frame_start / fps` plus the downbeat's time in the file. A beat's phase is then `(Seconds − Downbeat) · BPM / 60`. A live source outputs 0.
- **The estimate:** the tempo detection already finds the first downbeat, as the accented beat of the bar, and keeps it unused. It becomes a source property, in seconds from the start of the file, that the user can override, like BPM and meter.
- **The beat grid is fitted over the whole file.** A phase computed from a fixed BPM drifts: an error of 0.1 % in the BPM, which is what 15 seconds of audio gives, is half a beat off after 4 minutes at 120 BPM. So, after today's estimate from the first 15 seconds, the onset peaks near each beat of the whole file (up to 10 minutes) are fitted with a straight line by least squares, which gives the BPM and the downbeat. When the fit is poor, the estimate stays.
- **Phases in Geometry Nodes** come from a **Beat Clock** preset (section 3). It turns Scene Time, BPM and Downbeat into the phase and index of the beat, the bar and the phrase (4, 8 or 16 bars).

### State and timeline

- **Measurements** keep the time model: a stateful one gives the same value on a frame whether that frame is reached by playing or by jumping (ARCHITECTURE.md §3, Time).
- **Controllers** follow Blender's simulation rules (probed on 5.2):
  - A frame in the simulation cache, or in a bake, is exact. Playing from the start fills the cache in order.
  - A jump ahead past the cache, or a frame skipped in playback, takes one step from the last state. With the step measured in Scene Time, that step covers the time that passed.
  - Without the cache, a jump back starts again from the first frame's state.
  - **A render reads the viewport's simulation.** Frames that are cached render exactly. Frames past the cache render frozen on the last state, or at the start state if nothing was played. So an exact render needs the controllers' simulations cached over the range or baked. ACE's **Bake** also bakes the simulations of the objects that use controllers, after the sources they read. `render_init` notes any controller that isn't baked, in the Baking panel.
  - A live session carries their state on, with the simulation cache on or off.
- **Recompiles** don't touch controllers: they hold no backend state.

### Core set order

1. Probes on Blender 5.2, done:
   - a Simulation Zone inside a group keeps one state per use, nested groups included;
   - a Menu socket on a group is set per node, and the inputs the chosen mode doesn't use are greyed out;
   - a `SINGLE` input fed a field is flagged invalid by Blender;
   - plus the jumps, renders, `is_valid` flag and shader read-back described above.
2. Registry traits and the four categories, every check on a category removed, and the add menu. Values don't change.
3. Ignored links shown as invalid, with their reasons in the N-panel.
4. The merged Stereo node, and Pitch and Brightness on 0..1.
5. Rolling Average, Stream Accumulator and `ACE_Values` removed; the native controller groups, and Envelope.
6. Running Total and Gate.
7. The Downbeat estimate and the beat-grid fit.
8. Controllers in shaders, through the generated Geometry Nodes tree.

Measurements get reference curves (ARCHITECTURE.md §10). Controllers are checked against a Python reference of the same equations, on a step and an impulse at 24 and 48 fps.

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
