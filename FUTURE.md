# ACE: Future Work

Planned changes that aren't built yet, in the order they should be done. Nothing here describes current behaviour; [ARCHITECTURE.md](ARCHITECTURE.md) does.

1. [Live output window](#1-live-output-window)
2. [Presets as assets](#2-presets-as-assets)
3. [Per-instance parameters](#3-per-instance-parameters), only if needed

The extension packaging, the reference-curve tests, the fixes and removals, the instance model (definitions, uses and shared signals), the time model (render, bake, live sessions, takes) and the core operator set (node roles, the four menus, 0..1 outputs, the controllers, the Downbeat and the beat-grid fit, ignored links) are built ([ARCHITECTURE.md](ARCHITECTURE.md) §13, §10, §8, §1 and §3; the time model in §3 and §8; the core set in §1, §3, §7 and §8). The reference curves of the nodes that remain must stay unchanged through the work below.

Left from the time model:

- **To check by hand** (they need a window): which thread runs the frame handlers during an F12 render, and during a Bake node's bake started from its button (both run as jobs; started from a script, they run on the main thread), and that the baked values are right; a live session of 30 minutes with a simulation zone, its memory with the simulation cache on and off, and that no frame is skipped.
- **Downstream Bake nodes** record what ACE streams, and ACE bakes nothing for them: Python can't tell whether a native bake holds data. If Blender exposes that one day, ACE could bake its sources when a Bake node is baked, and release them when it is deleted.
- **Takes** record device inputs only: UDP inputs, and MIDI and OSC once ACE reads them, come later.

Left from the core operator set:

- **To check by hand** (they need a window): the add menu (Input, Output, a separator, Audio, Controllers, with icons on the last two only); Envelope's Mode menu and the inputs it hides, in Geometry Nodes and shaders; Gate's socket colours (Bool, Bool, Int); the Source panel's First Downbeat row; ignored links drawn red, and their reasons on the selected node; a field into a controller drawn red, with Blender's own tooltip; that the `.ACE_Controllers` object draws nothing in the viewport.
- **Also to check by hand:** ACE nodes in the link-drag search from an Audio socket and in Add › Group, in both editors; Ctrl+C / Ctrl+V of ACE nodes from Geometry Nodes into a material and back (Blender's own "Cannot add node" warning may still show); that no ACE input is drawn greyed; the internals' layout (Tab into a group); a controller selected in the Node Editor showing its settings and description in the Operators panel.
- **Controllers in shaders** run only directly in a material's, the World's or a light's tree, not inside shader groups, and need ACE to render: their values reach the shader through Python. Baking their read-back values into the scene's F-curves, as measurements are, would let a .blend render them without ACE.
- **The meter** finds no bar in a file that starts with silence, or in fewer than 12 beats; the Downbeat is then 0, and the grid isn't fitted.

## 1. Live output window

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

## 2. Presets as assets

Ship ready-made setups as an asset library that a user drags in from the Asset Browser:

- Geometry Nodes groups: bass pulse scale, spectrum bars, beat flash, stereo sway, and a **Beat Clock**: it turns Scene Time, BPM and Downbeat (ARCHITECTURE.md §8, Tempo) into the phase and index of the beat, the bar and the phrase (4, 8 or 16 bars), a beat's phase being `(Seconds - Downbeat) * BPM / 60`.
- Materials: emission on level, hue on pitch.

Each preset is an ordinary user tree built from ACE nodes, so presets also exercise the instance model (ARCHITECTURE.md §1: nesting, reuse across objects). The asset library ships inside the extension and is registered when the extension loads.

## 3. Per-instance parameters

Uses of one ACE node share its sink names (ARCHITECTURE.md §1). Uses that compute different values, such as a group used twice with different band frequencies, are reported as conflicts on the node's settings in the Operators panel, and only one use reaches the sink. Different sources through the bundle already work in Geometry Nodes.

- **Geometry Nodes:** the bundle could carry a signal-key string that each ACE node extends with its operation and parameters (Value to String, Join Strings, possibly Hash Value), so Geometry Nodes computes the attribute names itself.
- **Shaders:** there are no string operations, so this would need generated copies of the groups.

Build this only if the conflict reports show a real need.

### Open questions

- Are different **parameters** per object needed (the same tree on several objects with different bands), or are different sources and different trees enough?
- Shaders: is a value per object (OBJECT attributes) ever needed, or is one global value per signal (View Layer, today) enough?
