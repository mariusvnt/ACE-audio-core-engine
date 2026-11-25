AudioCore Engine (ACE)
======================

AudioCore Engine is a Blender add-on that turns audio into per-frame values for
Geometry Nodes and shader materials. A sound file or a live input is analysed as
the timeline plays, and the results (loudness, pitch, spectral features, stereo
image, beat phase and more) drive procedural animation without keyframes.

Requirements
------------

- Blender 5.0 or later.
- Optional Python libraries, installed from the add-on preferences:
  - AudioFlux: spectral analysis (spectrum, pitch, onset).
  - SoundFile: reading WAV, FLAC and other file formats.
  - SoundDevice: live capture and preview playback.
  - Numba: faster filters and envelopes.

  NumPy ships with Blender. Without the optional libraries, ACE uses simpler
  built-in fallbacks.

Installation
------------

1. In Blender, open *Edit > Preferences > Add-ons* and install the add-on folder or zip.
2. Enable *AudioCore Engine*.
3. In the add-on preferences, install the optional libraries you want.

Quick start
-----------

1. In the 3D View sidebar, open the *AudioCore Engine* tab and add an audio
   source: a file, or a live input (sound device or UDP stream).
2. In a Geometry Nodes tree, open *Add > AudioCore Engine* and add the *Source*
   node, then the analysis nodes you need.
3. Connect them. Each output appears as a value the tree can use, refreshed
   every frame during playback.
4. To drive a material, add the same nodes to an object's shader tree.

Nodes
-----

- **Source**: a reference to one of the scene's audio sources.
- **Delta** nodes split the audio before analysis: frequency bands, mid/side,
  attack/release, the 12 notes of the chromagram, and percussive/harmonic
  components.
  A split frequency can be driven by other nodes: Value, Integer, Math, Clamp,
  Map Range, Scene Time and modifier inputs are followed.
- **Stream** nodes measure the audio each frame: Loudness (RMS, LUFS or Peak,
  0..1 between -70 dB and 0 dB), Stereo (width or balance), Spectral
  (centroid, flux or zero-crossing rate), a stereo vector, onset and pitch. The
  method of a node is switched in the sidebar's *Active Operators* panel; files
  made with the former one-method nodes are converted when opened.
- **Utilities**: source specs (frames per beat, beat phase), audio math, a
  rolling average, an accumulator and a preview sink for listening to a branch.

Preview and bake
----------------

- **Preview**: during playback, the audio reaching a preview sink, a Viewer
  node (or else the tree's audio output) can be heard through the preview device.
- **Bake**: a file source can be analysed over the scene's frame range ahead
  of time; playback then reads the stored values.

Development
-----------

`ARCHITECTURE.md` describes how the add-on is built. Tests run with
`python -m unittest discover -s tests`; `tests/run_in_blender.py` runs the
Blender-side checks headless. The perf gates are skipped unless `ACE_PERF=1`
is set.
