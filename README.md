Audio Core Engine (ACE)
=======================

Audio Core Engine is a Blender extension that turns audio into per-frame values for
Geometry Nodes and shader materials. A sound file or a live input is analysed as
the timeline plays, and the results (loudness, pitch, spectral features, stereo
image, beat phase and more) drive procedural animation without keyframes.

Requirements
------------

- Blender 5.2 or later, on Windows (x64), macOS (Apple silicon, macOS 12 or
  later) or Linux (x64).
- Nothing else to install. The libraries ACE uses ship inside the extension:
  - AudioFlux: spectral analysis (spectrum and spectral features).
  - Numba: faster filters and envelopes.
  - SoundDevice: live capture and preview playback. On Linux it needs the
    PortAudio system library (`libportaudio2` on Debian and Ubuntu).

  Audio files are read by Blender itself, so every format Blender opens works:
  WAV, MP3, FLAC, Ogg, AAC, AIFF and more.

Installation
------------

1. Download the zip for your platform, for example
   `audio_core_engine-0.5.0-windows_x64.zip`.
2. In Blender, open *Edit > Preferences > Get Extensions*, open the menu at the
   top right and choose *Install from Disk*, then pick the zip. Installing
   needs no network connection.
3. The extension's preferences list each library and whether it loaded. Two
   switches turn off Numba (applies after restarting Blender) and SoundDevice
   (live input and preview) if either causes trouble on your machine.

Quick start
-----------

1. In the 3D View sidebar, open the *Audio Core Engine* tab and, under *Audio
   Inputs*, add an input: a file, or a live input (sound device or UDP stream).
   Adding an input creates no object and no node tree.
2. In a Geometry Nodes tree, open *Add > Audio Core Engine* and add the *Source*
   node, then the analysis nodes you need. The Source node reads the active
   input; with several, choose another in the node's settings.
3. Connect them. Each output appears as a value the tree can use, refreshed
   every frame during playback.
4. To drive a material, add the same nodes to an object's shader tree.

Nodes
-----

- **Source**: a reference to one of the scene's audio sources.
- **Delta** nodes split the audio before analysis: frequency bands, mid/side,
  attack/release, the 12 notes of the chromagram, and percussive/harmonic
  components.
  A split frequency can be driven by other nodes, audio operator outputs
  included: Value, Integer, Math, Clamp, Map Range, Mix, Switch, Float Curve,
  Float to Integer, Scene Time, node groups and modifier inputs are followed.
- **Stream** nodes measure the audio each frame: Loudness (RMS, LUFS or Peak,
  0..1 between -70 dB and 0 dB), Stereo (width or balance), Spectral
  (centroid, flux or zero-crossing rate), a stereo vector, onset and pitch. The
  method of a node is switched in the sidebar's *Operators* panel; files
  made with the former one-method nodes are converted when opened.
- **Utilities**: source specs (frames per beat, beat phase), audio math, a
  rolling average over a time window, an accumulator and a preview sink for
  listening to a branch.

Preview and bake
----------------

- **Preview**: during playback, the audio reaching a preview sink, a Viewer
  node (or else the tree's audio output) can be heard through the preview device.
- **Bake**: a file source can be analysed over the scene's frame range ahead
  of time; playback then reads the stored values.

Development
-----------

`ARCHITECTURE.md` describes how the extension is built, `FUTURE.md` what is
planned.

- Build: `python tools/fetch_wheels.py` downloads the pinned wheels
  (`tools/wheels.txt`) and lists them in `blender_manifest.toml`; then
  `blender --command extension build --split-platforms --output-dir dist`.
- Unit tests: `python -m unittest discover -s tests`. The perf gates are skipped
  unless `ACE_PERF=1` is set.
- In Blender, headless: `python tests/run_in_blender.py --package` builds and
  installs the zip for this platform and runs the smoke test. With
  `--script tests/reference/blender_reference.py` it compares every node's
  output with the reference curves.
