Audio Core Engine
=================

Audio Core Engine (ACE) is a Blender extension that turns sound into animation.
It listens to an audio file or a live input and translates what it hears into
values that drive geometry, materials, lights and the World, frame by frame,
without keyframes.

Overview
--------

- **Input**: audio files in any format Blender reads, or live sound from a
  device or network stream.
- **Analysis**: loudness, pitch, brightness, transients, stereo image and
  tempo, over the full spectrum or within chosen frequency bands.
- **Integration**: a set of node groups for the Geometry Nodes and shader
  editors, which also work nested inside your own groups.
- **Preview and bake**: listen to any part of the analysis during playback, or
  bake a file over the frame range into the .blend itself. A baked scene
  renders exactly what playback shows, even on a computer without ACE, and a
  Bake node placed after ACE nodes gets what it reads baked.
- **Live**: a live session plays every frame, never drops one and never loops,
  and can be recorded as a take to render the show later.

Requirements
------------

Blender 5.2 or later on Windows, macOS (Apple silicon) or Linux. All
dependencies ship with the extension; live input on Linux also needs the
PortAudio system library.

Installation
------------

In Blender, open *Edit > Preferences > Get Extensions*, choose *Install from
Disk* and select the zip for your platform.

Documentation
-------------

`ARCHITECTURE.md` describes the system design, `FUTURE.md` the roadmap.
