"""
Measure threads vs worker-process DSP inside Blender, at real-time pace.

Run headless from the repo root:
  python tests/run_in_blender.py --script tests/blender_perf.py [--extra-path LIBS] -- \
      --nodes 24,36,50 --frames 240 --out tests/artifacts/blender_perf.json [--profile 50]

For each node count and mode it builds one file source feeding N stream
nodes, plays `frames` frames at the scene fps and records, per frame:
  - ace_ms: time ACE spends in the frame handler (main thread);
  - frame_ms: the whole frame_set(), Blender's own evaluation included;
  - whether worker values were usable (processes mode), else the local fallback ran.
"""

import argparse
import cProfile
import importlib
import io
import json
import math
import os
import pstats
import statistics
import sys
import tempfile
import time
import wave
from array import array

import bpy  # type: ignore

ADDON = "AudioCoreEngine"
NODE_TYPES = ("rms", "peak", "lufs", "spectral_centroid", "spectral_flux", "zero_crossing_rate", "onset", "pitch")


def _args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", default="1,12,24,36,50")
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--out", default="")
    p.add_argument("--profile", type=int, default=0, help="cProfile threads mode at this node count")
    return p.parse_args(argv)


def _write_music_like_wav(path, seconds, sample_rate=48000):
    """Tones, level changes and clicks, so every node type has work to do."""
    samples = array("h")
    for i in range(int(seconds * sample_rate)):
        t = i / sample_rate
        beat = (t * 2.0) % 1.0
        env = 0.15 + 0.55 * (1.0 - beat) ** 3
        tone = math.sin(2 * math.pi * (220 + 110 * math.floor(t % 4)) * t)
        click = 0.8 if beat < 0.004 else 0.0
        samples.append(int(32767 * max(-1.0, min(1.0, env * tone + click))))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def _percentiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"p50": round(statistics.median(ordered), 3), "p95": round(pick(0.95), 3), "max": round(ordered[-1], 3)}


def _build_scene(n_nodes, wav_path):
    smoke = _smoke_helpers()
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, 100000
    bpy.ops.audiocore.add_audio_source()
    ace = scene.audiocore
    src = ace.audio_sources[ace.active_audio_source_index]
    src.source_type = "FILE"
    src.filepath = wav_path
    mesh = bpy.data.meshes.new("ACE_Perf_Mesh")
    obj = bpy.data.objects.new("ACE_Perf_Object", mesh)
    scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new("ACE_Perf_GN", "GeometryNodeTree")
    tree.is_modifier = True
    obj.modifiers.new("ACE Perf", "NODES").node_group = tree
    source = smoke._add_ace_node(ADDON, tree, "source_mngmt", "src", "Source")
    for i in range(n_nodes):
        node_type = NODE_TYPES[i % len(NODE_TYPES)]
        node = smoke._add_ace_node(ADDON, tree, "stream", node_type, f"{node_type} {i}")
        tree.links.new(source.outputs["Audio"], node.inputs["Audio"])
    return scene, src


def _smoke_helpers():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_integration_smoke.py")
    spec = importlib.util.spec_from_file_location("ace_smoke_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(n_nodes, mode, frames, wav_path, profile=False):
    engine_mod = importlib.import_module(f"{ADDON}.core.engine.engine")
    handlers = importlib.import_module(f"{ADDON}.core.engine.handlers")
    scene, _src = _build_scene(n_nodes, wav_path)
    scene.audiocore.engine_settings.ace_parallel_mode = mode
    scene.audiocore.engine_settings.auto_compile = True
    engine = engine_mod.ACEAudioEngine.ensure_instance(scene)
    engine.compile_scene_frontends(scene)

    # Let the sample-rate probe and its one recompile settle.
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        scene.frame_set(1 + int((time.monotonic() - t0) * 24) % 24)
        handlers.auto_compile_timer()
        time.sleep(1 / 24)
    cold_start_s = 0.0
    if mode == "processes_dsp":
        t0 = time.monotonic()
        f = 1
        while engine.parallel._parallel_result_accepts == 0 and time.monotonic() - t0 < 60:
            scene.frame_set(1 + f % 2)
            f += 1
            time.sleep(1 / 24)
        cold_start_s = round(time.monotonic() - t0, 2)

    ace_ms, frame_ms = [], []
    usable = {"yes": 0, "no": 0}
    original_execute = engine.frame.execute_next_frame
    original_usable = engine.parallel._parallel_has_usable_values_for_graph_frame

    def timed_execute(*a, **k):
        t = time.perf_counter()
        try:
            return original_execute(*a, **k)
        finally:
            ace_ms.append((time.perf_counter() - t) * 1000.0)

    def counted_usable(*a, **k):
        ok = original_usable(*a, **k)
        usable["yes" if ok else "no"] += 1
        return ok

    engine.frame.execute_next_frame = timed_execute
    engine.parallel._parallel_has_usable_values_for_graph_frame = counted_usable
    stale_before = engine.parallel._parallel_stale_job_drops
    accepts_before = engine.parallel._parallel_result_accepts
    profiler = cProfile.Profile() if profile else None
    start_frame = 100
    t0 = time.monotonic()
    if profiler:
        profiler.enable()
    try:
        for i in range(frames):
            target = t0 + i / 24.0
            time.sleep(max(0.0, target - time.monotonic()))
            t = time.perf_counter()
            scene.frame_set(start_frame + i)
            frame_ms.append((time.perf_counter() - t) * 1000.0)
    finally:
        if profiler:
            profiler.disable()
        engine.frame.execute_next_frame = original_execute
        engine.parallel._parallel_has_usable_values_for_graph_frame = original_usable
    late_frames = sum(1 for v in frame_ms if v > 1000.0 / 24.0)
    total_usable = usable["yes"] + usable["no"]
    result = {
        "nodes": n_nodes,
        "mode": mode,
        "frames": frames,
        "ace_ms": _percentiles(ace_ms),
        "frame_ms": _percentiles(frame_ms),
        "late_frames": late_frames,
        "cold_start_s": cold_start_s,
    }
    if mode == "processes_dsp":
        result["fresh_rate"] = round(usable["yes"] / total_usable, 3) if total_usable else 0.0
        result["accepted"] = engine.parallel._parallel_result_accepts - accepts_before
        result["stale_job_drops"] = engine.parallel._parallel_stale_job_drops - stale_before
    if profiler:
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(30)
        result["profile_cumulative"] = buf.getvalue()
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("tottime").print_stats(30)
        result["profile_tottime"] = buf.getvalue()
    engine.parallel._shutdown_parallel_pool(wait=True)
    return result


def main():
    args = _args()
    wav_path = os.path.join(tempfile.mkdtemp(prefix="ace_perf_"), "music_like.wav")
    _write_music_like_wav(wav_path, seconds=args.frames / 24.0 + 12.0)
    results = []
    for n in [int(v) for v in args.nodes.split(",") if v.strip()]:
        for mode in ("threads_io", "processes_dsp"):
            r = _run(n, mode, args.frames, wav_path, profile=(mode == "threads_io" and n == args.profile))
            results.append(r)
            line = {k: v for k, v in r.items() if not k.startswith("profile")}
            print("[ACE perf]", json.dumps(line))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"blender": bpy.app.version_string, "results": results}, f, indent=2)
        print("[ACE perf] wrote", args.out)


if __name__ == "__main__":
    main()
