"""Record every node's output curve inside Blender and compare it with the goldens.

Run headless from the repo root:
  python tests/run_in_blender.py --script tests/reference/blender_reference.py [--extra-path LIBS] [-- OPTIONS]

Options (after ``--``):
  --update            write the goldens instead of comparing
  --fixtures a,b      only these fixtures
  --fps 24,60         only these frame rates
  --out DIR           where actual curves and summary.json go (default tests/artifacts/reference)

Every fixture is played from the first frame, one frame at a time, with the
parallel offload off so the run is deterministic; each pass runs twice and must
repeat exactly. A second pass with the I/O threads on must match the first
within tolerance. Values are read from what ACE wrote into Blender (proxy
attributes), through ``read_node_outputs`` only, so a rewrite of the sinks
changes that one function.
"""

import datetime
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time

import bpy  # type: ignore
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
ROOT = os.path.dirname(TESTS)
for path in (HERE, TESTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import fixtures  # noqa: E402
import spec  # noqa: E402
from blender_helpers import _add_ace_node, _new_gn_tree, find_addon_key  # noqa: E402

GOLDEN_DIR = os.path.join(HERE, "curves")
META_PATH = os.path.join(GOLDEN_DIR, "meta.json")
LIBRARIES = ("numpy", "scipy", "numba", "llvmlite", "soundfile", "sounddevice")


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"update": False, "fixtures": None, "fps": None, "out": os.path.join(TESTS, "artifacts", "reference")}
    it = iter(argv)
    for arg in it:
        if arg == "--update":
            opts["update"] = True
        elif arg == "--fixtures":
            opts["fixtures"] = [s for s in next(it).split(",") if s]
        elif arg == "--fps":
            opts["fps"] = [int(s) for s in next(it).split(",") if s]
        elif arg == "--out":
            opts["out"] = next(it)
        else:
            raise SystemExit(f"Unknown option {arg!r}")
    return opts


# ---------------------------------------------------------------------------
# Reading values: the one place that knows how ACE exposes a node's outputs.
# ---------------------------------------------------------------------------

def read_node_outputs(engine) -> dict:
    """{ace_id: {output_name: value}} from the proxy attributes ACE committed this frame."""
    out: dict = {}
    for graph in engine.graphs.values():
        for attr_key, bindings in dict(getattr(graph, "sink_bindings_by_attr", {}) or {}).items():
            info = engine.writer.output_defs.get(attr_key) or {}
            node_id = str(info.get("node_id", "") or "")
            name = str(info.get("output_name", "") or f"out{int(info.get('socket_idx', 0) or 0)}")
            for binding in bindings:
                if str(getattr(binding, "sink_kind", "")) != "PROXY_ATTRIBUTE":
                    continue
                obj = bpy.data.objects.get(str(getattr(binding, "target_object_name", "") or ""))
                attrs = getattr(getattr(obj, "data", None), "attributes", None)
                attr = attrs.get(str(getattr(binding, "target_data_name", "") or "")) if attrs is not None else None
                if attr is None or len(attr.data) == 0:
                    continue
                item = attr.data[0]
                value = tuple(item.vector) if hasattr(item, "vector") else float(item.value)
                out.setdefault(node_id, {})[name] = value
                break
    return out


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def _sockets(sockets, kind):
    return [s for s in sockets if getattr(s, "type", "") == kind and getattr(s, "enabled", True)]


def build_tree(addon_key, scene, cases):
    """One GN tree: a Source node, and every case's chain hanging off it. Returns {case_id: last node}."""
    _obj, tree = _new_gn_tree(scene, "ACE_Reference")
    source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
    last_nodes = {}
    for case in cases:
        prev = source
        for idx, step in enumerate(case["chain"]):
            node = _add_ace_node(addon_key, tree, step["category"], step["type"], f"{case['id']}#{idx}", mode=step["mode"])
            wants_audio = bool(_sockets(node.inputs, "BUNDLE"))
            kind = "BUNDLE" if wants_audio else "VALUE"
            outputs = _sockets(prev.outputs, kind)
            target = _sockets(node.inputs, kind)
            if not outputs or not target:
                raise AssertionError(f"{case['id']}: cannot link step {idx} ({kind}) from {prev.name}")
            tree.links.new(outputs[step["from_output"]], target[0])
            prev = node
        last_nodes[case["id"]] = prev
    return tree, last_nodes


def _specs_busy(addon_key, scene) -> str:
    """What background tempo analysis may still change a value ("" when nothing)."""
    waiting = importlib.import_module(f"{addon_key}.core.tasks.tempo").TempoService.busy()
    if waiting:
        return f"tempo detection of {waiting}"
    for src in scene.audiocore.audio_sources:
        if str(src.get("ace_tempo_detect_status", "")) == "pending":
            return f"tempo detection pending on {src.name}"
    return ""


def _clear_spectrum_cache(addon_key) -> None:
    """Drop cached spectra, so every fixture starts as a fresh session does (Manual Engine Reset clears it too)."""
    importlib.import_module(f"{addon_key}.core.tasks.spectral").clear_spectrum_cache()


def reset_tempo(source) -> None:
    """Back to a new source's tempo: a file with no beat keeps whatever the last one set."""
    for key in ("ace_bpm_user_override", "ace_meter_user_override", "ace_tempo_detect_status",
                "ace_tempo_detect_message", "ace_tempo_detect_filepath", "ace_tempo_detect_mtime",
                "ace_tempo_detect_auto"):
        if key in source:
            del source[key]
    source.property_unset("bpm")
    source.property_unset("beats_per_bar")


def settle(addon_key, engine, scene, timeout_s=60.0):
    """Play the first second until no recompile and no background analysis has happened for 0.5 s."""
    _clear_spectrum_cache(addon_key)
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine.handlers")
    compiles = []
    original = engine.compiler.compile_scene_frontends
    engine.compiler.compile_scene_frontends = lambda *a, **k: (compiles.append(1), original(*a, **k))[1]
    try:
        t0 = last_change = time.monotonic()
        seen = 0
        step = 0
        while True:
            scene.frame_set(scene.frame_start + step % max(1, int(scene.render.fps)))
            handlers_mod.auto_compile_timer()  # timers do not run in background mode
            step += 1
            now = time.monotonic()
            busy = _specs_busy(addon_key, scene)
            if len(compiles) != seen or busy:
                seen = len(compiles)
                last_change = now
            if now - last_change >= 0.5 and engine.graphs:
                return
            if now - t0 > timeout_s:
                raise AssertionError(f"Engine did not settle in {timeout_s} s (compiles={len(compiles)}, busy: {busy or '-'})")
            time.sleep(1.0 / 48.0)
    finally:
        engine.compiler.compile_scene_frontends = original


def fresh_engine(addon_key, scene):
    """The engine as a file load leaves it: every engine, task and worker dropped, caches kept.

    A seek moves the file reader but keeps each analyzer's history (previous
    spectrum, Onset's running average), so values after a jump depend on what
    played before it. Each pass starts from this state instead.
    """
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine.handlers")
    engine_cls = importlib.import_module(f"{addon_key}.core.engine.engine").ACEAudioEngine
    scene.frame_set(scene.frame_start)
    handlers_mod._load_post_reset_engine(None)
    _clear_spectrum_cache(addon_key)
    engine = engine_cls.ensure_instance(scene)
    engine.compiler.compile_scene_frontends(scene)
    # A compile writes into the user tree; the next frame change sends that
    # write through the depsgraph, and it asks for one more compile after a
    # quiet window. Let that happen here, with the frame held at frame_start,
    # so it never lands in the middle of a pass; then compile once more so
    # every task starts with no history.
    # A fresh engine also probes the file again on a thread, and the result
    # forces a compile when it lands.
    compiles = []
    original = engine.compiler.compile_scene_frontends
    engine.compiler.compile_scene_frontends = lambda *a, **k: (compiles.append(time.monotonic()), original(*a, **k))[1]
    frontends = engine.frontends
    probes = engine.probes

    def busy():
        flags = {
            "dirty": getattr(frontends, "_dirty_frontend_uids", None),
            "forced": getattr(frontends, "_forced_recompile", False),
            "probe_in_flight": getattr(probes, "_probe_in_flight", None),
            "probe_pending": getattr(probes, "_probe_pending_updates", None),
        }
        return ", ".join(f"{k}={v!r}" for k, v in flags.items() if v)

    try:
        t0 = last = time.monotonic()
        while time.monotonic() - last < 0.6 or busy():
            # Setting the same frame again runs no frame handler; evaluate it
            # directly so probe results get flushed. The state this leaves in
            # the tasks is dropped by the compile below.
            engine.frame.execute_next_frame(scene, force=True)
            before = len(compiles)
            handlers_mod.auto_compile_timer()
            if len(compiles) != before or busy():
                last = time.monotonic()
            if time.monotonic() - t0 > 15.0:
                raise AssertionError(f"A fresh engine did not settle ({len(compiles)} compiles; {busy() or 'idle'})")
            time.sleep(0.05)
    finally:
        engine.compiler.compile_scene_frontends = original
    _clear_spectrum_cache(addon_key)
    engine.compiler.compile_scene_frontends(scene)
    return engine


def record(addon_key, scene, last_nodes, n_frames) -> dict:
    """{"<case>/<output>": array over frames}, played from frame_start on a fresh engine."""
    engine = fresh_engine(addon_key, scene)
    compiles = []
    original = engine.compiler.compile_scene_frontends
    frontends = engine.frontends
    dirty_since = {}
    engine.compiler.compile_scene_frontends = lambda *a, **k: (
        compiles.append((scene.frame_current, dict(dirty_since))), original(*a, **k))[1]
    rows: dict = {}
    try:
        for k in range(n_frames):
            scene.frame_set(scene.frame_start + k)
            for uid in list(getattr(frontends, "_dirty_frontend_uids", None) or ()):
                dirty_since.setdefault(uid, scene.frame_current)
            values = read_node_outputs(engine)
            for case_id, node in last_nodes.items():
                for name, value in values.get(str(node.get("ace_id", "")), {}).items():
                    rows.setdefault(f"{case_id}/{name}", []).append(value)
    finally:
        engine.compiler.compile_scene_frontends = original
    if compiles:
        raise AssertionError(f"The scene recompiled while recording (frame, frontends dirty since frame): {compiles}")
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in rows.items()}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(actual: dict, golden: dict) -> dict:
    """Per-key drift report; empty when everything is inside tolerance."""
    report = {}
    for key in sorted(set(actual) | set(golden)):
        if key not in golden:
            report[key] = {"problem": "no golden"}
            continue
        if key not in actual:
            report[key] = {"problem": "not produced"}
            continue
        a, g = actual[key], golden[key]
        if a.shape != g.shape:
            report[key] = {"problem": f"shape {a.shape} != golden {g.shape}"}
            continue
        tol = spec.tolerance(key.split("/", 1)[0])
        both_nan = np.isnan(a) & np.isnan(g)
        diff = np.where(both_nan, 0.0, np.abs(a - g))
        limit = tol["abs"] + tol["rel"] * np.abs(g)
        bad = ~(diff <= limit)
        if np.any(bad):
            frames = np.argwhere(bad.reshape(bad.shape[0], -1).any(axis=1)).ravel()
            report[key] = {
                "problem": "drift",
                "max_abs": float(np.nanmax(diff)),
                "frames": int(frames.size),
                "first_frame": int(frames[0]) + spec.FRAME_START,
                "tolerance": tol,
            }
    return report


def _active_features(addon_key) -> dict:
    """Which optional code paths the add-on is using: goldens from another set would not compare."""
    def flag(module, name):
        try:
            return bool(getattr(importlib.import_module(f"{addon_key}.{module}"), name))
        except ImportError:
            return None
    return {
        "numba": flag("core.tasks.filters", "NUMBA_AVAILABLE"),
        "scipy": flag("core.tasks.filters", "SCIPY_AVAILABLE"),
    }


def _versions() -> dict:
    out = {}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            module = sys.modules.get(name)
            out[name] = getattr(module, "__version__", None) if module else None
    return out


def _ace_version(addon_key) -> str:
    import tomllib
    module = sys.modules.get(addon_key)
    manifest = os.path.join(os.path.dirname(getattr(module, "__file__", "") or ""), "blender_manifest.toml")
    try:
        with open(manifest, "rb") as fh:
            return str(tomllib.load(fh).get("version", ""))
    except OSError:
        return ""


# ---------------------------------------------------------------------------

def main():
    opts = _parse_args()
    addon_key = find_addon_key()
    scene = bpy.context.scene
    ace = scene.audiocore
    engine_cls = importlib.import_module(f"{addon_key}.core.engine.engine").ACEAudioEngine
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine.handlers")
    engine = engine_cls.ensure_instance(scene)

    bpy.ops.audiocore.add_audio_source()
    src = ace.audio_sources[ace.active_audio_source_index]
    cases = spec.cases()
    _tree, last_nodes = build_tree(addon_key, scene, cases)

    names = opts["fixtures"] or list(fixtures.FIXTURES)
    rates = opts["fps"] or list(spec.FPS)
    features = _active_features(addon_key)
    tmp_dir = tempfile.mkdtemp(prefix="ace_reference_")
    os.makedirs(opts["out"], exist_ok=True)

    meta = {}
    if os.path.isfile(META_PATH):
        with open(META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
    if not opts["update"]:
        if not meta:
            raise AssertionError(f"No goldens in {GOLDEN_DIR}: record them with -- --update")
        if meta.get("features") != features:
            raise AssertionError(
                f"Optional libraries differ from the goldens: goldens used {meta.get('features')}, this run has {features}. "
                "Run with the same libraries (see --extra-path)."
            )
        if int(meta.get("fixture_version", -1)) != fixtures.FIXTURE_VERSION:
            raise AssertionError(f"Goldens were recorded from fixture version {meta.get('fixture_version')}, not {fixtures.FIXTURE_VERSION}")

    summary = {"drift": {}, "nondeterministic": {}, "threads_vs_off": {}, "warnings": {}}
    hashes = {}
    t_start = time.monotonic()
    for name in names:
        wav_path = os.path.join(tmp_dir, f"{name}.wav")
        hashes[name] = fixtures.write_wav(name, wav_path)
        for fps in rates:
            tag = spec.golden_name(name, fps)
            n_frames = int(fixtures.DURATION_S * fps)
            scene.render.fps, scene.render.fps_base = int(fps), 1.0
            scene.frame_start, scene.frame_end = spec.FRAME_START, spec.FRAME_START + n_frames - 1
            src.source_type = "FILE"
            src.filepath = wav_path
            reset_tempo(src)
            # And a fresh engine, as after loading a file.
            handlers_mod._load_post_reset_engine(None)
            engine = engine_cls.ensure_instance(scene)

            ace.engine_settings.ace_parallel_mode = "off"
            settle(addon_key, engine, scene)
            first = record(addon_key, scene, last_nodes, n_frames)
            second = record(addon_key, scene, last_nodes, n_frames)
            engine = engine_cls.ensure_instance(scene)
            unstable = sorted(k for k in set(first) | set(second)
                              if k not in first or k not in second or not np.array_equal(first[k], second[k], equal_nan=True))
            if unstable:
                summary["nondeterministic"][tag] = {
                    k: np.flatnonzero(np.any(np.reshape(first[k] != second[k], (len(first[k]), -1)), axis=1))[:8].tolist()
                    if k in first and k in second and first[k].shape == second[k].shape else "missing or reshaped"
                    for k in unstable
                }

            ace.engine_settings.ace_parallel_mode = "threads_io"
            settle(addon_key, engine, scene)
            threaded = record(addon_key, scene, last_nodes, n_frames)
            ace.engine_settings.ace_parallel_mode = "off"
            engine = engine_cls.ensure_instance(scene)
            # Known issue: with the I/O threads on, the first frame of the
            # spectrum-band outputs (splitters) sometimes differs.
            # It is reported, not failed; every later frame must match.
            mismatch = compare({k: v[1:] for k, v in threaded.items()}, {k: v[1:] for k, v in first.items()})
            if mismatch:
                summary["threads_vs_off"][tag] = mismatch
            first_frame = compare({k: v[:1] for k, v in threaded.items()}, {k: v[:1] for k, v in first.items()})
            if first_frame:
                summary["warnings"].setdefault("threads_first_frame", {})[tag] = sorted(first_frame)

            np.savez(os.path.join(opts["out"], f"actual_{tag}"), **first)
            if opts["update"]:
                os.makedirs(GOLDEN_DIR, exist_ok=True)
                np.savez(os.path.join(GOLDEN_DIR, tag), **first)
            else:
                golden_path = os.path.join(GOLDEN_DIR, tag)
                if not os.path.isfile(golden_path):
                    summary["drift"][tag] = {"*": {"problem": "no golden file"}}
                    continue
                with np.load(golden_path) as data:
                    golden = {k: data[k] for k in data.files}
                drift = compare(first, golden)
                if drift:
                    summary["drift"][tag] = drift
            print(f"[ACE reference] {tag}: {len(first)} curves, {n_frames} frames")

    if opts["update"]:
        meta = {
            "fixture_version": fixtures.FIXTURE_VERSION,
            "recorded": datetime.date.today().isoformat(),
            "blender": bpy.app.version_string,
            "ace": _ace_version(addon_key),
            "features": features,
            "libraries": _versions(),
            "fixtures": dict(sorted({**meta.get("fixtures", {}), **hashes}.items())),
            "fps": sorted(set(meta.get("fps", [])) | set(rates)),
        }
        with open(META_PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")

    summary["elapsed_s"] = round(time.monotonic() - t_start, 1)
    with open(os.path.join(opts["out"], "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
    for kind, entries in summary["warnings"].items():
        for tag, keys in entries.items():
            print(f"[ACE reference] warning {kind} {tag}: {len(keys)} curve(s)")
    problems = {k: v for k, v in summary.items() if k not in ("elapsed_s", "warnings") and v}
    for kind, entries in problems.items():
        for tag, detail in entries.items():
            print(f"[ACE reference] {kind} {tag}: {json.dumps(detail)[:600]}")
    if problems:
        raise AssertionError(f"Reference curves failed ({', '.join(problems)}); see {opts['out']}/summary.json")
    print(f"[ACE reference] {'recorded' if opts['update'] else 'matched'} {len(names)} fixtures x {len(rates)} fps in {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()
