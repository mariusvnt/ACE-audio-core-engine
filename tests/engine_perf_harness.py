import concurrent.futures
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from module_loader import load_module, PKG_NAME


ENGINE_PERF_CONTRACT_VERSION = "ace-engine-perf-v1"
ENGINE_PERF_REQUIRED_SCENARIOS = (1, 4, 12, 24, 36, 50)
ENGINE_PERF_MODES = ("threads_io", "processes_dsp")
ENGINE_PERF_FRAMES = 120
ENGINE_PERF_WARMUP_FRAMES = 24
ENGINE_PERF_STRICT_DSP_RATIO_MAX = 1.10
ENGINE_PERF_STRICT_DSP_SCENARIOS = (24, 36, 50)
ENGINE_PERF_STRICT_QUEUE_DROP_RATE_MAX = 0.03
ENGINE_PERF_STRICT_FRESH_RESULT_RATE_MIN = 0.90
ENGINE_PERF_DEFAULT_FRAME_PACE_HZ = 60.0
ENGINE_PERF_DEFAULT_DRAIN_TIMEOUT_S = 2.0
ENGINE_PERF_SLA_SAMPLE_RATE_HZ = 96000
ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS = 15.0
ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS = 10.0
ENGINE_PERF_SLA_MACHINE_PROFILE = {
    "cpu": "AMD Ryzen 7",
    "gpu": "NVIDIA RTX 3060 Ti",
    "ram_gb": 16,
}


def _engine_perf_strict_enabled() -> bool:
    raw = str(os.environ.get("ACE_ENGINE_PERF_GATE_STRICT", "")).strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    ci_raw = str(os.environ.get("CI", "")).strip().lower()
    return ci_raw in {"1", "true", "yes", "on"}


def _safe_percentile(values, q: float) -> float:
    if not values:
        return 0.0
    if np is None:
        ordered = sorted(float(v) for v in values)
        idx = int(round((q / 100.0) * max(len(ordered) - 1, 0)))
        return float(ordered[idx])
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _stage_stats_ms_from_seconds(values_s: List[float]) -> Dict[str, float]:
    if not values_s:
        return {"count": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    vals = [max(float(v), 0.0) for v in values_s]
    return {
        "count": float(len(vals)),
        "mean_ms": float((sum(vals) / max(len(vals), 1)) * 1000.0),
        "p95_ms": float(_safe_percentile(vals, 95.0) * 1000.0),
        "max_ms": float(max(vals) * 1000.0),
    }


def engine_perf_contract() -> Dict[str, Any]:
    return {
        "version": ENGINE_PERF_CONTRACT_VERSION,
        "required_scenarios": [int(v) for v in ENGINE_PERF_REQUIRED_SCENARIOS],
        "modes": list(ENGINE_PERF_MODES),
        "strict_gate": {
            "dsp_p95_ratio_max": float(ENGINE_PERF_STRICT_DSP_RATIO_MAX),
            "dsp_ratio_scenarios": [int(v) for v in ENGINE_PERF_STRICT_DSP_SCENARIOS],
            "parallel_timeouts_max": 0.0,
            "parallel_exceptions_max": 0.0,
            "parallel_queue_drop_rate_max": float(ENGINE_PERF_STRICT_QUEUE_DROP_RATE_MAX),
            "parallel_fresh_result_rate_min": float(ENGINE_PERF_STRICT_FRESH_RESULT_RATE_MIN),
        },
        "sla_topology": {
            "sample_rate_hz": int(ENGINE_PERF_SLA_SAMPLE_RATE_HZ),
            "pass_total_latency_p95_ms": float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS),
            "ideal_total_latency_p95_ms": float(ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS),
            "topology": (
                "3 sources -> specs -> 5-band splitters -> stream terminals; "
                "cross-source multiply -> chromagram; "
                "chroma[0] preview + all chroma outputs to streams"
            ),
            "machine_profile": dict(ENGINE_PERF_SLA_MACHINE_PROFILE),
        },
    }


def _install_fake_bpy():
    if "bpy" in sys.modules:
        return

    bpy_mod = types.ModuleType("bpy")
    bpy_mod.__dict__["_ace_fake_bpy"] = True

    handlers_mod = types.ModuleType("bpy.app.handlers")

    def _persistent(fn):
        return fn

    handlers_mod.persistent = _persistent
    for name in (
        "load_post",
        "undo_post",
        "redo_post",
        "frame_change_pre",
        "frame_change_post",
        "render_pre",
        "render_post",
        "render_cancel",
        "depsgraph_update_post",
    ):
        setattr(handlers_mod, name, [])

    timers_ns = types.SimpleNamespace(
        is_registered=lambda _fn: False,
        register=lambda *_args, **_kwargs: None,
        unregister=lambda *_args, **_kwargs: None,
    )
    app_ns = types.SimpleNamespace(
        handlers=handlers_mod,
        timers=timers_ns,
        is_job_running=lambda _name: False,
    )
    wm_ns = types.SimpleNamespace(windows=[])
    bpy_mod.app = app_ns
    bpy_mod.context = types.SimpleNamespace(scene=None, window_manager=wm_ns)
    bpy_mod.data = types.SimpleNamespace(objects={})
    bpy_mod.types = types.SimpleNamespace(
        NodeTree=type("NodeTree", (), {}),
        Object=type("Object", (), {}),
        Scene=type("Scene", (), {}),
    )

    app_mod = types.ModuleType("bpy.app")
    app_mod.handlers = handlers_mod
    app_mod.timers = timers_ns
    app_mod.is_job_running = app_ns.is_job_running

    sys.modules["bpy"] = bpy_mod
    sys.modules["bpy.app"] = app_mod
    sys.modules["bpy.app.handlers"] = handlers_mod


def _install_engine_import_stubs():
    core_pkg = types.ModuleType(f"{PKG_NAME}.core")
    core_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault(f"{PKG_NAME}.core", core_pkg)
    nodes_pkg = types.ModuleType(f"{PKG_NAME}.nodes")
    nodes_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault(f"{PKG_NAME}.nodes", nodes_pkg)

    properties_mod = types.ModuleType(f"{PKG_NAME}.properties")

    class ACEAudioSource:  # pragma: no cover - type placeholder for imports
        pass

    properties_mod.ACEAudioSource = ACEAudioSource
    sys.modules[f"{PKG_NAME}.properties"] = properties_mod

    dummy_tree_mod = types.ModuleType(f"{PKG_NAME}.nodes.dummy_tree")

    class ACEDummyTreeReader:  # pragma: no cover - compile path not used here
        def __init__(self, *_args, **_kwargs):
            self.bl_tree = _kwargs.get("blender_node_tree")

        def build(self):
            return None

    dummy_tree_mod.ACEDummyTreeReader = ACEDummyTreeReader
    dummy_tree_mod.ensure_frontend_uid = lambda tree, _kind, _existing: getattr(tree, "ace_frontend_uid", "")
    dummy_tree_mod.FRONTEND_KIND_GEOMETRY = "GEOMETRY_NODES"
    dummy_tree_mod.FRONTEND_KIND_SHADER = "SHADER_NODES"
    sys.modules[f"{PKG_NAME}.nodes.dummy_tree"] = dummy_tree_mod
    sys.modules[f"{PKG_NAME}.dummy_tree"] = dummy_tree_mod

    shader_tree_mod = types.ModuleType(f"{PKG_NAME}.nodes.shader_tree")

    class ACEShaderTreeReader(ACEDummyTreeReader):
        pass

    shader_tree_mod.ACEShaderTreeReader = ACEShaderTreeReader
    sys.modules[f"{PKG_NAME}.nodes.shader_tree"] = shader_tree_mod

    proxy_mod = types.ModuleType(f"{PKG_NAME}.core.proxy_management")

    class ACEProxyManager:  # pragma: no cover - runtime harness uses a dedicated writer stub
        COLLECTION_NAME = "ACE_Internals"

        def __init__(self, _abstract_tree):
            self.output_defs = {}

        @staticmethod
        def ensure_proxy_for_source(_source, scene=None):  # noqa: ARG004
            return None

        def ensure_attributes_exist(self, _proxy, _tree, active_keys=None, prune_existing=False):  # noqa: ARG002
            return None

        def update_proxy_object(self, _proxy, _results, active_keys=None):  # noqa: ARG002
            return None

        def ensure_sink_layout(self, _graph, active_keys=None):  # noqa: ARG002
            return set()

        def commit_results(self, _graph, _results, active_keys=None):  # noqa: ARG002
            return set()

        def clear_runtime_caches(self):
            return None

    proxy_mod.ACEProxyManager = ACEProxyManager
    sys.modules[f"{PKG_NAME}.core.proxy_management"] = proxy_mod
    sys.modules[f"{PKG_NAME}.proxy_management"] = proxy_mod


def _load_engine_stack():
    _install_fake_bpy()
    _install_engine_import_stubs()
    for name in (
        "core.audio_engine",
        "core.audio_tasks_registry",
        "core.parallel_dsp",
        "core.audio_influx",
        "core.abstract_tree",
        "audio_engine",
        "audio_tasks_registry",
        "parallel_dsp",
        "audio_influx",
        "abstract_tree",
    ):
        sys.modules.pop(f"{PKG_NAME}.{name}", None)
    ae = load_module("audio_engine")
    atr = load_module("audio_tasks_registry")
    abs_mod = load_module("abstract_tree")
    return ae, atr, abs_mod


class _FakeRender:
    fps = 24
    fps_base = 1.0
    use_lock_interface = True


class _FakeEngineSettings:
    def __init__(self, parallel_mode: str):
        self.auto_compile = True
        self.ace_perf_enabled = True
        self.ace_scheduler_mode = "demand"
        self.ace_timeslice_mode = "fixed"
        self.ace_parallel_mode = str(parallel_mode)
        self.ace_preview_block_size = "256"
        self.preview_numba_turbo = True


class _FakeSource:
    def __init__(self, stable_id: str, sample_rate: int = 48000):
        self.name = stable_id
        self.stable_id = stable_id
        self.proxy_object_name = f"ACE_PROXY_{stable_id}"
        self.source_type = "FILE"
        self.filepath = ""
        self.flow_identifier = ""
        self.live_latency_ms = 10
        self.sample_rate = int(sample_rate)
        self.duration = 12.0

    def __getitem__(self, _key):
        raise KeyError

    def __setitem__(self, _key, _value):
        return None

    def get(self, _key, default=None):
        return default


class _FakeAudioCore:
    def __init__(self, sources: List[_FakeSource], parallel_mode: str):
        self.audio_sources = list(sources)
        self.active_audio_source_index = 0
        self.engine_settings = _FakeEngineSettings(parallel_mode=parallel_mode)
        self.mode = "FILE"


class _FakeScene:
    def __init__(self, sources: List[_FakeSource], parallel_mode: str):
        self.frame_current = 1
        self.frame_start = 1
        self.frame_end = 250
        self.name = "ACE_EnginePerfScene"
        self.render = _FakeRender()
        self.audiocore = _FakeAudioCore(sources, parallel_mode=parallel_mode)

    def as_pointer(self):
        return 11001


class _WriterStub:
    def __init__(self, output_defs: Dict[str, Dict[str, Any]]):
        self.output_defs = dict(output_defs)
        self._last = {}

    def ensure_attributes_exist(self, _proxy, _tree, active_keys=None, prune_existing=False):  # noqa: ARG002
        return None

    def update_proxy_object(self, _proxy, results, active_keys=None):  # noqa: ARG002
        self._last = dict(results or {})

    def ensure_sink_layout(self, _graph, active_keys=None):  # noqa: ARG002
        return set()

    def commit_results(self, _graph, results, active_keys=None):  # noqa: ARG002
        self._last = dict(results or {})
        return set()

    def clear_runtime_caches(self):
        self._last = {}


def _bench_source_task_cls(atr):
    class BenchSourceTask(atr.SourceTask):
        def __init__(self, node_id, params, block_size):  # noqa: ARG002
            super().__init__(node_id, {}, block_size)
            self.sample_rate = 48000
            self.source_ready = True
            self.live_ready = True
            self.initialized = True
            t = np.arange(int(self.block_size), dtype=np.float32) / float(self.sample_rate)
            self._base = np.sin((2.0 * math.pi * 220.0 * t)).astype(np.float32, copy=False)
            self._mono = np.empty(int(self.block_size), dtype=np.float32)
            self._stereo = np.empty((2, int(self.block_size)), dtype=np.float32)

        def _read_next_block(self):
            t0 = time.perf_counter()
            try:
                frame_idx = int(getattr(self, "_current_frame_request", 0) or 0)
                n = int(self.block_size)
                shift = int(frame_idx) % max(n, 1)
                if shift <= 0:
                    np.copyto(self._mono, self._base)
                else:
                    self._mono[:shift] = self._base[-shift:]
                    self._mono[shift:] = self._base[:-shift]
                self._stereo[0, :] = self._mono
                self._stereo[1, 0] = self._mono[-1]
                self._stereo[1, 1:] = self._mono[:-1]
                latency_ms = (float(n) / float(max(self.sample_rate, 1))) * 1000.0
                return atr.AudioChunk(
                    self._mono.astype(np.float32, copy=False),
                    self._stereo.astype(np.float32, copy=False),
                    int(self.sample_rate),
                    provider=None,
                    frame_index=frame_idx,
                    lineage=(self.node_id,),
                    latency_ms=float(latency_ms),
                )
            finally:
                engine = getattr(self, "engine_ref", None)
                perf_add = getattr(engine, "_perf_add", None)
                if callable(perf_add):
                    perf_add("source_read", time.perf_counter() - t0)

    return BenchSourceTask


def _configure_bench_source_task(task, sample_rate: int, freq_hz: float):
    sr = int(max(sample_rate, 1))
    task.sample_rate = sr
    t = np.arange(int(task.block_size), dtype=np.float32) / float(sr)
    task._base = np.sin((2.0 * math.pi * float(freq_hz) * t)).astype(np.float32, copy=False)


def _build_engine_graph(ae, atr, abs_mod, operator_count: int, parallel_mode: str):
    source = _FakeSource("engine_perf_src")
    scene = _FakeScene([source], parallel_mode=parallel_mode)
    graph = atr.ACEAudioGraph(source, source.proxy_object_name)
    graph.source_uid = source.stable_id
    graph.block_size = 1024
    graph.scheduler_mode = "demand"
    graph.timeslice_mode = "fixed"
    graph.manual_preview_task_ids = []
    graph.auto_preview_task_ids = []
    graph.preview_task_ids = []

    BenchSourceTask = _bench_source_task_cls(atr)
    source_task = BenchSourceTask("src_node", {}, block_size=graph.block_size)
    source_task.source_context = source
    _configure_bench_source_task(source_task, sample_rate=int(source.sample_rate), freq_hz=220.0)
    graph.tasks["src_node"] = source_task
    graph.source_task_ids = ["src_node"]
    graph.file_source_task_ids = ["src_node"]
    graph.live_source_task_ids = []

    stream_types = (
        ("rms", atr.RMSTask),
        ("pitch", atr.PitchTask),
        ("spectral_centroid", atr.SpectralCentroidTask),
        ("spectral_flux", atr.SpectralFluxTask),
        ("onset", atr.OnsetTask),
        ("zero_crossing_rate", atr.ZeroCrossTask),
        ("stereo_width", atr.StereoWidthTask),
        ("stereo_balance", atr.StereoBalanceTask),
    )

    abstract_tree = abs_mod.ACEAbstractTree()
    abstract_tree.add_node(
        abs_mod.ACEAbstractNode(
            id="src_node",
            name="Source",
            category="source_mngmt",
            type="src",
            parameters={},
        )
    )

    output_attr_keys: List[str] = []
    output_defs: Dict[str, Dict[str, Any]] = {}
    for idx in range(int(max(operator_count, 1))):
        stype, cls = stream_types[idx % len(stream_types)]
        node_id = f"{stype}_{idx}"
        task = cls(node_id, {}, block_size=graph.block_size)
        task.connect_input(0, source_task, 0)
        graph.tasks[node_id] = task
        attr_key = f"{node_id}:0"
        output_attr_keys.append(attr_key)
        output_defs[attr_key] = {"node_id": node_id, "type": "FLOAT"}
        abstract_tree.add_node(
            abs_mod.ACEAbstractNode(
                id=node_id,
                name=node_id,
                category="stream",
                type=stype,
                parameters={},
            )
        )
        abstract_tree.add_link(abs_mod.ACEAbstractLink("src_node", node_id, 0, 0))

    graph.output_attr_keys = output_attr_keys
    graph.output_slot_specs = [(k, k.partition(":")[0], 0) for k in output_attr_keys]
    graph.output_slot_specs_by_subset = {}

    engine = ae.ACEAudioEngine(scene_key=scene.as_pointer())
    engine.scene_name = scene.name
    engine.writer = _WriterStub(output_defs=output_defs)
    engine.abstract_tree_ref = abstract_tree
    engine.graphs = {source.stable_id: graph}
    engine._last_attr_sync_frame[source.stable_id] = -10_000_000
    engine._rebuild_graph_execution_order(graph)
    engine._prepare_parallel_graph_ir(graph, abstract_tree)
    proxy_obj = object()
    graph.get_proxy_object = lambda: proxy_obj  # type: ignore[method-assign]
    engine._parallel_mode = types.MethodType(lambda _self, _scene_or_settings=None, _mode=parallel_mode: str(_mode), engine)

    use_real_processes = str(os.environ.get("ACE_ENGINE_PERF_USE_REAL_PROCESSES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if parallel_mode == "processes_dsp" and (not use_real_processes):
        def _ensure_parallel_pool_threaded(self, scene_obj):  # noqa: ANN001
            if self._parallel_mode(scene_obj) != "processes_dsp":
                self._shutdown_parallel_pool(wait=False)
                return False
            if self._parallel_pool_failed:
                return False
            if self._parallel_executor is not None:
                return True
            # The worker-side graph cache is process-isolated by design; when the
            # harness emulates process mode with threads, keep it single-worker to
            # avoid cross-thread reuse of mutable analysis state.
            self._parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._mark_parallel_ir_required_for_all_graphs()
            return True

        engine._ensure_parallel_pool = types.MethodType(_ensure_parallel_pool_threaded, engine)

    return engine, scene


def _build_sla_engine_graph(ae, atr, abs_mod, parallel_mode: str):
    sources = [
        _FakeSource(f"engine_perf_src_{idx+1}", sample_rate=ENGINE_PERF_SLA_SAMPLE_RATE_HZ)
        for idx in range(3)
    ]
    scene = _FakeScene(sources, parallel_mode=parallel_mode)
    graph = atr.ACEAudioGraph(sources[0], sources[0].proxy_object_name)
    graph.source_uid = "sla_multi_source"
    graph.source_context = sources[0]
    graph.block_size = 1024
    graph.scheduler_mode = "demand"
    graph.timeslice_mode = "fixed"
    graph.manual_preview_task_ids = []
    graph.auto_preview_task_ids = []
    graph.preview_task_ids = []

    BenchSourceTask = _bench_source_task_cls(atr)
    abstract_tree = abs_mod.ACEAbstractTree()
    output_attr_keys: List[str] = []
    output_defs: Dict[str, Dict[str, Any]] = {}
    source_task_ids: List[str] = []
    file_source_task_ids: List[str] = []
    specs_task_ids: List[str] = []
    split_freqs = (180.0, 420.0, 1200.0, 3600.0)

    for idx, source in enumerate(sources):
        src_node_id = f"src_{idx}"
        src_task = BenchSourceTask(src_node_id, {}, block_size=graph.block_size)
        src_task.source_context = source
        _configure_bench_source_task(src_task, sample_rate=ENGINE_PERF_SLA_SAMPLE_RATE_HZ, freq_hz=(220.0 * float(idx + 1)))
        graph.tasks[src_node_id] = src_task
        source_task_ids.append(src_node_id)
        file_source_task_ids.append(src_node_id)
        abstract_tree.add_node(
            abs_mod.ACEAbstractNode(
                id=src_node_id,
                name=src_node_id,
                category="source_mngmt",
                type="src",
                parameters={},
            )
        )

        specs_id = f"specs_{idx}"
        specs_task = atr.SourceSpecsTask(specs_id, {}, block_size=graph.block_size)
        specs_task.source_context = source
        specs_task.connect_input(0, src_task, 0)
        graph.tasks[specs_id] = specs_task
        specs_task_ids.append(specs_id)
        abstract_tree.add_node(
            abs_mod.ACEAbstractNode(
                id=specs_id,
                name=specs_id,
                category="utilities",
                type="src_specs",
                parameters={},
            )
        )
        abstract_tree.add_link(abs_mod.ACEAbstractLink(src_node_id, specs_id, 0, 0))

        split_id = f"split_{idx}"
        split_params = {f"in_{i + 1}": float(freq) for i, freq in enumerate(split_freqs)}
        split_task = atr.FrequencySplitterTask(split_id, split_params, block_size=graph.block_size)
        split_task.connect_input(0, specs_task, 2)
        graph.tasks[split_id] = split_task
        abstract_tree.add_node(
            abs_mod.ACEAbstractNode(
                id=split_id,
                name=split_id,
                category="delta",
                type="frequency_splitter",
                parameters=split_params,
            )
        )
        abstract_tree.add_link(abs_mod.ACEAbstractLink(specs_id, split_id, 2, 0))

        for band_idx in range(5):
            stream_id = f"src{idx}_band{band_idx}_rms"
            stream_task = atr.RMSTask(stream_id, {}, block_size=graph.block_size)
            stream_task.connect_input(0, split_task, band_idx)
            graph.tasks[stream_id] = stream_task
            attr_key = f"{stream_id}:0"
            output_attr_keys.append(attr_key)
            output_defs[attr_key] = {"node_id": stream_id, "type": "FLOAT"}
            abstract_tree.add_node(
                abs_mod.ACEAbstractNode(
                    id=stream_id,
                    name=stream_id,
                    category="stream",
                    type="rms",
                    parameters={},
                )
            )
            abstract_tree.add_link(abs_mod.ACEAbstractLink(split_id, stream_id, band_idx, 0))

    math_id = "math_mul_3src"
    math_task = atr.AudioMathTask(math_id, {"op": "multiply"}, block_size=graph.block_size)
    for idx, specs_id in enumerate(specs_task_ids):
        specs_task = graph.tasks.get(specs_id)
        if specs_task is None:
            continue
        math_task.connect_input(idx, specs_task, 2)
        abstract_tree.add_link(abs_mod.ACEAbstractLink(specs_id, math_id, 2, idx))
    graph.tasks[math_id] = math_task
    abstract_tree.add_node(
        abs_mod.ACEAbstractNode(
            id=math_id,
            name=math_id,
            category="utilities",
            type="audio_math",
            parameters={"op": "multiply"},
        )
    )

    chroma_id = "chroma_mix"
    chroma_task = atr.ChromagramTask(chroma_id, {}, block_size=graph.block_size)
    chroma_task.connect_input(0, math_task, 0)
    graph.tasks[chroma_id] = chroma_task
    abstract_tree.add_node(
        abs_mod.ACEAbstractNode(
            id=chroma_id,
            name=chroma_id,
            category="delta",
            type="chromagram",
            parameters={},
        )
    )
    abstract_tree.add_link(abs_mod.ACEAbstractLink(math_id, chroma_id, 0, 0))

    preview_id = "sla_preview_chroma0"
    preview_task = atr.PreviewSinkTask(preview_id, {"preview_enabled": True}, block_size=graph.block_size)
    preview_task.connect_input(0, chroma_task, 0)
    graph.tasks[preview_id] = preview_task
    graph.manual_preview_task_ids = [preview_id]
    graph.preview_task_ids = [preview_id]
    abstract_tree.add_node(
        abs_mod.ACEAbstractNode(
            id=preview_id,
            name=preview_id,
            category="utilities",
            type="preview_sink",
            parameters={"preview_enabled": True},
        )
    )
    abstract_tree.add_link(abs_mod.ACEAbstractLink(chroma_id, preview_id, 0, 0))

    for chroma_idx in range(12):
        stream_id = f"chroma_{chroma_idx}_rms"
        stream_task = atr.RMSTask(stream_id, {}, block_size=graph.block_size)
        stream_task.connect_input(0, chroma_task, chroma_idx)
        graph.tasks[stream_id] = stream_task
        attr_key = f"{stream_id}:0"
        output_attr_keys.append(attr_key)
        output_defs[attr_key] = {"node_id": stream_id, "type": "FLOAT"}
        abstract_tree.add_node(
            abs_mod.ACEAbstractNode(
                id=stream_id,
                name=stream_id,
                category="stream",
                type="rms",
                parameters={},
            )
        )
        abstract_tree.add_link(abs_mod.ACEAbstractLink(chroma_id, stream_id, chroma_idx, 0))

    graph.source_task_ids = source_task_ids
    graph.file_source_task_ids = file_source_task_ids
    graph.live_source_task_ids = []
    graph.output_attr_keys = output_attr_keys
    graph.output_slot_specs = [(k, k.partition(":")[0], 0) for k in output_attr_keys]
    graph.output_slot_specs_by_subset = {}

    engine = ae.ACEAudioEngine(scene_key=scene.as_pointer())
    engine.scene_name = scene.name
    engine.writer = _WriterStub(output_defs=output_defs)
    engine.abstract_tree_ref = abstract_tree
    engine.graphs = {graph.source_uid: graph}
    engine._last_attr_sync_frame[graph.source_uid] = -10_000_000
    engine._rebuild_graph_execution_order(graph)
    engine._prepare_parallel_graph_ir(graph, abstract_tree)
    proxy_obj = object()
    graph.get_proxy_object = lambda: proxy_obj  # type: ignore[method-assign]
    engine._parallel_mode = types.MethodType(lambda _self, _scene_or_settings=None, _mode=parallel_mode: str(_mode), engine)

    use_real_processes = str(os.environ.get("ACE_ENGINE_PERF_USE_REAL_PROCESSES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if parallel_mode == "processes_dsp" and (not use_real_processes):
        def _ensure_parallel_pool_threaded(self, scene_obj):  # noqa: ANN001
            if self._parallel_mode(scene_obj) != "processes_dsp":
                self._shutdown_parallel_pool(wait=False)
                return False
            if self._parallel_pool_failed:
                return False
            if self._parallel_executor is not None:
                return True
            # The worker-side graph cache is process-isolated by design; when the
            # harness emulates process mode with threads, keep it single-worker to
            # avoid cross-thread reuse of mutable analysis state.
            self._parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._mark_parallel_ir_required_for_all_graphs()
            return True

        engine._ensure_parallel_pool = types.MethodType(_ensure_parallel_pool_threaded, engine)

    return engine, scene


def _run_engine_benchmark(
    ae,
    atr,
    abs_mod,
    operator_count: int,
    mode: str,
    frames: int,
    warmup_frames: int,
    build_graph_fn=None,
    preview_playing: bool = False,
) -> Dict[str, Any]:
    if callable(build_graph_fn):
        engine, scene = build_graph_fn(
            ae=ae,
            atr=atr,
            abs_mod=abs_mod,
            parallel_mode=mode,
        )
    else:
        engine, scene = _build_engine_graph(
            ae=ae,
            atr=atr,
            abs_mod=abs_mod,
            operator_count=operator_count,
            parallel_mode=mode,
        )
    original_is_animation_playing = ae.ACEAudioEngine._is_animation_playing
    bpy_mod = sys.modules.get("bpy")
    prev_context_scene = getattr(getattr(bpy_mod, "context", None), "scene", None) if bpy_mod else None
    ae.ACEAudioEngine._is_animation_playing = staticmethod(lambda: bool(preview_playing))
    frame_pace_hz = float(os.environ.get("ACE_ENGINE_PERF_FRAME_PACE_HZ", str(ENGINE_PERF_DEFAULT_FRAME_PACE_HZ)) or ENGINE_PERF_DEFAULT_FRAME_PACE_HZ)
    frame_interval_s = (1.0 / frame_pace_hz) if frame_pace_hz > 0.0 else 0.0
    drain_timeout_s = float(os.environ.get("ACE_ENGINE_PERF_DRAIN_TIMEOUT_S", str(ENGINE_PERF_DEFAULT_DRAIN_TIMEOUT_S)) or ENGINE_PERF_DEFAULT_DRAIN_TIMEOUT_S)

    submit_counters = {"attempts": 0, "success": 0}
    result_counters = {"requests": 0, "fresh": 0}
    parallel_wait_samples: List[float] = []
    completion_counters = {"done": 0}
    completion_elapsed_samples: List[float] = []
    seen_completed_futures: set[int] = set()
    original_submit = engine._parallel_submit_graph_job
    original_values_for_frame = engine._parallel_values_for_graph_frame
    original_poll_futures = engine._parallel_poll_futures

    def _submit_instrumented(*args, **kwargs):
        submit_counters["attempts"] += 1
        ok = bool(original_submit(*args, **kwargs))
        if ok:
            submit_counters["success"] += 1
        return ok

    def _values_for_frame_instrumented(graph, graph_key, frame_index):
        result_counters["requests"] += 1
        frame_map = engine._parallel_results.get(str(graph_key), {}) or {}
        if int(frame_index) in frame_map:
            result_counters["fresh"] += 1
        t0 = time.perf_counter()
        out = original_values_for_frame(graph, graph_key, frame_index)
        parallel_wait_samples.append(max(time.perf_counter() - t0, 0.0))
        return out

    def _poll_futures_instrumented(scene_obj, current_frame):  # noqa: ANN001
        for fut in list(getattr(engine, "_parallel_future_meta", {}).keys()):
            fut_key = id(fut)
            if fut_key in seen_completed_futures or (not fut.done()):
                continue
            try:
                payload = fut.result(timeout=0.0)
            except Exception:
                seen_completed_futures.add(fut_key)
                continue
            seen_completed_futures.add(fut_key)
            if isinstance(payload, dict) and (not bool(payload.get("needs_graph_ir", False))):
                completion_counters["done"] += 1
                elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
                if elapsed_s > 0.0:
                    completion_elapsed_samples.append(elapsed_s)
        return original_poll_futures(scene_obj, current_frame)

    if mode == "processes_dsp":
        engine._parallel_submit_graph_job = _submit_instrumented  # type: ignore[assignment]
        engine._parallel_values_for_graph_frame = _values_for_frame_instrumented  # type: ignore[assignment]
        engine._parallel_poll_futures = _poll_futures_instrumented  # type: ignore[assignment]

    def _maybe_pace(next_due_ts: float) -> float:
        if frame_interval_s <= 0.0:
            return time.perf_counter()
        now_ts = time.perf_counter()
        if next_due_ts > now_ts:
            time.sleep(next_due_ts - now_ts)
        return next_due_ts + frame_interval_s

    def _drive_frames(start_frame: int, num_frames: int, next_due_ts: float) -> float:
        frame_cursor = int(start_frame)
        due = float(next_due_ts)
        for _ in range(int(max(num_frames, 0))):
            frame_cursor += 1
            scene.frame_current = int(frame_cursor)
            engine.execute_next_frame(scene, force=False)
            due = _maybe_pace(due)
        return due

    def _drain_parallel_futures():
        if mode != "processes_dsp":
            return
        deadline = time.perf_counter() + max(float(drain_timeout_s), 0.1)
        while bool(getattr(engine, "_parallel_future_meta", {})) and time.perf_counter() < deadline:
            try:
                engine._parallel_poll_futures(scene, current_frame=int(scene.frame_current))
            except Exception:
                break
            time.sleep(0.001)
        try:
            engine._parallel_poll_futures(scene, current_frame=int(scene.frame_current))
        except Exception:
            pass

    try:
        if bpy_mod is not None and hasattr(bpy_mod, "context"):
            bpy_mod.context.scene = scene

        next_due = time.perf_counter()
        _ = _drive_frames(start_frame=0, num_frames=int(max(warmup_frames, 0)), next_due_ts=next_due)

        # Measure only post-warmup frames.
        engine.get_perf_snapshot(reset=True)
        engine.get_runtime_diagnostics(reset=True)
        engine.get_dsp_node_profile_snapshot(reset=True)
        submit_counters["attempts"] = 0
        submit_counters["success"] = 0
        result_counters["requests"] = 0
        result_counters["fresh"] = 0

        next_due = time.perf_counter()
        _drive_frames(
            start_frame=int(max(warmup_frames, 0)),
            num_frames=int(max(frames, 1)),
            next_due_ts=next_due,
        )
        _drain_parallel_futures()

        perf = engine.get_perf_snapshot(reset=False)
        diag = engine.get_runtime_diagnostics(reset=False)
        dsp_profile = engine.get_dsp_node_profile_snapshot(reset=False)
        dsp_stage_source = "engine_dsp"
        if mode == "processes_dsp":
            dsp_stage = dict(perf.get("dsp", {}) or {})
            dsp_stage_p95 = float(dsp_stage.get("p95_ms", 0.0) or 0.0)
            if dsp_stage_p95 <= 0.0 and completion_elapsed_samples:
                perf["dsp"] = _stage_stats_ms_from_seconds(completion_elapsed_samples)
                dsp_stage_source = "parallel_worker_elapsed"
            elif float(dsp_stage.get("count", 0.0) or 0.0) <= 0.0 and parallel_wait_samples:
                perf["dsp"] = _stage_stats_ms_from_seconds(parallel_wait_samples)
                dsp_stage_source = "parallel_wait_fallback"
        graph = next(iter(engine.graphs.values()))
        submit_attempts = float(submit_counters["attempts"])
        submit_success = float(submit_counters["success"])
        result_requests = float(result_counters["requests"])
        fresh_results = float(result_counters["fresh"])
        completed_results = float(completion_counters["done"])
        accepted_results = float(diag.get("parallel_result_accepts", 0.0) or 0.0)
        queue_drops = float(diag.get("parallel_queue_pressure_drops", 0.0) or 0.0)
        queue_drop_rate = (
            float(queue_drops / submit_attempts)
            if submit_attempts > 0.0
            else 0.0
        )
        fresh_rate = (
            float(completed_results / submit_success)
            if (mode == "processes_dsp" and submit_success > 0.0)
            else (
                float(fresh_results / result_requests)
                if result_requests > 0.0
                else 1.0
            )
        )
        result = {
            "stages": perf,
            "realtime": {
                "parallel_job_timeouts": float(diag.get("parallel_job_timeouts", 0.0) or 0.0),
                "parallel_job_exceptions": float(diag.get("parallel_job_exceptions", 0.0) or 0.0),
                "parallel_pool_restarts": float(diag.get("parallel_pool_restarts", 0.0) or 0.0),
                "parallel_queue_pressure_drops": queue_drops,
                "parallel_queue_drop_rate": float(queue_drop_rate),
                "parallel_disabled_sources": float(diag.get("parallel_disabled_sources", 0.0) or 0.0),
                "parallel_submit_attempts": submit_attempts,
                "parallel_submit_success": submit_success,
                "parallel_submit_success_rate": float(submit_success / submit_attempts) if submit_attempts > 0.0 else 0.0,
                "parallel_result_requests": result_requests,
                "parallel_fresh_results": fresh_results,
                "parallel_completed_results": completed_results,
                "parallel_accepted_results": accepted_results,
                "parallel_fresh_result_rate": float(fresh_rate),
            },
            "parallel_meta": {
                "parallel_offload_attr_count": float(len(getattr(graph, "parallel_offload_attr_keys", []) or [])),
                "parallel_local_attr_count": float(len(getattr(graph, "parallel_local_attr_keys", []) or [])),
                "parallel_shards": float(len(getattr(graph, "parallel_output_shards", []) or [])),
                "dsp_stage_source": str(dsp_stage_source),
            },
            "dsp_hotspots": dsp_profile,
        }
        stage_total = 0.0
        for stage in ("source_read", "dsp", "preview", "proxy_write", "ui_sync"):
            stage_total += float(dict(perf.get(stage, {}) or {}).get("p95_ms", 0.0) or 0.0)
        result["total_latency_p95_ms"] = float(stage_total)
        return result
    finally:
        ae.ACEAudioEngine._is_animation_playing = original_is_animation_playing
        if mode == "processes_dsp":
            engine._parallel_submit_graph_job = original_submit  # type: ignore[assignment]
            engine._parallel_values_for_graph_frame = original_values_for_frame  # type: ignore[assignment]
            engine._parallel_poll_futures = original_poll_futures  # type: ignore[assignment]
        if bpy_mod is not None and hasattr(bpy_mod, "context"):
            bpy_mod.context.scene = prev_context_scene
        try:
            engine._shutdown_graphs(wait_parallel_pool=True, clear_bake_cache=True)
        except Exception:
            pass


def generate_engine_perf_report(
    frames: int = ENGINE_PERF_FRAMES,
    warmup_frames: int = ENGINE_PERF_WARMUP_FRAMES,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("NumPy is required for engine perf harness")

    ae, atr, abs_mod = _load_engine_stack()
    strict_mode = _engine_perf_strict_enabled()
    scenarios: Dict[str, Dict[str, Any]] = {}
    strict_dsp_scenarios = {int(v) for v in ENGINE_PERF_STRICT_DSP_SCENARIOS}
    for operator_count in ENGINE_PERF_REQUIRED_SCENARIOS:
        sid = str(operator_count)
        scenarios[sid] = {}
        threads_res = _run_engine_benchmark(
            ae=ae,
            atr=atr,
            abs_mod=abs_mod,
            operator_count=operator_count,
            mode="threads_io",
            frames=frames,
            warmup_frames=warmup_frames,
        )
        proc_res = _run_engine_benchmark(
            ae=ae,
            atr=atr,
            abs_mod=abs_mod,
            operator_count=operator_count,
            mode="processes_dsp",
            frames=frames,
            warmup_frames=warmup_frames,
        )
        scenarios[sid]["threads_io"] = threads_res
        scenarios[sid]["processes_dsp"] = proc_res

        threads_dsp_p95 = float(threads_res["stages"]["dsp"]["p95_ms"])
        proc_dsp_p95 = float(proc_res["stages"]["dsp"]["p95_ms"])
        valid_dsp_metrics = bool(threads_dsp_p95 > 0.0 and proc_dsp_p95 > 0.0)
        ratio = (proc_dsp_p95 / max(threads_dsp_p95, 1e-9)) if valid_dsp_metrics else 9999.0
        scenarios[sid]["comparison"] = {
            "dsp_p95_ratio_processes_vs_threads": float(ratio),
            "dsp_p95_delta_ms": float(proc_dsp_p95 - threads_dsp_p95),
            "dsp_metrics_valid": bool(valid_dsp_metrics),
            "threads_dsp_p95_ms": float(threads_dsp_p95),
            "processes_dsp_p95_ms": float(proc_dsp_p95),
        }
        # Strict DSP validity is only required for the contract scenarios used by
        # strict gate assertions (24/36/50), not for low-load smoke scenarios.
        if strict_mode and int(operator_count) in strict_dsp_scenarios and (not valid_dsp_metrics):
            raise RuntimeError(
                "Invalid DSP metrics in strict engine perf mode "
                f"(scenario={sid}, threads={threads_dsp_p95:.6f}, processes={proc_dsp_p95:.6f})"
            )

    sla_modes: Dict[str, Dict[str, Any]] = {}
    for mode in ENGINE_PERF_MODES:
        sla_modes[str(mode)] = _run_engine_benchmark(
            ae=ae,
            atr=atr,
            abs_mod=abs_mod,
            operator_count=0,
            mode=str(mode),
            frames=frames,
            warmup_frames=warmup_frames,
            build_graph_fn=_build_sla_engine_graph,
            preview_playing=True,
        )
    sla_process = dict(sla_modes.get("processes_dsp", {}) or {})
    sla_total = float(sla_process.get("total_latency_p95_ms", 0.0) or 0.0)
    sla_eval = {
        "sample_rate_hz": int(ENGINE_PERF_SLA_SAMPLE_RATE_HZ),
        "pass_threshold_ms": float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS),
        "ideal_threshold_ms": float(ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS),
        "total_latency_p95_ms": float(sla_total),
        "passes": bool(sla_total > 0.0 and sla_total < float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS)),
        "ideal": bool(sla_total > 0.0 and sla_total < float(ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS)),
    }
    if strict_mode:
        if sla_total <= 0.0:
            raise RuntimeError("Invalid SLA total latency metric in strict engine perf mode")
        if sla_total >= float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS):
            raise RuntimeError(
                "SLA latency contract failed in strict engine perf mode "
                f"(total_latency_p95_ms={sla_total:.3f}, limit={float(ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS):.3f})"
            )

    report = {
        "contract": engine_perf_contract(),
        "machine_profile": dict(ENGINE_PERF_SLA_MACHINE_PROFILE),
        "frames": int(frames),
        "warmup_frames": int(warmup_frames),
        "scenarios": scenarios,
        "sla_topology": {
            "modes": sla_modes,
            "evaluation": sla_eval,
        },
    }
    target = output_path or Path(__file__).with_name("engine_perf_report.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate_engine_perf_report(), indent=2))
