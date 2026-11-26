"""What the reference-curve tests record, and how far a curve may drift.

The cases come from ``ACE_NODES`` in vars.py: every node type, and every mode
of a mode node, is either a case here or listed in EXCLUDED with the reason,
so a new node fails the coverage test until it gets a golden.

A case is a chain of nodes hanging off one Source node. Each step names the
node and which output of the previous step feeds it; the values recorded are
every numeric output of the chain's last node. The empty chain of SOURCE_CASE
records the Source node's own outputs: the tempo detected for the file.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FPS = (24, 30, 60)
FRAME_START = 1

# Node types with no reference curve, and why.
EXCLUDED: Dict[Tuple[str, str], str] = {
    ("source_mngmt", "src"): "the root of every case; its tempo is SOURCE_CASE",
    ("utilities", "preview_sink"): "an audio monitor: it outputs no values",
}

# The Source node itself: BPM and Beats per Bar.
SOURCE_CASE = "source_mngmt.src"

# How many outputs a delta node has once placed (the frequency splitter's
# runtime variant is built with this many bands).
SPLITTER_BANDS = 2

# Tolerance per case id prefix: |actual - golden| <= abs + rel * |golden|.
# The longest matching prefix wins.
DEFAULT_TOLERANCE = {"abs": 1e-5, "rel": 1e-4}
TOLERANCES: Dict[str, Dict[str, float]] = {
    "stream.pitch": {"abs": 1e-3, "rel": 1e-4},             # Hz
    "stream.brightness": {"abs": 1e-2, "rel": 1e-4},         # Hz
    "delta.frequency_splitter.out00.brightness": {"abs": 1e-2, "rel": 1e-4},
    "delta.frequency_splitter.out01.brightness": {"abs": 1e-2, "rel": 1e-4},
    SOURCE_CASE: {"abs": 1e-3, "rel": 1e-4},                 # BPM
    "utilities.stream_accumulator": {"abs": 1e-4, "rel": 1e-4},  # a running sum
}


def tolerance(case_id: str) -> Dict[str, float]:
    best = ""
    for prefix in TOLERANCES:
        if (case_id == prefix or case_id.startswith(prefix + ".")) and len(prefix) > len(best):
            best = prefix
    return dict(TOLERANCES.get(best, DEFAULT_TOLERANCE))


def _load_ace_nodes():
    """ACE_NODES from vars.py, read without importing the add-on (vars.py has no bpy, but its package does)."""
    path = Path(__file__).resolve().parents[2] / "vars.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "ACE_NODES":
            return ast.literal_eval(node.value)
    raise RuntimeError("ACE_NODES not found in vars.py")


def _step(category: str, node_type: str, mode: Optional[str] = None, from_output: int = 0) -> dict:
    """One node of a chain. *from_output* indexes the previous node's outputs of the kind this node takes."""
    return {"category": category, "type": node_type, "mode": mode, "from_output": int(from_output)}


def cases() -> List[dict]:
    """Every reference case, in a stable order."""
    nodes = _load_ace_nodes()
    out: List[dict] = []

    stream = nodes["stream"]["nodes"]
    for node_type, node_def in stream.items():
        modes = [m[0] for m in node_def.get("modes") or []] or [None]
        for mode in modes:
            case_id = f"stream.{node_type}.{mode}" if mode else f"stream.{node_type}"
            out.append({"id": case_id, "chain": [_step("stream", node_type, mode)]})

    for node_type, node_def in nodes["delta"]["nodes"].items():
        count = SPLITTER_BANDS if node_type == "frequency_splitter" else len(node_def["outputs"])
        for idx in range(count):
            out.append({
                "id": f"delta.{node_type}.out{idx:02d}.rms",
                "chain": [_step("delta", node_type), _step("stream", "loudness", "rms", from_output=idx)],
            })

    # Measurements that must stay inside a band: one case per band and measurement.
    banded = [("stereo", mode) for mode in ("width", "balance")] + [("stereo_vector", None), ("brightness", None)]
    for idx in range(SPLITTER_BANDS):
        for node_type, mode in banded:
            if node_type not in stream:
                continue
            suffix = f"{node_type}.{mode}" if mode else node_type
            out.append({
                "id": f"delta.frequency_splitter.out{idx:02d}.{suffix}",
                "chain": [_step("delta", "frequency_splitter"), _step("stream", node_type, mode, from_output=idx)],
            })

    utilities = nodes["utilities"]["nodes"]
    if "util_rolling_avg" in utilities:
        out.append({
            "id": "utilities.util_rolling_avg",
            "chain": [_step("stream", "loudness", "rms"), _step("utilities", "util_rolling_avg")],
        })
    if "stream_accumulator" in utilities:
        out.append({
            "id": "utilities.stream_accumulator",
            "chain": [_step("stream", "onset"), _step("utilities", "stream_accumulator")],
        })
    out.append({"id": SOURCE_CASE, "chain": []})
    return out


def uncovered_node_types() -> List[Tuple[str, str]]:
    """Node types (and modes) in ACE_NODES that are neither a case nor excluded."""
    nodes = _load_ace_nodes()
    have = {(s["category"], s["type"], s["mode"]) for c in cases() for s in c["chain"]}
    missing = []
    for category, cat in nodes.items():
        for node_type, node_def in cat["nodes"].items():
            if (category, node_type) in EXCLUDED:
                continue
            modes = [m[0] for m in node_def.get("modes") or []] or [None]
            for mode in modes:
                if not any(h[0] == category and h[1] == node_type and (mode is None or h[2] == mode) for h in have):
                    missing.append((category, node_type if mode is None else f"{node_type}.{mode}"))
    return missing


def golden_name(fixture: str, fps: int) -> str:
    return f"{fixture}__{int(fps)}fps.npz"
