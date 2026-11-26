#!/usr/bin/env python3
"""Plot every curve that drifted from its golden, as one HTML page.

Reads what blender_reference.py left in tests/artifacts/reference (summary.json
and actual_*.npz) next to the goldens, and writes report.html there. Plots are
inline SVG, so this needs only NumPy.

Usage: python tests/reference/report.py [ARTIFACTS_DIR]
"""

import html
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "curves")
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "artifacts", "reference")
W, H, PAD = 640, 180, 28


def _polyline(values, lo, hi, n):
    xs = PAD + (W - 2 * PAD) * np.arange(len(values)) / max(n - 1, 1)
    span = (hi - lo) or 1.0
    ys = H - PAD - (H - 2 * PAD) * (np.asarray(values, dtype=float) - lo) / span
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys) if np.isfinite(y))


def _plot(golden, actual, bad_frames):
    golden = np.asarray(golden, dtype=float).reshape(len(golden), -1)
    actual = np.asarray(actual, dtype=float).reshape(len(actual), -1)
    both = np.concatenate([golden.ravel(), actual.ravel()])
    both = both[np.isfinite(both)]
    lo, hi = (float(both.min()), float(both.max())) if both.size else (0.0, 1.0)
    n = max(len(golden), len(actual))
    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">',
             f'<rect x="{PAD}" y="{PAD}" width="{W - 2 * PAD}" height="{H - 2 * PAD}" class="frame"/>']
    for f in bad_frames:
        x = PAD + (W - 2 * PAD) * f / max(n - 1, 1)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD}" y2="{H - PAD}" class="bad"/>')
    for col in range(golden.shape[1]):
        parts.append(f'<polyline points="{_polyline(golden[:, col], lo, hi, n)}" class="golden"/>')
    for col in range(actual.shape[1]):
        parts.append(f'<polyline points="{_polyline(actual[:, col], lo, hi, n)}" class="actual"/>')
    parts.append(f'<text x="{PAD}" y="{PAD - 8}">{hi:.4g}</text><text x="{PAD}" y="{H - 8}">{lo:.4g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build(artifacts: str) -> str:
    with open(os.path.join(artifacts, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    sections = []
    for kind in ("drift", "threads_vs_off"):
        for tag, entries in sorted(summary.get(kind, {}).items()):
            actual_path = os.path.join(artifacts, f"actual_{tag}")
            golden_path = os.path.join(GOLDEN_DIR, tag)
            actual = dict(np.load(actual_path)) if os.path.isfile(actual_path) else {}
            golden = dict(np.load(golden_path)) if os.path.isfile(golden_path) else {}
            for key, info in sorted(entries.items()):
                title = f"{html.escape(tag)} · {html.escape(key)}"
                detail = html.escape(json.dumps(info))
                if key in actual and key in golden and actual[key].shape == golden[key].shape:
                    a, g = actual[key], golden[key]
                    tol = info.get("tolerance") or {"abs": 0.0, "rel": 0.0}
                    diff = np.abs(a - g).reshape(len(a), -1).max(axis=1)
                    limit = tol["abs"] + tol["rel"] * np.abs(g).reshape(len(g), -1).max(axis=1)
                    figure = _plot(g, a, np.nonzero(diff > limit)[0])
                else:
                    figure = "<p>No curve to plot.</p>"
                sections.append(f"<section><h2>{kind}: {title}</h2><p class='detail'>{detail}</p>{figure}</section>")
    nondet = summary.get("nondeterministic", {})
    if nondet:
        items = "".join(f"<li>{html.escape(t)}: {html.escape(', '.join(k))}</li>" for t, k in sorted(nondet.items()))
        sections.insert(0, f"<section><h2>Not repeatable</h2><ul>{items}</ul></section>")
    body = "".join(sections) or "<p>Every curve matched its golden.</p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>ACE reference curves</title><style>"
        "body{font:14px system-ui,sans-serif;margin:24px;max-width:720px;color:#222}"
        "h2{font-size:14px;margin:20px 0 4px}.detail{color:#666;font:12px ui-monospace,monospace;word-break:break-all}"
        "svg text{font:11px system-ui;fill:#666}.frame{fill:none;stroke:#ddd}"
        ".golden{fill:none;stroke:#888;stroke-width:1.5}.actual{fill:none;stroke:#c33;stroke-width:1.2}"
        ".bad{stroke:#fbb;stroke-width:2}"
        "@media(prefers-color-scheme:dark){body{background:#161616;color:#ddd}.frame{stroke:#333}.golden{stroke:#999}.bad{stroke:#622}}"
        "</style></head><body><h1>ACE reference curves</h1>"
        "<p>Grey: golden. Red: this run. Shaded frames are outside tolerance.</p>"
        f"{body}</body></html>"
    )


def main() -> int:
    artifacts = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    page = build(artifacts)
    out = os.path.join(artifacts, "report.html")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
