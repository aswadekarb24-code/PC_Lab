#!/usr/bin/env python3
"""Compares the queue-based parallel build (v2) against the original serial/parallel runs (v1).

Usage: python3 make_queue_results.py [v1_results.csv] [v2_results.csv]
Writes figures/queue_*.tex (table bodies) and figures/queue_stats.json (numbers quoted in the text).
"""
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
files = sorted(glob.glob(str(HERE.parent / "image_pipeline/results/*.csv")))
v1_path = sys.argv[1] if len(sys.argv) > 1 else files[-2]
v2_path = sys.argv[2] if len(sys.argv) > 2 else files[-1]


def load(path, mode):
    d = {}
    for r in csv.DictReader(open(path)):
        assert r["status"] == "ok"
        if r["mode"] == mode:
            d[(int(r["num_images"]), int(r["width"]), int(r["height"]), int(r["intermediate_frames"]))] = r
    return d


S, P1, P2 = load(v1_path, "serial"), load(v1_path, "parallel"), load(v2_path, "parallel")
keys = sorted(P2)
assert set(keys) == set(P1) == set(S), "run grids differ"
f = lambda d, k, c: float(d[k][c])
ratio = lambda a, b, k, c="pipeline_ms": f(a, k, c) / f(b, k, c)
agg = lambda xs: {"min": min(xs), "max": max(xs), "mean": st.mean(xs)}
fmt = lambda v: f"{v:,.0f}".replace(",", "{,}")

stats = {"runs": len(keys), "v1": Path(v1_path).name, "v2": Path(v2_path).name,
         "v2_vs_serial": agg([ratio(S, P2, k) for k in keys]),
         "v1_vs_serial": agg([ratio(S, P1, k) for k in keys]),
         "v2_vs_v1": agg([ratio(P1, P2, k) for k in keys]),
         "v2_vs_v1_encode": agg([ratio(P1, P2, k, "encoding_ms") for k in keys]),
         "total_s": {"serial": sum(f(S, k, "pipeline_ms") for k in keys) / 3.6e6,
                     "v1": sum(f(P1, k, "pipeline_ms") for k in keys) / 3.6e6,
                     "v2": sum(f(P2, k, "pipeline_ms") for k in keys) / 3.6e6},
         "v2_faster_runs": sum(f(P2, k, "pipeline_ms") < f(P1, k, "pipeline_ms") for k in keys)}
stats["v2_enc_share"] = agg([f(P2, k, "encoding_ms") / f(P2, k, "pipeline_ms") for k in keys])
big = (1000, 1920, 1080, 30)
stats["longest"] = {"frames": (big[0] - 1) * (big[3] + 1) + 1, "serial_s": f(S, big, "pipeline_ms") / 1e3,
                    "v1_s": f(P1, big, "pipeline_ms") / 1e3, "v2_s": f(P2, big, "pipeline_ms") / 1e3}
stats["by_res"] = {}
rows = []
for w, h in [(640, 360), (1280, 720), (1920, 1080), (1024, 768)]:
    ks = [k for k in keys if k[1:3] == (w, h)]
    e = {
        "v1_enc": st.mean(f(P1, k, "encoding_ms") / ((k[0] - 1) * (k[3] + 1) + 1) for k in ks),
        "v2_enc": st.mean(f(P2, k, "encoding_ms") / ((k[0] - 1) * (k[3] + 1) + 1) for k in ks),
        "v2_vs_v1": st.mean(ratio(P1, P2, k) for k in ks),
        "v2_vs_serial": st.mean(ratio(S, P2, k) for k in ks),
        "v1_vs_serial": st.mean(ratio(S, P1, k) for k in ks)}
    stats["by_res"][f"{w}x{h}"] = e
    rows.append(f"${w}{{\\times}}{h}$ & {e['v1_enc']:.2f} & {e['v2_enc']:.2f} & {e['v1_vs_serial']:.2f} "
                f"& {e['v2_vs_v1']:.2f} & {e['v2_vs_serial']:.2f}")
(OUT / "queue_by_res.tex").write_text(" \\\\\n".join(rows) + "\n")

rows = []
for n in sorted({k[0] for k in keys}):
    k = (n, 1920, 1080, 20)
    rows.append(f"{n} & {fmt(f(S, k, 'pipeline_ms'))} & {fmt(f(P1, k, 'pipeline_ms'))} & {fmt(f(P2, k, 'pipeline_ms'))} "
                f"& {ratio(S, P1, k):.2f} & {ratio(S, P2, k):.2f}")
(OUT / "queue_main.tex").write_text(" \\\\\n".join(rows) + "\n")

(OUT / "queue_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
print((OUT / "queue_main.tex").read_text()); print((OUT / "queue_by_res.tex").read_text())
