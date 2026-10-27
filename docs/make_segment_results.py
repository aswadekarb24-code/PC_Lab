#!/usr/bin/env python3
"""Master comparison of serial, P1 (sync), P2 (queue), P3 (queue + parallel segments).

Usage: python3 make_segment_results.py base.csv p2.csv p3.csv p3_6threads.csv
(base.csv holds the serial and P1 rows; typically figures/results_merged.csv)
Reads image_pipeline/results/*.csv (+ figures/segment_pilot.csv) and writes
figures/segment_*.tex, figures/master_*.tex, figures/master_stats.json and figures/master_comparison.pdf.
"""
import csv, glob, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
files = sys.argv[1:5]
assert len(files) == 4, "pass: base.csv p2.csv p3.csv p3_6threads.csv"
P_THREADS, P_CORES = 16, 10          # hardware threads / physical cores of the CPU (GPU not counted)


def load(path, mode):
    d = {}
    for r in csv.DictReader(open(path)):
        assert r["status"] == "ok"
        if r["mode"] == mode:
            d[(int(r["num_images"]), int(r["width"]), int(r["height"]), int(r["intermediate_frames"]))] = r
    return d


V = {"S": load(files[0], "serial"), "P1": load(files[0], "parallel"),
     "P2": load(files[1], "parallel"), "P3": load(files[2], "parallel"),
     "P3s": load(files[3], "parallel")}
keys = sorted(V["P3"])
assert all(sorted(v) == keys for v in V.values())
f = lambda v, k, c="pipeline_ms": float(V[v][k][c])
frames = lambda k: (k[0] - 1) * (k[3] + 1) + 1
fmt = lambda x: f"{x:,.0f}".replace(",", "{,}")
agg = lambda xs: {"min": min(xs), "max": max(xs), "mean": st.mean(xs)}
tot = {v: sum(f(v, k) for k in keys) for v in V}                  # ms over the 60 configurations
ratio = lambda a, b, ks=keys: [f(a, k) / f(b, k) for k in ks]
karp = lambda psi, p: (1 / psi - 1 / p) / (1 - 1 / p)

stats = {"files": [Path(x).name for x in files], "configs": len(keys)}
# ---- master metrics
master = {}
PV = (("S", 1), ("P1", P_THREADS), ("P2", P_THREADS), ("P3", P_THREADS), ("P3s", 6))
for v, p in PV:
    psi = tot["S"] / tot[v]
    big = (1000, 1920, 1080, 30)
    m = {"p": p, "hours": tot[v] / 3.6e6, "psi": psi, "per_config": agg(ratio("S", v)),
         "longest_s": f(v, big) / 1e3,
         "enc_ms_frame_1080": st.mean(f(v, k, "encoding_ms") / frames(k) for k in keys if k[1] == 1920),
         "enc_share": agg([f(v, k, "encoding_ms") / f(v, k) for k in keys])}
    if v != "S":
        m.update({"E16": psi / p, "E10": psi / min(p, P_CORES), "cost_ratio": p * tot[v] / tot["S"],
                  "cost_thread_h": p * tot[v] / 3.6e6, "karp_flatt": karp(psi, p),
                  "amdahl_bound": 1 / karp(psi, p), "karp_flatt_p10": karp(psi, min(p, P_CORES))})
    else:
        m.update({"E16": 1.0, "cost_ratio": 1.0, "cost_thread_h": tot["S"] / 3.6e6})
    master[v] = m
stats["master"] = master
stats["p3_vs_p2"] = agg(ratio("P2", "P3")); stats["p3_vs_p1"] = agg(ratio("P1", "P3"))
stats["p3_stage"] = {"feature": agg([f("S", k, "feature_extraction_ms") / f("P3", k, "feature_extraction_ms") for k in keys]),
                     "seq": agg([f("S", k, "sequencing_ms") / f("P3", k, "sequencing_ms") for k in keys]),
                     "feature_share": agg([f("P3", k, "feature_extraction_ms") / f("P3", k) for k in keys]),
                     "seq_share": agg([f("P3", k, "sequencing_ms") / f("P3", k) for k in keys])}
stats["p3_by_N"] = {}
for n in sorted({k[0] for k in keys}):
    ks = [k for k in keys if k[0] == n]
    psi = sum(f("S", k) for k in ks) / sum(f("P3", k) for k in ks)
    stats["p3_by_N"][n] = {"psi": psi, "E16": psi / P_THREADS, "karp_flatt": karp(psi, P_THREADS)}
stats["p3_by_F"] = {}
for fr in (10, 20, 30):
    ks = [k for k in keys if k[3] == fr]
    stats["p3_by_F"][fr] = sum(f("S", k) for k in ks) / sum(f("P3", k) for k in ks)
slow = [k for k in keys if f("S", k) / f("P3", k) < 1]
stats["p3_slower_than_serial"] = [list(k) for k in slow]
stats["p3_vs_serial_excl_slow"] = agg([f("S", k) / f("P3", k) for k in keys if k not in slow])

# ---- tables
rows = []
for n in sorted({k[0] for k in keys}):
    k = (n, 1920, 1080, 20)
    rows.append(f"{n} & {fmt(f('P2', k))} & {fmt(f('P3', k))} & {f('P2', k) / f('P3', k):.2f} & {f('S', k) / f('P3', k):.2f}")
(OUT / "segment_main.tex").write_text(" \\\\\n".join(rows) + "\n")

rows = []
for w, h in [(640, 360), (1280, 720), (1920, 1080), (1024, 768)]:
    ks = [k for k in keys if k[1:3] == (w, h)]
    e2 = st.mean(f("P2", k, "encoding_ms") / frames(k) for k in ks)
    e3 = st.mean(f("P3", k, "encoding_ms") / frames(k) for k in ks)
    rows.append(f"${w}{{\\times}}{h}$ & {e2:.2f} & {e3:.2f} & {st.mean(ratio('P2', 'P3', ks)):.2f} & {st.mean(ratio('S', 'P3', ks)):.2f}")
(OUT / "segment_by_res.tex").write_text(" \\\\\n".join(rows) + "\n")

pil = defaultdict(list)
for r in csv.DictReader(open(OUT / "segment_pilot.csv")):
    pil[int(r["segments"])].append(float(r["encoding_ms"]))
med = {k: st.median(v) for k, v in pil.items()}
rows = [f"{k} & {med[k] / 1e3:.2f} & {med[1] / med[k]:.2f} & {med[1] / (k * med[k]):.2f}" for k in sorted(med)]
(OUT / "segment_pilot.tex").write_text(" \\\\\n".join(rows) + "\n")
stats["pilot"] = {k: {"median_ms": med[k], "speedup": med[1] / med[k], "eff": med[1] / (k * med[k])} for k in sorted(med)}

names = {"S": "Serial", "P1": "P$_1$ sync", "P2": "P$_2$ queue", "P3": "P$_3$ segments", "P3s": "P$_3$, 6 threads"}
rows = []
for v in V:
    m = master[v]
    rows.append(f"{names[v]} & {m['hours']:.2f} & {m['psi']:.2f} & {m['per_config']['min']:.2f}--{m['per_config']['max']:.2f} "
                f"& {m['longest_s']:.0f} & {m['enc_ms_frame_1080']:.2f}")
(OUT / "master_perf.tex").write_text(" \\\\\n".join(rows) + "\n")
rows = []
for v in ("P1", "P2", "P3", "P3s"):
    m = master[v]
    rows.append(f"{names[v]} & {m['p']} & {m['psi']:.2f} & {m['E16']:.3f} & {m['E10']:.3f} & {m['cost_ratio']:.1f} "
                f"& {m['karp_flatt']:.3f} & {m['amdahl_bound']:.1f}")
(OUT / "master_metrics.tex").write_text(" \\\\\n".join(rows) + "\n")
# ---- final comparison: serial vs P3 (16 threads, 12 segments) vs P3 (6 threads, 6 segments)
rows = []
for n in sorted({k[0] for k in keys}):
    k = (n, 1920, 1080, 20)
    rows.append(f"{n} & {fmt(f('S', k))} & {fmt(f('P3', k))} & {fmt(f('P3s', k))} & {f('S', k) / f('P3', k):.2f} & {f('S', k) / f('P3s', k):.2f}")
(OUT / "final_main.tex").write_text(" \\\\\n".join(rows) + "\n")
stats["p3s"] = {"by_N": {}, "by_F": {}, "by_res": {}}
for n in sorted({k[0] for k in keys}):
    ks = [k for k in keys if k[0] == n]
    psi = sum(f("S", k) for k in ks) / sum(f("P3s", k) for k in ks)
    stats["p3s"]["by_N"][n] = {"psi": psi, "E": psi / 6, "e": karp(psi, 6)}
for fr in (10, 20, 30):
    ks = [k for k in keys if k[3] == fr]
    stats["p3s"]["by_F"][fr] = sum(f("S", k) for k in ks) / sum(f("P3s", k) for k in ks)
for w, h in [(640, 360), (1280, 720), (1920, 1080), (1024, 768)]:
    ks = [k for k in keys if k[1:3] == (w, h)]
    stats["p3s"]["by_res"][f"{w}x{h}"] = sum(f("S", k) for k in ks) / sum(f("P3s", k) for k in ks)
worst = min(keys, key=lambda k: f("S", k) / f("P3s", k))
stats["p3s"]["worst"] = {"key": list(worst), "psi": f("S", worst) / f("P3s", worst)}
stats["p3s"]["vs_p3"] = agg(ratio("P3", "P3s")); stats["p3s"]["vs_p3_total"] = tot["P3"] / tot["P3s"]
stats["p3s"]["stage"] = {"feature": agg([f("S", k, "feature_extraction_ms") / f("P3s", k, "feature_extraction_ms") for k in keys]),
                         "seq": agg([f("S", k, "sequencing_ms") / f("P3s", k, "sequencing_ms") for k in keys])}
for v in ("P1", "P2", "P3", "P3s"):
    stats[f"enc_stage_total_speedup_{v}"] = sum(f("S", k, "encoding_ms") for k in keys) / sum(f(v, k, "encoding_ms") for k in keys)
stats["enc_stage_1080_mean"] = {v: st.mean(f("S", k, "encoding_ms") / f(v, k, "encoding_ms") for k in keys if k[1] == 1920) for v in ("P3", "P3s")}
(OUT / "master_stats.json").write_text(json.dumps(stats, indent=2, default=str))

# ---- figure
BLUE, ORANGE, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6b6b6b"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42})
style = {"S": (GREY, "s", "Serial"), "P1": (BLUE, "o", "P$_1$ (sync)"), "P2": (ORANGE, "^", "P$_2$ (queue)"),
         "P3": (GREEN, "D", "P$_3$ (segments)"), "P3s": (PURPLE, "v", "P$_3$, 6 threads")}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
# (a) pipeline time vs frames, 1920x1080
ks = [k for k in keys if k[1:3] == (1920, 1080)]
for v, (c, mk, lab) in style.items():
    ax[0].plot([frames(k) for k in ks], [f(v, k) / 1e3 for k in ks], mk, color=c, label=lab, ms=3.5,
               mfc=c if v == "S" else "none", mew=1.1, ls="none")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Total video frames"); ax[0].set_ylabel("Pipeline time (s)")
ax[0].set_title("(a) Time, $1920{\\times}1080$", loc="left"); ax[0].legend(loc="upper left", handletextpad=0.2, borderaxespad=0)
# (b) speedup vs N
Ns = sorted({k[0] for k in keys})
for v in ("P1", "P2", "P3", "P3s"):
    c, mk, lab = style[v]
    ys = [sum(f("S", k) for k in keys if k[0] == n) / sum(f(v, k) for k in keys if k[0] == n) for n in Ns]
    ax[1].plot(Ns, ys, mk + "-", color=c, label=lab)
    ax[1].annotate(lab, (Ns[-1], ys[-1]), xytext=(-2, -11 if v == "P1" else (-11 if v == "P3s" else 5)), textcoords="offset points",
                   ha="right", color=INK, fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xticks(Ns); ax[1].set_xticklabels([str(n) for n in Ns], fontsize=7)
ax[1].minorticks_off()
ax[1].set_ylim(0, 8); ax[1].set_xlabel("Images in dataset ($N$)"); ax[1].set_ylabel("Speedup over serial")
ax[1].set_title("(b) End-to-end speedup", loc="left")
# (c) K scaling pilot
Ks = sorted(med)
ax[2].plot(Ks, [med[k] / 1e3 for k in Ks], "D-", color=GREEN, label="measured (median of 3)")
ax[2].plot(Ks, [med[1] / k / 1e3 for k in Ks], "--", color=GREY, label="ideal $T_1/K$")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xticks(Ks); ax[2].set_xticklabels([str(k) for k in Ks], fontsize=7)
ax[2].minorticks_off()
ax[2].set_xlabel("Segments $K$"); ax[2].set_ylabel("Blend + encode (s)")
ax[2].set_title("(c) Segment scaling", loc="left"); ax[2].legend(loc="lower left", handletextpad=0.3, borderaxespad=0)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "master_comparison.pdf")
print(json.dumps({k: stats[k] for k in ("master", "p3_vs_p2", "p3_vs_p1", "p3_stage", "p3_by_N", "p3_by_F", "p3_slower_than_serial", "p3_vs_serial_excl_slow", "pilot")}, indent=1, default=str))
for t in ("segment_main", "segment_by_res", "segment_pilot", "master_perf", "master_metrics"):
    print(t); print((OUT / f"{t}.tex").read_text())
