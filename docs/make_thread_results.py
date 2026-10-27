#!/usr/bin/env python3
"""Analysis of figures/thread_scaling.csv (OpenMP thread scaling of the analysis stages).

Writes figures/thread_scaling.tex (table body), figures/thread_stats.json and figures/thread_scaling.pdf.
Speedups use the true single-threaded serial build (--cv-threads 1) as baseline, medians over repetitions,
means over the four (N, resolution) configurations.
"""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
P_CORES, P_PERF = 10, 6                      # physical cores / performance cores of the CPU
med = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(OUT / "thread_scaling.csv")):
    med[(int(r["num_images"]), int(r["width"]))][r["config"]].append((float(r["feature_ms"]), float(r["sequencing_ms"])))
cfg = sorted(med)
T = lambda c, label, stage: st.median(x[stage] for x in med[c][label])
threads = sorted({int(l[3:]) for c in cfg for l in med[c] if l.startswith("omp")})
karp = lambda psi, p: (1 / psi - 1 / p) / (1 - 1 / p) if p > 1 else 0.0

stats = {"configs": [list(c) for c in cfg], "threads": threads, "stage": {}}
rows = []
for t in threads:
    row = {}
    for stage, name in ((0, "feature"), (1, "seq")):
        psi = st.mean(T(c, "serial_cv1", stage) / T(c, f"omp{t}", stage) for c in cfg)
        row[name] = {"psi": psi, "E": psi / t, "e": karp(psi, t), "E_cores": psi / min(t, P_CORES)}
    stats["stage"][t] = row
    f, s = row["feature"], row["seq"]
    rows.append(f"{t} & {f['psi']:.2f} & {f['E']:.2f} & {f['e']:.3f} & {s['psi']:.2f} & {s['E']:.2f} & {s['e']:.3f}")
(OUT / "thread_scaling.tex").write_text(" \\\\\n".join(rows) + "\n")

# baseline effect of OpenCV's own pool
stats["cvdefault_vs_cv1"] = {"feature_ratio": {f"{c[0]}x{c[1]}": T(c, "serial_cv1", 0) / T(c, "serial_cvdefault", 0) for c in cfg},
                             "omp16_vs_cvdefault": st.mean(T(c, "serial_cvdefault", 0) / T(c, "omp16", 0) for c in cfg),
                             "omp16_vs_cv1": st.mean(T(c, "serial_cv1", 0) / T(c, "omp16", 0) for c in cfg),
                             "seq16_vs_cv1": st.mean(T(c, "serial_cv1", 1) / T(c, "omp16", 1) for c in cfg)}
stats["cvdefault_vs_cv1"]["feature_ratio_mean"] = st.mean(stats["cvdefault_vs_cv1"]["feature_ratio"].values())
# ---- independent encoder copies (encode_bench): weak scaling, fixed frames per copy
encd = defaultdict(list)
for r in csv.DictReader(open(OUT / "encoder_scaling.csv")):
    encd[int(r["copies"])].append(float(r["fps"]))
efps = {t: st.median(v) for t, v in encd.items()}
enc = {t: {"fps": efps[t], "speedup": efps[t] / efps[1], "E": efps[t] / efps[1] / t} for t in sorted(efps)}
stats["encoder"] = enc
(OUT / "encoder_scaling.tex").write_text(" \\\\\n".join(
    f"{t} & {enc[t]['fps']:.0f} & {enc[t]['speedup']:.2f} & {enc[t]['E']:.2f}" for t in sorted(enc)) + "\n")
(OUT / "thread_stats.json").write_text(json.dumps(stats, indent=1))

# ---- figure
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
for a in ax:
    for x, lab in ((P_PERF, "6 P-cores"), (P_CORES, "10 cores")):
        a.axvline(x, color=GRID, lw=1.0, zorder=0)
for key, c, mk, lab in (("feature", BLUE, "o", "Feature extraction"), ("seq", ORANGE, "^", "Similarity + ordering")):
    ax[0].plot(threads, [stats["stage"][t][key]["psi"] for t in threads], mk + "-", color=c, label=lab)
    ax[1].plot(threads, [stats["stage"][t][key]["E"] for t in threads], mk + "-", color=c, label=lab)
et = sorted(enc)
ax[0].plot(et, [enc[t]["speedup"] for t in et], "D-", color=GREEN, label="Independent encodes")
ax[1].plot(et, [enc[t]["E"] for t in et], "D-", color=GREEN, label="Independent encodes")
ax[0].plot(threads, threads, "--", color=GREY, label="ideal")
ax[0].set_xlabel("OpenMP threads"); ax[0].set_ylabel("Speedup over 1 thread")
ax[0].set_title("(a) Stage speedup", loc="left"); ax[0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0)
ax[1].set_ylim(0, 1.1); ax[1].set_xlabel("OpenMP threads"); ax[1].set_ylabel("Efficiency $\\psi/t$")
ax[1].set_title("(b) Efficiency", loc="left")
for a in ax:
    a.set_xticks(threads); a.set_xticklabels([str(t) for t in threads], fontsize=7)
    a.annotate("6", (P_PERF, a.get_ylim()[0]), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=6, color=MUTED, annotation_clip=False)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "thread_scaling.pdf")
print(json.dumps(stats, indent=1)); print((OUT / "thread_scaling.tex").read_text())
