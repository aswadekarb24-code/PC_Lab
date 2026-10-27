#!/usr/bin/env python3
"""Builds the figures/tables data for the paper from image_pipeline/results/*.csv.

Usage: python3 make_figures.py [results.csv]
Writes figures/*.pdf and figures/stats.json (numbers quoted in the text).
"""
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(str(HERE.parent / "image_pipeline/results/*.csv")))[-1]

# Okabe-Ito (colorblind safe); each series also gets its own marker
BLUE, ORANGE, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6b6b6b"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})

rows = list(csv.DictReader(open(src)))
assert all(r["status"] == "ok" for r in rows), "failed runs present"
data = defaultdict(dict)  # (n, w, h, f) -> {mode: row}
for r in rows:
    data[(int(r["num_images"]), int(r["width"]), int(r["height"]), int(r["intermediate_frames"]))][r["mode"]] = r
num = lambda r, c: float(r[c])
keys = sorted(data)
Ns = sorted({k[0] for k in keys})
RES = [(640, 360), (1024, 768), (1280, 720), (1920, 1080)]
RES_ORDER = [(640, 360), (1280, 720), (1920, 1080), (1024, 768)]
FR = [10, 20, 30]


def get(k, mode, col):
    return num(data[k][mode], col)


def speed(k, col):
    return get(k, "serial", col) / get(k, "parallel", col)


stats = {"runs": len(rows), "datasets": Ns}
# ---- aggregate numbers
stats["feature_speedup"] = {"min": min(speed(k, "feature_extraction_ms") for k in keys),
                            "max": max(speed(k, "feature_extraction_ms") for k in keys),
                            "mean": st.mean(speed(k, "feature_extraction_ms") for k in keys)}
stats["seq_speedup"] = {"min": min(speed(k, "sequencing_ms") for k in keys),
                        "max": max(speed(k, "sequencing_ms") for k in keys),
                        "mean": st.mean(speed(k, "sequencing_ms") for k in keys)}
stats["total_speedup"] = {"min": min(speed(k, "pipeline_ms") for k in keys),
                          "max": max(speed(k, "pipeline_ms") for k in keys),
                          "mean": st.mean(speed(k, "pipeline_ms") for k in keys)}
stats["encode_speedup"] = {"min": min(speed(k, "encoding_ms") for k in keys),
                           "max": max(speed(k, "encoding_ms") for k in keys),
                           "mean": st.mean(speed(k, "encoding_ms") for k in keys)}
enc_share_s = [get(k, "serial", "encoding_ms") / get(k, "serial", "pipeline_ms") for k in keys]
enc_share_p = [get(k, "parallel", "encoding_ms") / get(k, "parallel", "pipeline_ms") for k in keys]
stats["encode_share_serial"] = {"min": min(enc_share_s), "max": max(enc_share_s), "mean": st.mean(enc_share_s)}
stats["encode_share_parallel"] = {"min": min(enc_share_p), "max": max(enc_share_p), "mean": st.mean(enc_share_p)}
# speedup obtained from stages 1-2 alone (blend+encode time left at its serial value)
def stage12_only(k):
    s_tot = get(k, "serial", "pipeline_ms")
    saved = (get(k, "serial", "feature_extraction_ms") + get(k, "serial", "sequencing_ms")
             - get(k, "parallel", "feature_extraction_ms") - get(k, "parallel", "sequencing_ms"))
    return s_tot / (s_tot - saved)


bound = [stage12_only(k) for k in keys]
stats["stage12_only_speedup"] = {"min": min(bound), "max": max(bound), "mean": st.mean(bound)}
stats["best_total_speedup_key"] = max(keys, key=lambda k: speed(k, "pipeline_ms"))
stats["max_serial_total_ms"] = max(get(k, "serial", "pipeline_ms") for k in keys)
stats["sum_serial_h"] = sum(get(k, "serial", "pipeline_ms") for k in keys) / 3.6e6
stats["sum_parallel_h"] = sum(get(k, "parallel", "pipeline_ms") for k in keys) / 3.6e6
stats["mean_speedup_by_res"] = {f"{w}x{h}": st.mean(speed(k, "pipeline_ms") for k in keys if (k[1], k[2]) == (w, h))
                                for w, h in RES}
stats["feature_speedup_by_res"] = {f"{w}x{h}": st.mean(speed(k, "feature_extraction_ms") for k in keys if (k[1], k[2]) == (w, h))
                                   for w, h in RES}
stats["seq_speedup_by_N"] = {n: st.mean(speed(k, "sequencing_ms") for k in keys if k[0] == n) for n in Ns}
stats["total_speedup_by_N"] = {n: st.mean(speed(k, "pipeline_ms") for k in keys if k[0] == n) for n in Ns}
stats["feature_serial_ms_per_img_by_res"] = {
    f"{w}x{h}": st.mean(get(k, "serial", "feature_extraction_ms") / k[0] for k in keys if (k[1], k[2]) == (w, h)) for w, h in RES}
stats["feature_parallel_ms_per_img_by_res"] = {
    f"{w}x{h}": st.mean(get(k, "parallel", "feature_extraction_ms") / k[0] for k in keys if (k[1], k[2]) == (w, h)) for w, h in RES}
stats["serial_ms_per_frame"] = st.mean(get(k, "serial", "encoding_ms") / get(k, "serial", "total_frames") for k in keys if k[1:3] == (1920, 1080))
stats["parallel_ms_per_frame"] = st.mean(get(k, "parallel", "encoding_ms") / get(k, "parallel", "total_frames") for k in keys if k[1:3] == (1920, 1080))
# per-frame encode cost vs pixels (linearity check)
stats["encode_ms_per_frame_by_res"] = {
    f"{w}x{h}": st.mean(get(k, "parallel", "encoding_ms") / get(k, "parallel", "total_frames") for k in keys if (k[1], k[2]) == (w, h))
    for w, h in RES}
json.dump(stats, open(OUT / "stats.json", "w"), indent=2, default=str)


def kfmt(x, _):
    return f"{x/1000:g}k" if x >= 1000 else f"{x:g}"


def label_axes(ax, xl, yl):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)


# ---- Fig A: stage breakdown (stacked horizontal bars), largest dataset, each resolution, 20 frames
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"wspace": 0.28})
colors = {"feature_extraction_ms": BLUE, "sequencing_ms": GREEN, "encoding_ms": ORANGE}
names = {"feature_extraction_ms": "Feature extraction", "sequencing_ms": "Similarity + ordering",
         "encoding_ms": "Blend + encode"}
hatch = {"feature_extraction_ms": "", "sequencing_ms": "", "encoding_ms": "///"}
for ax, n in zip(axes, [100, 1000]):
    ylabels, ypos = [], []
    y = 0
    for (w, h) in RES_ORDER[:3]:
        k = (n, w, h, 20)
        for mode in ("serial", "parallel"):
            left = 0
            tot = get(k, mode, "pipeline_ms") / 1000
            for c in ("encoding_ms", "feature_extraction_ms", "sequencing_ms"):
                v = get(k, mode, c) / 1000
                ax.barh(y, v, left=left, color=colors[c], hatch=hatch[c], edgecolor="white", linewidth=0.6,
                        height=0.7, label=names[c] if (y == 0) else None)
                left += v
            ax.text(tot + max(ax.get_xlim()[1], 1) * 0.0, y, f" {tot:.1f}s", va="center", fontsize=6.5, color=MUTED)
            ylabels.append(f"{w}×{h} {mode[:3]}.")
            ypos.append(y)
            y += 1
        y += 0.5
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.14)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"({'a' if n == 100 else 'b'}) {n} images, 20 intermediate frames", loc="left")
    ax.grid(axis="y", visible=False)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
fig.savefig(OUT / "breakdown.pdf")
plt.close(fig)

# ---- Fig B: stage speedups vs N (feature, sequencing) per resolution
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw={"wspace": 0.28})
mk = {(640, 360): ("o", BLUE), (1280, 720): ("s", GREEN), (1920, 1080): ("^", ORANGE), (1024, 768): ("D", PURPLE)}
for ax, col, ttl in ((axes[0], "feature_extraction_ms", "(a) Feature extraction (OpenMP, 16 threads)"),
                     (axes[1], "sequencing_ms", "(b) Similarity matrix + ordering (OpenMP)")):
    for (w, h) in RES_ORDER:
        ys = [st.mean(speed(k, col) for k in keys if k[0] == n and (k[1], k[2]) == (w, h)) for n in Ns]
        m, c = mk[(w, h)]
        ax.plot(Ns, ys, marker=m, color=c, label=f"{w}×{h}")
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_title(ttl, loc="left")
    label_axes(ax, "Images in dataset ($N$)", "Speedup (serial / parallel)")
axes[0].legend(ncol=2, loc="lower right", title="Resolution", title_fontsize=7)
fig.savefig(OUT / "stage_speedup.pdf")
plt.close(fig)

# ---- Fig C: end-to-end time vs total frames (serial/parallel), per resolution (a), and total speedup (b)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={"wspace": 0.28})
ax = axes[0]
for (w, h) in RES_ORDER:
    m, c = mk[(w, h)]
    pts = sorted((get(k, "serial", "total_frames"), get(k, "serial", "pipeline_ms") / 1000,
                  get(k, "parallel", "pipeline_ms") / 1000) for k in keys if (k[1], k[2]) == (w, h))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, ms=3, lw=0, color=c, alpha=0.9, label=f"{w}×{h}")
    ax.plot([p[0] for p in pts], [p[2] for p in pts], marker=m, ms=3, lw=0, color=c, mfc="white", mew=0.9, alpha=0.9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax.set_title("(a) Total time vs. frames written", loc="left")
label_axes(ax, "Total video frames", "Pipeline time (s)")
ax.legend(loc="upper left", ncol=1, title="filled = serial, open = parallel", title_fontsize=6.5)
ax = axes[1]
for (w, h) in RES_ORDER:
    m, c = mk[(w, h)]
    ys = [st.mean(speed(k, "pipeline_ms") for k in keys if k[0] == n and (k[1], k[2]) == (w, h)) for n in Ns]
    ax.plot(Ns, ys, marker=m, color=c, label=f"{w}×{h}")
ab = [st.mean(stage12_only(k) for k in keys if k[0] == n) for n in Ns]
ax.plot(Ns, ab, color=GREY, ls="--", lw=1.1, label="Stages 1–2 only")
ax.set_xscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.minorticks_off()
ax.set_ylim(1.0, 1.21)
ax.set_title("(b) End-to-end speedup", loc="left")
label_axes(ax, "Images in dataset ($N$)", "Speedup (serial / parallel)")
ax.legend(ncol=3, loc="upper center", fontsize=6.3, columnspacing=0.9, handlelength=1.6)
fig.savefig(OUT / "total_time_speedup.pdf")
plt.close(fig)

# ---- Fig D: encoding share (%) of serial pipeline, per dataset and resolution
fig, ax = plt.subplots(figsize=(3.45, 2.2))
import numpy as np
x = np.arange(len(Ns))
wd = 0.2
for i, (w, h) in enumerate(RES_ORDER):
    m, c = mk[(w, h)]
    ys = [100 * st.mean(get(k, "serial", "encoding_ms") / get(k, "serial", "pipeline_ms") for k in keys
                        if k[0] == n and (k[1], k[2]) == (w, h)) for n in Ns]
    ax.bar(x + (i - 1.5) * wd, ys, wd * 0.88, color=c, label=f"{w}×{h}")
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in Ns])
ax.grid(axis="x", visible=False)
label_axes(ax, "Images in dataset ($N$)", "Blend+encode share (%)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, columnspacing=0.8, handlelength=1.0)
fig.savefig(OUT / "encode_share.pdf")
plt.close(fig)

# ---- LaTeX table fragments
def ms(x):
    return f"{x:,.0f}".replace(",", "{,}")


with open(OUT / "table_main.tex", "w") as f:
    for n in Ns:
        k = (n, 1920, 1080, 20)
        s, p = data[k]["serial"], data[k]["parallel"]
        f.write(f"{n} & {num(s,'total_frames'):.0f} & {ms(num(s,'feature_extraction_ms'))} & {ms(num(p,'feature_extraction_ms'))}"
                f" & {ms(num(s,'sequencing_ms'))} & {ms(num(p,'sequencing_ms'))}"
                f" & {ms(num(s,'encoding_ms'))} & {ms(num(p,'encoding_ms'))}"
                f" & {ms(num(s,'pipeline_ms'))} & {ms(num(p,'pipeline_ms'))} & {speed(k,'pipeline_ms'):.2f} \\\\\n")

with open(OUT / "table_speedup.tex", "w") as f:
    for w, h in RES_ORDER:
        sel = [k for k in keys if (k[1], k[2]) == (w, h)]
        f.write(f"{w}$\\times${h} & {st.mean(speed(k,'feature_extraction_ms') for k in sel):.2f}"
                f" & {st.mean(speed(k,'sequencing_ms') for k in sel):.2f}"
                f" & {st.mean(speed(k,'encoding_ms') for k in sel):.2f}"
                f" & {st.mean(speed(k,'pipeline_ms') for k in sel):.2f}"
                f" & {100*st.mean(get(k,'serial','encoding_ms')/get(k,'serial','pipeline_ms') for k in sel):.1f} \\\\\n")

with open(OUT / "table_frames.tex", "w") as f:  # effect of intermediate frames, N=500, 1280x720
    for fr in FR:
        k = (500, 1280, 720, fr)
        s, p = data[k]["serial"], data[k]["parallel"]
        f.write(f"{fr} & {num(s,'total_frames'):.0f} & {ms(num(s,'pipeline_ms'))} & {ms(num(p,'pipeline_ms'))}"
                f" & {speed(k,'pipeline_ms'):.2f} & {num(p,'encoding_ms')/num(p,'total_frames'):.2f} \\\\\n")

# rows end with "\\" except the last (the .tex file supplies it after \input)
for name in ("table_main", "table_speedup", "table_frames"):
    p = OUT / f"{name}.tex"
    p.write_text(p.read_text().rstrip().removesuffix("\\\\").rstrip() + "\n")

print(json.dumps(stats, indent=1, default=str))
