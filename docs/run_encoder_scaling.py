#!/usr/bin/env python3
"""Encoder scaling: T fully independent concurrent encodes (no GPU, queue or join) with encode_bench.

The source clip is generated with pipeline_parallel (landscape_100, 1280x720, F=20) and its first --frames
frames are re-encoded by every copy with the same mp4v writer used in the pipeline.

Usage: python3 run_encoder_scaling.py [--reps 3] [--frames 500]
Writes figures/encoder_scaling.csv (copies, rep, frames, wall_ms, fps).
"""
import argparse, csv, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "image_pipeline"
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--frames", type=int, default=500)
ap.add_argument("--copies", default="1,2,4,6,8,10,12,16")
a = ap.parse_args()

rows = []
with tempfile.TemporaryDirectory() as td:
    src = Path(td) / "clip.mp4"
    subprocess.run([str(IMG / "build/pipeline_parallel"), "--dataset", str(IMG / "data/landscape_100"), "--width", "1280",
                    "--height", "720", "--frames", "20", "--segments", "12", "--output", str(src)],
                   check=True, capture_output=True)
    for rep in range(a.reps):                       # interleave copy counts across repetitions
        for t in map(int, a.copies.split(",")):
            r = subprocess.run([str(IMG / "build/encode_bench"), "--input", str(src), "--frames", str(a.frames),
                                "--copies", str(t), "--output-dir", td], check=True, capture_output=True, text=True)
            c, n, ms, fps = r.stdout.split()
            rows.append({"copies": c, "rep": rep, "frames": n, "wall_ms": ms, "fps": fps})
            print(c, rep, n, round(float(ms)), round(float(fps)), flush=True)
out = HERE / "figures/encoder_scaling.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
print("wrote", out)
