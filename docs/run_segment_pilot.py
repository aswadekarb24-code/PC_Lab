#!/usr/bin/env python3
"""Segment-count pilot: runs pipeline_parallel for several --segments values, repeated, on one setting.

Usage: python3 run_segment_pilot.py [--reps 3]
Writes figures/segment_pilot.csv (segments, rep, encoding_ms, pipeline_ms).
"""
import argparse, csv, json, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "image_pipeline"
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--dataset", default=str(IMG / "data/landscape_100"))
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--frames", type=int, default=20)
ap.add_argument("--ks", default="1,2,4,6,8,12,16")
a = ap.parse_args()

rows = []
with tempfile.TemporaryDirectory() as td:
    # interleave K values across repetitions so thermal drift does not favour one K
    for rep in range(a.reps):
        for k in map(int, a.ks.split(",")):
            m = Path(td) / "m.json"
            subprocess.run([str(IMG / "build/pipeline_parallel"), "--dataset", a.dataset, "--width", str(a.width),
                            "--height", str(a.height), "--frames", str(a.frames), "--segments", str(k),
                            "--output", str(Path(td) / "o.mp4"), "--metrics", str(m)],
                           check=True, capture_output=True)
            j = json.loads(m.read_text())
            enc = j["encoding_ms"]
            rows.append({"segments": k, "rep": rep, "encoding_ms": enc,
                         "pipeline_ms": j["feature_extraction_ms"] + j["sequencing_ms"] + enc})
            print(k, rep, round(enc), flush=True)
out = HERE / "figures/segment_pilot.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
print("wrote", out)
