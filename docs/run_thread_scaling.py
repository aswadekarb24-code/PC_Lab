#!/usr/bin/env python3
"""OpenMP thread-scaling study of the analysis stages (feature extraction, similarity + ordering).

Runs pipeline_parallel with --threads T (and OpenCV's own pool forced to 1 thread so OpenMP is the only
parallelism in these stages) and pipeline_serial with --cv-threads 1 (true single-threaded baseline) and with
OpenCV's default pool (the baseline used in the earlier sweeps). F=0 and --segments 12 keep the encode short;
only the stage times (not encoding) are used.

Usage: python3 run_thread_scaling.py [--reps 3]
Writes figures/thread_scaling.csv.
"""
import argparse, csv, json, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "image_pipeline"
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--threads", default="1,2,4,6,8,10,12,16")
ap.add_argument("--datasets", default="landscape_500,landscape_1000")
ap.add_argument("--sizes", default="1280x720,1920x1080")
a = ap.parse_args()

rows = []
with tempfile.TemporaryDirectory() as td:
    m, o = Path(td) / "m.json", str(Path(td) / "o.mp4")
    for ds in a.datasets.split(","):
        n = int(ds.split("_")[1])
        for size in a.sizes.split(","):
            w, h = size.split("x")
            base = ["--dataset", str(IMG / "data" / ds), "--width", w, "--height", h, "--frames", "0",
                    "--output", o, "--metrics", str(m)]
            configs = [("serial", "serial_cv1", ["--cv-threads", "1"]), ("serial", "serial_cvdefault", [])]
            configs += [("parallel", f"omp{t}", ["--threads", str(t), "--cv-threads", "1", "--segments", "12"])
                        for t in map(int, a.threads.split(","))]
            for rep in range(a.reps):          # interleave configurations across repetitions
                for exe, label, extra in configs:
                    subprocess.run([str(IMG / f"build/pipeline_{exe}"), *base, *extra], check=True, capture_output=True)
                    j = json.loads(m.read_text())
                    rows.append({"dataset": ds, "num_images": n, "width": w, "height": h, "rep": rep, "config": label,
                                 "threads": j["threads"], "cv_threads": j["cv_threads"],
                                 "feature_ms": j["feature_extraction_ms"], "sequencing_ms": j["sequencing_ms"]})
                    print(ds, size, rep, label, round(j["feature_extraction_ms"]), round(j["sequencing_ms"], 1), flush=True)
out = HERE / "figures/thread_scaling.csv"
with open(out, "w", newline="") as f:
    wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print("wrote", out)
