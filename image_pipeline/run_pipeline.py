#!/usr/bin/env python3
"""End-to-end pipeline runner.

Builds the C++ executables with CMake, then for every dataset in a JSON config
runs both pipeline_serial and pipeline_parallel for every combination of
(width, height) x intermediateFrames. Invalid datasets/settings are reported and
skipped; only a build failure or an unusable config file stops the script.

Usage: python3 run_pipeline.py [--config datasets.json] [--build-dir build] [--skip-build]
"""
import argparse
import csv
import datetime
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # keep in sync with listImages() in src/common.hpp
MIN_IMAGES = 20
MODES = ("serial", "parallel")



def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass
class Dataset:
    name: str
    path: Path
    output_path: Path
    num_images: int
    sizes: list = field(default_factory=list)   # [(w, h)]
    frames: list = field(default_factory=list)  # [int]
    keep_video: tuple = None  # (w, h, frames): only this setting's videos are kept; None = keep all


@dataclass
class RunResult:
    dataset: str
    width: int
    height: int
    frames: int
    mode: str
    ok: bool
    wall_s: float = 0.0
    feature_ms: float = None
    order_ms: float = None
    encode_ms: float = None
    total_frames: int = None
    error: str = ""
    num_images: int = None

    @property
    def pipeline_ms(self):
        parts = [self.feature_ms, self.order_ms, self.encode_ms]
        return sum(parts) if all(p is not None for p in parts) else None


# ---------------------------------------------------------------- build
def build(build_dir: Path) -> bool:
    print("=" * 60)
    print(" [1/3] Building executables with CMake")
    print("=" * 60)
    configure = ["cmake", "-S", str(ROOT), "-B", str(build_dir)]
    if shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    for cmd in (configure, ["cmake", "--build", str(build_dir)]):
        print("[BUILD] " + " ".join(cmd))
        try:
            r = subprocess.run(cmd)
        except FileNotFoundError:
            print("[ERROR] cmake not found in PATH.")
            return False
        if r.returncode != 0:
            print(f"[ERROR] Command failed with exit code {r.returncode}.")
            return False
    return True


# ---------------------------------------------------------------- validation
def count_images(path: Path) -> int:
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def validate_config(config_path: Path):
    """Returns (datasets, skipped) where skipped is a list of message strings."""
    skipped = []

    def skip(msg):
        print(f"[SKIP] {msg}")
        skipped.append(msg)

    def warn(msg):
        print(f"[WARN] {msg}")
        skipped.append(msg)

    try:
        cfg = json.loads(config_path.read_text())
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        return None, skipped
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ERROR] Cannot read config '{config_path}': {e}")
        return None, skipped

    entries = cfg.get("datasets") if isinstance(cfg, dict) else None
    if not isinstance(entries, list) or not entries:
        print("[ERROR] Config must be an object with a non-empty 'datasets' list.")
        return None, skipped

    datasets = []
    for idx, e in enumerate(entries, 1):
        label = f"dataset #{idx}"
        if not isinstance(e, dict):
            skip(f"{label}: entry is not a JSON object")
            continue
        name = e.get("name") if isinstance(e.get("name"), str) and e.get("name") else None
        label = name or (Path(e["path"]).name if isinstance(e.get("path"), str) else label)
        name = name or label

        path_s, out_s = e.get("path"), e.get("output_path")
        if not isinstance(path_s, str) or not path_s:
            skip(f"{label}: 'path' missing or not a string")
            continue
        if not isinstance(out_s, str) or not out_s:
            skip(f"{label}: 'output_path' missing or not a string")
            continue

        path = Path(path_s).expanduser()
        if not path.is_dir():
            skip(f"{label}: dataset path does not exist or is not a directory: {path}")
            continue
        try:
            n = count_images(path)
        except OSError as ex:
            skip(f"{label}: cannot read dataset path ({ex})")
            continue
        if n < MIN_IMAGES:
            skip(f"{label}: only {n} images found in {path} (need at least {MIN_IMAGES})")
            continue

        out_path = Path(out_s).expanduser()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            skip(f"{label}: cannot create output directory {out_path.parent} ({ex})")
            continue

        sizes = []
        wh = e.get("widthHeight")
        if not isinstance(wh, list):
            skip(f"{label}: 'widthHeight' missing or not a list")
            continue
        for item in wh:
            w = item.get("width") if isinstance(item, dict) else None
            h = item.get("height") if isinstance(item, dict) else None
            if not (is_int(w) and is_int(h)) or w <= 0 or h <= 0:
                warn(f"{label}: ignoring invalid width/height setting {item!r} "
                     "(need an object with positive integer 'width' and 'height')")
            elif (w, h) not in sizes:
                sizes.append((w, h))

        frames = []
        fr = e.get("intermediateFrames")
        if not isinstance(fr, list):
            skip(f"{label}: 'intermediateFrames' missing or not a list")
            continue
        for f in fr:
            if not is_int(f) or f < 0:
                warn(f"{label}: ignoring invalid intermediateFrames value {f!r} "
                     "(need a non-negative integer)")
            elif f not in frames:
                frames.append(f)

        if not sizes or not frames:
            skip(f"{label}: no valid "
                 f"{'width/height' if not sizes else 'intermediateFrames'} settings left")
            continue

        keep = None
        kv = e.get("keepVideo")
        if kv is not None:
            w, h, f = (kv.get("width"), kv.get("height"), kv.get("intermediateFrames")) \
                if isinstance(kv, dict) else (None, None, None)
            if not (is_int(w) and is_int(h) and is_int(f)) or w <= 0 or h <= 0 or f < 0:
                warn(f"{label}: ignoring invalid 'keepVideo' {kv!r} (need positive integer "
                     "'width'/'height' and non-negative 'intermediateFrames'); keeping all videos")
            else:
                keep = (w, h, f)
                if (w, h) not in sizes or f not in frames:
                    warn(f"{label}: 'keepVideo' {w}x{h} f{f} is not one of the dataset's settings; "
                         "ALL its videos will be deleted")

        datasets.append(Dataset(name, path, out_path, n, sizes, frames, keep))
    return datasets, skipped


# ---------------------------------------------------------------- execution
CSV_FIELDS = ["timestamp", "dataset", "num_images", "mode", "width", "height",
              "intermediate_frames", "total_frames", "feature_extraction_ms",
              "sequencing_ms", "encoding_ms", "pipeline_ms", "wall_s", "status", "error"]


def result_row(r: "RunResult") -> dict:
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": r.dataset, "num_images": r.num_images, "mode": r.mode,
        "width": r.width, "height": r.height, "intermediate_frames": r.frames,
        "total_frames": r.total_frames, "feature_extraction_ms": r.feature_ms,
        "sequencing_ms": r.order_ms, "encoding_ms": r.encode_ms,
        "pipeline_ms": round(r.pipeline_ms, 3) if r.pipeline_ms is not None else None, "wall_s": round(r.wall_s, 2),
        "status": "ok" if r.ok else "failed", "error": r.error,
    }


class ResultStore:
    """Appends every run to results.csv and results.jsonl immediately (crash/sleep safe)."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = directory / f"results_{stamp}.csv"
        self.jsonl_path = directory / f"results_{stamp}.jsonl"
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writeheader()

    def add(self, r: "RunResult"):
        row = result_row(r)
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writerow(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")


def video_path(ds: Dataset, w: int, h: int, frames: int, mode: str) -> Path:
    stem = f"{ds.output_path.stem}_{w}x{h}_f{frames}_{mode}"
    return ds.output_path.with_name(stem + (ds.output_path.suffix or ".mp4"))


def prune_video(ds: Dataset, w: int, h: int, frames: int, mode: str) -> bool:
    """Delete this run's video unless it matches the dataset's keepVideo setting."""
    if ds.keep_video is None or ds.keep_video == (w, h, frames):
        return False
    try:
        video_path(ds, w, h, frames, mode).unlink(missing_ok=True)
        return True
    except OSError as ex:
        print(f"      [WARN] could not delete video: {ex}")
        return False


def run_one(exe: Path, ds: Dataset, w: int, h: int, frames: int, mode: str) -> RunResult:
    stem = f"{ds.output_path.stem}_{w}x{h}_f{frames}_{mode}"
    out = video_path(ds, w, h, frames, mode)
    metrics = ds.output_path.with_name(stem + ".metrics.json")
    metrics.unlink(missing_ok=True)
    cmd = [str(exe), "--dataset", str(ds.path), "--width", str(w), "--height", str(h),
           "--frames", str(frames), "--output", str(out), "--metrics", str(metrics)]
    res = RunResult(ds.name, w, h, frames, mode, ok=False, num_images=ds.num_images)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as ex:
        res.error = f"could not launch {exe.name}: {ex}"
        return res
    res.wall_s = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr.strip() or r.stdout.strip()).splitlines()
        res.error = f"exit code {r.returncode}" + (f": {tail[-1]}" if tail else "")
        return res
    try:
        m = json.loads(metrics.read_text())
        res.feature_ms = float(m["feature_extraction_ms"])
        res.order_ms = float(m["sequencing_ms"])
        res.encode_ms = float(m["encoding_ms"])
        res.total_frames = int(m["total_frames"])
        res.num_images = int(m.get("num_images", ds.num_images))
    except (OSError, ValueError, KeyError) as ex:
        res.error = f"ran OK but metrics file unreadable ({ex})"
        return res
    res.ok = True
    return res


def print_summary(results, skipped):
    print()
    print("=" * 60)
    print(" Summary")
    print("=" * 60)
    by_key = {}
    for r in results:
        by_key.setdefault((r.dataset, r.width, r.height, r.frames), {})[r.mode] = r

    header = f"{'dataset':<16}{'size':<11}{'frames':>6}{'serial ms':>12}{'parallel ms':>13}{'speedup':>9}"
    print(header)
    print("-" * len(header))
    for (name, w, h, f), modes in by_key.items():
        s, p = modes.get("serial"), modes.get("parallel")
        sm = s.pipeline_ms if s and s.ok else None
        pm = p.pipeline_ms if p and p.ok else None
        speed = f"{sm / pm:.2f}x" if sm and pm else "-"
        print(f"{name:<16}{f'{w}x{h}':<11}{f:>6}"
              f"{(f'{sm:.0f}' if sm else 'FAILED'):>12}"
              f"{(f'{pm:.0f}' if pm else 'FAILED'):>13}{speed:>9}")

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for r in failed:
            print(f"  {r.dataset} {r.width}x{r.height} f{r.frames} {r.mode}: {r.error}")
    if skipped:
        print(f"\nSkipped / ignored ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
    print(f"\nRuns: {sum(r.ok for r in results)} succeeded, {len(failed)} failed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "datasets.json"))
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"),
                    help="where results_<timestamp>.csv/.jsonl are written")
    ap.add_argument("--skip-build", action="store_true", help="reuse existing executables")
    args = ap.parse_args()

    build_dir = Path(args.build_dir).resolve()
    if not args.skip_build and not build(build_dir):
        return 1
    exes = {m: build_dir / f"pipeline_{m}" for m in MODES}
    missing = [str(p) for p in exes.values() if not p.is_file()]
    if missing:
        print(f"[ERROR] Missing executable(s): {', '.join(missing)}")
        return 1

    print("=" * 60)
    print(" [2/3] Validating configuration")
    print("=" * 60)
    datasets, skipped = validate_config(Path(args.config))
    if datasets is None:
        return 1
    if not datasets:
        print("[ERROR] No valid datasets to process.")
        print_summary([], skipped)
        return 1

    total_runs = sum(len(d.sizes) * len(d.frames) * len(MODES) for d in datasets)
    print(f"[INFO] {len(datasets)} valid dataset(s), {total_runs} runs planned.")

    print("=" * 60)
    print(" [3/3] Running pipelines")
    print("=" * 60)
    results = []
    store = ResultStore(Path(args.results_dir))
    print(f"[INFO] Saving every run's timings to {store.csv_path}")
    done = 0
    start = time.time()
    for di, ds in enumerate(datasets, 1):
        combos = list(itertools.product(ds.sizes, ds.frames))
        print(f"\n[dataset {di}/{len(datasets)}] {ds.name}: {ds.num_images} images, "
              f"{len(combos)} settings x {len(MODES)} modes = {len(combos) * len(MODES)} runs")
        for (w, h), f in combos:
            for mode in MODES:
                done += 1
                print(f"  [run {done}/{total_runs}] {mode:<8} {w}x{h} frames={f} ... ", end="", flush=True)
                r = run_one(exes[mode], ds, w, h, f, mode)
                results.append(r)
                store.add(r)
                pruned = prune_video(ds, w, h, f, mode)
                elapsed = time.time() - start
                eta = elapsed / done * (total_runs - done)
                if r.ok:
                    pm = r.pipeline_ms
                    detail = (f"{pm:.0f} ms pipeline "
                              f"(features {r.feature_ms:.0f}, order {r.order_ms:.0f}, "
                              f"encode {r.encode_ms:.0f}), {r.total_frames} frames"
                              if pm is not None else "done")
                    print(f"OK {r.wall_s:.1f}s | {detail}")
                else:
                    print(f"FAILED {r.wall_s:.1f}s | {r.error}")
                print(f"      elapsed {fmt_time(elapsed)}, ETA {fmt_time(eta)}"
                      + ("  (video pruned)" if pruned else ""), flush=True)

    print_summary(results, skipped)
    print(f"Results saved to {store.csv_path} and {store.jsonl_path}")
    print(f"Total time: {fmt_time(time.time() - start)}")
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.")
        sys.exit(130)
