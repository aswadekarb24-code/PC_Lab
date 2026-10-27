#!/usr/bin/env python3
"""One-off dataset setup.

Downloads a multi-class Kaggle landscape dataset via kagglehub, builds flat
class-balanced subsets of ~100/200/500/750/1000 images under data/, and writes
datasets.json (absolute paths) for run_pipeline.py.

Requires: pip install kagglehub
"""
import json
import random
import shutil
import sys
from pathlib import Path

import kagglehub

KAGGLE_SLUG = "utkarshsaxenadn/landscape-recognition-image-dataset-12k-images"
SIZES = [100, 200, 500, 750, 1000]
WIDTH_HEIGHT = [
    {"width": 640, "height": 360},
    {"width": 1280, "height": 720},
    {"width": 1920, "height": 1080},
    {"width": 1024, "height": 768},
]
INTERMEDIATE_FRAMES = [10, 20, 30]
KEEP_VIDEO = {"width": 1920, "height": 1080, "intermediateFrames": 20}
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """Pick the 'Training Data' folder (or fall back to any folder of class subdirs)."""
    candidates = [p for p in root.rglob("*") if p.is_dir() and "train" in p.name.lower()
                  and "tfrecord" not in p.name.lower()]
    base = candidates[0] if candidates else root
    classes = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if imgs:
            classes[d.name] = imgs
    return classes


def main() -> int:
    print(f"[INFO] Downloading {KAGGLE_SLUG} (cached after first run)...")
    src = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    classes = find_class_dirs(src)
    if len(classes) < 2:
        print(f"[ERROR] Expected multiple class folders under {src}, found {list(classes)}")
        return 1
    print(f"[INFO] Classes: " + ", ".join(f"{k} ({len(v)})" for k, v in classes.items()))

    # Remove old datasets
    for old in ("archive", "archive.zip", "raw_images"):
        p = DATA_DIR / old
        if p.is_dir():
            shutil.rmtree(p)
            print(f"[INFO] Removed {p}")
        elif p.exists():
            p.unlink()
            print(f"[INFO] Removed {p}")
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    rng = random.Random(SEED)
    datasets = []
    for total in SIZES:
        name = f"landscape_{total}"
        out = DATA_DIR / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()

        per_class = total // len(classes)
        chosen = []
        for cname, imgs in classes.items():
            chosen += [(cname, p) for p in rng.sample(imgs, min(per_class, len(imgs)))]
        for cname, p in chosen:
            shutil.copy2(p, out / f"{cname}_{p.name}")
        print(f"[INFO] {name}: {len(chosen)} images")

        datasets.append({
            "name": name,
            "path": str(out),
            "output_path": str(OUTPUT_DIR / f"{name}.mp4"),
            "widthHeight": WIDTH_HEIGHT,
            "intermediateFrames": INTERMEDIATE_FRAMES,
            # only this setting's videos are kept on disk; the rest are deleted after timing
            "keepVideo": KEEP_VIDEO,
        })

    cfg = ROOT / "datasets.json"
    cfg.write_text(json.dumps({"datasets": datasets}, indent=2) + "\n")
    print(f"[SUCCESS] Wrote {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
