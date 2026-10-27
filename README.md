# Similarity-Ordered Photo Transition Videos (Serial vs OpenMP/CUDA)

Turns a folder of photos into a smooth slideshow video. Images are described by HSV colour histograms, ordered so that neighbours look alike (greedy nearest-neighbour on histogram correlation), and joined by cross-dissolve transition frames. The same pipeline is built twice for comparison:

| Executable | Source | What it uses |
|---|---|---|
| `pipeline_serial` | `src/main_serial.cpp` | Single-threaded CPU, OpenCV only |
| `pipeline_parallel` | `src/main.cpp`, `src/kernels.cu` | OpenMP (feature extraction, similarity matrix) + CUDA (frame blending) + a producer–consumer queue that overlaps GPU blending with video encoding, optionally split into parallel video segments |

Both share `src/common.hpp` (argument parsing, image listing, HSV histogram, greedy ordering), so timing differences come only from how the stages are executed. The write-up is in `docs/research_paper.pdf`.

## Pipeline stages

1. **Feature extraction**: decode, resize, 3-D HSV histogram per image (OpenMP, dynamic schedule).
2. **Similarity matrix**: pairwise histogram correlation (OpenMP over rows).
3. **Ordering**: greedy nearest-neighbour tour (sequential).
4. **Blend + encode**: `F` cross-dissolve frames between each consecutive pair, written to an MP4 (`mp4v`, 60 fps).
   - Serial: blend on the CPU, then encode, frame by frame.
   - Parallel: the main thread (producer) uploads each image pair and queues CUDA blend kernels and async device-to-host copies on 8 per-slot streams into pinned buffers; an encoder thread (consumer) pops frames in order and writes them to the `VideoWriter`. A fixed pool of 8 slots bounds the frames in flight. The queue is `src/blocking_queue.hpp`.
   - With `--segments K` (K > 1) the ordered sequence is split into K contiguous ranges of image pairs. Each range runs its own producer/consumer pair and `VideoWriter`, writing its own temporary file; the files are then joined losslessly with `ffmpeg -f concat -c copy`. This uses several cores for encoding instead of one. `K = 1` (default) writes a single file and needs no ffmpeg.

A video has `(N - 1) * (F + 1) + 1` frames for `N` images and `F` intermediate frames.

## Requirements

- Linux, CMake >= 3.18 (Ninja is used if installed), a C++17 compiler with OpenMP
- OpenCV (C++ development package)
- `ffmpeg` in `PATH` (only for `--segments` > 1)
- CUDA Toolkit and an NVIDIA GPU (needed for `pipeline_parallel`; the project is configured with the CUDA language, so the toolkit is needed to build both targets)
- Python 3 for the scripts; `pip install kagglehub` for `setup_datasets.py` only

## Quick start

All commands run from `image_pipeline/`.

```bash
cd image_pipeline

# 1. One-off: download the Kaggle dataset, build the subsets, write datasets.json
python3 setup_datasets.py

# 2. Build both executables and run the whole benchmark sweep
python3 run_pipeline.py
```

## `setup_datasets.py`

Downloads "Landscape Recognition | Image Dataset (12k Images)" through `kagglehub` (Kaggle credentials may be needed; cached after the first run). It then:

- samples a class-balanced subset of **100, 200, 500, 750 and 1000** images (fixed seed 42) and copies them flat into `data/landscape_<N>/`,
- creates `output/`,
- (re)writes `datasets.json` with **absolute paths** for this machine.

Re-run it after moving the repository, since the paths in `datasets.json` are absolute. Dataset sizes, resolutions, frame counts and the kept video are constants at the top of the script.

## `datasets.json`

Configuration read by `run_pipeline.py`:

```json
{
  "datasets": [
    {
      "name": "landscape_100",
      "path": "/abs/path/to/data/landscape_100",
      "output_path": "/abs/path/to/output/landscape_100.mp4",
      "widthHeight": [{"width": 640, "height": 360}, {"width": 1920, "height": 1080}],
      "intermediateFrames": [10, 20, 30],
      "keepVideo": {"width": 1920, "height": 1080, "intermediateFrames": 20}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | Label used in logs and results |
| `path` | Folder of `.jpg` / `.jpeg` / `.png` images (at least 20) |
| `output_path` | Base path for videos; each run writes `<stem>_<W>x<H>_f<F>_<mode>.mp4` next to it, plus a `.metrics.json` |
| `widthHeight` | Output resolutions to sweep |
| `intermediateFrames` | Transition lengths `F` to sweep (0 allowed) |
| `keepVideo` | Optional. Only the videos of this `(width, height, frames)` setting are kept; all others are deleted after their timings are saved. Omit to keep everything |

Every dataset is run for every resolution x `F` combination in both modes. The checked-in file lists 5 datasets x 4 resolutions x 3 lengths x 2 modes = 120 runs.

## `run_pipeline.py`

```bash
python3 run_pipeline.py [--config datasets.json] [--build-dir build] \
                        [--results-dir results] [--skip-build] \
                        [--mode both|serial|parallel] [--segments K] \
                        [--threads T] [--cv-threads M]
```

1. Builds both executables with CMake into `build/` (Release by default; skip with `--skip-build` to reuse existing ones).
2. Validates the config. Bad entries are reported and skipped (missing path, fewer than 20 images, invalid size or frame count); only a build failure or an unusable config stops the script.
3. Runs every configuration (`--mode parallel` or `--mode serial` runs only that executable; default `both`; `--segments K` and `--threads T` are passed to the parallel executable only, `--cv-threads M` to both), printing progress and ETA, then a serial-vs-parallel speedup table.

Each run is appended immediately to `results/results_<timestamp>.csv` and `.jsonl` (per-stage times, frame count, wall time, status), so an interrupted sweep keeps its data. Videos go to `output/`.

## Running an executable directly

```bash
./build/pipeline_parallel --dataset data/landscape_100 --width 1280 --height 720 \
                          --frames 20 --output out.mp4 --metrics out.json
```

| Option | Default | Notes |
|---|---|---|
| `--dataset <path>` | `../data/raw_images` | Image folder |
| `--width`, `--height` | 1280, 720 | Must be given together |
| `--frames <int>` | 30 | Intermediate frames per image pair |
| `--output <path>` | `output_transition_video.mp4` | |
| `--metrics <path>` | none | Writes stage timings (ms) as JSON |
| `--segments <int>` | 1 | Parallel build only: concurrent video segments, joined with ffmpeg |
| `--threads <int>` | OpenMP default | Parallel build only: OpenMP threads (default: all logical CPUs, or `OMP_NUM_THREADS`) |
| `--cv-threads <int>` | OpenCV default | Both builds: size of OpenCV's own internal thread pool (default: all logical CPUs) |
| `--help` | | |

`--threads` sets the OpenMP team size of the parallel build; the serial build ignores it. **OpenCV has its own thread pool**, which is active by default in *both* builds (inside `resize`, `cvtColor`, `calcHist`, ...), so `pipeline_serial` is single-threaded in its own code but not in OpenCV calls. Pass `--cv-threads 1` to switch that off (a truly single-threaded serial baseline; also the clean setting for OpenMP scaling studies). The earlier benchmark sweeps used the OpenCV default. The number of frame buffers per segment is fixed at 8 (`kNumSlots` in `src/main.cpp`).

Invalid values exit with code 2. `pipeline_serial` accepts the same options but ignores `--segments` and `--threads`. The `blend + encode` time (`encoding_ms`) covers blending, host-device transfers, codec work and file output.

## Repository layout

```
image_pipeline/
  src/                 main.cpp (parallel), main_serial.cpp, kernels.cu, common.hpp, blocking_queue.hpp, encode_bench.cpp
  CMakeLists.txt       builds pipeline_serial, pipeline_parallel and encode_bench
  setup_datasets.py    dataset download + datasets.json generation
  datasets.json        benchmark configuration (generated, machine-specific paths)
  run_pipeline.py      build + benchmark driver
  data/ output/ results/ build/   generated, git-ignored
  logs/run.log         log of a previous sweep
docs/                  research_paper.tex/.pdf, figure and table generation (make_figures.py, make_queue_results.py,
                       make_segment_results.py), experiment scripts (run_segment_pilot.py, run_thread_scaling.py, run_encoder_scaling.py), proposal
papers/                reference papers
```

## Final benchmark settings used in the paper

```bash
cd image_pipeline
python3 run_pipeline.py --mode serial   --cv-threads 1                  # strictly single-threaded baseline
python3 run_pipeline.py --mode parallel --segments 12                   # P3, 16 threads
python3 run_pipeline.py --mode parallel --segments 6 --threads 6        # P3, six threads
```
Analysis scripts in `docs/` (`make_figures.py`, `make_queue_results.py`, `make_segment_results.py`, `make_thread_results.py`) turn the `results/*.csv` files into the paper's tables and figures; `docs/figures/results_merged.csv` holds the strict serial baseline together with the first parallel sweep.

`encode_bench` (built with the other targets) runs T independent concurrent encodes of one video clip to measure encoder scaling: `./build/encode_bench --input clip.mp4 --frames 500 --copies 8`; `docs/run_encoder_scaling.py` automates a sweep.
