# anomaly/ - brick defect detection with Anomalib

Trains [Anomalib](https://github.com/open-edge-platform/anomalib) models on **good bricks only** and scores
how far a new image deviates. Defective bricks are never used for training, only for checking the result.

Images and checkpoints stay on the machine that trains. They are not committed.

## Scripts

| Script | What it does |
|---|---|
| `sort_defects.py` | Trains one model on `good/`, saves it, and scores the images in `test_good/` |
| `show_scores.py` | Scores images with a saved model and shows them in an HTML page with score and verdict |
| `train.py` | Trains PaDiM and PatchCore side by side, prints the score per test image, optional OpenVINO export |

Models available in `sort_defects.py` and `show_scores.py`: `padim`, `patchcore`, `anomalydino`, `dinomaly`.
The inspection loop in [`sorter/`](../sorter/README.md) uses **AnomalyDINO** (DINOv2 ViT-B, 448 px).

## Setup

Anomalib gets its own Python environment, separate from Physical AI Studio:

```bash
uv venv --python 3.11 ~/anomalib-env
source ~/anomalib-env/bin/activate          # Windows: anomalib-env\Scripts\activate
uv pip install "anomalib[openvino,cpu]"
```

The first run downloads a pretrained backbone, so it needs internet.

## Image folders (outside the repo)

```
bricks-data/
  good/        training images, good bricks only
  test_good/   good bricks that are NOT in good/
  defect/      defective bricks, for evaluation only
```

- `.jpg` or `.png`, file names do not matter.
- Same camera position and lighting for all images.
- Include variation in `good/`: position, rotation, brick upside down.
- Capture images with the tools in `inspection/`.

## Train and check

```bash
python anomaly/sort_defects.py train --data ~/bricks-data --model anomalydino   # saves <data>/model_anomalydino.ckpt
python anomaly/sort_defects.py check --data ~/bricks-data --model anomalydino   # scores test_good/, moves nothing
python anomaly/show_scores.py        --data ~/bricks-data --model anomalydino   # HTML page for test_good/ and defect/
```

- The threshold comes from good images only: Anomalib holds back part of `good/` and normalises the
  score so that 0.5 is the limit. `--threshold 0.6` raises the limit (fewer images flagged).
- `check --move` moves flagged images into `defect/`, `--move --copy` copies them instead.
- `--device xpu` or `--device cpu` picks the device; the default `auto` uses the Intel GPU if available.
- Use the same `--model` for `train`, `check` and `show_scores.py`.

## Compare PaDiM and PatchCore

```bash
python anomaly/train.py --data ~/bricks-data                 # PaDiM, then PatchCore
python anomaly/train.py --data ~/bricks-data --models padim  # one model only
python anomaly/train.py --data ~/bricks-data --export        # also export to OpenVINO
```

For each model the script trains, tests, and prints the anomaly score per test image, sorted, with the
highest good and the lowest defect score. If the highest good score is below the lowest defect score, a
threshold between them separates the classes cleanly.

Checkpoints and exports land in a `results/` folder in the directory you start the script from. That
folder is git-ignored.

## Result on our station

13 wrist-camera test images at threshold 0.5 with AnomalyDINO: all 8 marked bricks were flagged and none of
the 5 clean ones. The set is small and comes from one lighting setup, so this shows that it works on our
station, not a general accuracy.
