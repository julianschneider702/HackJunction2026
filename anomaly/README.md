# anomaly/ - brick defect detection (Anomalib)

Trains Anomalib models on **good bricks only** and scores how far a new brick deviates. Images stay on the machine that trains; they are not committed.

> `train.py` has not been run yet. The Anomalib calls follow the API of the installed version (2.7.0), but expect small fixes on the first real run.

## Setup (Windows, station laptop)

```powershell
uv venv --python 3.11 anomalib-env
anomalib-env\Scripts\activate
uv pip install "anomalib[openvino,cpu]"
```

## Image folders (outside the repo)

```
bricks-data/
  good/        training images, good bricks only (about 30-40)
  test_good/   good bricks NOT in good/ (about 10)
  defect/      defective bricks, evaluation only (about 15-20)
```

- `.jpg` or `.png`, file names do not matter.
- Same camera position and lighting for all images, crop tightly around the brick.
- Include variation in `good/`: position, rotation, brick upside down.
- Defect images are never used for training.
- Keep a separate set of images aside that is not used for choosing the threshold, and check once with it at the end.

## Run

```powershell
python anomaly/train.py --data D:\bricks-data                 # PaDiM, then PatchCore
python anomaly/train.py --data D:\bricks-data --models padim  # one model only
python anomaly/train.py --data D:\bricks-data --export        # also export to OpenVINO
```

For each model the script trains, tests, and prints the anomaly score per test image, sorted, with the highest good and lowest defect score. If the highest good score is below the lowest defect score, a threshold between them separates the classes cleanly.

## Sort new images (`sort_defects.py`)

Trains on `good/` only, then checks every image in `test_good/` and moves the ones flagged as defect into `defect/`. The threshold comes from part of `good/` that Anomalib holds back (normalised to 0.5); `defect/` is never used for training.

```powershell
python anomaly/sort_defects.py train --data D:\bricks-data            # learn from good/, save <data>/model_padim.ckpt
python anomaly/sort_defects.py check --data D:\bricks-data --dry-run  # show scores, move nothing
python anomaly/sort_defects.py check --data D:\bricks-data            # move flagged images to defect/
```

- `--threshold 0.6` raises the limit (fewer images flagged), `--copy` copies instead of moving.
- `--model patchcore` switches the model; use the same value for `train` and `check`.
- Tested on Linux with Anomalib 2.7.0: 77 good images (478x423 screenshots), PaDiM trains in a few seconds on CPU.

## Notes

- First run downloads a pretrained backbone, so it needs internet.
- PaDiM takes about a minute and PatchCore a few minutes (rough estimates).
- Checkpoints and exports land in a `results/` folder in the directory you start the script from, which is git-ignored.
- Run the script from a folder outside the repo or from `anomaly/`; the ignore rules cover both.
