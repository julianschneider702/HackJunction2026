"""Train on good bricks, check test_good/, move every image flagged as defect into defect/.

Expected folders under --data:
    good/        training images, good bricks only
    test_good/   images to check (the model sorts these)
    defect/      images flagged as defect are moved here (created if missing)

The threshold comes from good images only: Anomalib holds back part of good/ and
uses the highest anomaly score among those as the limit (normalised to 0.5).
Images in defect/ are never used for training or for the threshold.

Usage:
    python sort_defects.py train --data D:/bricks-data   # only learn from good/, save model
    python sort_defects.py check --data D:/bricks-data   # use saved model, sort test_good/
    python sort_defects.py --data D:/bricks-data         # both in one go
Options:
    --model patchcore     other model (train and check must use the same one)
    --dry-run             only show, move nothing
    --threshold 0.6       higher limit = fewer defects
The trained model is saved as <data>/model_<name>.ckpt.
"""

import argparse
import shutil
from pathlib import Path

from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

MODELS = {"padim": Padim, "patchcore": Patchcore}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def images_in(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.is_dir() else []


def unique_target(folder: Path, name: str) -> Path:
    target = folder / name
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = folder / f"{stem}_{i}{suffix}"
        i += 1
    return target


def train(root: Path, name: str) -> Path:
    n_good = len(images_in(root / "good"))
    if n_good < 5:
        raise SystemExit(f"{root / 'good'}: only {n_good} images, need at least 5 (better 30+).")
    print(f"training {name} on {n_good} good images")
    # Anomalib splits off part of good/ for validation and threshold
    datamodule = Folder(name="bricks", root=root, normal_dir="good", num_workers=0)
    model = MODELS[name]()
    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)
    saved = root / f"model_{name}.ckpt"
    shutil.copy2(engine.best_model_path, saved)
    print(f"\nmodel saved: {saved}")
    return saved


def check(root: Path, name: str, threshold: float, copy: bool, dry_run: bool):
    test_dir, defect_dir = root / "test_good", root / "defect"
    saved = root / f"model_{name}.ckpt"
    if not saved.exists():
        raise SystemExit(f"no trained model at {saved}, run 'train' first.")
    to_check = images_in(test_dir)
    if not to_check:
        raise SystemExit(f"{test_dir}: no images to check.")
    print(f"checking {len(to_check)} images in {test_dir} with {saved.name}")

    results = []
    for batch in Engine().predict(model=MODELS[name](), dataset=PredictDataset(path=test_dir),
                                  ckpt_path=str(saved)):
        for path, score in zip(batch.image_path, batch.pred_score):
            results.append((float(score), Path(path)))
    results.sort(reverse=True)

    defects = [(s, p) for s, p in results if s > threshold]
    print(f"\nscore  result  file   (limit {threshold:.2f})")
    for score, path in results:
        print(f"{score:.3f}  {'DEFECT' if score > threshold else 'ok    '}  {path.name}")

    if dry_run:
        print(f"\n{len(defects)} of {len(results)} would be flagged as defect (dry run, nothing moved).")
        return
    defect_dir.mkdir(parents=True, exist_ok=True)
    for _, path in defects:
        target = unique_target(defect_dir, path.name)
        (shutil.copy2 if copy else shutil.move)(str(path), str(target))
    print(f"\n{len(defects)} of {len(results)} {'copied' if copy else 'moved'} to {defect_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all", choices=["train", "check", "all"])
    parser.add_argument("--data", default="data", help="folder containing good/, test_good/, defect/")
    parser.add_argument("--model", default="padim", choices=MODELS)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="normalised score limit, above it = defect (default 0.5)")
    parser.add_argument("--copy", action="store_true", help="copy instead of move")
    parser.add_argument("--dry-run", action="store_true", help="only print the result, move nothing")
    args = parser.parse_args()

    root = Path(args.data)
    if args.mode in ("train", "all"):
        train(root, args.model)
    if args.mode in ("check", "all"):
        check(root, args.model, args.threshold, args.copy, args.dry_run)

if __name__ == "__main__":
    main()
