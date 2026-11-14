"""Train and compare Anomalib models on Duplo brick images.

Expected folders under --data (default: ./data):
    good/        training images, good bricks only
    test_good/   good bricks NOT used in training
    defect/      defective bricks, evaluation only

Usage:
    python train.py --data D:/anomaly-data
    python train.py --models padim patchcore --export
"""

import argparse

from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

MODELS = {"padim": Padim, "patchcore": Patchcore}


def scores_per_image(engine, model, datamodule, ckpt):
    """Print anomaly score per test image and a threshold hint."""
    good, bad = [], []
    for batch in engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt):
        for path, label, score in zip(batch.image_path, batch.gt_label, batch.pred_score):
            (bad if int(label) else good).append((float(score), path))
    for name, rows in (("test_good", good), ("defect", bad)):
        print(f"\n{name} (score, file), sorted:")
        for score, path in sorted(rows):
            print(f"  {score:.3f}  {path}")
    if good and bad:
        hi_good, lo_bad = max(good)[0], min(bad)[0]
        print(f"\nhighest good score: {hi_good:.3f} | lowest defect score: {lo_bad:.3f}")
        print("clean separation, threshold between them" if hi_good < lo_bad
              else "scores overlap: more/better images or a different model needed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="folder containing good/, test_good/, defect/")
    parser.add_argument("--models", nargs="+", default=["padim", "patchcore"], choices=MODELS)
    parser.add_argument("--export", action="store_true", help="export best checkpoint to OpenVINO")
    args = parser.parse_args()

    for name in args.models:
        print(f"\n===== {name} =====")
        datamodule = Folder(
            name="bricks",
            root=args.data,
            normal_dir="good",
            normal_test_dir="test_good",
            abnormal_dir="defect",
            num_workers=0,  # safer on Windows
        )
        model = MODELS[name]()
        engine = Engine()
        engine.fit(datamodule=datamodule, model=model)
        engine.test(datamodule=datamodule, model=model, ckpt_path=engine.best_model_path)
        scores_per_image(engine, model, datamodule, engine.best_model_path)
        if args.export:
            engine.export(model=model, export_type=ExportType.OPENVINO,
                          ckpt_path=engine.best_model_path)


if __name__ == "__main__":
    main()
