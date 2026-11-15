"""Score images with the saved model and show them one below the other with score and DEFECT/ok.

Nothing is moved. Creates <data>/scores.html and opens it in the browser.

Usage:
    python show_scores.py --data ~/bricks-data                     # test_good/ and defect/
    python show_scores.py --data ~/bricks-data --folders good      # any subfolders
"""

import argparse
import base64
import html
import subprocess
from pathlib import Path

from anomalib.data import PredictDataset

from sort_defects import MODELS, images_in, make_engine


def score_folder(folder: Path, name: str, saved: Path, device: str):
    out = []
    for batch in make_engine(device).predict(model=MODELS[name](), dataset=PredictDataset(path=folder),
                                             ckpt_path=str(saved)):
        for path, score in zip(batch.image_path, batch.pred_score):
            out.append((float(score), Path(path)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--folders", nargs="+", default=["test_good", "defect"])
    ap.add_argument("--model", default="padim", choices=MODELS)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto", choices=["auto", "xpu", "cpu"])
    args = ap.parse_args()

    root = Path(args.data).expanduser()
    saved = root / f"model_{args.model}.ckpt"
    if not saved.exists():
        raise SystemExit(f"no trained model at {saved}, run 'sort_defects.py train' first.")

    results = []
    for f in args.folders:
        folder = root / f
        if images_in(folder):
            results += [(s, p, f) for s, p in score_folder(folder, args.model, saved, args.device)]
    if not results:
        raise SystemExit("no images found.")
    results.sort(key=lambda r: -r[0])

    rows = []
    for score, path, folder in results:
        bad = score > args.threshold
        img = base64.b64encode(path.read_bytes()).decode()
        mime = "png" if path.suffix.lower() == ".png" else "jpeg"
        rows.append(f"""<div class="row {'bad' if bad else 'ok'}">
  <img src="data:image/{mime};base64,{img}">
  <div class="info"><div class="score">{score:.3f}</div>
    <div class="tag">{'DEFECT' if bad else 'ok'}</div>
    <div class="name">{html.escape(folder)}/{html.escape(path.name)}</div></div></div>""")

    n_bad = sum(r[0] > args.threshold for r in results)
    page = f"""<!doctype html><meta charset="utf-8"><title>Anomaly Scores</title>
<style>
body{{font-family:sans-serif;background:#1e1e1e;color:#eee;margin:20px}}
.row{{display:flex;align-items:center;gap:24px;margin:12px 0;padding:10px;border-left:8px solid;background:#2a2a2a}}
.row.bad{{border-color:#e5484d}} .row.ok{{border-color:#30a46c}}
img{{max-height:300px;max-width:60vw}}
.score{{font-size:40px;font-weight:bold}}
.bad .tag{{color:#e5484d;font-size:28px;font-weight:bold}} .ok .tag{{color:#30a46c;font-size:28px;font-weight:bold}}
.name{{color:#aaa;margin-top:6px}}
</style>
<h2>{len(results)} images, {n_bad} DEFECT (limit {args.threshold:.2f}, model {args.model})</h2>
{''.join(rows)}"""
    out = root / "scores.html"
    out.write_text(page)
    print(f"\n{out}")
    subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
