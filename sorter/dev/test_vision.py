"""Offline check of the stroke detector and the scene tracker on saved images."""
import glob, os, sys, cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vision
cfg = vision.config()
os.chdir(os.path.dirname(HERE))   # the sample globs below are relative to sorter/
OUT = os.path.join(os.path.dirname(HERE), "samples", "out_" + __import__("time").strftime("%H%M%S")); os.makedirs(OUT, exist_ok=True)
def rgb(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
groups = {
  "good (soll 0)": sorted(glob.glob(os.path.expanduser("~/bricks-data/good/*.png"))),
  "defect (soll >0)": sorted(glob.glob(os.path.expanduser("~/bricks-data/defect/*.png"))),
  "test_good": sorted(glob.glob(os.path.expanduser("~/bricks-data/test_good/*.png"))),
  "ds wrist hold": sorted(glob.glob("samples/ds/*wrist_hold.png")),
  "ds webcam": sorted(glob.glob("samples/ds/*webcam*.png")),
  "live": sorted(glob.glob("samples/live_w*.png")),
}
for g, files in groups.items():
    hits = []
    for f in files:
        img = rgb(f)
        if img.shape[1] != 640:  # screenshots: bring to camera scale
            img = cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))
        s = vision.find_strokes(img, cfg)
        if s:
            hits.append((os.path.basename(f), [(round(x.length), round(x.width,1), round(x.ring_colour,2), round(x.ink), round(x.ring_hue,2), round(x.width_ratio,2)) for x in s]))
            b = cv2.cvtColor(img, cv2.COLOR_RGB2BGR); vision.draw_strokes(b, s)
            cv2.imwrite(f"{OUT}/{g.split()[0]}_{os.path.basename(f)}", b)
    print(f"{g:18s} {len(hits)}/{len(files)} mit Strich", hits[:6])
