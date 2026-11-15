"""Live debug window for the policy (SmolVLA / ACT) while Studio drives the arm.

For every prediction it shows:
  - the camera images the model received
  - the planned motion (action chunk) per joint; the dot is the current position
  - camera influence: the same input goes through the model again with one camera blacked out.
    If the motion barely changes, the model ignores that camera.
    "x Rauschen" (x noise) = change relative to the model's normal random variation
    (below ~1.5x the camera hardly matters).

Needs the debug dump of our Studio build (/dev/shm/physicalai-debug). On-screen labels are German.

Run (before or while the model runs in Studio):
    station/studio-python tools/policy_debug_view.py
Keys: q quit, a camera influence on/off, s save image, c swap colours (RGB/BGR)
"""

import argparse
import glob
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

DEBUG_DIR = Path("/dev/shm/physicalai-debug")
LATEST = DEBUG_DIR / "latest.npz"
MODELS = Path.home() / ".local/share/physicalai/models"
W = 1500
CAM_W, CAM_H = 480, 360
COLORS = [(80, 160, 255), (80, 220, 120), (255, 170, 60), (220, 90, 220), (90, 220, 220), (200, 200, 200)]


def newest_model() -> Path | None:
    dirs = [Path(p).parents[2] for p in glob.glob(str(MODELS / "*/exports/torch/manifest.json"))]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def load(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:  # file replaced while reading - next round
        return None


class Influence(threading.Thread):
    """Runs the model a few extra times on the latest observation, each camera blacked out once."""

    def __init__(self, model_dir: Path, device: str):
        super().__init__(daemon=True)
        self.model_dir, self.device = model_dir, device
        self.enabled = True
        self.status = f"lade Modell {model_dir.name[:8]} ..."
        self.result: dict | None = None
        self._job: dict | None = None
        self._lock = threading.Lock()

    def submit(self, data: dict):
        with self._lock:
            self._job = data

    def _seed(self, s: int):
        import torch
        torch.manual_seed(s)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(s)

    def run(self):
        try:
            from physicalai.inference import InferenceModel
            model = InferenceModel(self.model_dir / "exports" / "torch", backend="torch", device=self.device)
        except Exception as e:  # noqa: BLE001
            self.status = f"Modell laden fehlgeschlagen: {e}"[:120]
            return
        self.status = "bereit"
        while True:
            with self._lock:
                job, self._job = self._job, None
            if job is None or not self.enabled:
                time.sleep(0.05)
                continue
            t = time.monotonic()
            obs = {"state": job["state"][None].astype(np.float32)}
            cams = [k[4:] for k in job if k.startswith("img:")]
            for c in cams:
                obs[c] = job["img:" + c][None]
            if str(job["task"]):
                obs["task"] = [str(job["task"])]

            def run_with(o, seed):
                self._seed(seed)
                return model.predict_action_chunk(o)

            base = run_with(obs, 0)
            noise = float(np.abs(run_with(obs, 1) - base).mean())
            per_cam = {}
            for c in cams:
                o = dict(obs)
                o[c] = np.zeros_like(obs[c])
                per_cam[c] = float(np.abs(run_with(o, 0) - base).mean())
            self.result = {"noise": noise, "cams": per_cam, "stamp": float(job["stamp"]),
                           "seconds": time.monotonic() - t}
            self.status = f"berechnet in {self.result['seconds']:.1f}s"


def draw_chunk(canvas, x0, y0, w, h, data):
    actions, state = data["actions"], data["state"]
    names = [str(n) for n in data["joint_names"]] or [f"j{i}" for i in range(actions.shape[1])]
    n = actions.shape[1]
    rows = (n + 2) // 3
    pw, ph = w // 3, h // rows
    for j in range(n):
        px, py = x0 + (j % 3) * pw, y0 + (j // 3) * ph
        cv2.rectangle(canvas, (px + 4, py + 4), (px + pw - 4, py + ph - 4), (60, 60, 60), 1)
        vals = np.concatenate([[state[j]], actions[:, j]])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * 0.15, 2.0)
        lo, hi = lo - pad, hi + pad

        def pt(i, v):
            return (int(px + 14 + i * (pw - 28) / max(len(actions), 1)),
                    int(py + ph - 14 - (v - lo) / (hi - lo) * (ph - 50)))
        pts = np.array([pt(i + 1, v) for i, v in enumerate(actions[:, j])], np.int32)
        cv2.polylines(canvas, [pts], False, COLORS[j % len(COLORS)], 2, cv2.LINE_AA)
        cv2.circle(canvas, pt(0, state[j]), 6, (255, 255, 255), -1, cv2.LINE_AA)
        label = f"{names[j]}: jetzt {state[j]:.0f} -> Ziel {actions[-1, j]:.0f}"
        cv2.putText(canvas, label, (px + 12, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


def window_open() -> bool:
    try:
        return cv2.getWindowProperty("Policy Debug", cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:  # OpenCV 5 + Qt cannot report this; then only q closes
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Modell-ID oder Ordner (Standard: neuestes Modell mit torch-Export)")
    ap.add_argument("--device", default="xpu", help="Geraet fuer den Kamera-Einfluss (xpu oder cpu)")
    ap.add_argument("--no-influence", action="store_true", help="nur anzeigen, Modell nicht extra laden")
    ap.add_argument("--save-once", help="Test: ein Bild rendern, als PNG speichern, kein Fenster")
    args = ap.parse_args()

    DEBUG_DIR.mkdir(exist_ok=True)
    (DEBUG_DIR / "enabled").touch()

    infl = None
    if not args.no_influence:
        mdir = Path(args.model) if args.model and os.path.isdir(args.model) else (
            MODELS / args.model if args.model else newest_model())
        if mdir and (mdir / "exports/torch/manifest.json").exists():
            infl = Influence(mdir, args.device)
            infl.start()
        else:
            print("Kein Modell mit torch-Export gefunden, Kamera-Einfluss aus.")

    if not args.save_once:
        cv2.namedWindow("Policy Debug", cv2.WINDOW_NORMAL)
    t_start = time.time()
    last_mtime, data, count, swap_rgb = 0.0, None, 0, True
    stamps: list[float] = []
    try:
        while True:
            try:
                mtime = LATEST.stat().st_mtime
            except FileNotFoundError:
                mtime = 0.0
            if mtime != last_mtime:
                new = load(LATEST)
                if new is not None:
                    data, last_mtime, count = new, mtime, count + 1
                    stamps = [s for s in stamps if time.time() - s < 5] + [time.time()]
                    if infl:
                        infl.submit(data)

            canvas = np.full((CAM_H + 420, W, 3), 25, np.uint8)
            if data is None:
                cv2.putText(canvas, "Warte auf das Modell ... in Studio das Modell starten.", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(canvas, "(Studio muss nach dem Einbau einmal das Modell neu laden)", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)
            else:
                cams = sorted(k[4:] for k in data if k.startswith("img:"))
                res = infl.result if infl else None
                for i, c in enumerate(cams):
                    img = data["img:" + c]
                    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                        img = np.transpose(img, (1, 2, 0))
                    if img.dtype != np.uint8:
                        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
                    if swap_rgb:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    x = 10 + i * (CAM_W + 10)
                    canvas[10:10 + CAM_H, x:x + CAM_W] = cv2.resize(img, (CAM_W, CAM_H))
                    cv2.rectangle(canvas, (x, 10), (x + CAM_W, 40), (0, 0, 0), -1)
                    cv2.putText(canvas, c.removeprefix("images."), (x + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    if res and c in res["cams"]:
                        ratio = res["cams"][c] / max(res["noise"], 1e-6)
                        col = (60, 200, 60) if ratio >= 3 else (0, 200, 255) if ratio >= 1.5 else (60, 60, 230)
                        txt = f"Einfluss {res['cams'][c]:.1f}  ({ratio:.1f}x Rauschen)"
                        bar = int(min(ratio / 10, 1) * (CAM_W - 16))
                        cv2.rectangle(canvas, (x, CAM_H - 22), (x + CAM_W, CAM_H + 10), (0, 0, 0), -1)
                        cv2.rectangle(canvas, (x + 8, CAM_H - 16), (x + 8 + bar, CAM_H + 4), col, -1)
                        cv2.putText(canvas, txt, (x + 12, CAM_H), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                y = CAM_H + 20
                hz = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6) if len(stamps) > 1 else 0
                age = time.time() - float(data["stamp"])
                info = (f"Task: \"{data['task']}\"   |   Vorhersage {count}, {float(data['seconds']) * 1000:.0f} ms, "
                        f"{hz:.1f}/s, vor {age:.1f}s")
                cv2.putText(canvas, info, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                            cv2.LINE_AA)
                status = ("Kamera-Einfluss: " + (infl.status if infl.enabled else "aus (Taste a)")) if infl else \
                    "Kamera-Einfluss: aus"
                cv2.putText(canvas, status + "   |   q beenden, a Einfluss, s speichern, c Farben", (10, y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
                draw_chunk(canvas, 10, y + 50, W - 20, 420 - 70, data)

            if args.save_once:
                if data is not None and (not infl or infl.result or time.time() - t_start > 120):
                    cv2.imwrite(args.save_once, canvas)
                    print("gespeichert:", args.save_once)
                    break
                time.sleep(0.2)
                continue
            cv2.imshow("Policy Debug", canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or not window_open():
                break
            if key == ord("a") and infl:
                infl.enabled = not infl.enabled
            if key == ord("c"):
                swap_rgb = not swap_rgb
            if key == ord("s"):
                out = Path.home() / "Pictures" / f"policy_debug_{time.strftime('%H-%M-%S')}.png"
                out.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out), canvas)
                print("gespeichert:", out)
    finally:
        (DEBUG_DIR / "enabled").unlink(missing_ok=True)
        if not args.save_once:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
