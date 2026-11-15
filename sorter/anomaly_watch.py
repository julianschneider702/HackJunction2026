"""Watch a run started in the Studio web UI and check every picked brick for anomalies.

Read only: the Studio web UI keeps the arm (its policy and task decide the bowl). This script only
looks: it switches itself on when a run moves the arm, and for every brick the gripper picks up it
scores the wrist camera with the Anomalib model and reports verdict, score, image and cycle time to the
dashboard (frontend/hub.py) and to a small window.

To let the verdict choose the bowl, use policy_sort.py instead (that one drives the arm itself).

    station/studio-python sorter/anomaly_watch.py
Keys in the window: q quit
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402
from anomaly import Anomaly, AnomalyError  # noqa: E402
from common import LOG_DIR, load_config, stamp  # noqa: E402
from dashboard import Dashboard, heatmap_jpeg, jpeg  # noqa: E402
from studio import ACTION, GRIPPER, STATE, ReadOnlySession  # noqa: E402

WIN = "Anomalie-Beobachter"


class Watcher:
    def __init__(self, cfg, an: Anomaly, dash: Dashboard, log):
        self.cfg, self.an, self.dash, self.log = cfg, an, dash, log
        self.ac, self.pc = cfg["anomaly"], cfg["policy_sort"]
        self.closed_cmd = cfg["gripper"]["policy_closed_cmd"]
        self.run_id = stamp()
        self.img_dir = os.path.join(LOG_DIR, f"watch_{self.run_id}")
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(LOG_DIR, f"watch_{self.run_id}.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["zeit", "nr", "ergebnis", "scores", "zyklus_s"])
        self.nr = 0
        self.reset()
        self.last = None            # (rgb, scores, bad, amap) of the last checked brick, for the window
        self.running = False

    def reset(self):
        self.mode = "wait_grasp"    # wait_grasp -> wait_lift -> wait_release
        self.settled_since = None
        self.prev_g = None
        self.grasp_t = None
        self.lift0 = None
        self.cycle_t0 = time.monotonic()
        self.decision = None

    def tick(self, ro: ReadOnlySession, now: float):
        st, act = ro.store.read(STATE), ro.store.read(ACTION)
        if st is None or now - st.timestamp > 1.0:
            return
        s = np.asarray(st.value, dtype=np.float64)
        a = np.asarray(act.value, dtype=np.float64) if act is not None else s
        # a run (model or teleop) writes ACTION all the time; idle sessions do not
        active = act is not None and abs(now - act.timestamp) < 1.5
        if active != self.running:
            self.running = active
            self.log("Run erkannt - Anomalie-Prüfung aktiv" if active else "Run beendet - warte auf den nächsten")
            self.dash.robot("running" if active else "idle")
            self.dash.phase("perceive" if active else "idle", detail="Policy greift" if active else None)
            self.reset()
        if not active:
            return
        g = s[GRIPPER]
        if self.mode == "wait_grasp":
            if a[GRIPPER] < self.closed_cmd and g < 15 and self.prev_g is not None and abs(g - self.prev_g) < 0.5:
                self.settled_since = self.settled_since or now
                if now - self.settled_since >= self.cfg["pick"]["settle_s"]:
                    self.mode, self.grasp_t, self.lift0 = "wait_lift", now, s[1]
                    self.dash.phase("execute", detail="Stein gegriffen, hebt an")
            else:
                self.settled_since = None
        elif self.mode == "wait_lift":
            if a[GRIPPER] > self.closed_cmd + 10:          # opened again without lifting: new attempt
                self.reset()
            elif self.lift0 - s[1] >= self.pc["lift_delta"] or now - self.grasp_t >= self.pc["max_after_grasp_s"]:
                self.check(ro)
                self.mode = "wait_release"
        elif self.mode == "wait_release":
            if g > self.pc["release_gripper"]:
                self.finish(now)
        self.prev_g = g

    def check(self, ro: ReadOnlySession):
        """Right after the lift (still above the box, like the training photos): score a few wrist frames."""
        self.dash.phase("detect", detail="Wrist-Kamera: Anomalie-Prüfung")
        frames, last_ts = [], None
        deadline = time.monotonic() + 1.5
        while len(frames) < self.ac["wrist_frames"] and time.monotonic() < deadline:
            f = ro.store.read_observation("wrist")
            if f is not None and f.timestamp != last_ts:
                frames.append(np.array(f.value, copy=True))
                last_ts = f.timestamp
            time.sleep(0.04)
        if not frames:
            self.log("! keine Bilder von der Wrist-Kamera")
            return
        try:
            scores, maps = self.an.score("wrist", frames, maps=True)
        except AnomalyError as e:
            self.log(f"! Anomalie-Prüfung fehlgeschlagen: {e}")
            return
        self.dash.latency("anomaly", self.an.last_ms, device="GPU")
        need = min(self.ac["wrist_hits"], len(scores))
        kth = sorted(scores, reverse=True)[need - 1]
        bad = kth > self.ac["wrist_threshold"]
        i = max(range(len(scores)), key=lambda k: scores[k])
        self.nr += 1
        for k, (f, sc) in enumerate(zip(frames, scores, strict=False)):
            cv2.imwrite(os.path.join(self.img_dir, f"{self.nr:03d}_wrist{k}_{sc:.3f}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        self.decision = ("defective" if bad else "good", kth, scores, frames[i], None if maps is None else maps[i])
        self.last = (frames[i], scores, bad, None if maps is None else maps[i])
        self.log(f"Stein {self.nr}: {'DEFEKT' if bad else 'gut'}  Scores " + ", ".join(f"{x:.2f}" for x in scores)
                 + f"  (Grenze {self.ac['wrist_threshold']})")
        self.dash.phase("reason", detail=f"{'defekt' if bad else 'gut'} (Score {kth:.2f})")

    def finish(self, now: float):
        duration = now - self.cycle_t0
        if self.decision is not None:
            label, kth, scores, rgb, amap = self.decision
            self.dash.cycle(duration_s=round(duration, 2), anomaly_score=round(float(kth), 4),
                            threshold=self.ac["wrist_threshold"], decision=label,
                            cycle_id=f"watch-{self.run_id}-{self.nr:03d}", image=jpeg(rgb), heatmap=heatmap_jpeg(rgb, amap))
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([time.strftime("%H:%M:%S"), self.nr, "defekt" if label == "defective" else "gut",
                                        " ".join(f"{x:.3f}" for x in scores), f"{duration:.1f}"])
            self.log(f"  abgelegt nach {duration:.1f} s")
        self.dash.phase("perceive", detail="Policy greift")
        self.reset()


def newest_store():
    names = glob.glob("/dev/shm/pai-*")
    return max(names, key=os.path.getmtime) if names else None


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    cfg = load_config()
    lines = []

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        lines.append(line)

    os.makedirs(LOG_DIR, exist_ok=True)
    an = Anomaly({"wrist": cfg["anomaly"]["wrist_model"]}, log=log)
    dash = Dashboard(cfg["dashboard"], log=log)
    dash.models([{"role": "anomaly", "name": os.path.basename(os.path.expanduser(cfg["anomaly"]["wrist_model"])),
                  "device": "GPU", "precision": "FP32", "runtime": "PyTorch (Anomalib)"}])
    w = Watcher(cfg, an, dash, log)
    ro, ro_name = None, None
    ui.window(WIN)
    log("Bereit. In Studio den Run starten - die Prüfung schaltet sich dann von selbst ein. q = Ende")
    try:
        while True:
            now = time.monotonic()
            name = newest_store()
            if name != ro_name:
                if ro is not None:
                    ro.close()
                    ro = None
                    log("Studio-Sitzung beendet")
                if name is not None:
                    try:
                        ro = ReadOnlySession()
                        log("Mit Studio-Sitzung verbunden")
                    except Exception as e:  # noqa: BLE001
                        log(f"Verbinden fehlgeschlagen: {e}")
                ro_name = name
                w.running = False
                w.reset()
            if ro is not None:
                try:
                    w.tick(ro, now)
                except Exception as e:  # noqa: BLE001 - a vanished store must not end the watcher
                    log(f"! {e}")
                    ro, ro_name = None, None
            # window: live wrist image + the last checked brick
            live = None
            if ro is not None:
                f = ro.store.read_observation("wrist")
                live = None if f is None else np.array(f.value, copy=True)
            tiles = {"Wrist live": ui.to_bgr(live)}
            if w.last is not None:
                rgb, scores, bad, _ = w.last
                b = ui.to_bgr(rgb)
                cv2.putText(b, ("DEFEKT " if bad else "GUT ") + f"{sorted(scores, reverse=True)[min(cfg['anomaly']['wrist_hits'], len(scores)) - 1]:.2f}",
                            (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 255) if bad else (80, 220, 80), 3)
                tiles["Letzter Stein"] = b
            head = ("RUN LÄUFT - Prüfung aktiv" if w.running else
                    "Warte auf einen Run in Studio" if ro is not None else "Warte auf Studio-Sitzung")
            cv2.imshow(WIN, ui.compose(tiles, [f"Steine geprüft: {w.nr}   Log: {w.csv_path}"] + lines[-6:], head,
                                       (80, 220, 80) if w.running else (0, 200, 255)))
            if ui.key(30) == ord("q"):
                break
    finally:
        dash.robot("idle")
        dash.phase("idle")
        dash.flush()
        if ro is not None:
            ro.close()
        an.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
