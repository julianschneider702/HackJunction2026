"""Policy picks and places; the script checks each brick in between.

One cycle:
  1. The policy (SmolVLA in Studio) picks a brick with the task "put the brick in the yellow bowl".
  2. As soon as the brick is lifted, the policy is stopped and the arm holds still.
  3. Wrist camera + anomaly model (Anomalib): bad -> yellow bowl.
  4. Not bad: the arm carries the brick in front of the webcam (inspect pose) and turns the wrist through
     all angles. The second anomaly model checks every angle (until it exists: the stroke detector).
     Bad -> yellow, otherwise blue. The arm then returns to where the policy was stopped.
  5. The policy takes over again with "put the brick in the <colour> bowl" and puts it there.
     The script stops the policy once the gripper has opened, drives home, next brick.

Run (Studio running, the session in the Studio web UI closed):
    station/studio-python sorter/policy_sort.py
Test the check without the policy (you put the brick into the gripper by hand, nothing is placed):
    ... policy_sort.py --manual
Keys: SPACE pause/continue   q stop (a held brick goes back into the box)   ESC emergency hold
Dashboard (frontend/hub.py, if running): every brick with score, image and heatmap; its Stop button pauses
the arm like SPACE, its Resume button continues.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402
import vision  # noqa: E402
from anomaly import Anomaly, AnomalyError  # noqa: E402
from common import LOG_DIR, joint_limits, load_calibration, load_config, pose, stamp  # noqa: E402
from dashboard import Dashboard, heatmap_jpeg, jpeg  # noqa: E402
from sort_bricks import Shared, Stop, draw_frame  # noqa: E402
from studio import GRIPPER, Arm, MotionError, Session, StudioError, find_model  # noqa: E402

WIN = "Policy-Sortierer"


class Paused(RuntimeError):
    """Pause while the policy was driving: the policy is stopped, the cycle starts over after the pause."""


class PolicySorter:
    def __init__(self, cfg, cal, sess, arm, anomaly: Anomaly, vcfg, sh: Shared, manual: bool, dash=None):
        self.cfg, self.cal, self.sess, self.arm, self.an, self.vcfg, self.sh = cfg, cal, sess, arm, anomaly, vcfg, sh
        self.dash = dash if dash is not None else Dashboard({"enabled": False}, log=lambda m: None)
        self._dash_stop = False
        self._cycle_id = None
        self.regrasps = 0
        # (score, rgb, anomaly map) per camera, for the dashboard. score = the k-th highest frame score of a check,
        # k = hits needed, so "score > threshold" is exactly the decision
        self.best_wrist = None
        self.best_webcam = None
        self.pc, self.ac = cfg["policy_sort"], cfg["anomaly"]
        self.manual = manual
        self.home = pose(cal, "home")
        self.inspect_pose = pose(cal, "inspect")
        self.open_g, self.close_g = cfg["gripper"]["open"], cfg["gripper"]["close"]
        self.holding = False
        self.nr = 0
        self.run_id = stamp()
        self.img_dir = os.path.join(LOG_DIR, f"policy_{self.run_id}")
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(LOG_DIR, f"policy_{self.run_id}.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["zeit", "nr", "ergebnis", "schale", "gefunden_durch", "wrist_scores",
                                    "webcam_max", "notiz"])
        if not self.an.has("webcam"):
            sh.log("Hinweis: kein Webcam-Anomalie-Modell - Prüfung vor der Webcam nutzt den Strich-Detektor")
        if cal["webcam_roi_inspect"] is None:
            sh.log("Hinweis: kein Prüfbereich (calibrate.py, Taste r) - Webcam prüft das ganze Bild")

    # ------------------------------------------------------------ helpers
    def dashboard_control(self):
        """Stop / Resume buttons in the dashboard act like SPACE."""
        req = self.dash.stop_requested
        if req and not self._dash_stop:
            self._dash_stop = True
            if not self.sh.paused:
                self.sh.paused = True
                self.sh.log("Stop aus dem Dashboard - PAUSE (dort Resume = weiter)")
        elif not req and self._dash_stop:
            self._dash_stop = False
            if self.sh.paused:
                self.sh.paused = False
                self.sh.log("Weiter (Dashboard)")

    def check(self):
        self.dashboard_control()
        if self.sh.paused and self.arm.control_active and not self.sh.emergency:
            # never wait while the policy still drives the arm
            self.stop_policy(gripper=self.arm.last_command()[GRIPPER])
            raise Paused("Pause während die Policy fuhr - Policy angehalten")
        if self.sh.paused and not self.sh.stop:
            self.dash.robot("stopped", "Pause")
            while self.sh.paused and not self.sh.stop and not self.sh.emergency:
                self.dashboard_control()
                time.sleep(0.1)
            if not self.sh.stop and not self.sh.emergency:
                self.dash.robot("running")
        if self.sh.stop or self.sh.emergency:
            raise Stop()

    def step(self, text, colour=(255, 255, 255)):
        self.sh.step, self.sh.colour = text, colour
        self.sh.log(text)

    def with_g(self, p, g):
        q = np.array(p, dtype=np.float64).copy()
        q[GRIPPER] = g
        return q

    def move(self, target, speed_key="move"):
        self.check()
        self.arm.move(target, speed=self.cfg["speed"][speed_key], check_gripper=not self.holding, tol=5.0)

    def save(self, name, rgb):
        if self.ac["save_images"]:
            cv2.imwrite(os.path.join(self.img_dir, f"{self.nr:03d}_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def stop_policy(self, gripper=None):
        if self.arm.control_active:
            try:
                self.sess.set_control(None)
            finally:
                self.arm.control_active = False
        self.arm.hold(gripper=gripper)

    def cycle_id(self) -> str:
        """Same id for the whole cycle of one brick (set when the pick starts)."""
        return self._cycle_id or f"{self.run_id}-{self.nr + 1:03d}"

    @property
    def anomaly_device(self) -> str:
        return getattr(self.an, "device_name", None) or "GPU"

    def write_csv(self, result, bowl, found_by, wrist, webcam_max, note=""):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%H:%M:%S"), self.nr, result, bowl, found_by,
                                    " ".join(f"{s:.3f}" for s in wrist),
                                    "" if webcam_max is None else f"{webcam_max:.3f}", note])

    # ------------------------------------------------------------ steps
    def go_home_open(self):
        """Home above the box, gripper open. A held brick is only let go there (falls back into the box)."""
        self.step("Nach Home, Greifer auf")
        g = self.close_g if self.holding else self.arm.last_command()[GRIPPER]
        cur = self.arm.state()
        if self.holding and np.abs(cur[:5] - self.home[:5]).max() > 25:
            up = self.with_g(cur, g)
            up[1] += self.pc["raise_before_inspect"]
            self.move(up, "carry")
        self.move(self.with_g(self.home, g), "carry" if self.holding else "move")
        self.arm.gripper(self.open_g, wait=0.6)
        self.holding = False

    def pick_and_lift(self):
        """Policy picks; stopped as soon as the brick is lifted. Returns the pose it stopped in, or None."""
        task = self.pc["pick_task"]
        self.step(f"Modell greift: \"{task}\"", (255, 220, 120))
        self.dash.phase("execute", cycle_id=self.cycle_id(), detail="Policy greift")
        pk = self.cfg["pick"]
        closed_cmd = self.cfg["gripper"]["policy_closed_cmd"]
        self.sess.set_control({"type": "model", "task": task, "hz": pk["hz"]})
        self.arm.control_active = True
        lifted = False
        try:
            t0 = time.monotonic()
            settled_since, prev, grasp_t, lift0 = None, None, None, None
            while time.monotonic() - t0 < pk["timeout_s"]:
                self.check()
                st, cmd = self.arm.state(), self.arm.last_command()
                g = st[GRIPPER]
                if grasp_t is None:
                    if cmd[GRIPPER] < closed_cmd and g < 15 and prev is not None and abs(g - prev) < 0.5:
                        settled_since = settled_since or time.monotonic()
                        if time.monotonic() - settled_since >= pk["settle_s"]:
                            grasp_t, lift0 = time.monotonic(), st[1]
                            self.sh.log(f"Gegriffen (Greifer {g:.1f}) - warte aufs Anheben")
                    else:
                        settled_since = None
                else:
                    if cmd[GRIPPER] > closed_cmd + 10:      # policy opened again: new attempt
                        self.sh.log("Modell hat wieder geöffnet - greift neu")
                        grasp_t, settled_since = None, None
                    elif lift0 - st[1] >= self.pc["lift_delta"] or \
                            time.monotonic() - grasp_t >= self.pc["max_after_grasp_s"]:
                        lifted = True
                        break
                prev = g
                time.sleep(1 / 60)
        finally:
            self.stop_policy(gripper=self.close_g if lifted else None)
        if not lifted:
            self.sh.log("Modell hat nicht gegriffen (Zeit abgelaufen)")
            return None
        self.holding = True
        time.sleep(0.3)
        return self.arm.state()

    def is_holding(self) -> bool:
        g = self.arm.state()[GRIPPER]
        if self.cal.get("empty_gripper") is not None:
            limit = self.cal["empty_gripper"] + self.cfg["gripper"]["holding_margin"]
        else:
            limit = self.pc["empty_gripper_max"]
        holding = g > limit
        self.sh.log(f"{'Stein im Greifer' if holding else 'Greifer LEER'} (Greifer {g:.1f}, leer bis {limit:.1f})")
        return holding

    def wrist_check(self) -> tuple[bool, list[float]]:
        self.step("Wrist-Kamera: Anomalie?")
        self.dash.phase("detect", cycle_id=self.cycle_id(), detail="Wrist-Kamera")
        frames = self.arm.frames("wrist", self.ac["wrist_frames"], gap=0.1)
        scores, maps = self.an.score("wrist", frames, maps=True)
        self.dash.latency("anomaly", self.an.last_ms, device=self.anomaly_device)
        i = max(range(len(scores)), key=lambda k: scores[k])
        kth = sorted(scores, reverse=True)[min(self.ac["wrist_hits"], len(scores)) - 1]
        self.best_wrist = (kth, frames[i], None if maps is None else maps[i])
        for i, (f, s) in enumerate(zip(frames, scores)):
            self.save(f"wrist{i}_{s:.3f}", f)
        hits = sum(s > self.ac["wrist_threshold"] for s in scores)
        self.sh.log("Wrist-Scores: " + ", ".join(f"{s:.2f}" for s in scores)
                    + f"  (Grenze {self.ac['wrist_threshold']}, {hits}/{len(scores)} drüber)")
        return hits >= self.ac["wrist_hits"], scores

    def webcam_bad(self, frames, roll) -> tuple[int, float]:
        """How many frames look bad, and the highest score (stroke detector: 1.0 per hit)."""
        roi = self.cal["webcam_roi_inspect"]
        if self.an.has("webcam"):
            imgs = frames
            if roi is not None and self.ac["webcam_use_roi"]:
                x, y, w, h = roi
                imgs = [f[y:y + h, x:x + w] for f in frames]
            scores, maps = self.an.score("webcam", imgs, maps=True)
            self.dash.latency("anomaly", self.an.last_ms, device=self.anomaly_device)
            for i, (f, s) in enumerate(zip(frames, scores)):
                self.save(f"webcam_roll{roll:+04d}_{i}_{s:.3f}", f)
            i = max(range(len(scores)), key=lambda k: scores[k])
            kth = sorted(scores, reverse=True)[min(self.cfg["inspect"]["hits_needed"], len(scores)) - 1]
            if self.best_webcam is None or kth >= self.best_webcam[0]:
                self.best_webcam = (kth, imgs[i], None if maps is None else maps[i])
            return sum(s > self.ac["webcam_threshold"] for s in scores), kth
        r = None if roi is None else (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
        hits = [vision.find_strokes(f, self.vcfg, r) for f in frames]
        for i, f in enumerate(frames):
            self.save(f"webcam_roll{roll:+04d}_{i}_{'strich' if hits[i] else 'ok'}", f)
        if any(hits):
            self.sh.marks["webcam"] = (time.time(), max(hits, key=len))
            k = max(range(len(hits)), key=lambda j: len(hits[j]))
            bgr = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR)
            vision.draw_strokes(bgr, hits[k])
            self.best_webcam = (None, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None)
        n = sum(1 for h in hits if h)
        return n, float(n > 0)

    def inspect(self, stop_pose) -> tuple[bool, float]:
        """Carry the brick in front of the webcam, turn it, come back to stop_pose."""
        self.step("Prüfung vor der Webcam: Stein drehen", (120, 220, 255))
        self.dash.phase("detect", cycle_id=self.cycle_id(), detail="Drehung vor der Webcam")
        ic = self.cfg["inspect"]
        up = self.with_g(stop_pose, self.close_g)
        up[1] += self.pc["raise_before_inspect"]
        self.move(up, "carry")
        self.move(self.with_g(self.inspect_pose, self.close_g), "carry")
        start = self.inspect_pose[4]
        rolls = sorted(ic["roll"], reverse=abs(start - max(ic["roll"])) < abs(start - min(ic["roll"])))
        bad, best = False, 0.0
        for flex in ic["flex_offsets"]:
            for roll in rolls:
                target = self.with_g(self.inspect_pose, self.close_g)
                target[3] += flex
                target[4] = roll
                self.move(target, "inspect")
                time.sleep(ic["settle_s"])
                frames = self.arm.frames("webcam", ic["frames"], gap=0.06)
                n, top = self.webcam_bad(frames, roll)
                best = max(best, top)
                self.sh.log(f"  Drehung {roll:+d}: {n}/{len(frames)} auffällig (Score {top:.2f})")
                if n >= ic["hits_needed"]:
                    bad = True
                    break
            if bad:
                break
        self.sh.log("Zurück zur Übergabe-Pose")
        self.move(self.with_g(self.inspect_pose, self.close_g), "carry")
        self.move(up, "carry")
        self.move(self.with_g(stop_pose, self.close_g), "carry")
        time.sleep(0.2)
        return bad, best

    def place_with_policy(self, colour):
        task = self.pc["place_task"].format(colour=colour)
        self.step(f"Modell legt ab: \"{task}\"", (0, 220, 255) if colour == "yellow" else (255, 140, 0))
        self.dash.phase("execute", cycle_id=self.cycle_id(), detail=f"Policy legt in {colour} ab")
        self.sess.set_control({"type": "model", "task": task, "hz": self.cfg["pick"]["hz"]})
        self.arm.control_active = True
        released = False
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.pc["place_timeout_s"]:
                self.check()
                if self.arm.state()[GRIPPER] > self.pc["release_gripper"]:
                    released = True
                    end = time.monotonic() + self.pc["after_release_s"]
                    while time.monotonic() < end:
                        self.check()
                        time.sleep(0.05)
                    break
                time.sleep(1 / 30)
        finally:
            self.stop_policy()
        if not released:
            raise RuntimeError(f"Modell hat in {self.pc['place_timeout_s']:.0f}s nicht abgelegt")
        self.holding = False

    def manual_load(self):
        """--manual: you put a brick into the open gripper, SPACE closes it."""
        self.step("Stein zwischen die Greiferfinger halten, dann LEERTASTE", (0, 200, 255))
        self.sh.paused = True
        self.check()
        self.arm.gripper(self.close_g, wait=0.8)
        self.holding = True
        return self.arm.state()

    def report_cycle(self, duration, bad, found_by, wrist, colour):
        """One finished brick to the dashboard: score + threshold of the camera that decided, image, heatmap."""
        webcam_decided = found_by == "webcam" or (not found_by and self.an.has("webcam"))
        if webcam_decided and self.best_webcam is not None:
            score, rgb, amap = self.best_webcam
            thr = self.ac["webcam_threshold"] if self.an.has("webcam") else None   # stroke detector: no score
        else:
            score, rgb, amap = self.best_wrist or (None, None, None)
            thr = self.ac["wrist_threshold"]
            if found_by == "webcam" and self.best_webcam is not None:          # stroke found: show that image
                rgb, amap = self.best_webcam[1], None
        placed = None if self.manual else ("defective" if colour == self.pc["bad_bowl"] else "good")
        self.dash.cycle(duration_s=round(duration, 2), anomaly_score=None if score is None else round(float(score), 4),
                        threshold=thr, decision="defective" if bad else "good", placed=placed,
                        regrasps=self.regrasps, cycle_id=self.cycle_id(),
                        instruction=None if self.manual else self.pc["place_task"].format(colour=colour),
                        image=jpeg(rgb), heatmap=heatmap_jpeg(rgb, amap))

    # ------------------------------------------------------------ main loop
    def run(self):
        empty_in_row, failed_picks = 0, 0
        while self.nr < self.cfg["max_bricks"]:
            try:
                self.check()
                self.go_home_open()
                self.best_wrist = self.best_webcam = None
                self._cycle_id = f"{self.run_id}-{self.nr + 1:03d}"
                t_cycle = time.monotonic()
                stop_pose = self.manual_load() if self.manual else self.pick_and_lift()
                if self.manual:
                    t_cycle = time.monotonic()      # do not count the time the brick was loaded by hand
                if stop_pose is None:
                    self.regrasps += 1
                    failed_picks += 1
                    if failed_picks >= self.cfg["pick"]["max_attempts"]:
                        failed_picks = 0
                        raise RuntimeError("Modell greift nicht")
                    continue
                failed_picks = 0
                if not self.is_holding():
                    self.regrasps += 1
                    self.sh.counts["leer"] += 1
                    empty_in_row += 1
                    self.write_csv("leer", "", "", [], None)
                    if empty_in_row >= self.cfg["max_empty_in_row"]:
                        self.go_home_open()
                        self.sh.log(f"! {empty_in_row}x leer gegriffen - Box leer? LEERTASTE = weiter, q = Ende")
                        self.sh.paused = True
                        empty_in_row = 0
                    continue
                empty_in_row = 0
                self.nr += 1
                found_by, webcam_max = "", None
                bad, wrist = self.wrist_check()
                if bad:
                    found_by = "wrist"
                else:
                    bad, webcam_max = self.inspect(stop_pose)
                    found_by = "webcam" if bad else ""
                colour = self.pc["bad_bowl"] if bad else self.pc["good_bowl"]
                self.dash.phase("reason", cycle_id=self.cycle_id(),
                                detail=f"{'defekt' if bad else 'gut'} ({found_by or 'alle Seiten sauber'}) -> {colour}")
                if bad:
                    self.step(f"Stein {self.nr}: BAD ({found_by}) -> {colour}", (60, 60, 255))
                else:
                    self.step(f"Stein {self.nr}: GUT (von allen Seiten unauffällig) -> {colour}", (80, 220, 80))
                if self.manual:
                    self.sh.log(f"(--manual: würde in {colour} ablegen; Stein fällt zurück in die Box)")
                else:
                    self.place_with_policy(colour)
                self.sh.counts["defekt" if bad else "gut"] += 1
                self.write_csv("bad" if bad else "gut", colour, found_by, wrist, webcam_max,
                               "manual" if self.manual else "")
                self.report_cycle(time.monotonic() - t_cycle, bad, found_by, wrist, colour)
                self.regrasps = 0
            except Stop:
                break
            except (MotionError, StudioError, RuntimeError, AnomalyError) as e:
                self.sh.log(f"! {e} - PAUSE. Prüfen, dann LEERTASTE (ein gehaltener Stein kommt zurück in die Box)")
                if not isinstance(e, Paused):
                    self.dash.robot("error", str(e)[:200])
                try:
                    self.stop_policy(gripper=self.close_g if self.holding else None)
                except Exception:  # noqa: BLE001
                    pass
                self.sh.paused = True
                try:
                    self.check()
                    self.go_home_open()
                except Stop:
                    break
                except Exception as e2:  # noqa: BLE001
                    self.sh.log(f"! Zurückfahren fehlgeschlagen: {e2}")
        if not self.sh.emergency:
            try:
                self.sh.stop, self.sh.paused = False, False
                self.go_home_open()
            except Exception as e:  # noqa: BLE001
                self.sh.log(f"! Parken fehlgeschlagen: {e}")
        c = self.sh.counts
        self.dash.phase("idle")
        self.dash.robot("stopped" if (self.sh.emergency or self._dash_stop) else "idle",
                        "Not-Halt" if self.sh.emergency else None)
        self.step(f"Fertig: {c['defekt']} bad, {c['gut']} gut, {c['leer']} leer. Log: {self.csv_path}")


def emergency_stop(sh, arm, sess):
    sh.emergency, sh.paused = True, False
    sh.log("NOT-HALT")
    try:
        if arm.control_active:
            sess.set_control(None)
            arm.control_active = False
        arm.hold()
    except Exception:  # noqa: BLE001
        pass


def run_headless(th, sh, arm, sess):
    """No window: log lines go to the terminal, ENTER = pause/continue, q+ENTER = stop, Ctrl+C = stop, 2x = emergency."""
    print("Läuft ohne Fenster.  ENTER = Pause/weiter   q + ENTER = Stop   Ctrl+C = Stop (2x = Not-Halt)", flush=True)

    def keys():
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                sh.log("Stop angefordert - fahre in Ruhe zurück")
                sh.stop, sh.paused = True, False
                return
            sh.paused = not sh.paused
            sh.log("Pause" if sh.paused else "Weiter")

    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=keys, daemon=True).start()
    interrupts = 0
    while th.is_alive():                  # sh.log already prints every line to the terminal
        try:
            th.join(timeout=0.2)
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts == 1:
                sh.log("Stop angefordert (Ctrl+C) - fahre in Ruhe zurück. Nochmal Ctrl+C = Not-Halt")
                sh.stop, sh.paused = True, False
            else:
                emergency_stop(sh, arm, sess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", action="store_true",
                    help="ohne Policy: Stein von Hand in den Greifer, nur prüfen, nichts ablegen")
    ap.add_argument("--window", action="store_true",
                    help="Debug-Fenster mit den drei Kameras zeigen (sonst nur Terminal + Dashboard)")
    args = ap.parse_args()
    cfg, cal = load_config(), load_calibration()
    if pose(cal, "inspect") is None:
        sys.exit("Prüf-Pose fehlt (poses/pose_*.json oder calibrate.py, Taste i)")
    vcfg = vision.config(cfg["vision"])
    tracker = vision.SceneTracker(vcfg)
    sh = Shared()
    os.makedirs(LOG_DIR, exist_ok=True)

    an = Anomaly({"wrist": cfg["anomaly"]["wrist_model"], "webcam": cfg["anomaly"]["webcam_model"]}, log=sh.log)
    if not an.has("wrist"):
        sys.exit("kein Wrist-Anomalie-Modell (config.json -> anomaly.wrist_model)")
    sess = None
    try:
        sh.log("Verbinde mit Studio ...")
        sess = Session(cfg["project_id"], log=sh.log)
        sess.load_environment(cfg["environment_id"])
        if not args.manual:
            model = find_model(cfg["model"], cfg["project_id"])
            sh.log(f"Lade Policy {model['name']} ({cfg['backend']}, {cfg['device']}) ...")
            sess.load_model(model["id"], cfg["backend"], cfg["device"])
        arm = Arm(sess, joint_limits(cal, cfg), max_speed=cfg["speed"]["move"])
        dash = Dashboard(cfg["dashboard"], log=sh.log)
        rows = [{"role": "anomaly", "name": os.path.basename(os.path.expanduser(cfg["anomaly"]["wrist_model"])),
                 "device": "GPU", "precision": "FP32", "runtime": "PyTorch (Anomalib)"}]
        if not args.manual:
            rows.insert(0, {"role": "policy", "name": model["name"], "device": "GPU" if cfg["device"] in ("auto", "xpu", "gpu", "GPU") else cfg["device"],
                            "precision": "BF16" if cfg["backend"] == "torch" else None,
                            "runtime": "PyTorch" if cfg["backend"] == "torch" else cfg["backend"]})
        dash.models(rows)
        if dash.stop_requested:
            sh.log("Dashboard: Stop ist noch gedrückt - erst dort Resume, dann geht es los")
        dash.robot("running")
        sorter = PolicySorter(cfg, cal, sess, arm, an, vcfg, sh, args.manual, dash)

        def worker():
            try:
                sorter.run()
            except Exception:  # noqa: BLE001
                sh.log("! Absturz: " + traceback.format_exc().splitlines()[-1])
                traceback.print_exc()
            finally:
                sh.stop = True

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        if not args.window:
            run_headless(th, sh, arm, sess)
            return
        ui.window(WIN)
        while th.is_alive():
            rs, wc, wr = draw_frame(arm, tracker, cal, vcfg, sh, live_detect=False)
            c = sh.counts
            head = ("PAUSE - " if sh.paused else "") + sh.step
            lines = [f"bad {c['defekt']}   gut {c['gut']}   leer gegriffen {c['leer']}      "
                     "LEERTASTE Pause/weiter   q Stop   ESC Not-Halt"] + sh.lines[-7:]
            cv2.imshow(WIN, ui.compose({"RealSense": rs, "Webcam": wc, "Wrist": wr}, lines, head,
                                       (0, 200, 255) if sh.paused else sh.colour))
            k = ui.key(40)
            if k == ord(" "):
                sh.paused = not sh.paused
                sh.log("Pause" if sh.paused else "Weiter")
            elif k == ord("q"):
                sh.log("Stop angefordert - fahre in Ruhe zurück")
                sh.stop, sh.paused = True, False
            elif k == 27:
                emergency_stop(sh, arm, sess)
        th.join(timeout=1)
        while True:
            rs, wc, wr = draw_frame(arm, tracker, cal, vcfg, sh, live_detect=False)
            cv2.imshow(WIN, ui.compose({"RealSense": rs, "Webcam": wc, "Wrist": wr}, sh.lines[-8:],
                                       sh.step + "  (q schliesst)"))
            if ui.key(80) == ord("q"):
                break
    finally:
        try:
            dash.flush()
        except NameError:
            pass
        an.close()
        if sess is not None:
            sess.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
