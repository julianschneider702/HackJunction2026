"""Brick sorter: find marked (damaged) bricks and sort them into the bowls.

One cycle:
  1. Webcam looks into the collection box for a marked brick (black marker stroke/dot).
     If found and `direct_pick_task` is set, the policy is asked for that brick; otherwise a random pick.
  2. The policy (SmolVLA in Studio) picks a brick. As soon as the gripper has closed, the script takes over,
     lifts to the transit pose and checks with the wrist camera that a brick is held, and whether it shows a mark.
  3. No mark yet: carry it to the inspect pose in front of the webcam and turn the wrist through all angles;
     the webcam looks for a mark.
  4. Marked -> blue bowl, clean -> yellow bowl. The bowls are tracked live by the RealSense, so they may be
     moved, even during the run; the drop pose comes from the taught drop points (calibrate.py).

Run (Studio running, session in the web UI closed, calibrate.py done):
    station/studio-python sorter/sort_bricks.py
Vision test without moving anything (Studio web UI session open, any mode):
    ... sort_bricks.py --test
Keys: SPACE pause/continue   q stop (puts a held brick back into the box)   ESC emergency hold
"""

from __future__ import annotations

import argparse
import csv
import glob
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
from common import (LOG_DIR, PlaceMap, colour_name, joint_limits, load_calibration, load_config,  # noqa: E402
                    pose, stamp)
from studio import GRIPPER, Arm, MotionError, ReadOnlySession, Session, StudioError, find_model  # noqa: E402

WIN = "Stein-Sortierer"


class Stop(Exception):
    pass


class Shared:
    """What the worker tells the window."""

    def __init__(self):
        self.step = "Start"
        self.lines: list[str] = []
        self.marks: dict[str, tuple[float, list]] = {}  # camera -> (time, strokes)
        self.counts = {"gut": 0, "defekt": 0, "leer": 0}
        self.paused = False
        self.stop = False
        self.emergency = False
        self.colour = (255, 255, 255)

    def log(self, msg: str):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        self.lines.append(line)


class Sorter:
    def __init__(self, cfg, cal, sess, arm, tracker, vcfg, sh: Shared):
        self.cfg, self.cal, self.sess, self.arm, self.tracker, self.vcfg, self.sh = cfg, cal, sess, arm, tracker, vcfg, sh
        self.place_map = PlaceMap(cal["place_points"], cfg["place"]["hull_margin_px"])
        self.home = pose(cal, "home")
        self.transit = pose(cal, "transit")
        self.inspect_pose = pose(cal, "inspect")
        self.open_g = cfg["gripper"]["open"]
        self.close_g = cfg["gripper"]["close"]
        self.empty_ref = None
        if cal["empty_ref"] and os.path.exists(cal["empty_ref"]):
            self.empty_ref = cv2.cvtColor(cv2.imread(cal["empty_ref"]), cv2.COLOR_BGR2RGB)
        self.holding = False
        os.makedirs(LOG_DIR, exist_ok=True)
        self.run_id = stamp()
        self.csv_path = os.path.join(LOG_DIR, f"run_{self.run_id}.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["zeit", "nr", "ergebnis", "gefunden_durch", "schale_px", "notiz"])
        if cal["webcam_roi_tray"] is None:
            sh.log("Hinweis: kein Box-Bereich (calibrate.py, Taste m) - Schritt 1 wird übersprungen")
        if cal["webcam_roi_inspect"] is None:
            sh.log("Hinweis: kein Prüfbereich (calibrate.py, Taste r) - Webcam prüft das ganze Bild")
        if self.empty_ref is None and cal["empty_gripper"] is None:
            sh.log("Hinweis: keine Leer-Referenz (calibrate.py, Taste t) - Greif-Prüfung nimmt immer 'Stein da' an")

    # ------------------------------------------------------------ helpers
    def check(self):
        while self.sh.paused and not self.sh.stop:
            time.sleep(0.1)
        if self.sh.stop or self.sh.emergency:
            raise Stop()

    def step(self, text, colour=(255, 255, 255)):
        self.sh.step, self.sh.colour = text, colour
        self.sh.log(text)

    def with_g(self, p, g):
        q = np.array(p, dtype=np.float64).copy()
        q[GRIPPER] = g
        return q

    def move(self, target, speed_key="move", holding=None):
        self.check()
        holding = self.holding if holding is None else holding
        self.arm.move(target, speed=self.cfg["speed"][speed_key], check_gripper=not holding, tol=5.0)

    def evidence(self, name, rgb, strokes):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vision.draw_strokes(bgr, strokes)
        cv2.imwrite(os.path.join(LOG_DIR, f"{self.run_id}_{time.strftime('%H%M%S')}_{name}.jpg"), bgr)

    def strokes_in(self, rgb, roi=None, mask=None):
        r = None if roi is None else (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
        found = vision.find_strokes(rgb, self.vcfg, r)
        if mask is not None:
            found = [s for s in found if mask[int(min(s.centre[1], mask.shape[0] - 1)), int(min(s.centre[0], mask.shape[1] - 1))]]
        return found

    def write_csv(self, nr, result, source, px, note=""):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%H:%M:%S"), nr, result, source,
                                    "" if px is None else f"{px[0]:.0f},{px[1]:.0f}", note])

    # ------------------------------------------------------------ steps
    def go_home_open(self):
        """Home above the box with the gripper open; a held brick falls back into the box."""
        self.step("Zurueck nach Home, Greifer auf")
        cur = self.arm.state()
        # keep the gripper as it is until above the box, so nothing is dropped on the way
        g = self.close_g if self.holding else self.arm.last_command()[GRIPPER]
        if np.abs(cur[:5] - self.home[:5]).max() > 25:
            self.move(self.with_g(self.transit, g))
        self.move(self.with_g(self.home, g), holding=True)
        self.arm.gripper(self.open_g, wait=0.6)
        self.holding = False

    def scan_box(self):
        """Step 1: marked brick visible in the box? Returns its colour name or None."""
        roi = self.cal["webcam_roi_tray"]
        if roi is None:
            return None
        frames = self.arm.frames("webcam", self.cfg["scan"]["frames"], gap=0.1)
        hits = [self.strokes_in(f, roi) for f in frames]
        pts = [s.centre for h in hits for s in h]
        if not pts:
            return None
        # the same spot in enough frames
        best = max(pts, key=lambda p: sum(any(np.hypot(s.centre[0] - p[0], s.centre[1] - p[1]) < self.cfg["scan"]["same_place_px"]
                                               for s in h) for h in hits))
        n = sum(any(np.hypot(s.centre[0] - best[0], s.centre[1] - best[1]) < self.cfg["scan"]["same_place_px"] for s in h) for h in hits)
        if n < self.cfg["scan"]["hits_needed"]:
            return None
        i = max(range(len(hits)), key=lambda k: len(hits[k]))
        self.sh.marks["webcam"] = (time.time(), hits[i])
        self.evidence("box_markiert", frames[i], hits[i])
        s = min(hits[i], key=lambda s: np.hypot(s.centre[0] - best[0], s.centre[1] - best[1]))
        return self._colour_around(frames[i], s)

    def _colour_around(self, rgb, s):
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.drawContours(mask, [s.contour.astype(np.int32)], -1, 1, -1)
        ring = cv2.dilate(mask, np.ones((15, 15), np.uint8)).astype(bool) & ~mask.astype(bool)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[ring]
        return colour_name(float(np.median(hsv[:, 0])), float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2])))

    def pick(self, task) -> bool:
        """Step 2a: the policy picks; returns True once the gripper has closed (on something)."""
        self.step(f"Greifen mit dem Modell: \"{task}\"", (255, 220, 120))
        pc = self.cfg["pick"]
        closed_cmd = self.cfg["gripper"]["policy_closed_cmd"]
        self.sess.set_control({"type": "model", "task": task, "hz": pc["hz"]})
        self.arm.control_active = True
        grasped = False
        try:
            t0 = time.monotonic()
            settled_since, prev = None, None
            while time.monotonic() - t0 < pc["timeout_s"]:
                self.check()
                st = self.arm.state()
                cmd = self.arm.last_command()
                g = st[GRIPPER]
                if cmd[GRIPPER] < closed_cmd and g < 15 and prev is not None and abs(g - prev) < 0.5:
                    settled_since = settled_since or time.monotonic()
                    if time.monotonic() - settled_since >= pc["settle_s"]:
                        grasped = True
                        break
                else:
                    settled_since = None
                prev = g
                time.sleep(1 / 30)
            if grasped:
                time.sleep(pc["after_grasp_s"])   # let the policy lift a little, as in the demos
        finally:
            self.sess.set_control(None)
            self.arm.control_active = False
            self.arm.hold(gripper=self.close_g if grasped else None)
        if grasped:
            self.holding = True
        else:
            self.sh.log("Modell hat nicht gegriffen (Zeit abgelaufen)")
        return grasped

    def lift_to_transit(self):
        self.step("Anheben und zur Transit-Pose")
        cur = self.arm.state()
        up = self.with_g(cur, self.close_g)
        up[1] += self.cfg["pick"]["lift_shoulder"]
        self.move(up, "carry")
        self.move(self.with_g(self.transit, self.close_g), "carry")
        time.sleep(0.3)

    def is_holding(self) -> bool:
        st = self.arm.state()
        votes, notes = [], []
        if self.cal["empty_gripper"] is not None:
            g_hold = st[GRIPPER] > self.cal["empty_gripper"] + self.cfg["gripper"]["holding_margin"]
            votes.append(g_hold)
            notes.append(f"Greifer {st[GRIPPER]:.1f} (leer {self.cal['empty_gripper']:.1f})")
        if self.empty_ref is not None:
            d = np.median([vision.grip_change(self.empty_ref, f, self.vcfg) for f in self.arm.frames("wrist", 3)])
            noise = self.cal["grip_noise"] or 2.0
            hold = self.cal["grip_hold_diff"]
            thr = noise + 0.5 * (hold - noise) if hold and hold > 2.5 * noise else max(3 * noise, 8.0)
            votes.append(d > thr)
            notes.append(f"Bild {d:.1f} (Grenze {thr:.1f})")
        holding = any(votes) if votes else True
        self.sh.log(("Stein im Greifer" if holding else "Greifer LEER") + ("  [" + ", ".join(notes) + "]" if notes else ""))
        return holding

    def brick_mask(self, rgb):
        """Where the held brick is in the wrist image: what differs from the empty-gripper reference."""
        if self.empty_ref is None:
            return None
        a = cv2.GaussianBlur(cv2.cvtColor(self.empty_ref, cv2.COLOR_RGB2GRAY), (9, 9), 0).astype(np.int16)
        b = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 9), 0).astype(np.int16)
        m = (np.abs(a - b) > self.cfg["wrist_check"]["brick_diff"]).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return cv2.dilate((lab == big).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)

    def wrist_check(self) -> bool:
        """Step 2b: mark visible in the wrist camera while holding?"""
        self.step("Wrist-Kamera: Strich am Stein?")
        wc = self.cfg["wrist_check"]
        frames = self.arm.frames("wrist", wc["frames"], gap=0.1)
        hits = [self.strokes_in(f, mask=self.brick_mask(f)) for f in frames]
        n = sum(1 for h in hits if h)
        if n:
            i = max(range(len(hits)), key=lambda k: len(hits[k]))
            self.sh.marks["wrist"] = (time.time(), hits[i])
            self.evidence(f"wrist_{n}von{len(frames)}", frames[i], hits[i])
        return n >= wc["hits_needed"]

    def inspect(self) -> bool:
        """Step 3: turn the brick in front of the webcam."""
        self.step("Pruefung vor der Webcam: Stein drehen", (120, 220, 255))
        ic = self.cfg["inspect"]
        self.move(self.with_g(self.inspect_pose, self.close_g), "carry")
        # sweep from the end nearest to the inspect roll to the other end (no long detours)
        start = self.inspect_pose[4]
        rolls = sorted(ic["roll"], reverse=abs(start - max(ic["roll"])) < abs(start - min(ic["roll"])))
        roi = self.cal["webcam_roi_inspect"]
        for flex in ic["flex_offsets"]:
            for roll in rolls:
                target = self.with_g(self.inspect_pose, self.close_g)
                target[3] += flex
                target[4] = roll
                self.move(target, "inspect")
                time.sleep(ic["settle_s"])
                frames = self.arm.frames("webcam", ic["frames"], gap=0.06)
                hits = [self.strokes_in(f, roi) for f in frames]
                n = sum(1 for h in hits if h)
                if n:
                    i = max(range(len(hits)), key=lambda k: len(hits[k]))
                    self.sh.marks["webcam"] = (time.time(), hits[i])
                if n >= ic["hits_needed"]:
                    self.evidence(f"inspect_roll{roll}", frames[i], hits[i])
                    return True
        return False

    def bowl_pixel(self, colour):
        """Wait until the bowl is seen and inside the taught area; returns (pixel, joints)."""
        pc = self.cfg["place"]
        t0 = time.monotonic()
        warned = False
        while True:
            self.check()
            t = self.tracker.things[colour]
            reason = None
            if not t.fresh(time.monotonic(), pc["bowl_max_age_s"]):
                reason = f"{colour}-Schale nicht sichtbar"
            else:
                try:
                    return t.centre, self.place_map.joints(t.centre)
                except ValueError as e:
                    reason = str(e)
            if not warned:
                self.sh.log(f"Warte: {reason}")
                warned = True
            if time.monotonic() - t0 > pc["wait_bowl_s"]:
                self.sh.log(f"! {reason} - PAUSE. Schale richten, dann LEERTASTE")
                self.sh.paused = True
                t0 = time.monotonic()
                warned = False
            time.sleep(0.2)

    def place(self, colour):
        name = "GELB (gut)" if colour == "yellow" else "BLAU (defekt)"
        self.step(f"Ablegen in {name}", (0, 220, 255) if colour == "yellow" else (255, 140, 0))
        px, j = self.bowl_pixel(colour)
        self.move(self.with_g(self.transit, self.close_g), "carry")
        above = self.with_g(self.transit, self.close_g)
        above[0] = j[0]
        self.move(above, "carry")
        px2, j2 = self.bowl_pixel(colour)    # it may have moved meanwhile
        if np.hypot(px2[0] - px[0], px2[1] - px[1]) > self.cfg["place"]["replan_px"]:
            self.sh.log("Schale wurde verschoben - neu geplant")
            px, j = px2, j2
            above[0] = j[0]
            self.move(above, "carry")
        self.move(self.with_g(np.r_[j, 0.0], self.close_g), "carry")
        self.arm.gripper(self.open_g, wait=0.6)
        self.holding = False
        self.move(self.with_g(above, self.open_g))
        return px

    # ------------------------------------------------------------ main loop
    def run(self):
        nr, empty_in_row = 0, 0
        while nr < self.cfg["max_bricks"]:
            try:
                self.check()
                self.go_home_open()
                colour = self.scan_box()
                task = self.cfg["pick_task"]
                if colour:
                    if self.cfg["direct_pick_task"]:
                        task = self.cfg["direct_pick_task"].format(colour=colour)
                        self.sh.log(f"Schritt 1: markierter {colour} Stein in der Box -> gezielt greifen")
                    else:
                        self.sh.log(f"Schritt 1: markierter {colour} Stein in der Box gesehen "
                                    "(gezieltes Greifen braucht direct_pick_task + Training) -> zufaellig")
                attempts = 0
                while not self.pick(task):
                    attempts += 1
                    self.go_home_open()
                    if attempts >= self.cfg["pick"]["max_attempts"]:
                        raise RuntimeError("Modell greift nicht - PAUSE")
                self.lift_to_transit()
                if not self.is_holding():
                    self.sh.counts["leer"] += 1
                    empty_in_row += 1
                    self.write_csv(nr + 1, "leer", "", None)
                    if empty_in_row >= self.cfg["max_empty_in_row"]:
                        self.go_home_open()
                        self.sh.log(f"! {empty_in_row}x leer gegriffen - Box leer? LEERTASTE = weiter, q = Ende")
                        self.sh.paused = True
                        empty_in_row = 0
                    continue
                empty_in_row = 0
                nr += 1
                found_by = None
                if self.wrist_check():
                    found_by = "wrist"
                elif self.inspect():
                    found_by = "webcam-pruefung"
                defect = found_by is not None
                if defect:
                    self.step(f"Stein {nr}: DEFEKT (Strich gefunden: {found_by})", (60, 60, 255))
                else:
                    self.step(f"Stein {nr}: GUT (kein Strich von allen Seiten)", (80, 220, 80))
                px = self.place("blue" if defect else "yellow")
                self.sh.counts["defekt" if defect else "gut"] += 1
                self.write_csv(nr, "defekt" if defect else "gut", found_by or "", px, f"Schritt1={colour or '-'}")
            except Stop:
                break
            except (MotionError, StudioError, RuntimeError) as e:
                self.sh.log(f"! {e} - PAUSE. Pruefen, dann LEERTASTE (Stein wird in die Box zurueckgelegt)")
                if self.arm.control_active:
                    try:
                        self.sess.set_control(None)
                    except Exception:  # noqa: BLE001
                        pass
                    self.arm.control_active = False
                try:
                    self.arm.hold(gripper=self.close_g if self.holding else None)
                except StudioError:
                    pass
                self.sh.paused = True
                try:
                    self.check()
                    self.go_home_open()
                except Stop:
                    break
                except Exception as e2:  # noqa: BLE001
                    self.sh.log(f"! Zurueckfahren fehlgeschlagen: {e2}")
        # leaving: put a held brick back, unless it was an emergency stop
        if not self.sh.emergency:
            try:
                self.sh.stop = False
                self.go_home_open()
            except Exception as e:  # noqa: BLE001
                self.sh.log(f"! Parken fehlgeschlagen: {e}")
        self.step(f"Fertig: {self.sh.counts['gut']} gut, {self.sh.counts['defekt']} defekt. Log: {self.csv_path}")


# ---------------------------------------------------------------- window
def draw_frame(arm_like, tracker, cal, vcfg, sh: Shared, live_detect: bool):
    now = time.monotonic()
    frames = {}
    for cam in ("realsense", "webcam", "wrist"):
        try:
            f = arm_like.s.store.read_observation(cam) if hasattr(arm_like, "s") else arm_like.store.read_observation(cam)
            frames[cam] = None if f is None or now - f.timestamp > 2 else np.array(f.value, copy=True)
        except Exception:  # noqa: BLE001
            frames[cam] = None
    if frames["realsense"] is not None:
        tracker.update(frames["realsense"], now)
    rs, wc, wr = (ui.to_bgr(frames[c]) for c in ("realsense", "webcam", "wrist"))
    tracker.draw(rs, now)
    ui.draw_roi(wc, cal["webcam_roi_inspect"], "Pruefbereich", (0, 255, 255))
    ui.draw_roi(wc, cal["webcam_roi_tray"], "Box", (255, 255, 0))
    if live_detect:
        for cam, bgr in (("webcam", wc), ("wrist", wr)):
            if frames[cam] is not None:
                st = vision.find_strokes(frames[cam], vcfg)
                vision.draw_strokes(bgr, st)
                if st:
                    sh.marks[cam] = (time.time(), st)
    else:
        for cam, bgr in (("webcam", wc), ("wrist", wr)):
            t, st = sh.marks.get(cam, (0, []))
            if time.time() - t < 4:
                vision.draw_strokes(bgr, st)
    return rs, wc, wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="nur Erkennung anzeigen, nichts bewegen (Studio-UI-Sitzung offen)")
    args = ap.parse_args()
    cfg, cal = load_config(), load_calibration()
    vcfg = vision.config(cfg["vision"])
    tracker = vision.SceneTracker(vcfg)
    sh = Shared()
    ui.window(WIN)

    if args.test:
        ro, ro_name = None, None
        sh.log("Testmodus: nur Erkennung, keine Bewegung. q = Ende")
        try:
            while True:
                # (re)attach to whatever session the Studio web UI has open
                names = glob.glob("/dev/shm/pai-*")
                newest = max(names, key=os.path.getmtime) if names else None
                if newest != ro_name:
                    if ro is not None:
                        ro.close()
                        ro = None
                        sh.log("Studio-Sitzung beendet")
                    if newest is not None:
                        try:
                            ro = ReadOnlySession()
                            sh.log("Mit Studio-Sitzung verbunden")
                        except Exception as e:  # noqa: BLE001
                            sh.log(f"Verbinden fehlgeschlagen: {e}")
                    ro_name = newest
                if ro is None:
                    blank = ui.to_bgr(None)
                    cv2.imshow(WIN, ui.compose({"RealSense": blank, "Webcam": blank, "Wrist": blank},
                                               ["In Studio die Umgebung oeffnen (Robots -> Umgebung starten / Run model)."]
                                               + sh.lines[-4:], "TESTMODUS - warte auf Studio ..."))
                    if ui.key(500) == ord("q"):
                        break
                    continue
                rs, wc, wr = draw_frame(ro, tracker, cal, vcfg, sh, live_detect=True)
                th = tracker.things
                lines = [" | ".join(f"{k}: " + ("-" if t.centre is None else f"{t.centre[0]:.0f},{t.centre[1]:.0f} sichtbar {t.visible:.0%}")
                                    for k, t in th.items())] + sh.lines[-4:]
                cv2.imshow(WIN, ui.compose({"RealSense": rs, "Webcam": wc, "Wrist": wr}, lines, "TESTMODUS - Erkennung live"))
                if ui.key(60) == ord("q"):
                    break
        finally:
            if ro is not None:
                ro.close()
            cv2.destroyAllWindows()
        return

    missing = [p for p in ("transit", "inspect") if p not in cal["poses"]]
    if missing or len(cal["place_points"]) < 3:
        sys.exit(f"Kalibrierung unvollständig (fehlt: {missing or ''} Ablagepunkte {len(cal['place_points'])}/3+). "
                 "Erst calibrate.py ausführen.")
    sh.log("Verbinde mit Studio ...")
    sess = Session(cfg["project_id"], log=sh.log)
    sess.load_environment(cfg["environment_id"])
    model = find_model(cfg["model"], cfg["project_id"])
    sh.log(f"Lade Modell {model['name']} ({cfg['backend']}, {cfg['device']}) ...")
    sess.load_model(model["id"], cfg["backend"], cfg["device"])
    arm = Arm(sess, joint_limits(cal, cfg), max_speed=cfg["speed"]["move"])
    sorter = Sorter(cfg, cal, sess, arm, tracker, vcfg, sh)

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
    try:
        while th.is_alive():
            rs, wc, wr = draw_frame(arm, tracker, cal, vcfg, sh, live_detect=False)
            c = sh.counts
            head = ("PAUSE - " if sh.paused else "") + sh.step
            lines = [f"gut {c['gut']}   defekt {c['defekt']}   leer gegriffen {c['leer']}      "
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
                sh.emergency, sh.paused = True, False
                sh.log("NOT-HALT")
                try:
                    if arm.control_active:
                        sess.set_control(None)
                        arm.control_active = False
                    arm.hold()
                except Exception:  # noqa: BLE001
                    pass
        th.join(timeout=1)
        # keep the window open with the final message until q
        while True:
            rs, wc, wr = draw_frame(arm, tracker, cal, vcfg, sh, live_detect=False)
            cv2.imshow(WIN, ui.compose({"RealSense": rs, "Webcam": wc, "Wrist": wr}, sh.lines[-8:], sh.step + "  (q schliesst)"))
            if ui.key(80) == ord("q"):
                break
    finally:
        sess.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
