"""Teach the sorter: poses, bowl drop points, gripper references and webcam regions.

Run (Studio must be running; close the session in the Studio web UI first, the robot can only be in one):
    station/studio-python sorter/calibrate.py

You move the follower with the leader arm (teleop). Keys in the window:
  t   TRANSIT pose: high above everything, gripper CLOSED and EMPTY. Also takes the empty-gripper
      reference picture and value (needed to tell whether a brick is held).
  k   at the TRANSIT pose with a BRICK in the gripper: holding sample (sets the threshold)
  i   INSPECT pose: brick right in front of the webcam
  h   HOME pose (start of the training episodes); optional, default = training start pose
  y / b   take the YELLOW / BLUE bowl centre from the RealSense (arm out of the way!)
  SPACE   then save the current arm pose as "drop above that bowl centre" (gripper above the bowl middle)
  u   undo last drop point
  r   draw the webcam region where the held brick appears at the INSPECT pose (hold a brick there)
  m   draw the webcam region of the collection box (for step 1)
  q   quit (saved automatically after every change)
Teach 6-9 drop points: move a bowl to a new spot each time, cover everywhere the bowls may stand.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402
import vision  # noqa: E402
from common import HERE, PlaceMap, joint_limits, load_calibration, load_config, save_calibration  # noqa: E402
from studio import JOINTS, Arm, Session, StudioError  # noqa: E402

WIN = "Sorter Kalibrierung"


def main():
    cfg = load_config()
    cal = load_calibration()
    vcfg = vision.config(cfg["vision"])
    tracker = vision.SceneTracker(vcfg)
    log = []

    def say(msg):
        print(msg)
        log.append(msg)

    say("Verbinde mit Studio ...")
    sess = Session(cfg["project_id"], log=say)
    sess.load_environment(cfg["environment_id"])
    arm = Arm(sess, joint_limits(cal, cfg))
    sess.set_control({"type": "teleop"})
    arm.control_active = True
    say("Teleop aktiv: Follower mit dem Leader-Arm bewegen.")

    pending = None  # (bowl, pixel)
    ui.window(WIN)
    try:
        while True:
            now = time.monotonic()
            frames = {}
            for cam in ("realsense", "webcam", "wrist"):
                try:
                    frames[cam] = arm.frame(cam)
                except StudioError:
                    frames[cam] = None
            if frames["realsense"] is not None:
                tracker.update(frames["realsense"], now)
            rs, wc, wr = (ui.to_bgr(frames[c]) for c in ("realsense", "webcam", "wrist"))
            tracker.draw(rs, now)
            pm = PlaceMap(cal["place_points"], cfg["place"]["hull_margin_px"])
            for p in cal["place_points"]:
                col = (0, 220, 255) if p.get("bowl") == "yellow" else (255, 120, 0)
                cv2.circle(rs, tuple(int(v) for v in p["pixel"]), 5, col, -1)
            if pm.ok and len(pm.hull) >= 3:
                cv2.polylines(rs, [pm.hull.astype(np.int32)], True, (0, 255, 0), 1)
            if pending:
                cv2.drawMarker(rs, tuple(int(v) for v in pending[1]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
            ui.draw_roi(wc, cal["webcam_roi_inspect"], "Pruefbereich", (0, 255, 255))
            ui.draw_roi(wc, cal["webcam_roi_tray"], "Box", (255, 255, 0))
            x0, y0, x1, y1 = vision.grip_roi(wr.shape, vcfg)
            cv2.rectangle(wr, (x0, y0), (x1, y1), (0, 255, 0), 1)

            try:
                st = arm.state()
                joints_txt = "  ".join(f"{j[:8]} {v:6.1f}" for j, v in zip(JOINTS, st))
            except StudioError as e:
                st, joints_txt = None, str(e)
            loo = pm.leave_one_out()
            taught = ", ".join(k for k in ("transit", "inspect", "home") if k in cal["poses"]) or "-"
            lines = [joints_txt,
                     f"Posen: {taught}   Ablagepunkte: {len(cal['place_points'])}"
                     + (f"  (Fehler bei Vorhersage ~{loo:.1f})" if loo is not None else "")
                     + f"   Leer-Referenz: {'ja' if cal['empty_ref'] else 'NEIN'}"
                     + f"   Halte-Probe: {'ja' if cal['grip_hold_diff'] is not None else 'nein'}",
                     "t Transit+Leer-Ref | k Halte-Probe | i Pruefpose | h Home | y/b Schale, dann LEERTASTE | "
                     "u zurueck | r Pruefbereich | m Box | q Ende"] + log[-5:]
            header = ("Schale " + pending[0] + " gemerkt -> Greifer ueber die Mitte, dann LEERTASTE") if pending \
                else "Kalibrierung"
            cv2.imshow(WIN, ui.compose({"RealSense": rs, "Webcam": wc, "Wrist": wr}, lines, header))

            k = ui.key(30)
            if k == ord("q"):
                break
            if st is None:
                continue
            changed = True
            if k == ord("t"):
                if st[5] > 15:
                    say("! Greifer ist offen - erst schliessen (leer), dann t")
                    continue
                ref1 = arm.frame("wrist")
                time.sleep(1.0)
                ref2 = arm.frame("wrist")
                cal["poses"]["transit"] = st.tolist()
                path = os.path.join(HERE, "empty_ref.png")
                cv2.imwrite(path, cv2.cvtColor(ref2, cv2.COLOR_RGB2BGR))
                cal["empty_ref"] = path
                cal["grip_noise"] = vision.grip_change(ref1, ref2, vcfg)
                cal["empty_gripper"] = float(st[5])
                say(f"Transit gespeichert, Leer-Referenz: Rauschen {cal['grip_noise']:.1f}, Greifer leer {st[5]:.1f}")
            elif k == ord("k"):
                if not cal["empty_ref"] or "transit" not in cal["poses"]:
                    say("! erst t (Transit + Leer-Referenz)")
                    continue
                if np.abs(st[:5] - np.array(cal["poses"]["transit"][:5])).max() > 4:
                    say("! nicht in der Transit-Pose - genau dorthin fahren")
                    continue
                ref = cv2.cvtColor(cv2.imread(cal["empty_ref"]), cv2.COLOR_BGR2RGB)
                d = vision.grip_change(ref, arm.frame("wrist"), vcfg)
                cal["grip_hold_diff"] = d
                cal["holding_gripper"] = float(st[5])
                say(f"Halte-Probe: Bildabweichung {d:.1f} (leer {cal['grip_noise']:.1f}), Greifer {st[5]:.1f} "
                    f"(leer {cal['empty_gripper']:.1f})")
                if d < 2.5 * max(cal["grip_noise"], 1.0):
                    say("! Abweichung klein - Stein zu wenig sichtbar? Greif-Pruefung stuetzt sich dann auf den Greiferwert")
            elif k == ord("i"):
                cal["poses"]["inspect"] = st.tolist()
                say("Pruef-Pose gespeichert")
            elif k == ord("h"):
                cal["poses"]["home"] = st.tolist()
                say("Home gespeichert")
            elif k in (ord("y"), ord("b")):
                name = "yellow" if k == ord("y") else "blue"
                t = tracker.things[name]
                if not t.fresh(now) or t.visible < 0.9:
                    say(f"! {name}-Schale nicht frei sichtbar - Arm wegfahren")
                    continue
                pending = (name, t.centre)
                say(f"{name}-Schale bei {t.centre[0]:.0f},{t.centre[1]:.0f} gemerkt")
                changed = False
            elif k == ord(" "):
                if not pending:
                    say("! erst y oder b (Schalenmitte merken)")
                    continue
                cal["place_points"].append({"pixel": [float(pending[1][0]), float(pending[1][1])],
                                            "joints": st[:5].tolist(), "bowl": pending[0]})
                say(f"Ablagepunkt {len(cal['place_points'])} gespeichert")
                pending = None
            elif k == ord("u"):
                if cal["place_points"]:
                    cal["place_points"].pop()
                    say("letzten Ablagepunkt geloescht")
            elif k in (ord("r"), ord("m")):
                img = ui.to_bgr(frames["webcam"])
                roi = cv2.selectROI("Bereich ziehen, ENTER", img, showCrosshair=False)
                cv2.destroyWindow("Bereich ziehen, ENTER")
                if roi[2] > 10 and roi[3] > 10:
                    cal["webcam_roi_inspect" if k == ord("r") else "webcam_roi_tray"] = [int(v) for v in roi]
                    say("Bereich gespeichert")
            else:
                changed = False
            if changed:
                save_calibration(cal)
    finally:
        try:
            sess.set_control(None)
        except Exception:  # noqa: BLE001
            pass
        sess.close()
        cv2.destroyAllWindows()
        print("Kalibrierung gespeichert:", os.path.join(HERE, "calibration.json"))


if __name__ == "__main__":
    main()
