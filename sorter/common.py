"""Config, calibration file, joint limits and the RealSense-pixel -> arm-pose map."""

from __future__ import annotations

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
CALIB_PATH = os.path.join(HERE, "calibration.json")
LOG_DIR = os.path.join(HERE, "log")

DEFAULT_CONFIG = {
    "project_id": "857e9ecc-23bc-495f-ad0a-296dd2505dbd",
    "environment_id": "c87de068-b6bf-45f8-9a94-1cff921959e5",
    "model": None,                  # name or id; None = newest SmolVLA
    "backend": "torch",
    "device": "auto",
    "pick_task": "put the brick in the yellow bowl",
    # Step 1 (directed pick) only works once the policy is trained on such texts, e.g.
    # "pick up the {colour} brick". None = always a random pick, a marked brick seen first is only logged.
    "direct_pick_task": None,
    "speed": {"move": 40.0, "carry": 28.0, "inspect": 30.0},
    "gripper": {"open": 35.0, "close": 0.0, "policy_closed_cmd": 8.0, "holding_margin": 0.8},
    "pick": {"timeout_s": 25.0, "after_grasp_s": 0.5, "settle_s": 0.25, "max_attempts": 3,
             "lift_shoulder": -20.0, "hz": 30},
    "inspect": {"roll": [-95, -68, -41, -14, 13, 40, 67, 94], "flex_offsets": [0], "settle_s": 0.3,
                "frames": 3, "hits_needed": 2},
    "wrist_check": {"frames": 3, "hits_needed": 2, "brick_diff": 30},
    "scan": {"frames": 5, "hits_needed": 3, "same_place_px": 25},
    "place": {"hull_margin_px": 35, "bowl_max_age_s": 4.0, "wait_bowl_s": 30.0, "replan_px": 20},
    # policy_sort.py: the policy picks and places, the script checks in between
    "policy_sort": {
        "pick_task": "put the brick in the yellow bowl",
        "place_task": "put the brick in the {colour} bowl",
        "bad_bowl": "yellow", "good_bowl": "blue",
        "lift_delta": 6.0,            # shoulder_lift drop after the grasp that counts as "lifted"
        "max_after_grasp_s": 1.2,     # stop the policy at the latest this long after the grasp
        "raise_before_inspect": -15.0,  # shoulder_lift change before going to the webcam (clear the box)
        "release_gripper": 15.0,      # measured gripper above this = the policy let go of the brick
        "after_release_s": 1.0,       # let the policy pull back before it is stopped
        "place_timeout_s": 20.0,
        "empty_gripper_max": 2.0,     # closed gripper below this = nothing held (calibration overrides)
    },
    "anomaly": {
        "wrist_model": "~/bricks-data/model_anomalydino.ckpt",
        "webcam_model": None,         # second model (webcam, inspection); None = stroke detector as stand-in
        "wrist_threshold": 0.5, "webcam_threshold": 0.5,
        "wrist_frames": 3, "wrist_hits": 2,
        "webcam_use_roi": True,       # crop the webcam image to webcam_roi_inspect (if calibrated)
        "save_images": True,          # every checked image goes to log/<run>/ (training data for later)
    },
    "dashboard": {"enabled": True, "url": "http://127.0.0.1:8090", "frontend_dir": None},  # team hub (frontend/)
    "max_bricks": 100,
    "max_empty_in_row": 3,
    "vision": {},                   # overrides for vision.DEFAULTS
}

# joint range seen in the 400 training episodes (normalized units), widened by taught poses
DATASET_MIN = [-53.0, -99.2, -58.2, -16.2, -48.7, 0.0]
DATASET_MAX = [44.7, 59.5, 99.3, 99.6, 72.2, 60.4]
# start pose of the training episodes (median); the policy expects to start here
DATASET_HOME = [-3.6, 1.3, 10.0, 88.3, 4.2, 27.3]


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    with open(CONFIG_PATH) as f:
        return _deep_merge(DEFAULT_CONFIG, json.load(f))


def load_calibration() -> dict:
    cal = {"poses": {}, "place_points": [], "webcam_roi_tray": None, "webcam_roi_inspect": None,
           "empty_ref": None, "grip_noise": None, "grip_hold_diff": None,
           "empty_gripper": None, "holding_gripper": None}
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            cal.update(json.load(f))
    if "inspect" not in cal["poses"]:
        # the pose read from the arm on 2026-11-15 00:52 (brick in front of the webcam)
        import glob
        files = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "poses", "pose_*.json")))
        if files:
            with open(files[0]) as f:
                st = json.load(f)["state"]
            from studio import JOINTS
            cal["poses"]["inspect"] = [st[j]["normalized"] for j in JOINTS]
    return cal


def save_calibration(cal: dict):
    tmp = CALIB_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cal, f, indent=1)
    if os.path.exists(CALIB_PATH):
        os.replace(CALIB_PATH, CALIB_PATH + ".bak")
    os.replace(tmp, CALIB_PATH)


def pose(cal: dict, name: str) -> np.ndarray | None:
    p = cal["poses"].get(name)
    if p is None and name == "home":
        return np.array(DATASET_HOME)
    return None if p is None else np.array(p, dtype=np.float64)


def joint_limits(cal: dict, cfg: dict, margin: float = 5.0) -> dict[str, list[float]]:
    """Allowed range per joint: training range + every taught pose + the inspection roll sweep."""
    from studio import JOINTS
    lo, hi = np.array(DATASET_MIN), np.array(DATASET_MAX)
    extra = [np.array(p) for p in cal["poses"].values()]
    extra += [np.r_[pt["joints"], 0.0] for pt in cal["place_points"]]
    for e in extra:
        lo, hi = np.minimum(lo, e), np.maximum(hi, e)
    lo, hi = lo - margin, hi + margin
    ins = pose(cal, "inspect")
    if ins is not None:
        rolls = cfg["inspect"]["roll"]
        lo[4], hi[4] = min(lo[4], min(rolls)), max(hi[4], max(rolls))
        flex = cfg["inspect"]["flex_offsets"]
        lo[3], hi[3] = min(lo[3], ins[3] + min(flex)), max(hi[3], ins[3] + max(flex))
    lo, hi = np.clip(lo, -100, 100), np.clip(hi, -100, 100)
    lo[5], hi[5] = 0.0, max(hi[5], cfg["gripper"]["open"])
    return {j: [float(lo[i]), float(hi[i])] for i, j in enumerate(JOINTS)}


class PlaceMap:
    """RealSense pixel of a bowl centre -> the 5 arm joints that hold a brick above it.

    Learned from taught points; thin-plate spline with a linear part (>= 6 points) or an
    affine fit (3-5 points). Refuses pixels outside the taught area (plus a margin).
    """

    def __init__(self, points: list[dict], margin_px: float = 35.0):
        self.n = len(points)
        self.margin = margin_px
        self.ok = self.n >= 3
        if not self.ok:
            return
        self.P = np.array([p["pixel"] for p in points], dtype=np.float64)
        self.J = np.array([p["joints"] for p in points], dtype=np.float64)
        if self.n >= 6:
            from scipy.interpolate import RBFInterpolator
            self._f = RBFInterpolator(self.P, self.J, kernel="thin_plate_spline", degree=1, smoothing=2.0)
        else:
            A = np.c_[self.P, np.ones(self.n)]
            coef, *_ = np.linalg.lstsq(A, self.J, rcond=None)
            self._f = lambda q: np.c_[q, np.ones(len(q))] @ coef
        try:
            from scipy.spatial import ConvexHull
            self.hull = self.P[ConvexHull(self.P).vertices] if self.n >= 3 else self.P
        except Exception:  # noqa: BLE001 - collinear points
            self.hull = self.P

    def distance_outside(self, px) -> float:
        """0 inside the taught area, else pixels to its border."""
        import cv2
        d = cv2.pointPolygonTest(self.hull.astype(np.float32).reshape(-1, 1, 2), (float(px[0]), float(px[1])), True)
        return max(0.0, -d)

    def joints(self, px) -> np.ndarray:
        if not self.ok:
            raise ValueError("zu wenige Ablagepunkte kalibriert (mindestens 3, besser 6-9)")
        if self.distance_outside(px) > self.margin:
            raise ValueError(f"Schale liegt außerhalb des kalibrierten Bereichs ({self.distance_outside(px):.0f} px)")
        return np.asarray(self._f(np.array([px], dtype=np.float64))[0])

    def leave_one_out(self) -> float | None:
        """Mean joint error when each taught point is predicted from the others."""
        if self.n < 5:
            return None
        errs = []
        for i in range(self.n):
            pts = [{"pixel": self.P[j].tolist(), "joints": self.J[j].tolist()} for j in range(self.n) if j != i]
            m = PlaceMap(pts, margin_px=1e9)
            errs.append(np.abs(m.joints(self.P[i]) - self.J[i]).max())
        return float(np.mean(errs))


def colour_name(hue: float, sat: float = 255, val: float = 255) -> str:
    if sat < 50:
        return "white" if val > 150 else "grey"
    for upto, name in ((8, "red"), (18, "orange"), (34, "yellow"), (80, "green"), (95, "turquoise"),
                       (125, "blue"), (160, "purple"), (169, "pink"), (180, "red")):
        if hue <= upto:
            return name
    return "red"


def stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")
