"""Image checks for the brick sorter. All functions take RGB uint8 images (as the Studio store holds them).

- find_strokes(): black marker strokes on a coloured brick (wrist cam, webcam).
- SceneTracker:   yellow bowl, blue bowl and the silver collection box in the RealSense top view,
                  smoothed over frames and robust to the arm covering parts of them.
- grip_change():  how much the image region between the gripper fingers changed between two
                  wrist frames. A held brick moves with the camera (small change), an empty
                  gripper sees the scene move away (large change).
All thresholds live in DEFAULTS and can be overridden from config.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULTS = {
    "stroke": {
        "v_max": 70,            # marker ink: darker than this (0-255)
        "chroma_max": 45,       # ...and nearly colourless (max-min of RGB)
        "min_area": 40,         # px at 640x480; small dots count too
        "max_area": 12000,
        "min_elongation": 0.0,  # marks can be dots, so no shape demand by default
        "min_length": 0,        # px centre-line length at 640 px width
        "ink_v_mean_max": 40,   # mean brightness inside the mark: real ink 15-35, look-alike shadows 47+
        "border_px": 4,         # strokes touching the image border are ignored (frames, shadows)
        "max_width": 28,        # px; thick dark blobs (servos, holes) are not strokes
        "ring_px": 7,           # ring around the stroke that must be brick colour
        "ring_colour_min": 0.85,  # real 0.90-0.96
        "ring_hue_min": 0.85,   # real 0.92-1.00, gaps between bricks ~0.5; share of the coloured ring within +-hue_tol of its main hue (one brick)
        "hue_tol": 12,
        "max_width_ratio": 2.3, # real 1.4-2.1; widest point / mean width; shadows are wedges, ink has even width
        "colour_s_min": 60,     # brick colour: saturation above this...
        "colour_v_min": 60,     # ...and not dark
        "white_v_min": 170,     # white/grey bricks count as brick colour if bright and...
        "white_s_max": 40,      # ...unsaturated; only used where use_white_bricks is true
        "use_white_bricks": False,
    },
    "scene": {
        "yellow": {"h": [18, 38], "s_min": 90, "v_min": 110},
        "blue": {"h": [92, 118], "s_min": 90, "v_min": 70},
        "bowl_min_area": 3500,     # px; bricks are far smaller
        "tray_v": [40, 115],       # rim: dark grey (towel ~160-190, black arm < 40)
        "tray_radius": [50, 110],  # px
        "tray_min_score": 0.15,
        "occluded_ratio": 0.6,     # visible area below this share of the best seen = partly covered
        "smooth": 0.35,            # new position weight per frame
    },
    "grip": {
        # region between the fingertips in the wrist image, as fractions (x0, y0, x1, y1)
        "roi": [0.30, 0.45, 0.70, 0.95],
        "hold_max_change": 18.0,   # mean abs grey difference; below = brick moved with the gripper
    },
}


def _merge(base: dict, override: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config(override: dict | None = None) -> dict:
    return _merge(DEFAULTS, override)


# ------------------------------------------------------------------ strokes
@dataclass
class Stroke:
    box: tuple[int, int, int, int]  # x, y, w, h
    centre: tuple[float, float]
    length: float
    width: float
    ring_colour: float
    contour: np.ndarray
    ink: float = 0.0
    ring_hue: float = 0.0
    width_ratio: float = 0.0


def find_strokes(rgb: np.ndarray, cfg: dict | None = None, roi: tuple[int, int, int, int] | None = None) -> list[Stroke]:
    """Black, thin, elongated marks that are surrounded by brick colour."""
    c = (cfg or DEFAULTS)["stroke"]
    img = rgb if roi is None else rgb[roi[1]:roi[3], roi[0]:roi[2]]
    scale = img.shape[1] / 640.0
    i16 = img.astype(np.int16)
    vmax = i16.max(axis=2)
    chroma = vmax - i16.min(axis=2)
    dark = ((vmax < c["v_max"]) & (chroma < c["chroma_max"])).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colour = (hsv[..., 1] > c["colour_s_min"]) & (hsv[..., 2] > c["colour_v_min"])
    if c["use_white_bricks"]:
        colour |= (hsv[..., 1] < c["white_s_max"]) & (hsv[..., 2] > c["white_v_min"])

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    a_min, a_max = c["min_area"] * scale ** 2, c["max_area"] * scale ** 2
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(c["ring_px"] * scale)) + 1,) * 2)
    found = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not a_min <= area <= a_max:
            continue
        pad = int(c["ring_px"] * scale) + 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(cnt, True)
        width = 2.0 * area / max(perim, 1.0)          # mean stroke width
        length = max(perim / 2.0 - width, 1.0)         # centre-line length
        if width > c["max_width"] * scale or length / max(width, 1.0) < c["min_elongation"]:
            continue
        if length < c["min_length"] * scale:
            continue
        b = c["border_px"]
        if x <= b or y <= b or x + w >= img.shape[1] - b or y + h >= img.shape[0] - b:
            continue
        if float(vmax[y0:y1, x0:x1][comp.astype(bool)].mean()) > c["ink_v_mean_max"]:
            continue
        ring = cv2.dilate(comp, ring_k).astype(bool) & ~dark[y0:y1, x0:x1].astype(bool)
        if ring.sum() == 0:
            continue
        ring_colour = float(colour[y0:y1, x0:x1][ring].mean())
        if ring_colour < c["ring_colour_min"]:
            continue
        hues = hsv[y0:y1, x0:x1, 0][ring & colour[y0:y1, x0:x1]].astype(np.int16)
        hist = np.bincount(hues, minlength=180)
        main = int(np.argmax(np.convolve(np.r_[hist[-c["hue_tol"]:], hist, hist[:c["hue_tol"]]],
                                         np.ones(2 * c["hue_tol"] + 1), "valid")))
        dh = np.abs(hues - main)
        ring_hue = float((np.minimum(dh, 180 - dh) <= c["hue_tol"]).mean())
        if ring_hue < c["ring_hue_min"]:
            continue
        widest = 2.0 * float(cv2.distanceTransform(comp, cv2.DIST_L2, 3).max())
        width_ratio = widest / max(width, 1.0)
        if width_ratio > c["max_width_ratio"]:
            continue
        ox, oy = (roi[0], roi[1]) if roi else (0, 0)
        m = cv2.moments(cnt)
        cx = (m["m10"] / m["m00"] if m["m00"] else w / 2) + x0 + ox
        cy = (m["m01"] / m["m00"] if m["m00"] else h / 2) + y0 + oy
        ink = float(vmax[y0:y1, x0:x1][comp.astype(bool)].mean())
        found.append(Stroke((int(x + ox), int(y + oy), int(w), int(h)), (cx, cy), length, width, ring_colour,
                            cnt + np.array([x0 + ox, y0 + oy]), ink, ring_hue, width_ratio))
    return found


def draw_strokes(bgr: np.ndarray, strokes: list[Stroke]) -> None:
    for s in strokes:
        cv2.drawContours(bgr, [s.contour], -1, (0, 0, 255), 2)
        x, y, w, h = s.box
        cv2.rectangle(bgr, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 2)
        cv2.putText(bgr, "STRICH", (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


# ------------------------------------------------------------------ scene (RealSense top view)
@dataclass
class Thing:
    centre: tuple[float, float] | None = None   # smoothed pixel position
    radius: float = 0.0
    area_best: float = 0.0
    visible: float = 0.0                        # share of the best seen area visible now
    seen_at: float = -1.0                       # time of the last fresh measurement
    raw: tuple[float, float] | None = None

    def fresh(self, now: float, max_age: float = 1.5) -> bool:
        return self.centre is not None and now - self.seen_at <= max_age


def _filled_blob(mask: np.ndarray, min_area: float):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    if len(hull) >= 5:
        (ex, ey), (ma, mi), _ = cv2.fitEllipse(hull)
        cx, cy, r = ex, ey, (ma + mi) / 4
    return (float(cx), float(cy)), float(r), float(cv2.contourArea(hull)), hull


class SceneTracker:
    """Keeps the latest position of the yellow bowl, blue bowl and the silver box.

    A position is only updated when enough of the object is visible; when the arm covers it,
    the last good position is kept and `visible` drops so callers can wait for a clear view.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or DEFAULTS)["scene"]
        self.things = {"yellow": Thing(), "blue": Thing(), "tray": Thing()}
        self.debug: dict = {}

    def update(self, rgb: np.ndarray, now: float) -> dict[str, Thing]:
        c = self.cfg
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        found = {}
        for name in ("yellow", "blue"):
            p = c[name]
            mask = ((h >= p["h"][0]) & (h <= p["h"][1]) & (s >= p["s_min"]) & (v >= p["v_min"])).astype(np.uint8)
            found[name] = _filled_blob(mask, c["bowl_min_area"])
        bowls = [(f[0][0], f[0][1], f[1]) for f in (found["yellow"], found["blue"]) if f]
        found["tray"] = self._tray(hsv, rgb, bowls)
        self.debug = found
        for name, res in found.items():
            t = self.things[name]
            if res is None:
                t.visible = 0.0
                continue
            centre, r, area, _ = res
            t.raw = centre
            t.area_best = max(t.area_best * 0.999, area)  # slowly forgets, so a smaller new spot is accepted
            t.visible = area / t.area_best if t.area_best else 0.0
            if t.visible < c["occluded_ratio"]:
                continue
            if t.centre is None or np.hypot(centre[0] - t.centre[0], centre[1] - t.centre[1]) > 0.5 * max(r, 1):
                t.centre = centre          # moved (or first sight): jump, do not crawl
            else:
                a = c["smooth"]
                t.centre = (t.centre[0] * (1 - a) + centre[0] * a, t.centre[1] * (1 - a) + centre[1] * a)
            t.radius, t.seen_at = r, now
        return self.things

    def _tray(self, hsv: np.ndarray, rgb: np.ndarray, bowls: list):
        """The box has a dark grey round rim: best circle whose rim band is dark grey and whose inside is not."""
        c = self.cfg
        v, s = hsv[..., 2], hsv[..., 1]
        gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (7, 7), 1.5)
        cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20, param1=80, param2=25,
                              minRadius=c["tray_radius"][0], maxRadius=c["tray_radius"][1])
        if cs is None:
            return None
        prev = self.things["tray"].centre
        best = None
        H, W = v.shape
        for cx, cy, r in cs[0][:40]:
            if any(np.hypot(cx - b[0], cy - b[1]) < b[2] * 0.9 for b in bowls):
                continue
            x0, x1 = max(0, int(cx - r * 1.05)), min(W, int(cx + r * 1.05) + 1)
            y0, y1 = max(0, int(cy - r * 1.05)), min(H, int(cy + r * 1.05) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.hypot(xx - cx, yy - cy)
            vv, ss = v[y0:y1, x0:x1], s[y0:y1, x0:x1]
            band, inner = (d > r * 0.88) & (d < r * 1.04), d < r * 0.75
            if band.sum() < 50 or not inner.any():
                continue
            rim = ((vv[band] >= c["tray_v"][0]) & (vv[band] <= c["tray_v"][1]) & (ss[band] < 90)).mean()
            score = rim - 0.5 * (vv[inner] < c["tray_v"][1]).mean()
            if prev is not None and np.hypot(cx - prev[0], cy - prev[1]) < 25:
                score += 0.1  # prefer staying put
            if best is None or score > best[0]:
                best = (score, float(cx), float(cy), float(r))
        if best is None or best[0] < c["tray_min_score"]:
            return None
        _, cx, cy, r = best
        return (cx, cy), r, float(np.pi * r * r), None

    def draw(self, bgr: np.ndarray, now: float) -> None:
        colours = {"yellow": (0, 220, 255), "blue": (255, 120, 0), "tray": (200, 200, 200)}
        labels = {"yellow": "GELB (gut)", "blue": "BLAU (defekt)", "tray": "Sammelbox"}
        for name, t in self.things.items():
            if t.centre is None:
                continue
            col = colours[name] if t.fresh(now) else (90, 90, 90)
            cx, cy = int(t.centre[0]), int(t.centre[1])
            cv2.circle(bgr, (cx, cy), int(t.radius), col, 2)
            cv2.drawMarker(bgr, (cx, cy), col, cv2.MARKER_CROSS, 18, 2)
            txt = labels[name] + ("" if t.visible >= self.cfg["occluded_ratio"] else " (verdeckt)")
            cv2.putText(bgr, txt, (cx - 50, cy - int(t.radius) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


# ------------------------------------------------------------------ grasp check
def grip_roi(shape, cfg: dict | None = None) -> tuple[int, int, int, int]:
    fx0, fy0, fx1, fy1 = (cfg or DEFAULTS)["grip"]["roi"]
    hgt, wid = shape[:2]
    return int(fx0 * wid), int(fy0 * hgt), int(fx1 * wid), int(fy1 * hgt)


def grip_change(rgb_a: np.ndarray, rgb_b: np.ndarray, cfg: dict | None = None) -> float:
    """Mean absolute difference (0-255) of the finger-gap region between two wrist frames."""
    x0, y0, x1, y1 = grip_roi(rgb_a.shape, cfg)
    a = cv2.GaussianBlur(cv2.cvtColor(rgb_a[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (9, 9), 0).astype(np.float32)
    b = cv2.GaussianBlur(cv2.cvtColor(rgb_b[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (9, 9), 0).astype(np.float32)
    return float(np.abs(a - b).mean())
