"""One OpenCV window: three cameras with overlays and a text panel."""

from __future__ import annotations

import cv2
import numpy as np

W, H = 480, 360
PANEL_H = 250


def to_bgr(rgb: np.ndarray | None) -> np.ndarray:
    if rgb is None:
        img = np.full((480, 640, 3), 40, np.uint8)
        cv2.putText(img, "kein Bild", (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        return img
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_roi(bgr, roi, label, colour=(255, 255, 0)):
    if roi:
        x, y, w, h = [int(v) for v in roi]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(bgr, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def compose(tiles: dict[str, np.ndarray], lines: list[str], header: str, header_colour=(255, 255, 255)) -> np.ndarray:
    canvas = np.full((H + PANEL_H, 3 * W + 40, 3), 25, np.uint8)
    for i, (name, bgr) in enumerate(tiles.items()):
        x = 10 + i * (W + 10)
        canvas[10:10 + H, x:x + W] = cv2.resize(bgr, (W, H))
        cv2.rectangle(canvas, (x, 10), (x + 130, 38), (0, 0, 0), -1)
        cv2.putText(canvas, name, (x + 6, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y = H + 45
    cv2.putText(canvas, header, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, header_colour, 2, cv2.LINE_AA)
    for line in lines[-8:]:
        y += 25
        cv2.putText(canvas, line[:170], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def window(name: str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 3 * W + 40, H + PANEL_H)


def key(delay_ms: int = 30) -> int:
    return cv2.waitKey(delay_ms) & 0xFF
