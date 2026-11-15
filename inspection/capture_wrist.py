#!/usr/bin/env python3
"""Live preview of the wrist camera; a key press saves the image.

Only GOOD bricks are photographed (training data for Anomalib).

Keys (in the preview window):
  SPACE      save image -> <out>/good/
  q / Esc    quit

Studio must not use the camera at the same time (close the environment page),
otherwise the device is busy.

Run (Studio's backend Python has OpenCV):
  station/studio-python inspection/capture_wrist.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

# Stable path of the wrist camera (USB2.0_CAM1), independent of /dev/videoN
WRIST_CAM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "wrist"
KEYS = {ord(" "): "good"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=WRIST_CAM, help="camera device (default: wrist camera)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # OpenCV opens V4L2 devices by number only: /dev/v4l/by-id/... -> /dev/videoN -> N
    real = Path(args.device).resolve()
    if not real.name.startswith("video"):
        print(f"{args.device} is not a video device (camera plugged in?)")
        return 1
    cap, ok = None, False
    for source, api in ((int(real.name.removeprefix("video")), cv2.CAP_V4L2), (str(real), cv2.CAP_FFMPEG)):
        cap = cv2.VideoCapture(source, api)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = cap.read()
        if cap.isOpened() and ok:
            break
        cap.release()
    if not ok:
        print(f"Cannot open camera {args.device}.")
        print("Is Studio using it? Close the environment page in the browser and start again.")
        return 1

    counts = {name: len(list((args.out / name).glob("*.png"))) for name in KEYS.values()}
    window = "Greiferkamera  |  Leertaste = gutes Bild speichern  |  q = Ende"
    flash = 0
    print(f"Saving to {args.out}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No image from the camera, quitting.")
            break

        view = frame.copy()
        status = "  ".join(f"{k}: {v}" for k, v in counts.items())
        cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if flash > 0:  # short white frame as confirmation
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (255, 255, 255), 8)
            flash -= 1
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in KEYS:
            label = KEYS[key]
            folder = args.out / label
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"wrist_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            cv2.imwrite(str(path), frame)  # original image without the overlay text
            counts[label] += 1
            flash = 5
            print(f"saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
