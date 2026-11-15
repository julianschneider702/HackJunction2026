#!/usr/bin/env python3
"""Save camera images from the running Studio while you teleoperate in Studio.

Studio shares the camera frames over shared memory (iceoryx2). This tool only
reads them, so the camera stays usable in Studio.

Precondition: the environment page is open in the browser (then the cameras run).

Keys (in the preview window):
  SPACE      save image -> <out>/good/
  q / Esc    quit

Run:
  station/studio-python inspection/capture_from_studio.py [--camera wrist|webcam|realsense]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from physicalai.capture.transport._shared_camera import SharedCamera

# Device paths of the cameras; Studio's service names are derived from them
USB_CAMS = {
    "wrist": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
    "webcam": "/dev/v4l/by-id/usb-Webcam_Webcam_Audio_01.00.00-video-index0",
}
REALSENSE_SERIAL = "243722065120"
DEFAULT_OUT = Path(__file__).resolve().parent / "data"


def service_name(camera: str) -> str:
    if camera == "realsense":
        return f"physicalai/camera/RealSenseCamera/{REALSENSE_SERIAL}/frame"
    index = Path(USB_CAMS[camera]).resolve().name.removeprefix("video")
    return f"physicalai/camera/UVCCamera/{index}/frame"


def connect(name: str) -> SharedCamera:
    """Wait until the Studio publishes this camera, then attach."""
    print(f"Waiting for Studio camera {name} ... (open the environment page in the browser)")
    while True:
        cam = SharedCamera(camera=None, service_name=name)
        try:
            cam.connect(timeout=2.0)
            return cam
        except Exception:  # noqa: BLE001 - publisher not up yet
            time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=["wrist", "webcam", "realsense"], default="wrist")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: inspection/data/<camera>)")
    args = parser.parse_args()

    out = (args.out or DEFAULT_OUT / args.camera) / "good"
    out.mkdir(parents=True, exist_ok=True)
    name = service_name(args.camera)
    cam = connect(name)
    print(f"Connected to {name}. Saving to {out}")

    window = f"Studio {args.camera}  |  Leertaste = gutes Bild speichern  |  q = Ende"
    count = len(list(out.glob("*.png")))
    flash = 0
    try:
        while True:
            try:
                frame = cam.read(timeout=2.0)
            except Exception:  # noqa: BLE001 - Studio closed the camera
                cam.disconnect()
                cam = connect(name)
                continue
            bgr = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
            view = bgr.copy()
            cv2.putText(view, f"saved: {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if flash > 0:
                cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (255, 255, 255), 8)
                flash -= 1
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                path = out / f"{args.camera}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                cv2.imwrite(str(path), bgr)
                count += 1
                flash = 5
                print(f"saved: {path}")
    finally:
        cam.disconnect()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
