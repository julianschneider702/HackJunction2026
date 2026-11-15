#!/usr/bin/env python3
"""Teleoperation without Studio: the follower follows the leader.

Uses the same calibration as Studio (fetched from the Studio API when the backend
runs, otherwise from the cache robot/calibration/studio_cache.json).
The arms are found by serial number, so the USB port does not matter.

No environment page may be open in Studio, otherwise the arms are busy.

Run:
  station/studio-python robot/scripts/teleop.py

Quit: Ctrl+C  (the follower holds its position)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from physicalai_so101_plugin.so101 import SO101
from serial.tools import list_ports

STUDIO = "http://localhost:7860/api"
CACHE = Path(__file__).resolve().parent.parent / "calibration" / "studio_cache.json"
NAMES = {"follower": "SO101 Follower", "leader": "SO101 Leader"}


def load_robots() -> dict[str, dict]:
    """Return {"follower": payload, "leader": payload} with calibration and serial number."""
    try:
        projects = json.load(urllib.request.urlopen(f"{STUDIO}/projects", timeout=2))
        projects = projects if isinstance(projects, list) else projects.get("items", [])
        robots = {}
        for project in projects:
            for r in json.load(urllib.request.urlopen(f"{STUDIO}/projects/{project['id']}/robots", timeout=2)):
                for role, name in NAMES.items():
                    if r["name"] == name:
                        robots[role] = r["payload"]
        if len(robots) == 2:
            CACHE.write_text(json.dumps(robots, indent=2))
            print("Calibration loaded from Studio.")
            return robots
    except OSError:
        pass
    if CACHE.exists():
        print(f"Studio not reachable, using the calibration from {CACHE.name}.")
        return json.loads(CACHE.read_text())
    sys.exit("Keine Kalibrierung gefunden: Studio-Backend einmal starten und das Skript erneut ausfuehren.")


def find_port(serial_number: str, fallback: str) -> str:
    for port in list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hz", type=float, default=50.0, help="control rate (default 50 Hz)")
    parser.add_argument("--ramp", type=float, default=2.0, help="seconds to blend gently to the leader pose at the start")
    args = parser.parse_args()

    robots = load_robots()
    arms = {}
    for role, payload in robots.items():
        port = find_port(payload.get("serial_number") or "", payload["connection_string"])
        arms[role] = SO101(port=port, calibration=payload["calibration"], role=role, unit=payload.get("unit", "normalized"))
        print(f"{NAMES[role]:15s} -> {port}")

    leader, follower = arms["leader"], arms["follower"]
    try:
        leader.connect()
        follower.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"Connection failed: {exc}")
        print("Are the arms in use by Studio? Close the environment page in the browser.")
        leader.disconnect()
        follower.disconnect()
        return 1

    period = 1.0 / args.hz
    start_pose = follower.get_observation().joint_positions.astype(np.float64)
    t0 = time.monotonic()
    print("Teleop running. Move the leader. Quit with Ctrl+C.")
    try:
        while True:
            tick = time.monotonic()
            target = leader.get_observation().joint_positions.astype(np.float64)
            # gentle start: blend from the start pose to the leader during the first --ramp seconds
            alpha = min(1.0, (tick - t0) / args.ramp) if args.ramp > 0 else 1.0
            follower.send_action(start_pose + alpha * (target - start_pose))
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        follower.disconnect()  # the follower holds its position
        leader.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
