#!/usr/bin/env python3
"""Read-only: print Present_Position ticks of both SO-101 arms (nothing is written).

Stop the Studio backend / leave the environment page first, it holds the serial ports.
Run with the Studio venv (needs scservo_sdk):
  python robot/scripts/read_ticks.py --follower PORT --leader PORT
Ports: macOS /dev/cu.usbmodem..., Linux /dev/ttyACM0..., Windows COM3...
"""

from __future__ import annotations

import argparse
import json

import scservo_sdk as scs

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PRESENT_POSITION = 56


def read_arm(port: str) -> dict[str, int | None]:
    handler, packet = scs.PortHandler(port), scs.PacketHandler(0)
    if not handler.openPort():
        raise SystemExit(f"cannot open {port}")
    handler.setBaudRate(1_000_000)
    values: dict[str, int | None] = {}
    for servo_id, name in enumerate(NAMES, 1):
        samples = []
        for _ in range(5):
            try:
                raw, result, _error = packet.read2ByteTxRx(handler, servo_id, PRESENT_POSITION)
            except IndexError:  # truncated reply, retry
                continue
            if result == 0:
                samples.append(-(raw & 0x7FFF) if raw & 0x8000 else raw)
        values[name] = round(sum(samples) / len(samples)) if samples else None
    handler.closePort()
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--follower", required=True)
    parser.add_argument("--leader", required=True)
    args = parser.parse_args()
    print(json.dumps({"follower": read_arm(args.follower), "leader": read_arm(args.leader)}))
