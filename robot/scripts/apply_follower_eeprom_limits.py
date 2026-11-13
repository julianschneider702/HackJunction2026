#!/usr/bin/env python3
"""Write the follower's min/max position limits (EEPROM regs 9/11) from robot/calibration/follower.json.

Only needed if the follower's servos were re-flashed or recalibrated with the Studio wizard:
the servos clamp every move to these limits, so they must match the Studio calibration.
Check first with --dry-run. Stop the Studio backend first (it holds the port).
  python robot/scripts/apply_follower_eeprom_limits.py --port PORT [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import scservo_sdk as scs

CALIBRATION = Path(__file__).resolve().parent.parent / "calibration" / "follower.json"
MIN_LIMIT, MAX_LIMIT, LOCK = 9, 11, 55
BODY_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def retry(call, *args):
    for _ in range(10):
        try:
            value = call(*args)
        except IndexError:  # truncated reply
            value = None
        if value is not None and value[-2 if len(value) == 3 else 0] == 0:
            return value
        time.sleep(0.03)
    raise SystemExit(f"servo did not answer: {call.__name__}{args[1:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="follower serial port")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    calibration = json.loads(CALIBRATION.read_text())
    handler, packet = scs.PortHandler(args.port), scs.PacketHandler(0)
    if not handler.openPort():
        raise SystemExit(f"cannot open {args.port}")
    handler.setBaudRate(1_000_000)
    for name in BODY_JOINTS:
        cal = calibration[name]
        servo, lo, hi = cal["id"], cal["range_min"], cal["range_max"]
        current = (retry(packet.read2ByteTxRx, handler, servo, MIN_LIMIT)[0],
                   retry(packet.read2ByteTxRx, handler, servo, MAX_LIMIT)[0])
        if args.dry_run or current == (lo, hi):
            print(f"{name:14s} eeprom={current} calibration={(lo, hi)} {'ok' if current == (lo, hi) else 'DIFFERS'}")
            continue
        retry(packet.write1ByteTxRx, handler, servo, LOCK, 0)
        retry(packet.write2ByteTxRx, handler, servo, MIN_LIMIT, lo)
        retry(packet.write2ByteTxRx, handler, servo, MAX_LIMIT, hi)
        retry(packet.write1ByteTxRx, handler, servo, LOCK, 1)
        print(f"{name:14s} {current} -> {(lo, hi)}")
    handler.closePort()


if __name__ == "__main__":
    main()
