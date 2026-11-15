#!/usr/bin/env python3
"""Feed the dashboard hub with made-up sorting cycles, for building and demoing the dashboard without the robot.

All events carry source="sim", so the dashboard shows a "Demo data" badge. Every number here
(scores, latencies, power, temperature, benchmark rows) is invented. Python standard library only.

    python frontend/hub.py            # terminal 1
    python frontend/simulate.py       # terminal 2
    python frontend/simulate.py --speed 4 --cycles 20
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import threading
import time
import zlib

from hub_client import DEFAULT_URL, HubClient

W, H = 200, 150
BRICK_COLOURS = [(196, 40, 28), (13, 105, 172), (245, 205, 47), (40, 127, 70)]
THRESHOLD = 0.5


def png(rgb: bytearray) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes(rgb[y * W * 3:(y + 1) * W * 3]) for y in range(H))
    header = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b"")


def heat_colour(v: float) -> tuple:
    """0 = dark blue, 0.5 = yellow, 1 = red."""
    v = max(0.0, min(1.0, v))
    if v < 0.5:
        t = v / 0.5
        return int(20 + 235 * t), int(40 + 190 * t), int(120 - 90 * t)
    t = (v - 0.5) / 0.5
    return 255, int(230 - 190 * t), int(30 - 10 * t)


def brick_images(rng: random.Random, defective: bool, score: float) -> tuple:
    """Draw a brick seen from above (with a marker scribble if defective) and a matching heatmap."""
    colour = rng.choice(BRICK_COLOURS)
    cx, cy = W / 2 + rng.uniform(-14, 14), H / 2 + rng.uniform(-10, 10)
    half_w, half_h = 62, 34
    angle = rng.uniform(-0.35, 0.35)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    studs = [(-45 + 30 * i, -16 + 32 * j) for i in range(4) for j in range(2)]
    spot = (rng.uniform(-40, 40), rng.uniform(-18, 18), rng.uniform(7, 12)) if defective else None

    image, heat = bytearray(W * H * 3), bytearray(W * H * 3)
    for y in range(H):
        for x in range(W):
            # brick coordinates
            dx, dy = x - cx, y - cy
            bx, by = dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a
            shade = 232 + ((x * 7 + y * 13) % 5)
            pixel = (shade, shade - 2, shade - 8)
            level = 0.04 + 0.03 * ((x * 3 + y * 5) % 4) / 4
            if abs(bx) <= half_w and abs(by) <= half_h:
                pixel = colour
                level = 0.12
                if any((bx - sx) ** 2 + (by - sy) ** 2 <= 100 for sx, sy in studs):
                    pixel = tuple(min(255, c + 34) for c in colour)
                if spot:
                    distance = math.hypot(bx - spot[0], by - spot[1])
                    if distance <= spot[2] and (int(bx + by) % 3 != 0):
                        pixel = (28, 26, 30)
                    level = max(level, score * math.exp(-(distance / (spot[2] * 1.9)) ** 2))
            i = (y * W + x) * 3
            image[i:i + 3] = bytes(pixel)
            heat[i:i + 3] = bytes(heat_colour(level))
    return png(image), png(heat)


class Simulator:
    def __init__(self, hub: HubClient, speed: float, seed: int | None):
        self.hub = hub
        self.speed = speed
        self.rng = random.Random(seed)
        self.phase = "idle"
        self.phase_args: tuple = (None, None)
        self.running = True
        self.power = 9.0
        self.temp = 52.0

    # ---- helpers

    def set_phase(self, phase: str, cycle_id: str | None = None, detail: str | None = None) -> None:
        self.phase, self.phase_args = phase, (cycle_id, detail)
        self.hub.phase(phase, cycle_id=cycle_id, detail=detail)

    def wait(self, seconds: float, policy_active: bool = False) -> None:
        """Sleep in small steps, send policy latencies meanwhile, and hold while a stop is requested."""
        end = time.monotonic() + seconds / self.speed
        while time.monotonic() < end:
            if self.hub.stop_requested:
                self.hub.robot("stopped", "Stop requested from the dashboard")
                previous, previous_args = self.phase, self.phase_args
                self.set_phase("idle", detail="Stopped by the operator")
                while self.hub.stop_requested:
                    time.sleep(0.2)
                self.set_phase(previous, *previous_args)
                self.hub.robot("running")
                end = time.monotonic() + seconds / self.speed  # start this step again
            if policy_active:
                for _ in range(3):
                    self.hub.latency("policy", max(20.0, self.rng.gauss(46, 4)), device="NPU")
            time.sleep(0.1)

    def telemetry_loop(self) -> None:
        load = {"idle": 9, "perceive": 13, "detect": 19, "reason": 22, "execute": 21}
        while self.running:
            target = load[self.phase] + self.rng.uniform(-1.5, 1.5)
            self.power += (target - self.power) * 0.35
            self.temp += (44 + self.power * 1.5 - self.temp) * 0.06
            busy = self.phase != "idle"
            self.hub.telemetry(
                power_w=round(self.power, 2), temp_c=round(self.temp, 1), temp_limit_c=100,
                cpu=self.rng.uniform(18, 42) if busy else self.rng.uniform(3, 9),
                gpu=self.rng.uniform(55, 80) if self.phase == "detect" else self.rng.uniform(2, 10),
                npu=self.rng.uniform(60, 85) if self.phase in ("reason", "execute") else self.rng.uniform(0, 4),
            )
            time.sleep(1.0)

    def announce(self) -> None:
        self.hub.models([
            {"role": "policy", "name": "SmolVLA (demo)", "device": "NPU", "precision": "INT8", "runtime": "OpenVINO"},
            {"role": "anomaly", "name": "PaDiM (demo)", "device": "GPU", "precision": "FP16", "runtime": "OpenVINO"},
        ])
        for model, device, precision, latency, power in [
            ("SmolVLA (demo)", "CPU", "FP32", 312.0, 38.0), ("SmolVLA (demo)", "GPU", "FP16", 92.0, 27.0),
            ("SmolVLA (demo)", "NPU", "INT8", 46.0, 14.0), ("PaDiM (demo)", "CPU", "FP32", 41.0, 31.0),
            ("PaDiM (demo)", "GPU", "FP16", 17.0, 19.0), ("PaDiM (demo)", "NPU", "INT8", 22.0, 11.0),
        ]:
            self.hub.benchmark(model, device, precision, latency, fps=round(1000 / latency, 1), power_w=power)
        self.hub.robot("running", "Simulator")

    # ---- one brick

    def score(self, defective: bool) -> float:
        mean, spread = (0.74, 0.12) if defective else (0.27, 0.09)
        self.hub.latency("anomaly", max(8.0, self.rng.gauss(17, 2)), device="GPU")
        return round(max(0.02, self.rng.gauss(mean, spread)), 3)

    def cycle(self, number: int) -> None:
        rng = self.rng
        cycle_id = f"sim-{number}"
        started = time.monotonic()
        defective = rng.random() < 0.35

        self.set_phase("perceive", cycle_id, "Locating the next brick")
        self.wait(rng.uniform(0.8, 1.4))
        self.set_phase("execute", cycle_id, "Picking the brick and showing it to the inspection camera")
        self.wait(rng.uniform(3.0, 4.5), policy_active=True)
        self.set_phase("detect", cycle_id, "Scoring the brick")
        self.wait(rng.uniform(0.6, 1.0))
        score = self.score(defective)
        regrasps = 0
        if abs(score - THRESHOLD) < 0.1:  # too close to call: look again
            regrasps = 1
            self.set_phase("execute", cycle_id, "Score near the threshold, re-grasping for a second look")
            self.wait(rng.uniform(2.0, 3.0), policy_active=True)
            self.set_phase("detect", cycle_id, "Scoring the brick again")
            self.wait(rng.uniform(0.6, 1.0))
            score = self.score(defective)
        decision = "defective" if score > THRESHOLD else "good"
        instruction = f"put the brick into the {decision} bowl"
        self.set_phase("reason", cycle_id, f"Instruction: {instruction}")
        self.wait(rng.uniform(0.4, 0.8), policy_active=True)
        self.set_phase("execute", cycle_id, f"Placing the brick in the {decision} bowl")
        self.wait(rng.uniform(3.0, 4.5), policy_active=True)

        roll = rng.random()
        placed = decision if roll < 0.94 else ("none" if roll < 0.97 else ("good" if decision == "defective" else "defective"))
        image, heatmap = brick_images(rng, defective, score)
        self.hub.cycle(
            cycle_id=cycle_id, duration_s=round((time.monotonic() - started) * self.speed, 2),
            anomaly_score=score, threshold=THRESHOLD, decision=decision, placed=placed,
            truth=("defective" if defective else "good") if rng.random() < 0.7 else None,
            regrasps=regrasps, instruction=instruction, image=image, heatmap=heatmap,
        )
        self.set_phase("idle", detail="Waiting for the next brick")
        self.wait(rng.uniform(0.6, 1.2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    parser.add_argument("--cycles", type=int, default=0, help="stop after this many bricks (0 = run until Ctrl+C)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    hub = HubClient(args.url, source="sim")
    hub.check_stop()
    if not hub.connected:
        raise SystemExit(f"hub not reachable at {args.url}. Start it first: python frontend/hub.py")
    sim = Simulator(hub, max(0.1, args.speed), args.seed)
    sim.announce()
    threading.Thread(target=sim.telemetry_loop, daemon=True).start()
    print(f"simulator: sending demo data to {args.url} (Ctrl+C to stop)")
    number = int(time.time()) % 100000  # distinct ids across runs
    done = 0
    try:
        while not args.cycles or done < args.cycles:
            number += 1
            sim.cycle(number)
            done += 1
    except KeyboardInterrupt:
        pass
    finally:
        sim.running = False
        hub.phase("idle")
        hub.robot("idle", "Simulator ended")
        hub.flush()


if __name__ == "__main__":
    main()
