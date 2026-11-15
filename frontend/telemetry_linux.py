#!/usr/bin/env python3
"""Send power, temperature and CPU / iGPU / NPU load of an Intel laptop to the dashboard hub (Linux only).

Reads kernel files, no extra packages:
    power        /sys/class/powercap/intel-rapl:*/energy_uj      (CPU package; root-only on most kernels)
    temperature  /sys/class/thermal/thermal_zone*/ (x86_pkg_temp) or hwmon coretemp
    CPU load     /proc/stat
    iGPU load    share of time the GPU was not idle (xe: gtidle/idle_residency_ms, i915: rc6_residency_ms)
    NPU load     /sys/class/accel/accel*/device/npu_busy_time_us  (intel_vpu driver)

    sudo python frontend/telemetry_linux.py            # sudo so the power counter is readable
    python frontend/telemetry_linux.py --once          # print one reading and exit

Not yet run on the station laptop: check `--once` there first. A value that cannot be read is left out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


class Sensors:
    """Reads counters and turns them into rates between two calls of sample()."""

    def __init__(self, root: str = "/"):
        self.root = Path(root)
        self._previous: dict = {}
        self._previous_time: float | None = None
        self.unreadable: list = []  # names of values that could not be read on the last sample

    # ---- raw counters

    def _rapl(self) -> tuple | None:
        for zone in sorted(self.root.glob("sys/class/powercap/intel-rapl:*")):
            if ":" in zone.name.split(":", 1)[1]:  # skip sub-zones like intel-rapl:0:0 (core, uncore)
                continue
            if _read(zone / "name") == "package-0":
                energy = _read_int(zone / "energy_uj")
                if energy is not None:
                    return energy, _read_int(zone / "max_energy_range_uj")
        return None

    def _cpu(self) -> tuple | None:
        text = _read(self.root / "proc/stat")
        if not text:
            return None
        fields = text.splitlines()[0].split()[1:]
        try:
            values = [int(v) for v in fields]
        except ValueError:
            return None
        if len(values) < 5:
            return None
        return sum(values[:8]), values[3] + values[4]  # total, idle + iowait

    def _gpu_idle_ms(self) -> list:
        paths = list(self.root.glob("sys/class/drm/card*/device/tile*/gt*/gtidle/idle_residency_ms"))
        paths += list(self.root.glob("sys/class/drm/card*/gt/gt*/rc6_residency_ms"))
        return [(str(p), v) for p in sorted(paths) if (v := _read_int(p)) is not None]

    def _npu_busy_us(self) -> int | None:
        for path in sorted(self.root.glob("sys/class/accel/accel*/device/npu_busy_time_us")):
            value = _read_int(path)
            if value is not None:
                return value
        return None

    def _temperature(self) -> tuple:
        """Returns (temp_c, limit_c); either may be None."""
        for zone in sorted(self.root.glob("sys/class/thermal/thermal_zone*")):
            if _read(zone / "type") != "x86_pkg_temp":
                continue
            temp = _read_int(zone / "temp")
            limit = None
            for kind in sorted(zone.glob("trip_point_*_type")):
                if _read(kind) == "critical":
                    limit = _read_int(kind.with_name(kind.name.replace("_type", "_temp")))
            if temp is not None:
                return temp / 1000, limit / 1000 if limit else None
        for hwmon in sorted(self.root.glob("sys/class/hwmon/hwmon*")):
            if _read(hwmon / "name") == "coretemp":
                temp, limit = _read_int(hwmon / "temp1_input"), _read_int(hwmon / "temp1_crit")
                if temp is not None:
                    return temp / 1000, limit / 1000 if limit else None
        return None, None

    # ---- rates

    def sample(self) -> dict:
        """One reading. Rates (power, loads) need a previous call, so the first sample only has temperature."""
        now = time.monotonic()
        current = {"rapl": self._rapl(), "cpu": self._cpu(), "gpu": self._gpu_idle_ms(), "npu": self._npu_busy_us()}
        previous, elapsed = self._previous, (now - self._previous_time) if self._previous_time else None
        self._previous, self._previous_time = current, now

        temp_c, limit_c = self._temperature()
        out = {"power_w": None, "temp_c": temp_c, "temp_limit_c": limit_c, "cpu": None, "gpu": None, "npu": None}
        if elapsed and elapsed > 0:
            if current["rapl"] and previous.get("rapl"):
                delta = current["rapl"][0] - previous["rapl"][0]
                if delta < 0 and current["rapl"][1]:  # counter wrapped around
                    delta += current["rapl"][1]
                if delta >= 0:
                    out["power_w"] = delta / 1e6 / elapsed
            if current["cpu"] and previous.get("cpu"):
                total = current["cpu"][0] - previous["cpu"][0]
                idle = current["cpu"][1] - previous["cpu"][1]
                if total > 0:
                    out["cpu"] = 100 * (1 - idle / total)
            before = dict(previous.get("gpu") or [])
            busy = [1 - (value - before[path]) / (elapsed * 1000) for path, value in current["gpu"] if path in before]
            if busy:
                out["gpu"] = 100 * max(busy)  # busiest engine group (render or media)
            if current["npu"] is not None and previous.get("npu") is not None:
                out["npu"] = 100 * (current["npu"] - previous["npu"]) / (elapsed * 1e6)
        for key in ("cpu", "gpu", "npu"):
            if out[key] is not None:
                out[key] = round(max(0.0, min(100.0, out[key])), 1)
        if out["power_w"] is not None:
            out["power_w"] = round(out["power_w"], 2)
        self.unreadable = [k for k, v in out.items() if v is None and k != "temp_limit_c"]
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    parser.add_argument("--once", action="store_true", help="print one reading as JSON and exit")
    parser.add_argument("--root", default="/", help=argparse.SUPPRESS)  # for tests with a fake /sys tree
    args = parser.parse_args()
    if not sys.platform.startswith("linux") and args.root == "/":
        raise SystemExit("telemetry_linux.py reads Linux kernel files; on this system send 'telemetry' events yourself.")

    sensors = Sensors(args.root)
    sensors.sample()
    time.sleep(max(0.2, args.interval))
    reading = sensors.sample()
    if sensors.unreadable:
        print("not readable here (left out): " + ", ".join(sensors.unreadable)
              + (". Power usually needs sudo." if "power_w" in sensors.unreadable else ""), file=sys.stderr)
    if args.once:
        print(json.dumps(reading, indent=2))
        return

    from hub_client import HubClient
    hub = HubClient(args.url)
    print(f"telemetry: sending a reading every {args.interval:g} s to {args.url} (Ctrl+C to stop)")
    try:
        while True:
            hub.telemetry(**reading)
            time.sleep(max(0.2, args.interval))
            reading = sensors.sample()
    except KeyboardInterrupt:
        hub.flush()


if __name__ == "__main__":
    main()
