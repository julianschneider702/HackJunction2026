#!/usr/bin/env python3
"""Measure an OpenVINO model on CPU, iGPU and NPU and send the results to the dashboard's comparison table.

    python frontend/benchmark_openvino.py path/to/model.xml --name SmolVLA
    python frontend/benchmark_openvino.py model.xml --name PaDiM --devices CPU GPU --runs 200
    python frontend/benchmark_openvino.py model.xml --name PaDiM --no-send     # only print

Needs the `openvino` package (the Studio environment and the Anomalib environment both have it).
The inputs are random data in the model's input shapes, so this measures inference speed only.
Power is the CPU package power during the run and is only measured on Linux with a readable
power counter (run with sudo); otherwise the column stays empty.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from hub_client import DEFAULT_URL, HubClient
from telemetry_linux import Sensors

DEVICE_ORDER = ("CPU", "GPU", "NPU")


def precision_of(model) -> str:
    """INT8 if the model is quantized, else FP16 or FP32 by what most of its weights are stored as."""
    weights = {"f16": 0, "f32": 0}
    for op in model.get_ops():
        kind = op.get_type_name()
        if kind == "FakeQuantize":
            return "INT8"
        if kind == "Constant":
            name = op.get_element_type().get_type_name()
            if name in weights:
                size = 1
                for dim in op.get_output_shape(0):
                    size *= dim
                weights[name] += size
    return "FP16" if weights["f16"] > weights["f32"] else "FP32"


def make_static(model) -> None:
    """Give every dynamic input dimension the size 1 (the NPU needs fixed shapes)."""
    shapes = {}
    for port in model.inputs:
        partial = port.get_partial_shape()
        if partial.is_dynamic:
            shapes[port.get_any_name()] = [dim.get_length() if dim.is_static else 1 for dim in partial]
    if shapes:
        print("dynamic input dimensions set to 1: " + ", ".join(f"{k} {v}" for k, v in shapes.items()))
        model.reshape(shapes)


def random_inputs(compiled) -> dict:
    import numpy as np

    rng = np.random.default_rng(0)
    inputs = {}
    for index, port in enumerate(compiled.inputs):
        dtype = port.get_element_type().to_dtype()
        shape = list(port.get_shape())
        if np.issubdtype(dtype, np.floating):
            inputs[index] = rng.random(shape, dtype=np.float32).astype(dtype)
        else:
            inputs[index] = np.ones(shape, dtype=dtype)  # ids / masks: any valid value will do
    return inputs


def measure(core, model, device: str, runs: int, warmup: int, sensors: Sensors | None) -> dict:
    started = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_s = time.perf_counter() - started
    request = compiled.create_infer_request()
    inputs = random_inputs(compiled)
    for _ in range(warmup):
        request.infer(inputs)
    if sensors:
        sensors.sample()
    latencies = []
    for _ in range(runs):
        started = time.perf_counter()
        request.infer(inputs)
        latencies.append((time.perf_counter() - started) * 1000)
    power = sensors.sample()["power_w"] if sensors else None
    ordered = sorted(latencies)
    return {
        "latency_ms": statistics.median(latencies),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "fps": 1000 / statistics.fmean(latencies),
        "power_w": power,
        "compile_s": compile_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", type=Path, help="OpenVINO model (.xml) or ONNX file")
    parser.add_argument("--name", help="name shown in the dashboard (default: the file name)")
    parser.add_argument("--devices", nargs="+", help="devices to measure (default: all of CPU, GPU, NPU that exist)")
    parser.add_argument("--precision", help="label like FP32, FP16, INT8 (default: read from the model)")
    parser.add_argument("--runs", type=int, default=100, help="timed inferences per device (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed inferences before (default: %(default)s)")
    parser.add_argument("--url", default=DEFAULT_URL, help="dashboard hub (default: %(default)s)")
    parser.add_argument("--no-send", action="store_true", help="print the results, send nothing")
    args = parser.parse_args()

    try:
        import openvino as ov
    except ImportError:
        print("The 'openvino' package is missing. Run this with the Studio or Anomalib Python environment.", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1

    core = ov.Core()
    model = core.read_model(args.model)
    make_static(model)
    name = args.name or args.model.stem
    precision = args.precision or precision_of(model)
    present = {d.split(".")[0] for d in core.available_devices}
    devices = [d.upper() for d in args.devices] if args.devices else [d for d in DEVICE_ORDER if d in present]
    sensors = Sensors() if sys.platform.startswith("linux") else None

    hub = None if args.no_send else HubClient(args.url)
    print(f"{name} ({precision}), {args.runs} runs per device. OpenVINO sees: {', '.join(core.available_devices)}")
    print(f"{'device':8}{'median ms':>11}{'p95 ms':>9}{'FPS':>8}{'power W':>9}{'compile s':>11}")
    failed = 0
    for device in devices:
        try:
            result = measure(core, model, device, args.runs, args.warmup, sensors)
        except Exception as error:  # noqa: BLE001 - a device that cannot run the model must not stop the others
            failed += 1
            lines = [line.strip() for line in str(error).splitlines() if line.strip()]
            reason = next((line for line in lines if not line.startswith("Exception from")), lines[0] if lines else "")
            print(f"{device:8}failed: {reason[:160]}")
            continue
        power = f"{result['power_w']:.1f}" if result["power_w"] is not None else "-"
        print(f"{device:8}{result['latency_ms']:>11.2f}{result['p95_ms']:>9.2f}{result['fps']:>8.1f}{power:>9}{result['compile_s']:>11.1f}")
        if hub and device in DEVICE_ORDER:
            hub.benchmark(name, device, precision, round(result["latency_ms"], 2), fps=round(result["fps"], 1),
                          power_w=round(result["power_w"], 1) if result["power_w"] is not None else None)
    if hub:
        hub.flush()
        print("sent to the dashboard" if hub.connected else f"dashboard hub not reachable at {args.url}: nothing sent")
    return 1 if failed == len(devices) else 0


if __name__ == "__main__":
    sys.exit(main())
