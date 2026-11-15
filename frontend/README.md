# frontend/ - live dashboard for the inspection cell

Shows what the judges score: sorting results and accuracy, cycle time, latency per inference step,
power and temperature, and which model runs on CPU / iGPU / NPU. Owner: Tobby.

Python standard library and plain HTML/JS/CSS only. Nothing to install, no build step, works offline.

## Run

The dashboard is a read-only display: no buttons, no camera view, no setup. Open it, it updates itself
every half second and hides panels that have no data yet (score chart, device comparison).

```bash
python frontend/hub.py          # dashboard at http://127.0.0.1:8090
```

Without the robot, add demo data in a second terminal (the dashboard then shows a "Demo data" badge):

```bash
python frontend/simulate.py     # --speed 3 for faster cycles
```

Tests: `python -m unittest discover -s frontend/tests`

## On the station laptop (real data)

Three terminals, from the repo root. Nothing here is simulated; do not start `simulate.py`,
and if demo bricks are still shown, stop the hub and delete `frontend/data/` once.

```bash
# 1) dashboard + power / temperature / CPU, iGPU, NPU load
bash frontend/start_laptop.sh

# 2) policy runs on the arm; every cycle a person judges appears in the dashboard (see robot/README.md)
cd ../physical-ai-studio/application/backend
uv run --extra xpu python ../../../HackJunction2026/robot/scripts/record_successes_gui.py --device NPU

# 3) device comparison table: one run per exported model
uv run --extra xpu python ../../../HackJunction2026/frontend/benchmark_openvino.py <model.xml> --name SmolVLA
```

Exported models are below Studio's storage folder: `find ~/.local/share/physicalai -name "*.xml"`.

What is real after these steps, and what is still missing:

| Dashboard part | Source | State |
|---|---|---|
| Bricks, cycle time, throughput, accuracy, image | `robot/scripts/record_successes*.py` | tested against a stand-in Studio server, not yet on the arm |
| Power, temperature, device load | `telemetry_linux.py` | tested with copied kernel files only; run `--once` on the laptop first |
| Device comparison table | `benchmark_openvino.py` | ran on a Mac CPU with a small test model; GPU / NPU untested |
| Anomaly score, heatmap, model decision | nothing sends them yet | needs the Anomalib model wired into the loop |
| Live latency per policy step | nothing sends it yet | Studio does not report it; the comparison table has the per-device latency instead |

## Files

| File | What it does |
|---|---|
| `hub.py` | Small HTTP server: takes events, keeps the live state, serves the dashboard. Cycles, models and benchmark rows survive a restart (`frontend/data/`, git-ignored); images, telemetry and latency samples do not. |
| `hub_client.py` | Helper for the sorting loop to send events. Never raises and never blocks the loop. |
| `web/` | The dashboard page (`index.html`, `app.js`, `styles.css`). |
| `simulate.py` | Sends invented cycles, latencies and readings. For development and for rehearsing the demo only. |
| `telemetry_linux.py` | Reads power, temperature and CPU / iGPU / NPU load from Linux kernel files and sends them. |
| `benchmark_openvino.py` | Times an OpenVINO model on CPU, iGPU and NPU and fills the device comparison table. |
| `start_laptop.sh` | Starts hub and `telemetry_linux.py` together on the station laptop. |
| `tests/` | Tests for hub, client and telemetry reader. |

## Connect the real pipeline

The interface is in [`docs/CONTRACT.md`](../docs/CONTRACT.md). In short, from the sorting loop:

```python
import sys; sys.path.insert(0, "frontend")
from hub_client import HubClient

hub = HubClient()
hub.robot("running")
hub.models([{"role": "policy", "name": "SmolVLA", "device": "NPU", "precision": "INT8", "runtime": "OpenVINO"},
            {"role": "anomaly", "name": "PaDiM", "device": "GPU", "precision": "FP16", "runtime": "OpenVINO"}])

hub.phase("detect", detail="scoring brick")
hub.latency("anomaly", 17.2, device="GPU")       # one call per inference, also for every policy step
hub.cycle(duration_s=14.2, anomaly_score=0.71, threshold=0.5, decision="defective", placed="defective",
          image=jpeg_bytes, heatmap=heatmap_jpeg_bytes)

if hub.stop_requested:                            # operator pressed Stop in the dashboard
    ...                                           # stop the arm safely, then:
    hub.robot("stopped")
```

For the device comparison table send one `hub.benchmark(model, device, precision, latency_ms, fps, power_w)`
per combination you measured (for example from OpenVINO `benchmark_app`).

## Things to know

- **Stop:** the dashboard has no Stop button any more. `POST /api/control` still works for scripts
  (`curl -X POST -d '{"command":"stop"}' .../api/control`), but the real stop is local: terminal, key press, power switch.
- **Sorting accuracy** needs the truth per brick. The loop sends `truth` (or a script posts it to
  `/api/cycles/<id>/verdict`). The accuracy tile only appears once at least one brick has it.
- **Telemetry on the laptop:** `sudo python frontend/telemetry_linux.py` (power needs root on most kernels).
  It has not run on the station laptop yet: try `python frontend/telemetry_linux.py --once` there first.
  It is Linux only; on Windows send `telemetry` events from your own measurement.
- **Other machines:** `python frontend/hub.py --host 0.0.0.0` makes the dashboard reachable in the network.
  There is no login, so only do that on a network you trust.
