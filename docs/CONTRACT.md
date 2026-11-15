# Interface contract: sorting loop and dashboard

The interface between everything that produces data (sorting loop, recording tools, benchmarks, telemetry) and the live dashboard in `frontend/`.
Every change gets a line in the change log at the end.

## Overview

```text
sorting loop (policy + Anomalib) ──POST /api/events──►  dashboard hub  ◄──GET /api/state── dashboard (browser)
telemetry collector              ──POST /api/events──►  (frontend/hub.py)
sorting loop                     ◄──stop_requested────
```

The **hub** (`frontend/hub.py`, Python stdlib only) keeps the live state in memory and serves the dashboard.
Everything that produces data (sorting loop, benchmarks, telemetry) only has to POST events to it.
Python code can use `frontend/hub_client.py` instead of raw HTTP.

## Conventions

- IDs: strings. Cycle ids are chosen by the sender (`cycle_id`) or assigned by the hub (`c-1`, `c-2`, ...).
- Timestamps: ISO 8601, UTC (`2026-11-15T14:30:00.000Z`). Every event may carry `ts`; the hub fills it in if missing.
- Errors: `{ "error": { "code": "<machine_code>", "message": "<human text>" } }` with a matching HTTP status
- Auth: none. The hub binds to `127.0.0.1` by default; everything runs on the station laptop.
- Base URL (local): `http://127.0.0.1:8090`
- Requests with a body must send `Content-Type: application/json`. Body limit 8 MB.
- Labels: `"good"` or `"defective"`. Devices: `"CPU"`, `"GPU"` (integrated Arc GPU), `"NPU"` (OpenVINO device names).
- Demo data: events from the simulator carry `"source": "sim"`; the dashboard then shows a "Demo data" badge. Real senders omit `source` or send `"live"`.

## Data model

### Event (what producers send)

Every event is a JSON object with a `type`. Optional on all events: `ts`, `source`.

| `type` | Fields | Meaning |
|---|---|---|
| `phase` | `phase` (required: `idle`, `perceive`, `detect`, `reason`, `execute`), `cycle_id`, `detail` | The loop entered a pipeline phase. `detail` is free text shown in the dashboard. |
| `cycle` | see table below | One brick is finished (placed or lost). |
| `inference` | `model` (required: `policy` or `anomaly`), `latency_ms` (required: number or list of numbers), `device` | Latency of one or more inference steps. Send policy steps in batches (about once per second). |
| `telemetry` | `power_w`, `temp_c`, `temp_limit_c`, `util` (`{ "cpu": 0-100, "gpu": 0-100, "npu": 0-100 }`) | Hardware reading. All fields optional, send what you can measure. |
| `models` | `models` (required: list of `{ "role": "policy"/"anomaly", "name", "device", "precision", "runtime" }`) | What is deployed where right now. Replaces the previous list. |
| `benchmark` | `model`, `device`, `precision`, `latency_ms` (all required), `fps`, `power_w` | One row of the device comparison table. Same `model` + `device` + `precision` overwrites the old row. |
| `robot` | `state` (required: `idle`, `running`, `stopped`, `error`), `message` | State of the arm / loop. Send `stopped` after you honoured a stop request. |

### `cycle` event fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `cycle_id` | string | no | Hub assigns `c-<n>` if missing |
| `duration_s` | number | yes | Pick to drop, in seconds |
| `anomaly_score` | number | no | Score of the final inspection |
| `threshold` | number | no | Score above it = defective |
| `decision` | label | no | What the anomaly model decided |
| `placed` | label or `"none"` | no | Bowl the brick ended up in; `"none"` = dropped / lost. Leave out if unknown. |
| `truth` | label | no | What the brick really was, if the sender knows. Otherwise the operator marks it in the dashboard. |
| `correct` | boolean | no | The sender knows whether the brick was sorted correctly (for example a person judged it) |
| `regrasps` | integer | no | How often the brick was picked up again (default 0) |
| `instruction` | string | no | Language instruction given to the policy |
| `image_b64` | string | no | Inspection camera image, base64 JPEG or PNG |
| `heatmap_b64` | string | no | Anomaly heatmap, base64 JPEG or PNG, same framing as the image |

Only `duration_s` is required: a sender without an anomaly model (the recording tools in `robot/scripts/`)
leaves out score, threshold and decision.

Whether a cycle counts as **correct** (field `correct` in the state, `true` / `false` / `null`):
`placed == truth` if the operator marked the brick in the dashboard and `placed` is known; otherwise the
sender's `correct`; otherwise `placed == truth` if both are known; otherwise unknown (not counted).

Example:

```json
{
  "type": "cycle",
  "duration_s": 14.2,
  "anomaly_score": 0.71,
  "threshold": 0.5,
  "decision": "defective",
  "placed": "defective",
  "regrasps": 0,
  "instruction": "put the brick into the defective bowl"
}
```

### State (what the dashboard reads)

`GET /api/state` returns one object: `seq` (increments on every change), `demo` (simulator data is on screen), `robot`, `control`, `phase`,
`stats`, `cycles` (newest first, without image data, with `has_image` / `has_heatmap`), `telemetry`
(`latest`, `history`), `inference` (per model: `device`, `count`, `last_ms`, `mean_ms`, `p95_ms`, `history`),
`models`, `benchmarks`. The exact shape is defined by `Hub.snapshot()` in `frontend/hub.py`.

## Endpoints / functions

### `POST /api/events`

- **Purpose:** send one event or several.
- **Request:** one event object, or `{ "events": [ ... ] }`.
- **Response (200):** `{ "ok": true, "accepted": <n>, "stop_requested": <bool> }`
- **Errors:** `400 invalid_json`, `400 invalid_event` (message names the field; nothing from that request is applied), `413 too_large`, `415 unsupported_media_type`

### `GET /api/state`

- **Purpose:** full dashboard state. `?since=<seq>` returns `{ "seq": <seq>, "unchanged": true }` if nothing changed.
- **Response (200):** state object (see above).

### `GET /api/control`

- **Purpose:** the sorting loop asks whether the operator pressed Stop.
- **Response (200):** `{ "stop_requested": <bool>, "updated_at": "<ts>" }`

The loop **must** check this (or the `stop_requested` field in the `POST /api/events` response) at least once
per control step, stop the arm safely when it is `true`, and then send a `robot` event with `state: "stopped"`.
The dashboard button is an additional stop, not a replacement for a local one (key press, power switch).

### `POST /api/control`

- **Purpose:** dashboard buttons.
- **Request:** `{ "command": "stop" | "resume" | "reset" }`. `reset` deletes all recorded cycles, statistics, telemetry and benchmarks.
- **Response (200):** `{ "ok": true, "stop_requested": <bool> }`
- **Errors:** `400 invalid_command`

### `POST /api/cycles/<cycle_id>/verdict`

- **Purpose:** operator marks what a brick really was.
- **Request:** `{ "truth": "good" | "defective" | null }` (`null` clears it)
- **Response (200):** `{ "ok": true }`
- **Errors:** `404 unknown_cycle`, `400 invalid_truth`

### `GET /api/cycles/<cycle_id>/image` and `/heatmap`

- **Purpose:** image data of a cycle (kept in memory for the newest 60 cycles).
- **Response (200):** `image/jpeg` or `image/png`
- **Errors:** `404 no_image`

### Python helper (`frontend/hub_client.py`)

```python
from hub_client import HubClient

hub = HubClient()                       # http://127.0.0.1:8090, never raises, sends in the background
hub.phase("detect", detail="scoring brick")
hub.latency("policy", 41.7, device="NPU")   # buffered, sent about once per second
hub.cycle(duration_s=14.2, anomaly_score=0.71, threshold=0.5, decision="defective",
          placed="defective", image=jpeg_bytes, heatmap=heatmap_jpeg_bytes)
if hub.stop_requested:                  # refreshed about four times per second
    ...                                 # stop the arm, then hub.robot("stopped")
```

## Change log

| Date | Who | Change |
|---|---|---|
| 2026-11-15 | Tobby | Initial version: dashboard hub events, state and control endpoints |
| 2026-11-15 | Tobby | `cycle`: only `duration_s` required, new optional `correct`; state cycles carry the computed `correct` |
