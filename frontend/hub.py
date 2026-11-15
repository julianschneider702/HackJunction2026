#!/usr/bin/env python3
"""Dashboard hub: collects events from the sorting loop and serves the dashboard.

Python standard library only. Interface: docs/CONTRACT.md.

    python frontend/hub.py                 # http://127.0.0.1:8090
    python frontend/hub.py --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"

MAX_BODY = 8 * 1024 * 1024
MAX_CYCLES = 200
MAX_IMAGES = 60
MAX_TELEMETRY = 300
MAX_LATENCY_SAMPLES = 500
MAX_LATENCY_HISTORY = 120
MAX_TEXT = 300
SIM_ACTIVE_S = 10  # demo badge stays on this long after the last simulator event

PHASES = ("idle", "perceive", "detect", "reason", "execute")
LABELS = ("good", "defective")
ROBOT_STATES = ("idle", "running", "stopped", "error")
INFERENCE_MODELS = ("policy", "anomaly")
DEVICE_ALIASES = {"CPU": "CPU", "GPU": "GPU", "IGPU": "GPU", "NPU": "NPU"}

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _bad(message: str) -> ApiError:
    return ApiError(400, "invalid_event", message)


def _number(ev: dict, key: str, required: bool = False, lo: float | None = None, hi: float | None = None):
    value = ev.get(key)
    if value is None:
        if required:
            raise _bad(f"'{key}' is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _bad(f"'{key}' must be a number")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise _bad(f"'{key}' is out of range")
    return value


def _text(ev: dict, key: str, required: bool = False):
    value = ev.get(key)
    if value is None:
        if required:
            raise _bad(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise _bad(f"'{key}' must be a string")
    return value[:MAX_TEXT]


def _boolean(ev: dict, key: str):
    value = ev.get(key)
    if value is not None and not isinstance(value, bool):
        raise _bad(f"'{key}' must be true or false")
    return value


def _choice(ev: dict, key: str, choices, required: bool = False):
    value = ev.get(key)
    if value is None:
        if required:
            raise _bad(f"'{key}' is required, one of: {', '.join(choices)}")
        return None
    if value not in choices:
        raise _bad(f"'{key}' must be one of: {', '.join(choices)}")
    return value


def _device(ev: dict, key: str = "device", required: bool = False):
    value = _text(ev, key, required)
    if value is None:
        return None
    device = DEVICE_ALIASES.get(value.upper())
    if device is None:
        raise _bad(f"'{key}' must be CPU, GPU or NPU")
    return device


def _image(ev: dict, key: str):
    """Decode a base64 JPEG/PNG. Returns (bytes, mime) or None."""
    value = ev.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad(f"'{key}' must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise _bad(f"'{key}' is not valid base64") from None
    if raw[:3] == b"\xff\xd8\xff":
        return raw, "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw, "image/png"
    raise _bad(f"'{key}' must be a JPEG or PNG image")


def _percentile(values, share: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))
    return ordered[index]


class Hub:
    """In-memory dashboard state. All public methods are thread-safe."""

    def __init__(self, data_dir: Path | None = None):
        self._lock = threading.Lock()
        self._data_dir = data_dir
        self._seq = 0
        self._stop_requested = False
        self._control_updated = now_iso()
        self._reset_data()
        self._load()

    # ---- state ---------------------------------------------------------------

    def _reset_data(self):
        self._last_sim = None  # time.monotonic() of the last simulator event
        self._robot = {"state": "idle", "message": None, "updated_at": now_iso()}
        self._phase = {"phase": "idle", "cycle_id": None, "detail": None, "since": now_iso()}
        self._cycles: deque = deque(maxlen=MAX_CYCLES)  # newest last
        self._images: dict = {}  # cycle id -> {"image": (bytes, mime), "heatmap": (bytes, mime)}
        self._next_cycle = 1
        self._telemetry: deque = deque(maxlen=MAX_TELEMETRY)
        self._samples = {m: deque(maxlen=MAX_LATENCY_SAMPLES) for m in INFERENCE_MODELS}
        self._latency_history = {m: deque(maxlen=MAX_LATENCY_HISTORY) for m in INFERENCE_MODELS}
        self._latency_count = {m: 0 for m in INFERENCE_MODELS}
        self._latency_device = {m: None for m in INFERENCE_MODELS}
        self._models: list = []
        self._benchmarks: dict = {}

    def _state_file(self) -> Path | None:
        return self._data_dir / "state.json" if self._data_dir else None

    def _load(self):
        path = self._state_file()
        if not path or not path.is_file():
            return
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            self._cycles.extend(saved.get("cycles", []))
            self._next_cycle = int(saved.get("next_cycle", len(self._cycles) + 1))
            self._models = list(saved.get("models", []))
            for row in saved.get("benchmarks", []):
                self._benchmarks[(row["model"], row["device"], row["precision"])] = row
        except (OSError, ValueError, KeyError, TypeError) as err:
            print(f"hub: could not read {path}, starting empty ({err})")
            self._reset_data()

    def _save(self):
        """Persist what should survive a restart (cycles, models, benchmarks). Caller holds the lock."""
        path = self._state_file()
        if not path:
            return
        payload = {
            "cycles": list(self._cycles),
            "next_cycle": self._next_cycle,
            "models": self._models,
            "benchmarks": list(self._benchmarks.values()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as err:
            print(f"hub: could not save {path} ({err})")

    # ---- events --------------------------------------------------------------

    def apply(self, events: list) -> int:
        """Validate all events first, then apply them. Raises ApiError, applies nothing on error."""
        prepared = [self._prepare(ev) for ev in events]
        with self._lock:
            persist = False
            for kind, data in prepared:
                persist = getattr(self, f"_apply_{kind}")(data) or persist
                if data["source"] == "sim":
                    self._last_sim = time.monotonic()
            self._seq += 1
            if persist:
                self._save()
        return len(prepared)

    def _prepare(self, ev) -> tuple:
        if not isinstance(ev, dict):
            raise _bad("an event must be a JSON object")
        kind = ev.get("type")
        builder = getattr(self, f"_prepare_{kind}", None) if isinstance(kind, str) else None
        if builder is None:
            raise _bad("'type' must be one of: phase, cycle, inference, telemetry, models, benchmark, robot")
        data = builder(ev)
        data["ts"] = _text(ev, "ts") or now_iso()
        data["source"] = _choice(ev, "source", ("live", "sim")) or "live"
        return kind, data

    def _prepare_phase(self, ev):
        return {
            "phase": _choice(ev, "phase", PHASES, required=True),
            "cycle_id": _text(ev, "cycle_id"),
            "detail": _text(ev, "detail"),
        }

    def _apply_phase(self, d):
        self._phase = {"phase": d["phase"], "cycle_id": d["cycle_id"], "detail": d["detail"], "since": d["ts"]}

    def _prepare_cycle(self, ev):
        regrasps = _number(ev, "regrasps", lo=0, hi=1000)
        if regrasps is not None and int(regrasps) != regrasps:
            raise _bad("'regrasps' must be an integer")
        return {
            "cycle_id": _text(ev, "cycle_id"),
            "duration_s": _number(ev, "duration_s", required=True, lo=0),
            "anomaly_score": _number(ev, "anomaly_score"),
            "threshold": _number(ev, "threshold"),
            "decision": _choice(ev, "decision", LABELS),
            "placed": _choice(ev, "placed", LABELS + ("none",)),
            "truth": _choice(ev, "truth", LABELS),
            "correct": _boolean(ev, "correct"),
            "regrasps": int(regrasps or 0),
            "instruction": _text(ev, "instruction"),
            "image": _image(ev, "image_b64"),
            "heatmap": _image(ev, "heatmap_b64"),
        }

    def _apply_cycle(self, d):
        cycle_id = d["cycle_id"]
        if not cycle_id:
            cycle_id = f"c-{self._next_cycle}"
        self._next_cycle += 1
        existing = next((c for c in self._cycles if c["id"] == cycle_id), None)
        if existing is not None:  # same id sent twice: replace instead of counting it twice
            self._cycles.remove(existing)
        self._cycles.append({
            "id": cycle_id,
            "finished_at": d["ts"],
            "duration_s": d["duration_s"],
            "anomaly_score": d["anomaly_score"],
            "threshold": d["threshold"],
            "decision": d["decision"],
            "placed": d["placed"],
            "truth": d["truth"],
            "truth_by": "sender" if d["truth"] else None,
            "reported_correct": d["correct"],
            "regrasps": d["regrasps"],
            "instruction": d["instruction"],
            "has_image": d["image"] is not None,
            "has_heatmap": d["heatmap"] is not None,
            "source": d["source"],
        })
        self._images.pop(cycle_id, None)
        if d["image"] or d["heatmap"]:
            self._images[cycle_id] = {"image": d["image"], "heatmap": d["heatmap"]}
            while len(self._images) > MAX_IMAGES:
                self._images.pop(next(iter(self._images)))
        return True

    def _prepare_inference(self, ev):
        raw = ev.get("latency_ms")
        values = raw if isinstance(raw, list) else [raw]
        if not values or len(values) > 5000:
            raise _bad("'latency_ms' must be a number or a list of 1-5000 numbers")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise _bad("'latency_ms' must be a non-negative number or a list of them")
        return {
            "model": _choice(ev, "model", INFERENCE_MODELS, required=True),
            "device": _device(ev),
            "values": [float(v) for v in values],
        }

    def _apply_inference(self, d):
        model = d["model"]
        self._samples[model].extend(d["values"])
        self._latency_count[model] += len(d["values"])
        self._latency_history[model].append({"ts": d["ts"], "ms": sum(d["values"]) / len(d["values"])})
        if d["device"]:
            self._latency_device[model] = d["device"]

    def _prepare_telemetry(self, ev):
        util_in = ev.get("util") or {}
        if not isinstance(util_in, dict):
            raise _bad("'util' must be an object")
        util = {key: _number(util_in, key, lo=0, hi=100) for key in ("cpu", "gpu", "npu")}
        return {
            "power_w": _number(ev, "power_w", lo=0),
            "temp_c": _number(ev, "temp_c"),
            "temp_limit_c": _number(ev, "temp_limit_c"),
            "util": util,
        }

    def _apply_telemetry(self, d):
        self._telemetry.append({k: d[k] for k in ("ts", "power_w", "temp_c", "temp_limit_c", "util")})

    def _prepare_models(self, ev):
        rows = ev.get("models")
        if not isinstance(rows, list) or len(rows) > 20:
            raise _bad("'models' must be a list (max 20 entries)")
        models = []
        for row in rows:
            if not isinstance(row, dict):
                raise _bad("each entry of 'models' must be an object")
            models.append({
                "role": _choice(row, "role", INFERENCE_MODELS, required=True),
                "name": _text(row, "name", required=True),
                "device": _device(row, required=True),
                "precision": _text(row, "precision"),
                "runtime": _text(row, "runtime"),
            })
        return {"models": models}

    def _apply_models(self, d):
        self._models = d["models"]
        return True

    def _prepare_benchmark(self, ev):
        return {
            "model": _text(ev, "model", required=True),
            "device": _device(ev, required=True),
            "precision": _text(ev, "precision", required=True),
            "latency_ms": _number(ev, "latency_ms", required=True, lo=0),
            "fps": _number(ev, "fps", lo=0),
            "power_w": _number(ev, "power_w", lo=0),
        }

    def _apply_benchmark(self, d):
        row = {k: d[k] for k in ("model", "device", "precision", "latency_ms", "fps", "power_w", "ts", "source")}
        self._benchmarks[(d["model"], d["device"], d["precision"])] = row
        return True

    def _prepare_robot(self, ev):
        return {"state": _choice(ev, "state", ROBOT_STATES, required=True), "message": _text(ev, "message")}

    def _apply_robot(self, d):
        self._robot = {"state": d["state"], "message": d["message"], "updated_at": d["ts"]}

    # ---- control, verdicts, images ---------------------------------------------

    def control(self) -> dict:
        with self._lock:
            return {"stop_requested": self._stop_requested, "updated_at": self._control_updated}

    def command(self, command) -> bool:
        if command not in ("stop", "resume", "reset"):
            raise ApiError(400, "invalid_command", "'command' must be stop, resume or reset")
        with self._lock:
            if command == "reset":
                self._reset_data()
                self._save()
            else:
                self._stop_requested = command == "stop"
                self._control_updated = now_iso()
            self._seq += 1
            return self._stop_requested

    def verdict(self, cycle_id: str, truth):
        if truth is not None and truth not in LABELS:
            raise ApiError(400, "invalid_truth", "'truth' must be good, defective or null")
        with self._lock:
            cycle = next((c for c in self._cycles if c["id"] == cycle_id), None)
            if cycle is None:
                raise ApiError(404, "unknown_cycle", f"no cycle '{cycle_id}'")
            cycle["truth"] = truth
            cycle["truth_by"] = "operator" if truth else None
            self._seq += 1
            self._save()

    def image(self, cycle_id: str, kind: str):
        with self._lock:
            entry = self._images.get(cycle_id, {}).get(kind)
        if entry is None:
            raise ApiError(404, "no_image", f"no {kind} for cycle '{cycle_id}'")
        return entry

    # ---- snapshot --------------------------------------------------------------

    @staticmethod
    def _correct(cycle: dict):
        """True / False / None (unknown). An operator's mark in the dashboard beats what the sender reported."""
        compared = cycle["placed"] == cycle["truth"] if cycle["truth"] and cycle["placed"] else None
        if cycle.get("truth_by") == "operator" and compared is not None:
            return compared
        reported = cycle.get("reported_correct")
        return reported if reported is not None else compared

    @classmethod
    def _stats(cls, cycles: list) -> dict:
        results = [cls._correct(c) for c in cycles]
        judged = [r for r in results if r is not None]
        correct = [r for r in judged if r]
        last10 = [c["duration_s"] for c in cycles[-10:]]
        avg10 = sum(last10) / len(last10) if last10 else None
        return {
            "total": len(cycles),
            "good": sum(1 for c in cycles if c["placed"] == "good"),
            "defective": sum(1 for c in cycles if c["placed"] == "defective"),
            "lost": sum(1 for c in cycles if c["placed"] == "none"),
            "judged": len(judged),
            "correct": len(correct),
            "accuracy": len(correct) / len(judged) if judged else None,
            "avg_cycle_s": sum(c["duration_s"] for c in cycles) / len(cycles) if cycles else None,
            "avg_cycle_s_last10": avg10,
            "bricks_per_min": 60 / avg10 if avg10 else None,
            "regrasps": sum(c["regrasps"] for c in cycles),
        }

    def _demo(self, cycles: list) -> bool:
        """True while simulator data is on screen: recent sim events, or stored sim cycles/benchmarks."""
        recent = self._last_sim is not None and time.monotonic() - self._last_sim < SIM_ACTIVE_S
        return (recent or any(c.get("source") == "sim" for c in cycles)
                or any(b.get("source") == "sim" for b in self._benchmarks.values()))

    def snapshot(self, since: int | None = None) -> dict:
        with self._lock:
            if since is not None and since == self._seq:
                return {"seq": self._seq, "unchanged": True}
            cycles = list(self._cycles)
            inference = {}
            for model in INFERENCE_MODELS:
                samples = self._samples[model]
                inference[model] = {
                    "device": self._latency_device[model],
                    "count": self._latency_count[model],
                    "last_ms": samples[-1] if samples else None,
                    "mean_ms": sum(samples) / len(samples) if samples else None,
                    "p95_ms": _percentile(samples, 0.95) if samples else None,
                    "history": list(self._latency_history[model]),
                }
            telemetry = list(self._telemetry)
            return {
                "seq": self._seq,
                "server_time": now_iso(),
                "demo": self._demo(cycles),
                "robot": dict(self._robot),
                "control": {"stop_requested": self._stop_requested, "updated_at": self._control_updated},
                "phase": dict(self._phase),
                "stats": self._stats(cycles),
                "cycles": [{**c, "correct": self._correct(c)} for c in reversed(cycles)],
                "telemetry": {"latest": telemetry[-1] if telemetry else None, "history": telemetry},
                "inference": inference,
                "models": list(self._models),
                "benchmarks": sorted(self._benchmarks.values(), key=lambda r: (r["model"], r["latency_ms"])),
            }


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DashboardHub/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the terminal quiet; errors are answered as JSON
            pass

        # ---- helpers ----

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200):
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _error(self, err: ApiError):
            # The request body may be unread at this point, so do not reuse the connection.
            self.close_connection = True
            self._json({"error": {"code": err.code, "message": err.message}}, err.status)

        def _body(self) -> dict:
            # Requiring JSON keeps other web pages from posting to the hub (no CORS headers are sent).
            if "application/json" not in (self.headers.get("Content-Type") or "").lower():
                raise ApiError(415, "unsupported_media_type", "send Content-Type: application/json")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError(400, "invalid_json", "bad Content-Length") from None
            if length > MAX_BODY:
                raise ApiError(413, "too_large", f"body is larger than {MAX_BODY} bytes")
            try:
                payload = json.loads(self.rfile.read(length) or b"null")
            except ValueError:
                raise ApiError(400, "invalid_json", "body is not valid JSON") from None
            if not isinstance(payload, dict):
                raise ApiError(400, "invalid_json", "body must be a JSON object")
            return payload

        def _static(self, path: str):
            relative = "index.html" if path in ("", "/") else unquote(path).lstrip("/")
            target = (WEB_DIR / relative).resolve()
            if WEB_DIR not in target.parents or not target.is_file() or target.suffix not in MIME:
                raise ApiError(404, "not_found", "no such page")
            self._send(200, target.read_bytes(), MIME[target.suffix])

        # ---- routes ----

        def do_GET(self):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            try:
                if url.path == "/api/state":
                    since = parse_qs(url.query).get("since", [None])[0]
                    self._json(hub.snapshot(int(since) if since and since.lstrip("-").isdigit() else None))
                elif url.path == "/api/control":
                    self._json(hub.control())
                elif len(parts) == 4 and parts[:2] == ["api", "cycles"] and parts[3] in ("image", "heatmap"):
                    data, mime = hub.image(parts[2], parts[3])
                    self._send(200, data, mime)
                elif parts[:1] == ["api"]:
                    raise ApiError(404, "not_found", "no such endpoint")
                else:
                    self._static(url.path)
            except ApiError as err:
                self._error(err)

        def do_POST(self):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            try:
                if url.path == "/api/events":
                    payload = self._body()
                    events = payload["events"] if "events" in payload else [payload]
                    if not isinstance(events, list):
                        raise _bad("'events' must be a list")
                    accepted = hub.apply(events)
                    self._json({"ok": True, "accepted": accepted, "stop_requested": hub.control()["stop_requested"]})
                elif url.path == "/api/control":
                    self._json({"ok": True, "stop_requested": hub.command(self._body().get("command"))})
                elif len(parts) == 4 and parts[:2] == ["api", "cycles"] and parts[3] == "verdict":
                    hub.verdict(parts[2], self._body().get("truth"))
                    self._json({"ok": True})
                else:
                    raise ApiError(404, "not_found", "no such endpoint")
            except ApiError as err:
                self._error(err)

    return Handler


def make_server(host: str, port: int, hub: Hub) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(hub))
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 makes the dashboard reachable from other machines")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    parser.add_argument("--data-dir", default=str(HERE / "data"), help="where cycles and benchmarks are kept across restarts")
    parser.add_argument("--no-persist", action="store_true", help="keep everything in memory only")
    args = parser.parse_args()

    hub = Hub(None if args.no_persist else Path(args.data_dir))
    server = make_server(args.host, args.port, hub)
    print(f"dashboard: http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
