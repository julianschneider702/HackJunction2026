"""Send events from the sorting loop to the dashboard hub. Python standard library only.

    from hub_client import HubClient
    hub = HubClient()
    hub.phase("perceive")
    hub.latency("policy", 41.7, device="NPU")
    hub.cycle(duration_s=14.2, anomaly_score=0.71, threshold=0.5,
              decision="defective", placed="defective", image=jpeg_bytes)
    if hub.stop_requested: ...

Nothing here raises or blocks: events go through a background thread, and if the hub is
down they are dropped. A dashboard problem must never stop or slow the robot loop.
Event fields: docs/CONTRACT.md.
"""

from __future__ import annotations

import atexit
import base64
import json
import queue
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8090"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route local traffic via a proxy


class HubClient:
    def __init__(self, url: str = DEFAULT_URL, source: str | None = None, timeout: float = 1.0,
                 latency_flush_s: float = 1.0, control_poll_s: float = 0.25):
        self.url = url.rstrip("/")
        self.source = source
        self.timeout = timeout
        self.stop_requested = False  # last answer from the hub; stays as it was while the hub is unreachable
        self.connected = False
        self.last_error: str | None = None  # most recent failure (hub down, or event rejected); not cleared
        self._latency_flush_s = latency_flush_s
        self._control_poll_s = control_poll_s  # how fast the Stop button is noticed while nothing is sent
        self._latencies: dict = {}  # (model, device) -> [ms, ...]
        self._latency_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._worker = threading.Thread(target=self._run, name="hub-client", daemon=True)
        self._worker.start()
        atexit.register(self.flush)

    # ---- events ----------------------------------------------------------------

    def send(self, type: str, **fields) -> None:
        """Queue one event. Fields with value None are left out."""
        event = {"type": type, **{k: v for k, v in fields.items() if v is not None}}
        if self.source:
            event["source"] = self.source
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass  # hub is down or slow: drop instead of blocking the loop

    def phase(self, phase: str, cycle_id: str | None = None, detail: str | None = None) -> None:
        self.send("phase", phase=phase, cycle_id=cycle_id, detail=detail)

    def cycle(self, duration_s: float, anomaly_score: float | None = None, threshold: float | None = None,
              decision: str | None = None, placed: str | None = None, truth: str | None = None,
              regrasps: int = 0, cycle_id: str | None = None, instruction: str | None = None,
              image: bytes | None = None, heatmap: bytes | None = None, correct: bool | None = None) -> None:
        """One finished brick. Only duration_s is required; send whatever else is known.

        image / heatmap: encoded JPEG or PNG bytes (e.g. cv2.imencode('.jpg', frame)[1].tobytes()).
        """
        self.send(
            "cycle", duration_s=duration_s, anomaly_score=anomaly_score, threshold=threshold, decision=decision,
            placed=placed, truth=truth, correct=correct, regrasps=regrasps, cycle_id=cycle_id, instruction=instruction,
            image_b64=base64.b64encode(image).decode("ascii") if image else None,
            heatmap_b64=base64.b64encode(heatmap).decode("ascii") if heatmap else None,
        )

    def latency(self, model: str, latency_ms: float, device: str | None = None) -> None:
        """Record one inference step. Buffered and sent about once per second, so it is cheap at 30 Hz."""
        with self._latency_lock:
            self._latencies.setdefault((model, device), []).append(float(latency_ms))

    def telemetry(self, power_w: float | None = None, temp_c: float | None = None,
                  temp_limit_c: float | None = None, cpu: float | None = None, gpu: float | None = None,
                  npu: float | None = None) -> None:
        util = {k: v for k, v in (("cpu", cpu), ("gpu", gpu), ("npu", npu)) if v is not None}
        self.send("telemetry", power_w=power_w, temp_c=temp_c, temp_limit_c=temp_limit_c, util=util or None)

    def models(self, models: list) -> None:
        """models: [{"role": "policy"|"anomaly", "name", "device", "precision", "runtime"}, ...]"""
        self.send("models", models=models)

    def benchmark(self, model: str, device: str, precision: str, latency_ms: float,
                  fps: float | None = None, power_w: float | None = None) -> None:
        self.send("benchmark", model=model, device=device, precision=precision, latency_ms=latency_ms,
                  fps=fps, power_w=power_w)

    def robot(self, state: str, message: str | None = None) -> None:
        self.send("robot", state=state, message=message)

    # ---- control -----------------------------------------------------------------

    def check_stop(self) -> bool:
        """Ask the hub right now (blocking, up to `timeout`). Usually reading `stop_requested` is enough."""
        try:
            with _OPENER.open(self.url + "/api/control", timeout=self.timeout) as response:
                self.stop_requested = bool(json.load(response).get("stop_requested"))
                self.connected = True
        except (OSError, ValueError):
            self.connected = False
        return self.stop_requested

    def flush(self, timeout: float = 3.0) -> None:
        """Wait until queued events are sent (call before the program exits)."""
        self._queue_latencies()
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)

    # ---- background sender ---------------------------------------------------------

    def _queue_latencies(self) -> None:
        with self._latency_lock:
            pending, self._latencies = self._latencies, {}
        for (model, device), values in pending.items():
            self.send("inference", model=model, device=device, latency_ms=values)

    def _run(self) -> None:
        last_flush = last_control = time.monotonic()
        while True:
            events = []
            try:
                events.append(self._queue.get(timeout=min(0.2, self._control_poll_s)))
                while len(events) < 50:
                    events.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            if events:
                self._post(events)  # the answer carries stop_requested
                last_control = time.monotonic()
                for _ in events:
                    self._queue.task_done()
            elif time.monotonic() - last_control >= self._control_poll_s:
                last_control = time.monotonic()
                self.check_stop()  # keep stop_requested fresh while the loop sends nothing
            if time.monotonic() - last_flush >= self._latency_flush_s:
                last_flush = time.monotonic()
                self._queue_latencies()

    def _post(self, events: list) -> None:
        request = urllib.request.Request(
            self.url + "/api/events", data=json.dumps({"events": events}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                self.stop_requested = bool(json.load(response).get("stop_requested"))
            self.connected = True
        except urllib.error.HTTPError as err:  # hub answered: an event was rejected
            self.connected = True
            try:
                self.last_error = json.load(err)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                self.last_error = f"HTTP {err.code}"
            err.close()
            if len(events) > 1:  # one bad event rejects the whole request: resend one by one
                for event in events:
                    self._post([event])
            else:
                print(f"hub_client: {events[0].get('type')} event rejected: {self.last_error}")
        except (OSError, ValueError) as err:
            self.connected, self.last_error = False, str(err)
