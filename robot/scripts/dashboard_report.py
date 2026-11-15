"""Report what the recording tools do to the dashboard hub (``frontend/hub.py``).

Used by ``record_successes.py`` and ``record_successes_gui.py``. Every policy run on the arm
becomes a real cycle in the dashboard: task, duration, whether the person judged it correct,
and a camera image. The dashboard's Stop button aborts the running cycle.

Reporting is optional. Without a running hub, or with ``--dashboard off``, the tools work as before.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
DEFAULT_URL = "http://127.0.0.1:8090"

DEFECTIVE_WORDS = ("defect", "defekt", "faulty", "broken", "kaputt", "bad", "schlecht", "reject")
GOOD_WORDS = ("good", "gut", "intact", "okay", " ok ")
DEVICES = {"CPU": "CPU", "GPU": "GPU", "NPU": "NPU"}


def bowl_for(task: str) -> str | None:
    """The bowl a task instruction names ("good" or "defective"), or None if it names neither."""
    text = f" {task.lower()} "
    if any(word in text for word in DEFECTIVE_WORDS):
        return "defective"
    if any(word in text for word in GOOD_WORDS):
        return "good"
    return None


class DashboardReporter:
    """Thin layer over ``HubClient``. Every method is safe to call when reporting is off."""

    def __init__(self, url: str | None = DEFAULT_URL, camera: str | None = None) -> None:
        self.hub: Any = None
        self.camera = camera
        self.frame: bytes | None = None  # newest JPEG of the chosen camera
        self.frame_key: str | None = None
        self.cameras: list[str] = []
        self.run_id = time.strftime("%H%M%S")
        self.cycle_number = 0
        self.cycle_id: str | None = None
        if not url or url.lower() == "off":
            return
        try:
            sys.path.insert(0, str(FRONTEND))
            from hub_client import HubClient  # noqa: PLC0415 - optional, lives in frontend/
        except ImportError:
            print(f"Dashboard: {FRONTEND / 'hub_client.py'} nicht gefunden, es wird nichts gemeldet.")
            return
        self.hub = HubClient(url)
        self.hub.check_stop()
        if self.hub.connected:
            print(f"Dashboard: verbunden mit {url}")
        else:
            print(f"Dashboard: {url} nicht erreichbar. Läuft frontend/hub.py? Es geht auch ohne weiter.")

    # ---- stop button ---------------------------------------------------------

    @property
    def stop_requested(self) -> bool:
        return bool(self.hub and self.hub.stop_requested)

    # ---- session ---------------------------------------------------------------

    def session_started(self, found: dict[str, Any], device: str) -> None:
        if not self.hub:
            return
        self.hub.robot("running", "Recording tool: a person starts and judges every cycle")
        self.hub.phase("idle", detail="Waiting for the next brick")
        known = DEVICES.get(device.upper().split(".")[0])
        if known:
            runtime = {"openvino": "OpenVINO", "onnx": "ONNX Runtime", "torch": "PyTorch"}.get(found["backend"], found["backend"])
            self.hub.models([{"role": "policy", "name": found["model"]["name"], "device": known, "runtime": runtime}])
        else:
            print(f"Dashboard: Gerät {device!r} ist kein festes Gerät. Mit --device CPU, GPU oder NPU "
                  "zeigt das Dashboard, wo das Modell läuft.")

    def session_closed(self) -> None:
        if not self.hub:
            return
        self.hub.phase("idle")
        self.hub.robot("idle", "Recording tool closed")
        self.hub.flush()

    # ---- cycles ------------------------------------------------------------------

    def cycle_started(self, task: str) -> None:
        if not self.hub:
            return
        self.cycle_number += 1
        self.cycle_id = f"run{self.run_id}-{self.cycle_number}"
        self.hub.robot("running")
        self.hub.phase("execute", cycle_id=self.cycle_id, detail=f"Policy runs: {task}")

    def cycle_finished(self, task: str, verdict: str, duration_s: float) -> None:
        """verdict: saved (judged correct), wrong, timeout, or anything else (aborted / failed: not counted)."""
        if not self.hub:
            return
        if verdict in ("saved", "wrong", "timeout"):
            bowl = bowl_for(task)
            correct = verdict == "saved"
            self.hub.cycle(
                cycle_id=self.cycle_id, duration_s=round(duration_s, 2), instruction=task,
                truth=bowl, placed=bowl if correct else None, correct=correct, image=self.frame,
            )
        self.hub.phase("idle", detail="Waiting for the next brick")

    def stopped_by_dashboard(self) -> None:
        if self.hub:
            self.hub.robot("stopped", "Stopped from the dashboard")

    def resumed(self) -> None:
        if self.hub:
            self.hub.robot("running")

    # ---- camera ------------------------------------------------------------------

    def on_frames(self, raw: bytes) -> None:
        """Keep the newest JPEG of the chosen camera from one of Studio's observation messages."""
        if not self.hub:
            return
        try:
            size = int.from_bytes(raw[:4], "big")
            header = json.loads(raw[4 : 4 + size])
            offset = 4 + size
            frames = header["frames"]
            self.cameras = [frame["key"] for frame in frames]
            wanted = self._pick_camera()
            for frame in frames:
                if frame["key"] == wanted:
                    self.frame, self.frame_key = raw[offset : offset + frame["size"]], wanted
                    return
                offset += frame["size"]
        except (KeyError, ValueError, TypeError, IndexError):
            return  # an unexpected message must never disturb the recording

    def _pick_camera(self) -> str | None:
        if not self.cameras:
            return None
        if self.camera:
            for key in self.cameras:
                if self.camera.lower() in key.lower():
                    return key
        return self.cameras[0]
