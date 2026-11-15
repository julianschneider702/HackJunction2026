"""Client for anomaly_worker.py: starts it in the anomalib venv and asks it for scores."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
import time
from multiprocessing.connection import Client

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANOMALIB_PYTHON = os.path.expanduser("~/anomalib-env/bin/python")


class AnomalyError(RuntimeError):
    pass


class Anomaly:
    def __init__(self, models: dict[str, str | None], log=print, python: str = ANOMALIB_PYTHON,
                 timeout: float = 180.0):
        self.models = {k: os.path.expanduser(v) for k, v in models.items() if v}
        for name, path in self.models.items():
            if not os.path.exists(path):
                raise AnomalyError(f"Anomalie-Modell {name!r} fehlt: {path}")
        self.log = log
        self.last_ms = None
        self.conn = None
        self.proc = None
        if not self.models:
            return
        self._dir = tempfile.mkdtemp(prefix="sorter-anomaly-")
        address = os.path.join(self._dir, "sock")
        key = secrets.token_hex(16)
        self._logfile = open(os.path.join(HERE, "log", "anomaly_worker.log"), "a")
        self.proc = subprocess.Popen([python, "-I", os.path.join(HERE, "anomaly_worker.py"), address,
                                      json.dumps(self.models)],
                                     env={**os.environ, "SORTER_AUTHKEY": key},
                                     stdout=self._logfile, stderr=subprocess.STDOUT)
        log(f"Lade Anomalie-Modelle: {', '.join(self.models)} ...")
        t0 = time.monotonic()
        while self.conn is None:
            if self.proc.poll() is not None:
                raise AnomalyError(f"Anomalie-Prozess beendet (Code {self.proc.returncode}), "
                                   f"siehe {self._logfile.name}")
            if time.monotonic() - t0 > timeout:
                self.close()
                raise AnomalyError("Anomalie-Prozess startet nicht (Zeitlimit)")
            if os.path.exists(address):
                try:
                    self.conn = Client(address, family="AF_UNIX", authkey=key.encode())
                except (ConnectionRefusedError, FileNotFoundError):
                    pass
            time.sleep(0.2)
        log(f"Anomalie-Modelle bereit ({time.monotonic() - t0:.0f}s)")

    def has(self, name: str) -> bool:
        return name in self.models

    def score(self, name: str, images: list[np.ndarray], maps: bool = False):
        """Normalized anomaly score per RGB uint8 image (all the same size).

        maps=True returns (scores, maps): uint8 anomaly maps (n, h, w), or None if the model has none.
        `last_ms` is the time per image of the last call.
        """
        if self.conn is None or name not in self.models:
            raise AnomalyError(f"kein Anomalie-Modell {name!r} geladen")
        t0 = time.monotonic()
        arr = np.ascontiguousarray(np.stack(images).astype(np.uint8))
        self.conn.send_bytes(json.dumps({"model": name, "shape": list(arr.shape), "maps": maps}).encode())
        self.conn.send_bytes(arr.tobytes())
        reply = json.loads(self.conn.recv_bytes())
        if "error" in reply:
            raise AnomalyError(reply["error"])
        amaps = None
        if "map_shape" in reply:
            amaps = np.frombuffer(self.conn.recv_bytes(), np.uint8).reshape(reply["map_shape"])
        self.last_ms = (time.monotonic() - t0) * 1000 / len(images)
        return (reply["scores"], amaps) if maps else reply["scores"]

    def close(self):
        if self.conn is not None:
            try:
                self.conn.send_bytes(json.dumps({"cmd": "quit"}).encode())
            except OSError:
                pass
            self.conn.close()
            self.conn = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
