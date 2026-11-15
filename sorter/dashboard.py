"""Optional link to the team dashboard hub (frontend/hub.py, http://127.0.0.1:8090).

Uses the team's hub_client.py (stdlib only, never raises or blocks). If it cannot be found or the
hub is not running, every call does nothing: the dashboard must never stop or slow the robot.
"""

from __future__ import annotations

import importlib.util
import os

import cv2
import numpy as np

SEARCH = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")]


class _Off:
    stop_requested = False
    connected = False

    def __getattr__(self, name):
        return lambda *a, **k: None


def _load_client_class(frontend_dir: str | None):
    for d in ([frontend_dir] if frontend_dir else []) + SEARCH:
        path = os.path.join(os.path.expanduser(d), "hub_client.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("hub_client", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.HubClient, path
    return None, None


class Dashboard:
    def __init__(self, cfg: dict, log=print):
        self.hub = _Off()
        if not cfg.get("enabled", True):
            log("Dashboard: aus (config.json -> dashboard.enabled)")
            return
        try:
            cls, path = _load_client_class(cfg.get("frontend_dir"))
            if cls is None:
                log("Dashboard: hub_client.py nicht gefunden - es wird nichts gemeldet")
                return
            self.hub = cls(cfg.get("url", "http://127.0.0.1:8090"))
            if self.hub.check_stop() is not None and self.hub.connected:
                log(f"Dashboard: verbunden ({cfg.get('url', 'http://127.0.0.1:8090')})")
            else:
                log("Dashboard: nicht erreichbar (läuft frontend/hub.py?) - es geht auch ohne weiter")
        except Exception as e:  # noqa: BLE001 - never let the dashboard stop the robot
            log(f"Dashboard: Fehler beim Laden ({e}) - es wird nichts gemeldet")
            self.hub = _Off()

    @property
    def stop_requested(self) -> bool:
        try:
            return bool(self.hub.stop_requested)
        except Exception:  # noqa: BLE001
            return False

    def __getattr__(self, name):
        fn = getattr(self.hub, name)

        def safe(*a, **k):
            try:
                return fn(*a, **k)
            except Exception:  # noqa: BLE001
                return None
        return safe


def jpeg(rgb: np.ndarray | None, quality: int = 85) -> bytes | None:
    if rgb is None:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def heatmap_jpeg(rgb: np.ndarray | None, amap: np.ndarray | None) -> bytes | None:
    """Anomaly map (uint8 0..255, any size) as a colour overlay on the image."""
    if rgb is None or amap is None:
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = cv2.resize(amap.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    over = cv2.addWeighted(bgr, 0.55, cv2.applyColorMap(m, cv2.COLORMAP_JET), 0.45, 0)
    ok, buf = cv2.imencode(".jpg", over, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None
