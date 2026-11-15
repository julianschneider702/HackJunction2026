"""Talk to a running Physical AI Studio: own runtime session, cameras and arm.

The session is opened over Studio's runtime websocket (the same one the web UI uses), so Studio keeps
owning the robot and cameras. Frames and joint states are read from the session's shared-memory store,
and scripted moves are written into that store's ACTION slot, which the robot worker follows
(it interpolates linearly to the target by the given due time). While a model or teleop control is
active, they write ACTION themselves and Arm refuses to move.

Run with Studio's backend Python: station/studio-python <script>
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import Future

import numpy as np

# Studio checkout: next to this repo, or wherever PHYSICAL_AI_STUDIO points
STUDIO_DIR = os.environ.get("PHYSICAL_AI_STUDIO") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "physical-ai-studio")
BACKEND_SRC = os.path.join(os.path.abspath(os.path.expanduser(STUDIO_DIR)), "application", "backend", "src")
sys.path.insert(0, BACKEND_SRC)
from runtime_session.store import (  # noqa: E402
    _BLOCK_SESSION_CHARS, ACTION, BLOCK_PREFIX, STATE, FeatureSpec, ImageSpec, Store)

API = "http://localhost:7860/api"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER = 5


class StudioError(RuntimeError):
    pass


class MotionError(RuntimeError):
    """The arm did not reach a target in time (blocked, collision, or a dead robot)."""


def get_json(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.load(r)


class Session:
    """One Studio runtime session for the lifetime of this object."""

    def __init__(self, project_id: str, log=print):
        import websockets  # Studio's venv has it
        self._ws_mod = websockets
        self.project_id = project_id
        self.log = log
        self.state: dict = {}
        self.statuses: dict[str, str] = {}
        self.errors: list[str] = []
        self.store: Store | None = None
        self._store_key = None
        self._pending: dict[str, Future] = {}
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="studio-session")
        self._thread.start()
        if not self._ready.wait(15):
            raise StudioError("Keine Verbindung zu Studio (läuft es auf localhost:7860?)")
        if self._connect_error:
            raise StudioError(f"Keine Verbindung zu Studio: {self._connect_error}")

    # ---------------------------------------------------------------- websocket thread
    def _run(self):
        self._connect_error = None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:  # noqa: BLE001
            self._connect_error = self._connect_error or repr(e)
            self._ready.set()

    async def _main(self):
        url = f"ws://localhost:7860/api/projects/{self.project_id}/runtime/ws?hz=1"
        try:
            self._ws = await self._ws_mod.connect(url, max_size=None, open_timeout=10)
        except Exception as e:  # noqa: BLE001
            self._connect_error = repr(e)
            self._ready.set()
            return
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue  # observation previews (JPEG); we read the store directly
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "ack":
                    fut = self._pending.pop(msg.get("request_id", ""), None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                elif kind == "state":
                    self.state = msg["data"]
                    self._ready.set()
                elif kind == "status":
                    self.statuses[msg["component"]] = msg["state"]
                    if msg.get("error"):
                        self.log(f"[Studio] {msg['component']}: {msg['state']} - {msg['error']}")
                elif kind == "error":
                    text = f"{msg.get('component')}: {msg.get('message')}"
                    self.errors.append(text)
                    self.log(f"[Studio Fehler] {text}")
        finally:
            self._closed = True
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(StudioError("Studio-Sitzung wurde geschlossen"))

    # ---------------------------------------------------------------- commands
    def command(self, command: str, timeout: float = 120, **fields) -> dict:
        if self._closed:
            raise StudioError("Studio-Sitzung ist geschlossen")
        rid = uuid.uuid4().hex
        fut: Future = Future()
        self._pending[rid] = fut
        payload = json.dumps({"command": command, "request_id": rid, **fields})
        asyncio.run_coroutine_threadsafe(self._ws.send(payload), self._loop).result(10)
        ack = fut.result(timeout)
        if not ack.get("ok"):
            err = ack.get("error") or {}
            raise StudioError(f"{command}: {err.get('message', ack)}")
        return ack.get("result") or {}

    def load_environment(self, environment_id: str):
        env = get_json(f"/projects/{self.project_id}/environments/{environment_id}")
        draft = {"id": env["id"], "name": env["name"], "robots": env["robots"], "cameras": env["cameras"]}
        self.command("load_environment", timeout=120, environment=draft, unload_incompatible=True)
        self._attach_store()

    def load_model(self, model_id: str, backend: str = "torch", device: str = "auto"):
        self.command("load_model", timeout=600, model_id=model_id, backend=backend, device=device)

    def set_control(self, control: dict | None):
        self.command("set_control", timeout=30, control=control)

    def close(self):
        try:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result(5)
        except Exception:  # noqa: BLE001
            pass
        if self.store is not None:
            self.store.close()
            self.store = None

    # ---------------------------------------------------------------- store
    def _attach_store(self):
        deadline = time.time() + 10
        while time.time() < deadline:
            env = (self.state or {}).get("environment")
            sid = (self.state or {}).get("session_id", "")
            if env and sid:
                names = glob.glob(f"/dev/shm/{BLOCK_PREFIX}*-{sid[:_BLOCK_SESSION_CHARS]}")
                if names:
                    spec = FeatureSpec(tuple(env["joint_names"]),
                                       {c["key"]: ImageSpec(c["height"], c["width"]) for c in env["cameras"]})
                    if self.store is not None:
                        self.store.close()
                    self.store = Store.attach(spec, os.path.basename(names[0]))
                    self.cameras = [c["key"] for c in env["cameras"]]
                    self.joint_names = list(env["joint_names"])
                    return
            time.sleep(0.1)
        raise StudioError("Speicher der Sitzung nicht gefunden (Umgebung geladen?)")


class Arm:
    """Reads state and frames from the session store and moves the follower by writing ACTION."""

    def __init__(self, session: Session, limits: dict[str, list[float]], max_speed: float = 45.0):
        self.s = session
        self.limits = np.array([limits[j] for j in JOINTS], dtype=np.float64)  # (6, 2) min/max
        self.max_speed = max_speed      # normalized units per second for the fastest joint
        self.control_active = False     # True while a model/teleop control writes ACTION
        self.lock = threading.RLock()

    # ------------------------------------------------------------ reading
    def state(self, max_age: float = 0.5) -> np.ndarray:
        s = self.s.store.read(STATE) if self.s.store else None
        if s is None or time.monotonic() - s.timestamp > max_age:
            raise StudioError("Keine aktuellen Gelenkwerte vom Arm")
        return np.array(s.value, dtype=np.float64)

    def last_command(self) -> np.ndarray:
        a = self.s.store.read(ACTION)
        return np.array(a.value, dtype=np.float64) if a is not None else self.state()

    def frame(self, camera: str, max_age: float = 1.0) -> np.ndarray:
        f = self.s.store.read_observation(camera) if self.s.store else None
        if f is None or time.monotonic() - f.timestamp > max_age:
            raise StudioError(f"Kamera {camera!r} liefert keine aktuellen Bilder")
        return np.array(f.value, copy=True)

    def frames(self, camera: str, n: int, gap: float = 0.08) -> list[np.ndarray]:
        out, last = [], None
        deadline = time.monotonic() + n * gap + 2.0
        while len(out) < n and time.monotonic() < deadline:
            f = self.s.store.read_observation(camera)
            if f is not None and f.timestamp != last and time.monotonic() - f.timestamp < 1.0:
                out.append(np.array(f.value, copy=True))
                last = f.timestamp
                time.sleep(gap)
            else:
                time.sleep(0.01)
        if len(out) < n:
            raise StudioError(f"Kamera {camera!r} liefert keine neuen Bilder")
        return out

    # ------------------------------------------------------------ moving
    def clamp(self, target) -> np.ndarray:
        t = np.asarray(target, dtype=np.float64)
        return np.clip(t, self.limits[:, 0], self.limits[:, 1])

    def send(self, target, duration: float):
        """Write one target; the robot worker gets there linearly in `duration` seconds."""
        if self.control_active:
            raise StudioError("Eine Steuerung (Modell/Teleop) ist aktiv - erst stoppen")
        t = self.clamp(target)
        self.s.store.write(ACTION, t, timestamp=time.monotonic() + max(duration, 0.05))
        return t

    def move(self, target, speed: float | None = None, tol: float = 4.0, check_gripper: bool = True,
             min_time: float = 0.4, extra: float = 2.0) -> np.ndarray:
        """Move to `target` and wait until the joints are there.

        Raises MotionError if the arm is not within `tol` (normalized units) after the planned
        time plus `extra` seconds; the arm is then told to hold where it is.
        """
        with self.lock:
            start = self.last_command()
            t = self.clamp(target)
            delta = np.abs(t - start)
            delta[GRIPPER] *= 0.5            # gripper may move faster
            duration = max(min_time, float(delta.max()) / (speed or self.max_speed))
            self.send(t, duration)
            joints = slice(0, 6) if check_gripper else slice(0, 5)
            deadline = time.monotonic() + duration + extra
            time.sleep(duration * 0.9)
            while True:
                err = np.abs(self.state()[joints] - t[joints])
                if err.max() <= tol:
                    return t
                if time.monotonic() > deadline:
                    here = self.state()
                    hold = here.copy()
                    hold[GRIPPER] = t[GRIPPER]   # keep the gripper command (do not drop a brick)
                    self.send(hold, 0.3)
                    worst = JOINTS[int(np.argmax(np.abs(here - t)[joints]))]
                    raise MotionError(f"Ziel nicht erreicht ({worst}: ist {here[JOINTS.index(worst)]:.1f}, "
                                      f"soll {t[JOINTS.index(worst)]:.1f})")
                time.sleep(0.03)

    def hold(self, gripper: float | None = None):
        here = self.state()
        if gripper is not None:
            here[GRIPPER] = gripper
        self.send(here, 0.25)
        return here

    def gripper(self, value: float, wait: float = 0.6):
        t = self.last_command()
        t[GRIPPER] = value
        self.send(t, 0.3)
        time.sleep(wait)


class ReadOnlySession:
    """Attach to the store of a session someone else opened (e.g. the Studio web UI), read only.

    For the vision test mode: frames and joint states, no commands, no moves.
    """

    def __init__(self, cameras=("webcam", "wrist", "realsense"), height=480, width=640):
        names = glob.glob(f"/dev/shm/{BLOCK_PREFIX}*")
        if not names:
            raise StudioError("Keine laufende Studio-Sitzung gefunden - in Studio die Umgebung öffnen")
        spec = FeatureSpec(tuple(f"{j}.pos" for j in JOINTS), {c: ImageSpec(height, width) for c in cameras})
        newest = max(names, key=os.path.getmtime)
        self.store = Store.attach(spec, os.path.basename(newest))
        self.cameras = list(cameras)

    def close(self):
        self.store.close()


def find_model(name_or_id: str | None, project_id: str) -> dict:
    models = get_json(f"/projects/{project_id}/models")
    if name_or_id:
        for m in models:
            if m["id"] == name_or_id or m["name"] == name_or_id:
                return m
        raise StudioError(f"Modell {name_or_id!r} nicht gefunden. Vorhanden: {[m['name'] for m in models]}")
    smol = [m for m in models if m.get("policy") == "smolvla"] or models
    return max(smol, key=lambda m: m.get("created_at", ""))
