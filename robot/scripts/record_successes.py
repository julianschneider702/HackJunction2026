#!/usr/bin/env python3
"""Let the trained policy sort bricks and keep only the cycles that went right.

Opens its own runtime session in Physical AI Studio (the same websocket the UI
uses), loads the environment, the model and a dataset, and then loops:

    pick a task -> the model drives the arm while an episode is recorded
    -> you judge: correct = the episode is saved, wrong = it is discarded

Saved episodes land in the dataset as ordinary training data, so the next
training run in Studio learns from the policy's own successful cycles. Every
cycle (saved or not) is also appended to a JSONL log with its task, result and
duration: that is the success rate and cycle time for the dashboard.

Needs the Studio backend running and the Studio venv (for ``websockets``):

    ~/.venvs/physicalai-studio/bin/python robot/scripts/record_successes.py --check
    ~/.venvs/physicalai-studio/bin/python robot/scripts/record_successes.py

Close the environment in the Studio UI first: only one session can hold the
arm and the cameras.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import queue
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard_report import DEFAULT_URL as DASHBOARD_URL
from dashboard_report import DashboardReporter

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "logs" / "success_recordings.jsonl"

# Loading copies the dataset into Studio's cache and unloading copies it back;
# with a few hundred episodes of video both take a while.
SLOW_COMMAND_S = 600.0
COMMAND_S = 60.0


class ToolError(Exception):
    """Something the user has to fix; printed without a traceback."""


# ---------------------------------------------------------------- REST helpers


def get_json(base: str, path: str) -> Any:
    try:
        with urllib.request.urlopen(base + path, timeout=15) as response:  # noqa: S310 - local Studio backend
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise ToolError(f"Studio antwortet mit {error.code} auf {path}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ToolError(f"Studio-Backend nicht erreichbar unter {base} ({error})") from error


def pick(items: list[dict[str, Any]], wanted: str | None, kind: str) -> dict[str, Any]:
    """The item whose id or name is ``wanted``; without ``wanted`` the only one there is."""
    if not items:
        raise ToolError(f"Kein(e) {kind} im Studio gefunden.")
    if wanted is None:
        if len(items) == 1:
            return items[0]
        names = ", ".join(repr(item["name"]) for item in items)
        raise ToolError(f"Mehrere {kind} vorhanden ({names}). Bitte eines mit der passenden Option angeben.")
    for item in items:
        if wanted in (item["id"], item["name"]):
            return item
    names = ", ".join(repr(item["name"]) for item in items)
    raise ToolError(f"{kind} {wanted!r} nicht gefunden. Vorhanden: {names}")


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    """Look up everything the session needs; touches no device."""
    base = args.url.rstrip("/")
    project = pick(get_json(base, "/api/projects"), args.project, "Projekt")
    project_id = project["id"]

    models = get_json(base, f"/api/projects/{project_id}/models")
    if args.model is None and len(models) > 1:
        model = max(models, key=lambda item: item.get("created_at") or "")  # newest
    else:
        model = pick(models, args.model, "Modell")
    backends = model.get("available_backends") or []
    if not backends:
        raise ToolError(f"Modell {model['name']!r} hat keinen Export, der sich laden lässt.")
    if args.backend is not None:
        if args.backend not in backends:
            raise ToolError(f"Backend {args.backend!r} gibt es für dieses Modell nicht. Vorhanden: {backends}")
        backend = args.backend
    else:
        backend = "openvino" if "openvino" in backends else backends[0]

    datasets = project.get("datasets") or []
    if args.dataset is not None:
        dataset = pick(datasets, args.dataset, "Dataset")
    else:
        # Default: the dataset the model was trained on, so the next training run includes the new episodes.
        trained_on = [item for item in datasets if item["id"] == model.get("dataset_id")]
        dataset = trained_on[0] if trained_on else pick(datasets, None, "Dataset")

    environments = get_json(base, f"/api/projects/{project_id}/environments")
    environment = pick(environments, args.environment or dataset.get("environment_id"), "Environment")

    if args.task:
        tasks = list(args.task)
    else:
        tasks = get_json(base, f"/api/models/{model['id']}/tasks")
        if not tasks:
            tasks = [dataset["default_task"]]

    episodes = get_json(base, f"/api/dataset/{dataset['id']}/episodes")
    return {
        "base": base,
        "project": project,
        "model": model,
        "backend": backend,
        "dataset": dataset,
        "environment": environment,
        "tasks": tasks,
        "episode_count": len(episodes),
    }


def describe(found: dict[str, Any], device: str) -> None:
    print(f"Projekt:      {found['project']['name']}")
    print(f"Environment:  {found['environment']['name']}")
    print(f"Modell:       {found['model']['name']} ({found['model']['policy']}, {found['backend']}, {device})")
    print(f"Dataset:      {found['dataset']['name']} ({found['episode_count']} Episoden)")
    for number, task in enumerate(found["tasks"], start=1):
        print(f"Task {number}:       {task}")


# ------------------------------------------------------------ session over ws


class Session:
    """One Studio runtime session: send a command, wait for its ack."""

    def __init__(self, socket: Any, on_binary: Any = None) -> None:
        self.socket = socket
        self.on_binary = on_binary
        """Called with each binary observation message (joints and camera JPEGs), if set."""
        self.acks: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for raw in self.socket:
                if isinstance(raw, bytes):
                    if self.on_binary is not None:
                        self.on_binary(raw)
                    continue
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "ack":
                    future = self.acks.pop(message["request_id"], None)
                    if future is not None and not future.done():
                        future.set_result(message)
                elif kind == "error":
                    print(f"\n[Studio] Fehler in {message['component']}: {message['message']}")
                    if not message.get("session_alive", True):
                        self._end(message["message"])
                elif kind == "status" and message["state"] == "failed":
                    print(f"\n[Studio] {message['component']} ausgefallen: {message.get('error')}")
        except Exception as error:  # noqa: BLE001 - whatever closed the socket ends the session
            self._end(f"Verbindung zum Studio verloren ({error})")
        else:
            self._end("Studio hat die Verbindung geschlossen")

    def _end(self, reason: str) -> None:
        if not self.ended.done():
            self.ended.set_result(reason)
        for future in self.acks.values():
            if not future.done():
                future.set_exception(ToolError(reason))
        self.acks.clear()

    async def command(self, name: str, timeout: float = COMMAND_S, **fields: Any) -> dict[str, Any] | None:
        """Send one command and return its result.

        Raises:
            ToolError: Studio refused the command, did not answer in time, or the session ended.
        """
        if self.ended.done():
            raise ToolError(self.ended.result())
        request_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.acks[request_id] = future
        await self.socket.send(json.dumps({"command": name, "request_id": request_id, **fields}))
        try:
            ack = await asyncio.wait_for(future, timeout)
        except TimeoutError as error:
            self.acks.pop(request_id, None)
            raise ToolError(f"Studio hat auf {name} nicht innerhalb von {timeout:.0f} s geantwortet") from error
        if not ack["ok"]:
            raise ToolError(f"{name} abgelehnt: {ack['error']['message']}")
        return ack.get("result")

    async def close(self) -> None:
        self.reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.reader


# ------------------------------------------------------------------ keyboard


class Keyboard:
    """Lines typed in the terminal, read on one thread so a timed-out wait never swallows the next line."""

    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in sys.stdin:
            self.lines.put(line.strip().lower())
        self.lines.put(None)  # stdin closed

    def drain(self) -> None:
        """Forget what was typed before the question was asked."""
        with contextlib.suppress(queue.Empty):
            while True:
                if self.lines.get_nowait() is None:
                    self.lines.put(None)
                    return

    async def ask(self, stop: asyncio.Event, timeout: float | None = None) -> str | None:
        """The next line; ``None`` on timeout, Ctrl+C or closed stdin."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return None
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if line is None:
                stop.set()
            return line
        return None


# --------------------------------------------------------------------- cycles


def log_cycle(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def choose_task(tasks: list[str], keyboard: Keyboard, stop: asyncio.Event) -> str | None:
    """The task for the next cycle, or ``None`` to finish."""
    while not stop.is_set():
        print()
        for number, task in enumerate(tasks, start=1):
            print(f"  [{number}] {task}")
        print("Stein hinlegen, dann Nummer + Enter zum Starten (q = beenden): ", end="", flush=True)
        keyboard.drain()
        answer = await keyboard.ask(stop)
        if answer is None or answer == "q":
            return None
        if answer == "" and len(tasks) == 1:
            return tasks[0]
        if answer.isdigit() and 1 <= int(answer) <= len(tasks):
            return tasks[int(answer) - 1]
        print("Bitte eine der Nummern eingeben.")
    return None


async def run_cycle(
    session: Session, task: str, args: argparse.Namespace, keyboard: Keyboard, stop: asyncio.Event,
    reporter: DashboardReporter,
) -> tuple[str, float]:
    """Run the model once with an episode recording. Returns the verdict and how long the arm ran.

    The verdict is ``saved``, ``wrong``, ``timeout`` (no answer in time) or ``aborted`` (Ctrl+C or
    the dashboard's Stop button). The arm is held still before this returns, whatever happened.
    """
    control = {"type": "model", "task": task, "hz": args.hz, "grace_s": args.grace}
    await session.command("set_control", control=control)
    started = time.monotonic()
    try:
        await session.command("start_episode", task=task)
        reporter.cycle_started(task)
        print(f"Läuft: {task!r}")
        print("  j + Enter = richtig einsortiert (speichern)   n + Enter = falsch (verwerfen)")
        keyboard.drain()
        verdict = "aborted"
        while not stop.is_set():
            left = args.max_seconds - (time.monotonic() - started)
            if reporter.stop_requested:
                print("Stop im Dashboard gedrückt: Arm hält an, Episode verworfen.")
                break
            if left <= 0:
                verdict = "timeout"
                print(f"Nach {args.max_seconds:.0f} s gestoppt, Episode verworfen.")
                break
            # Short waits, so the dashboard's Stop button is noticed within a quarter of a second.
            answer = await keyboard.ask(stop, timeout=min(left, 0.25))
            if answer is None:
                continue  # nothing typed yet; Ctrl+C or closed stdin end the loop through `stop`
            if answer in ("j", "y"):
                verdict = "saved"
                break
            if answer == "n":
                verdict = "wrong"
                break
            print("  j = richtig, n = falsch")
        duration = time.monotonic() - started
        # Save before the control changes: Studio discards an open episode when the control switches.
        if verdict == "saved":
            await session.command("save_episode", timeout=SLOW_COMMAND_S)
        else:
            await session.command("discard_episode")
        return verdict, duration
    finally:
        with contextlib.suppress(ToolError):
            await session.command("set_control", control=None)  # hold the arm where it is


async def run(args: argparse.Namespace) -> None:
    import websockets  # only the Studio venv has it; --check works without

    found = resolve(args)
    describe(found, args.device)
    environment = found["environment"]
    ws_url = found["base"].replace("http", "ws", 1) + f"/api/projects/{found['project']['id']}/runtime/ws?hz=1"

    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
    keyboard = Keyboard()
    saved = total = 0
    reporter = DashboardReporter(args.dashboard, args.dashboard_camera)

    async with websockets.connect(ws_url, max_size=None, ping_interval=None) as socket:
        session = Session(socket, on_binary=reporter.on_frames)
        try:
            print("\nVerbinde Arm und Kameras ...")
            draft = {key: environment[key] for key in ("id", "name", "robots", "cameras")}
            await session.command("load_environment", environment=draft)
            print("Lade Modell ...")
            await session.command(
                "load_model", timeout=SLOW_COMMAND_S, model_id=found["model"]["id"],
                backend=found["backend"], device=args.device,
            )  # fmt: skip
            print("Öffne Dataset (wird kopiert, das kann dauern) ...")
            await session.command("load_dataset", timeout=SLOW_COMMAND_S, dataset_id=found["dataset"]["id"])
            print("Bereit. Strg+C hält den Arm sofort an.")
            reporter.session_started(found, args.device)

            while not stop.is_set() and not session.ended.done():
                if reporter.stop_requested:
                    reporter.stopped_by_dashboard()
                    print("\nStop im Dashboard aktiv. Dort 'Resume' drücken, um weiterzumachen (Strg+C beendet).")
                    while reporter.stop_requested and not stop.is_set():
                        await asyncio.sleep(0.2)
                    reporter.resumed()
                    continue
                task = await choose_task(found["tasks"], keyboard, stop)
                if task is None:
                    break
                if reporter.stop_requested:
                    continue  # Stop was pressed while choosing: do not start the arm
                verdict, duration = await run_cycle(session, task, args, keyboard, stop, reporter)
                reporter.cycle_finished(task, verdict, duration)
                total += 1
                entry: dict[str, Any] = {
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "task": task,
                    "result": verdict,
                    "duration_s": round(duration, 2),
                    "model": found["model"]["name"],
                    "dataset": found["dataset"]["name"],
                }
                if verdict == "saved":
                    entry["episode_index"] = found["episode_count"] + saved
                    saved += 1
                    print(f"Gespeichert als Episode {entry['episode_index']} ({duration:.1f} s).")
                elif verdict == "wrong":
                    print("Verworfen.")
                log_cycle(args.log, entry)
                print(f"Bisher: {saved} von {total} Zyklen gespeichert.")
        finally:
            if not session.ended.done():
                with contextlib.suppress(ToolError):
                    await session.command("set_control", control=None)
                print("\nSchreibe das Dataset zurück, bitte nicht abbrechen ...")
                try:
                    await session.command("unload_dataset", timeout=SLOW_COMMAND_S)
                except ToolError as error:
                    print(f"Dataset konnte nicht sauber geschlossen werden: {error}")
            await session.close()
            reporter.session_closed()

    print(f"Fertig: {saved} neue Episode(n) in {found['dataset']['name']!r}, {total} Zyklen insgesamt.")
    print(f"Protokoll: {args.log}")
    if session.ended.done():
        raise ToolError(session.ended.result())


def parse_args(extra: list[tuple[tuple[str, ...], dict[str, Any]]] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:7860", help="Studio backend (default: %(default)s)")
    parser.add_argument("--project", help="project name or id (default: the only project)")
    parser.add_argument("--model", help="model name or id (default: the newest model)")
    parser.add_argument("--backend", help="model export to run (default: openvino if exported, else the first)")
    parser.add_argument("--device", default="auto", help="inference device, e.g. CPU, GPU, NPU (default: %(default)s)")
    parser.add_argument("--dataset", help="dataset to add episodes to (default: the one the model was trained on)")
    parser.add_argument("--environment", help="environment name or id (default: the dataset's environment)")
    parser.add_argument(
        "--task", action="append", help="task to offer; repeat for several (default: the tasks the model was trained on)"
    )
    parser.add_argument("--max-seconds", type=float, default=60.0, help="stop and discard a cycle after this long")
    parser.add_argument("--hz", type=float, default=30.0, help="how often the model's action is applied")
    parser.add_argument("--grace", type=float, default=3.0, help="seconds the arm takes to reach the model's first pose")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="cycle log, JSON lines (default: %(default)s)")
    parser.add_argument("--check", action="store_true", help="only show what would be used; connects to no device")
    parser.add_argument(
        "--dashboard", default=DASHBOARD_URL,
        help="dashboard hub that gets every cycle (frontend/hub.py); 'off' reports nothing (default: %(default)s)",
    )  # fmt: skip
    parser.add_argument(
        "--dashboard-camera", help="camera whose image goes to the dashboard, part of its name (default: the first)"
    )
    for flags, options in extra or []:
        parser.add_argument(*flags, **options)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            describe(resolve(args), args.device)
            print("\nAlles gefunden. Ohne --check starten, um aufzunehmen.")
        else:
            asyncio.run(run(args))
    except ToolError as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
