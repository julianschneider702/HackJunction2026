#!/usr/bin/env python3
"""Browser GUI for ``record_successes.py``: run the policy, watch the cameras, keep the good cycles.

Starts a small local web page (default http://localhost:8790) and opens it in
the browser. The page shows the live camera images, lets you pick the task and
start a cycle, and has "Richtig" / "Falsch" buttons: a correct cycle is saved
into the Studio dataset as a training episode, a wrong one is discarded.

    ~/.venvs/physicalai-studio/bin/python robot/scripts/record_successes_gui.py

Same rules as the terminal tool: the Studio backend must run, and the
environment must be closed in the Studio UI. Use "Beenden" in the page (or
Ctrl+C here) to stop: it writes the dataset back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import socket as socket_module
import sys
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any

from dashboard_report import DashboardReporter
from record_successes import SLOW_COMMAND_S, Session, ToolError, describe, log_cycle, parse_args, resolve

GUI_HZ = 10
"""Camera images per second in the page."""


class Station:
    """The recording loop behind the page: one Studio session, one cycle at a time."""

    def __init__(self, found: dict[str, Any], args: Any) -> None:
        self.found = found
        self.args = args
        self.reporter = DashboardReporter(args.dashboard, args.dashboard_camera)
        self.session: Session | None = None
        self.clients: set[Any] = set()
        self.lock = asyncio.Lock()
        self.stop = asyncio.Event()
        self.phase = "loading"  # loading, ready, running, closing, closed, error
        self.message = "Verbinde ..."
        self.task: str | None = None
        self.started = 0.0  # time.time() of the running cycle, for the page's timer
        self.cycle = 0
        self.saved = 0
        self.total = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------- to the page

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "state",
            "phase": self.phase,
            "message": self.message,
            "tasks": self.found["tasks"],
            "task": self.task,
            "started": self.started,
            "max_seconds": self.args.max_seconds,
            "saved": self.saved,
            "total": self.total,
            "history": self.history[-8:],
            "project": self.found["project"]["name"],
            "model": f"{self.found['model']['name']} ({self.found['backend']}, {self.args.device})",
            "dataset": self.found["dataset"]["name"],
            "episodes": self.found["episode_count"] + self.saved,
        }

    def publish(self, phase: str | None = None, message: str | None = None) -> None:
        from websockets.asyncio.server import broadcast

        if phase is not None:
            self.phase = phase
        if message is not None:
            self.message = message
        broadcast(self.clients, json.dumps(self.snapshot()))

    def forward_frames(self, raw: bytes) -> None:
        """Split Studio's observation message and send each JPEG on as ``[key length][key][jpeg]``."""
        from websockets.asyncio.server import broadcast

        self.reporter.on_frames(raw)
        if not self.clients:
            return
        size = int.from_bytes(raw[:4], "big")
        header = json.loads(raw[4 : 4 + size])
        offset = 4 + size
        for frame in header["frames"]:
            key = frame["key"].encode()
            broadcast(self.clients, bytes([len(key)]) + key + raw[offset : offset + frame["size"]])
            offset += frame["size"]

    # ---------------------------------------------------------------- commands

    async def open(self, session: Session) -> None:
        self.session = session
        environment = self.found["environment"]
        try:
            self.publish(message="Verbinde Arm und Kameras ...")
            draft = {key: environment[key] for key in ("id", "name", "robots", "cameras")}
            await session.command("load_environment", environment=draft)
            self.publish(message="Lade Modell ...")
            await session.command(
                "load_model", timeout=SLOW_COMMAND_S, model_id=self.found["model"]["id"],
                backend=self.found["backend"], device=self.args.device,
            )  # fmt: skip
            self.publish(message="Öffne Dataset (wird kopiert, das kann dauern) ...")
            await session.command("load_dataset", timeout=SLOW_COMMAND_S, dataset_id=self.found["dataset"]["id"])
        except ToolError as error:
            self.publish("error", str(error))
            return
        self.reporter.session_started(self.found, self.args.device)
        self.publish("ready", "Bereit. Stein hinlegen und Aufgabe starten.")

    async def watch_dashboard(self) -> None:
        """The dashboard's Stop button: abort the running cycle, and start none until Resume."""
        halted = False
        while True:
            await asyncio.sleep(0.2)
            try:
                if self.reporter.stop_requested and not halted:
                    halted = True
                    await self.finish("aborted")  # does nothing if no cycle is running
                    self.reporter.stopped_by_dashboard()
                    if self.phase == "ready":
                        self.publish(message="Im Dashboard gestoppt. Dort 'Resume' drücken, um weiterzumachen.")
                elif halted and not self.reporter.stop_requested:
                    halted = False
                    self.reporter.resumed()
                    if self.phase == "ready":
                        self.publish(message="Bereit. Stein hinlegen und Aufgabe starten.")
            except Exception as error:  # noqa: BLE001 - the Stop button must keep working after a hiccup
                print(f"Dashboard-Überwachung: {error}")

    async def start(self, task: str) -> None:
        async with self.lock:
            if self.phase != "ready" or self.session is None or task not in self.found["tasks"]:
                return
            if self.reporter.stop_requested:
                self.publish(message="Im Dashboard ist Stop aktiv. Dort erst 'Resume' drücken.")
                return
            control = {"type": "model", "task": task, "hz": self.args.hz, "grace_s": self.args.grace}
            try:
                await self.session.command("set_control", control=control)
                await self.session.command("start_episode", task=task)
            except ToolError as error:
                with contextlib.suppress(ToolError):
                    await self.session.command("set_control", control=None)
                self.publish(message=f"Start fehlgeschlagen: {error}")
                return
            self.cycle += 1
            self.task, self.started = task, time.time()
            self.reporter.cycle_started(task)
            self.publish("running", "Arm läuft. Danach bewerten.")
            asyncio.create_task(self._timeout(self.cycle))  # noqa: RUF006 - ends with the cycle

    async def _timeout(self, cycle: int) -> None:
        await asyncio.sleep(self.args.max_seconds)
        if self.cycle == cycle:
            await self.finish("timeout")

    async def finish(self, verdict: str) -> None:
        """End the running cycle: ``saved`` keeps the episode, anything else discards it. The arm then holds."""
        async with self.lock:
            if self.phase != "running" or self.session is None:
                return
            self.cycle += 1  # the timeout of this cycle no longer applies
            duration = time.time() - self.started
            note = {
                "saved": "Gespeichert.",
                "wrong": "Verworfen.",
                "aborted": "Gestoppt, Episode verworfen.",
                "timeout": f"Nach {self.args.max_seconds:.0f} s gestoppt, Episode verworfen.",
            }[verdict]
            try:
                # Save before the control changes: Studio discards an open episode when the control switches.
                if verdict == "saved":
                    await self.session.command("save_episode", timeout=SLOW_COMMAND_S)
                else:
                    await self.session.command("discard_episode")
            except ToolError as error:
                verdict, note = "failed", f"Speichern fehlgeschlagen: {error}"
            finally:
                with contextlib.suppress(ToolError):
                    await self.session.command("set_control", control=None)
            self.total += 1
            self.reporter.cycle_finished(self.task or "", verdict, duration)
            entry: dict[str, Any] = {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "task": self.task,
                "result": verdict,
                "duration_s": round(duration, 2),
                "model": self.found["model"]["name"],
                "dataset": self.found["dataset"]["name"],
            }
            if verdict == "saved":
                entry["episode_index"] = self.found["episode_count"] + self.saved
                self.saved += 1
            log_cycle(self.args.log, entry)
            self.history.append(entry)
            self.publish("ready", note)

    async def close(self) -> None:
        """Hold the arm and write the dataset back."""
        session = self.session
        if session is None or session.ended.done():
            return
        await self.finish("aborted")
        async with self.lock:
            self.publish("closing", "Schreibe das Dataset zurück, bitte warten ...")
            with contextlib.suppress(ToolError):
                await session.command("set_control", control=None)
            try:
                await session.command("unload_dataset", timeout=SLOW_COMMAND_S)
            except ToolError as error:
                self.publish("error", f"Dataset konnte nicht sauber geschlossen werden: {error}")
                return
            self.publish("closed", f"Beendet. {self.saved} neue Episode(n) im Dataset. Dieses Fenster kann zu.")

    # ---------------------------------------------------------------- the page

    async def serve_client(self, socket: Any) -> None:
        self.clients.add(socket)
        try:
            await socket.send(json.dumps(self.snapshot()))
            async for raw in socket:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                action = message.get("action")
                if action == "start":
                    await self.start(message.get("task", ""))
                elif action == "judge":
                    await self.finish("saved" if message.get("ok") else "wrong")
                elif action == "stop":
                    await self.finish("aborted")
                elif action == "quit":
                    self.stop.set()
        except Exception:  # noqa: BLE001, S110 - a closed browser tab is not an error
            pass
        finally:
            self.clients.discard(socket)


def page_response(_connection: Any, request: Any) -> Any:
    """Serve the page on ``/``; ``/ws`` goes on to the websocket."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    if request.path.split("?")[0] == "/ws":
        return None
    body = PAGE.encode()
    headers = Headers({"Content-Type": "text/html; charset=utf-8", "Content-Length": str(len(body))})
    return Response(HTTPStatus.OK, "OK", headers, body)


async def run(args: Any) -> None:
    import websockets
    from websockets.asyncio.server import serve

    found = resolve(args)
    describe(found, args.device)
    station = Station(found, args)
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, station.stop.set)
    with contextlib.closing(socket_module.socket()) as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise ToolError(f"Port {args.port} ist schon belegt. Mit --port einen anderen wählen.")
    studio = found["base"].replace("http", "ws", 1) + f"/api/projects/{found['project']['id']}/runtime/ws?hz={GUI_HZ}"
    page_url = f"http://localhost:{args.port}"

    async with serve(station.serve_client, "127.0.0.1", args.port, process_request=page_response):
        print(f"\nOberfläche: {page_url}  (Strg+C oder 'Beenden' in der Seite stoppt)")
        if not args.no_browser:
            webbrowser.open(page_url)
        async with websockets.connect(studio, max_size=None, ping_interval=None) as socket:
            session = Session(socket, on_binary=station.forward_frames)
            opening = asyncio.create_task(station.open(session))
            stopped = asyncio.create_task(station.stop.wait())
            watching = asyncio.create_task(station.watch_dashboard())
            try:
                await asyncio.wait({stopped, session.ended}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                opening.cancel()
                stopped.cancel()
                watching.cancel()
                if session.ended.done():
                    station.publish("error", session.ended.result())
                else:
                    print("Schreibe das Dataset zurück, bitte nicht abbrechen ...")
                    await station.close()
                await session.close()
                station.reporter.session_closed()
        await asyncio.sleep(0.3)  # let the page receive its last state

    print(f"Fertig: {station.saved} neue Episode(n) in {found['dataset']['name']!r}, {station.total} Zyklen insgesamt.")
    print(f"Protokoll: {args.log}")
    if session.ended.done():
        raise ToolError(session.ended.result())


PAGE = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erfolge aufnehmen</title>
<style>
  :root { --bg:#10151c; --panel:#19212b; --line:#2a3542; --text:#e8edf2; --dim:#93a1b0;
          --accent:#19b5e6; --good:#2fb170; --bad:#e0564f; --warn:#e2a93b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:16px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1100px; margin:0 auto; padding:20px 16px 40px; }
  header { display:flex; flex-wrap:wrap; gap:8px 24px; align-items:baseline; margin-bottom:16px; }
  h1 { font-size:22px; margin:0; }
  .meta { color:var(--dim); font-size:14px; }
  .meta b { color:var(--text); font-weight:600; }
  #status { padding:12px 16px; border-radius:10px; background:var(--panel); border:1px solid var(--line);
            border-left:5px solid var(--accent); margin-bottom:16px; display:flex; justify-content:space-between;
            gap:16px; align-items:center; }
  #status.running { border-left-color:var(--warn); }
  #status.error { border-left-color:var(--bad); }
  #status.closed { border-left-color:var(--dim); }
  #timer { font-variant-numeric:tabular-nums; font-size:22px; font-weight:600; white-space:nowrap; }
  #cams { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-bottom:16px; }
  figure { margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  figure img { display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#000; }
  figcaption { padding:6px 10px; font-size:13px; color:var(--dim); }
  .empty { color:var(--dim); padding:28px; text-align:center; border:1px dashed var(--line); border-radius:10px; }
  .row { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
  button { font:inherit; font-weight:600; color:var(--text); background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:16px 18px; cursor:pointer; flex:1 1 200px; min-height:64px; }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.35; cursor:default; }
  button small { display:block; font-weight:400; color:var(--dim); font-size:12px; margin-top:2px; }
  .task { text-align:left; }
  .good { background:var(--good); border-color:var(--good); color:#06170e; font-size:20px; }
  .bad { background:var(--bad); border-color:var(--bad); color:#1d0605; font-size:20px; }
  .good small, .bad small { color:inherit; opacity:.75; }
  .quiet { flex:0 0 auto; min-height:0; padding:10px 16px; font-weight:500; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:16px; }
  .stat { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .stat span { display:block; color:var(--dim); font-size:13px; }
  .stat b { font-size:24px; font-variant-numeric:tabular-nums; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:500; }
  .tag { padding:1px 8px; border-radius:99px; font-size:12px; font-weight:600; }
  .tag.saved { background:var(--good); color:#06170e; }
  .tag.other { background:var(--line); color:var(--text); }
  h2 { font-size:15px; color:var(--dim); font-weight:500; margin:20px 0 8px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Erfolge aufnehmen</h1>
    <span class="meta">Modell <b id="model">–</b></span>
    <span class="meta">Dataset <b id="dataset">–</b></span>
  </header>

  <div id="status"><span id="message">Verbinde ...</span><span id="timer"></span></div>

  <div id="cams"><div class="empty">Noch kein Kamerabild</div></div>

  <h2>1. Stein hinlegen und Aufgabe starten</h2>
  <div class="row" id="tasks"></div>

  <h2>2. Wenn der Arm fertig ist: bewerten</h2>
  <div class="row">
    <button class="good" id="yes" disabled>Richtig<small>speichern · Taste J</small></button>
    <button class="bad" id="no" disabled>Falsch<small>verwerfen · Taste N</small></button>
  </div>
  <div class="row">
    <button class="quiet" id="stop" disabled>Arm anhalten (Leertaste)</button>
    <button class="quiet" id="quit">Beenden und Dataset speichern</button>
  </div>

  <div class="stats">
    <div class="stat"><span>Gespeichert</span><b id="saved">0</b></div>
    <div class="stat"><span>Zyklen</span><b id="total">0</b></div>
    <div class="stat"><span>Trefferquote</span><b id="rate">–</b></div>
    <div class="stat"><span>Episoden im Dataset</span><b id="episodes">–</b></div>
  </div>

  <h2>Letzte Zyklen</h2>
  <table><thead><tr><th>Aufgabe</th><th>Ergebnis</th><th>Dauer</th><th>Episode</th></tr></thead>
  <tbody id="history"></tbody></table>
</main>
<script>
const $ = id => document.getElementById(id);
const LABEL = {saved:'gespeichert', wrong:'falsch', aborted:'gestoppt', timeout:'Zeit abgelaufen', failed:'Fehler'};
let state = null, socket = null, cams = {}, tasksShown = '';

function send(message) { if (socket && socket.readyState === 1) socket.send(JSON.stringify(message)); }

function render() {
  const s = state, running = s.phase === 'running', ready = s.phase === 'ready';
  $('model').textContent = s.model; $('dataset').textContent = s.dataset;
  $('message').textContent = running ? 'Läuft: ' + s.task : s.message;
  $('status').className = s.phase;
  const key = s.tasks.join('|');
  if (key !== tasksShown) {
    tasksShown = key; $('tasks').replaceChildren();
    s.tasks.forEach((task, i) => {
      const button = document.createElement('button'); button.className = 'task';
      button.textContent = task;
      const hint = document.createElement('small'); hint.textContent = 'starten · Taste ' + (i + 1);
      button.append(hint); button.onclick = () => send({action:'start', task});
      $('tasks').append(button);
    });
  }
  document.querySelectorAll('.task').forEach(button => button.disabled = !ready);
  $('yes').disabled = $('no').disabled = $('stop').disabled = !running;
  $('quit').disabled = s.phase === 'closing' || s.phase === 'closed';
  $('saved').textContent = s.saved; $('total').textContent = s.total;
  $('rate').textContent = s.total ? Math.round(100 * s.saved / s.total) + ' %' : '–';
  $('episodes').textContent = s.episodes;
  $('history').replaceChildren(...s.history.slice().reverse().map(entry => {
    const row = document.createElement('tr');
    const cells = [entry.task, null, entry.duration_s.toFixed(1) + ' s', entry.episode_index ?? ''];
    cells.forEach(value => {
      const cell = document.createElement('td');
      if (value === null) {
        const tag = document.createElement('span');
        tag.className = 'tag ' + (entry.result === 'saved' ? 'saved' : 'other');
        tag.textContent = LABEL[entry.result] || entry.result; cell.append(tag);
      } else cell.textContent = value;
      row.append(cell);
    });
    return row;
  }));
}

function tick() {
  if (state && state.phase === 'running') {
    const elapsed = Date.now() / 1000 - state.started;
    $('timer').textContent = elapsed.toFixed(1) + ' / ' + state.max_seconds.toFixed(0) + ' s';
  } else $('timer').textContent = '';
}
setInterval(tick, 100);

function showFrame(buffer) {
  const bytes = new Uint8Array(buffer), length = bytes[0];
  const key = new TextDecoder().decode(bytes.subarray(1, 1 + length));
  if (!cams[key]) {
    if (!Object.keys(cams).length) $('cams').replaceChildren();
    const figure = document.createElement('figure'), image = document.createElement('img');
    const caption = document.createElement('figcaption'); caption.textContent = key; image.alt = key;
    figure.append(image, caption); $('cams').append(figure); cams[key] = image;
  }
  const url = URL.createObjectURL(new Blob([bytes.subarray(1 + length)], {type:'image/jpeg'}));
  const image = cams[key], old = image.src;
  image.src = url; if (old.startsWith('blob:')) URL.revokeObjectURL(old);
}

function connect() {
  socket = new WebSocket('ws://' + location.host + '/ws');
  socket.binaryType = 'arraybuffer';
  socket.onmessage = event => {
    if (typeof event.data === 'string') { state = JSON.parse(event.data); render(); }
    else showFrame(event.data);
  };
  socket.onclose = () => {
    if (state && state.phase !== 'closed') {
      $('message').textContent = 'Verbindung zum Aufnahme-Programm getrennt.'; $('status').className = 'error';
      document.querySelectorAll('button').forEach(button => button.disabled = true);
    }
  };
}

$('yes').onclick = () => send({action:'judge', ok:true});
$('no').onclick = () => send({action:'judge', ok:false});
$('stop').onclick = () => send({action:'stop'});
$('quit').onclick = () => send({action:'quit'});
document.addEventListener('keydown', event => {
  if (!state || event.repeat || event.metaKey || event.ctrlKey) return;
  const key = event.key.toLowerCase();
  if (state.phase === 'running') {
    if (key === 'j') send({action:'judge', ok:true});
    else if (key === 'n') send({action:'judge', ok:false});
    else if (key === ' ') { event.preventDefault(); send({action:'stop'}); }
  } else if (state.phase === 'ready' && /^[1-9]$/.test(key) && state.tasks[key - 1]) {
    send({action:'start', task: state.tasks[key - 1]});
  }
});
connect();
</script>
</body>
</html>
"""


def main() -> int:
    extra = [
        (("--port",), {"type": int, "default": 8790, "help": "port of the local page (default: %(default)s)"}),
        (("--no-browser",), {"action": "store_true", "help": "do not open the page automatically"}),
    ]
    args = parse_args(extra)
    try:
        asyncio.run(run(args))
    except ToolError as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
