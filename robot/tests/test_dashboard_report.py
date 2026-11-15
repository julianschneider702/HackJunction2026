"""Tests for the dashboard reporting of the recording tools.

    python -m unittest discover -s robot/tests -v
"""

import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot" / "scripts"))
sys.path.insert(0, str(ROOT / "frontend"))

import hub as hub_module  # noqa: E402
from dashboard_report import DashboardReporter, bowl_for  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 40


def observation(frames: dict) -> bytes:
    header = json.dumps({"frames": [{"key": key, "size": len(data)} for key, data in frames.items()]}).encode()
    return len(header).to_bytes(4, "big") + header + b"".join(frames.values())


class BowlTests(unittest.TestCase):
    def test_bowl_from_instruction(self):
        self.assertEqual(bowl_for("Put the brick into the defective bowl"), "defective")
        self.assertEqual(bowl_for("Lege den Stein in die defekt-Schale"), "defective")
        self.assertEqual(bowl_for("put the brick into the good bowl"), "good")
        self.assertEqual(bowl_for("Stein in die gute Schale"), "good")
        self.assertIsNone(bowl_for("sort the brick"))


class ReporterTests(unittest.TestCase):
    def setUp(self):
        self.hub = hub_module.Hub(None)
        self.server = hub_module.make_server("127.0.0.1", 0, self.hub)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def state(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.url + "/api/state", timeout=5) as response:
            return json.load(response)

    def test_cycles_reach_the_hub(self):
        reporter = DashboardReporter(self.url, camera="wrist")
        found = {"backend": "openvino", "model": {"name": "SmolVLA"}}
        reporter.session_started(found, "npu")
        reporter.on_frames(observation({"webcam": JPEG + b"A", "wrist": JPEG + b"B"}))
        reporter.on_frames(b"garbage")  # must be ignored
        self.assertEqual(reporter.frame, JPEG + b"B")

        reporter.cycle_started("Put the brick into the good bowl")
        reporter.cycle_finished("Put the brick into the good bowl", "saved", 21.5)
        reporter.cycle_started("Put the brick into the defective bowl")
        reporter.cycle_finished("Put the brick into the defective bowl", "wrong", 30.0)
        reporter.cycle_started("sort the brick")
        reporter.cycle_finished("sort the brick", "timeout", 60.0)
        reporter.cycle_started("sort the brick")
        reporter.cycle_finished("sort the brick", "aborted", 3.0)  # stopped by a person: not a result
        reporter.hub.flush()

        state = self.state()
        self.assertFalse(state["demo"])
        self.assertEqual(state["models"], [{"role": "policy", "name": "SmolVLA", "device": "NPU", "precision": None, "runtime": "OpenVINO"}])
        stats = state["stats"]
        self.assertEqual((stats["total"], stats["good"], stats["judged"], stats["correct"]), (3, 1, 3, 1))
        newest_first = [(c["placed"], c["truth"], c["correct"], c["has_image"]) for c in state["cycles"]]
        self.assertEqual(newest_first, [(None, None, False, True), (None, "defective", False, True), ("good", "good", True, True)])
        self.assertEqual(state["phase"]["phase"], "idle")

        # the dashboard's Stop button reaches the tool within about a quarter of a second
        self.hub.command("stop")
        deadline = time.monotonic() + 2
        while not reporter.stop_requested and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(reporter.stop_requested)
        reporter.stopped_by_dashboard()
        reporter.session_closed()
        self.assertEqual(self.state()["robot"]["state"], "idle")

    def test_off_and_unreachable_do_nothing(self):
        for url in ("off", None, "http://127.0.0.1:9"):
            reporter = DashboardReporter(url)
            reporter.session_started({"backend": "openvino", "model": {"name": "m"}}, "auto")
            reporter.on_frames(observation({"webcam": JPEG}))
            reporter.cycle_started("t")
            reporter.cycle_finished("t", "saved", 1.0)
            self.assertFalse(reporter.stop_requested)


if __name__ == "__main__":
    unittest.main()
