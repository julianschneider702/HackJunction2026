"""Tests for the dashboard hub, its client and the telemetry reader.

    python -m unittest discover -s frontend/tests -v
"""

import base64
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hub as hub_module  # noqa: E402
from hub_client import HubClient  # noqa: E402
from simulate import brick_images  # noqa: E402
from telemetry_linux import Sensors  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CYCLE = {"type": "cycle", "duration_s": 12.5, "anomaly_score": 0.7, "threshold": 0.5,
         "decision": "defective", "placed": "defective"}


class HubServerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.start()

    def start(self):
        self.hub = hub_module.Hub(Path(self.tmp.name))
        self.server = hub_module.make_server("127.0.0.1", 0, self.hub)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, content_type="application/json"):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": content_type} if data is not None else {}
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with OPENER.open(req, timeout=5) as response:
                raw = response.read()
                kind = response.headers.get("Content-Type", "")
                return response.status, json.loads(raw) if "json" in kind else raw
        except urllib.error.HTTPError as err:
            with err:
                return err.code, json.loads(err.read())

    def post(self, event):
        return self.request("POST", "/api/events", event)

    def state(self):
        return self.request("GET", "/api/state")[1]


class EventTests(HubServerCase):
    def test_cycle_updates_stats(self):
        self.assertEqual(self.post(CYCLE)[0], 200)
        self.post({**CYCLE, "decision": "good", "placed": "good", "anomaly_score": 0.2, "duration_s": 7.5, "regrasps": 1})
        self.post({**CYCLE, "placed": "none"})
        state = self.state()
        stats = state["stats"]
        self.assertEqual((stats["total"], stats["good"], stats["defective"], stats["lost"]), (3, 1, 1, 1))
        self.assertEqual(stats["regrasps"], 1)
        self.assertAlmostEqual(stats["avg_cycle_s"], (12.5 + 7.5 + 12.5) / 3)
        self.assertAlmostEqual(stats["bricks_per_min"], 60 / stats["avg_cycle_s_last10"])
        self.assertIsNone(stats["accuracy"])
        self.assertEqual([c["id"] for c in state["cycles"]], ["c-3", "c-2", "c-1"])  # newest first
        self.assertFalse(state["demo"])

    def test_accuracy_from_sender_truth_and_operator_verdict(self):
        self.post({**CYCLE, "truth": "defective"})                      # correct
        self.post({**CYCLE, "truth": "good"})                           # wrong bowl
        self.post({**CYCLE, "placed": "none", "truth": "defective"})    # lost counts as wrong
        self.post(CYCLE)                                                # not judged yet
        stats = self.state()["stats"]
        self.assertEqual((stats["judged"], stats["correct"]), (3, 1))
        self.assertEqual(self.request("POST", "/api/cycles/c-4/verdict", {"truth": "defective"})[0], 200)
        stats = self.state()["stats"]
        self.assertEqual((stats["judged"], stats["correct"]), (4, 2))
        self.assertAlmostEqual(stats["accuracy"], 0.5)
        self.request("POST", "/api/cycles/c-4/verdict", {"truth": None})
        self.assertEqual(self.state()["stats"]["judged"], 3)
        self.assertEqual(self.request("POST", "/api/cycles/nope/verdict", {"truth": "good"})[1]["error"]["code"], "unknown_cycle")
        self.assertEqual(self.request("POST", "/api/cycles/c-1/verdict", {"truth": "maybe"})[1]["error"]["code"], "invalid_truth")

    def test_cycle_without_score_and_with_reported_result(self):
        # what the recording tool sends: no anomaly model, a person judged the result
        self.post({"type": "cycle", "duration_s": 20, "instruction": "sort", "truth": "good", "placed": "good", "correct": True})
        self.post({"type": "cycle", "duration_s": 30, "truth": "defective", "correct": False})
        self.post({"type": "cycle", "duration_s": 10})  # nothing known but the time
        state = self.state()
        stats = state["stats"]
        self.assertEqual((stats["total"], stats["good"], stats["defective"], stats["lost"]), (3, 1, 0, 0))
        self.assertEqual((stats["judged"], stats["correct"]), (2, 1))
        self.assertEqual([c["correct"] for c in state["cycles"]], [None, False, True])
        self.assertIsNone(state["cycles"][0]["anomaly_score"])
        # the operator's mark beats the reported result once the bowl is known
        self.request("POST", "/api/cycles/c-1/verdict", {"truth": "defective"})
        self.assertFalse(self.state()["cycles"][2]["correct"])
        self.assertEqual(self.post({"type": "cycle", "duration_s": 5, "correct": "yes"})[0], 400)

    def test_invalid_events_are_rejected_and_nothing_is_applied(self):
        bad = [
            {"type": "nope"},
            {**CYCLE, "decision": "broken"},
            {**CYCLE, "anomaly_score": "0.7"},
            {**CYCLE, "duration_s": -1},
            {**CYCLE, "regrasps": 1.5},
            {**CYCLE, "image_b64": "not base64!"},
            {**CYCLE, "image_b64": base64.b64encode(b"plain text").decode()},
            {"type": "phase", "phase": "dance"},
            {"type": "inference", "model": "policy", "latency_ms": []},
            {"type": "inference", "model": "policy", "latency_ms": [5, -1]},
            {"type": "telemetry", "util": {"cpu": 140}},
            {"type": "benchmark", "model": "m", "device": "TPU", "precision": "INT8", "latency_ms": 5},
            {"type": "models", "models": [{"role": "policy", "name": "x"}]},
            {"type": "robot", "state": "dancing"},
        ]
        for event in bad:
            status, body = self.post(event)
            self.assertEqual((status, body["error"]["code"]), (400, "invalid_event"), event)
        # one bad event rejects the whole request
        self.assertEqual(self.post({"events": [CYCLE, {"type": "nope"}]})[0], 400)
        self.assertEqual(self.state()["stats"]["total"], 0)

    def test_transport_errors(self):
        self.assertEqual(self.request("POST", "/api/events", CYCLE, content_type="text/plain")[0], 415)
        req = urllib.request.Request(self.url + "/api/events", data=b"{not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            OPENER.open(req, timeout=5)
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()
        self.assertEqual(self.request("GET", "/api/nope")[0], 404)
        self.assertEqual(self.request("POST", "/api/nope", {})[0], 404)

    def test_phase_robot_models_benchmarks(self):
        self.post({"events": [
            {"type": "phase", "phase": "detect", "cycle_id": "b7", "detail": "scoring"},
            {"type": "robot", "state": "running"},
            {"type": "models", "models": [{"role": "policy", "name": "SmolVLA", "device": "npu", "precision": "INT8", "runtime": "OpenVINO"}]},
            {"type": "benchmark", "model": "SmolVLA", "device": "CPU", "precision": "FP32", "latency_ms": 300},
            {"type": "benchmark", "model": "SmolVLA", "device": "iGPU", "precision": "FP16", "latency_ms": 90},
            {"type": "benchmark", "model": "SmolVLA", "device": "CPU", "precision": "FP32", "latency_ms": 280},
        ]})
        state = self.state()
        self.assertEqual((state["phase"]["phase"], state["phase"]["cycle_id"]), ("detect", "b7"))
        self.assertEqual(state["robot"]["state"], "running")
        self.assertEqual(state["models"][0]["device"], "NPU")
        self.assertEqual([(b["device"], b["latency_ms"]) for b in state["benchmarks"]], [("GPU", 90), ("CPU", 280)])

    def test_inference_and_telemetry(self):
        self.post({"type": "inference", "model": "policy", "device": "NPU", "latency_ms": list(range(1, 101))})
        self.post({"type": "inference", "model": "anomaly", "latency_ms": 17.5})
        self.post({"type": "telemetry", "power_w": 18.2, "temp_c": 71, "temp_limit_c": 100, "util": {"cpu": 30, "npu": 80}})
        state = self.state()
        policy = state["inference"]["policy"]
        self.assertEqual((policy["count"], policy["device"], policy["p95_ms"], policy["last_ms"]), (100, "NPU", 95, 100))
        self.assertAlmostEqual(policy["mean_ms"], 50.5)
        self.assertEqual(state["inference"]["anomaly"]["mean_ms"], 17.5)
        latest = state["telemetry"]["latest"]
        self.assertEqual((latest["power_w"], latest["util"]), (18.2, {"cpu": 30, "gpu": None, "npu": 80}))

    def test_images(self):
        image, heatmap = brick_images(__import__("random").Random(1), True, 0.8)
        self.post({**CYCLE, "cycle_id": "img 1", "image_b64": base64.b64encode(image).decode(),
                   "heatmap_b64": "data:image/png;base64," + base64.b64encode(heatmap).decode()})
        cycle = self.state()["cycles"][0]
        self.assertTrue(cycle["has_image"] and cycle["has_heatmap"])
        self.assertNotIn("image", cycle)
        status, body = self.request("GET", "/api/cycles/img%201/image")
        self.assertEqual((status, body), (200, image))
        self.assertEqual(self.request("GET", "/api/cycles/img%201/heatmap")[1], heatmap)
        self.assertEqual(self.request("GET", "/api/cycles/c-9/image")[1]["error"]["code"], "no_image")

    def test_control_and_since(self):
        self.assertFalse(self.request("GET", "/api/control")[1]["stop_requested"])
        self.assertTrue(self.request("POST", "/api/control", {"command": "stop"})[1]["stop_requested"])
        self.assertTrue(self.post(CYCLE)[1]["stop_requested"])  # the loop learns about it with every answer
        self.assertFalse(self.request("POST", "/api/control", {"command": "resume"})[1]["stop_requested"])
        self.assertEqual(self.request("POST", "/api/control", {"command": "explode"})[1]["error"]["code"], "invalid_command")
        seq = self.state()["seq"]
        self.assertEqual(self.request("GET", f"/api/state?since={seq}")[1], {"seq": seq, "unchanged": True})
        self.request("POST", "/api/control", {"command": "reset"})
        state = self.state()
        self.assertEqual((state["stats"]["total"], state["cycles"]), (0, []))

    def test_demo_flag_and_persistence(self):
        self.post({**CYCLE, "source": "sim", "truth": "defective"})
        self.post({"type": "benchmark", "model": "m", "device": "NPU", "precision": "INT8", "latency_ms": 40})
        self.assertTrue(self.state()["demo"])
        self.stop()
        self.start()  # a restarted hub keeps cycles and benchmarks
        state = self.state()
        self.assertEqual((state["stats"]["total"], state["stats"]["correct"], len(state["benchmarks"])), (1, 1, 1))
        self.assertTrue(state["demo"])
        self.post(CYCLE)
        self.assertEqual(self.state()["cycles"][0]["id"], "c-2")

    def test_static_files(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Inspection Cell", body)
        self.assertEqual(self.request("GET", "/app.js")[0], 200)
        for path in ("/../hub.py", "/%2e%2e/hub.py", "/..%2fhub.py", "/nope.html"):
            self.assertEqual(self.request("GET", path)[0], 404, path)


class ClientTests(HubServerCase):
    def test_client_sends_in_background_and_sees_stop(self):
        client = HubClient(self.url, source="sim", latency_flush_s=0.1)
        client.phase("execute", detail="placing")
        for value in (40, 50, 60):
            client.latency("policy", value, device="NPU")
        client.cycle(duration_s=9, anomaly_score=0.1, threshold=0.5, decision="good", placed="good", image=b"\xff\xd8\xff\xe0rest")
        client.telemetry(power_w=15, cpu=20)
        client.send("cycle", decision="good")  # invalid: must not take the valid events down with it
        client.flush()
        time.sleep(0.3)
        client.flush()
        state = self.state()
        self.assertEqual(state["phase"]["phase"], "execute")
        self.assertEqual(state["stats"]["total"], 1)
        self.assertTrue(state["cycles"][0]["has_image"])
        self.assertEqual(state["inference"]["policy"]["count"], 3)
        self.assertEqual(state["telemetry"]["latest"]["util"]["cpu"], 20)
        self.assertTrue(state["demo"])
        self.assertIn("duration_s", client.last_error)
        self.request("POST", "/api/control", {"command": "stop"})
        self.assertTrue(client.check_stop())

    def test_client_survives_missing_hub(self):
        client = HubClient("http://127.0.0.1:9", timeout=0.3)
        client.phase("idle")
        client.flush(timeout=2)
        self.assertFalse(client.connected)
        self.assertFalse(client.check_stop())


@unittest.skipIf(sys.platform == "win32", "simulates Linux sysfs paths, which Windows file names do not allow")
class SensorTests(unittest.TestCase):
    def test_rates_from_fake_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write(path, text):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"{text}\n")

            rapl, zone = "sys/class/powercap/intel-rapl:0", "sys/class/thermal/thermal_zone3"
            gpu = "sys/class/drm/card0/device/tile0/gt0/gtidle/idle_residency_ms"
            npu = "sys/class/accel/accel0/device/npu_busy_time_us"
            write(f"{rapl}/name", "package-0")
            write(f"{rapl}/max_energy_range_uj", 1_000_000_000)
            write(f"{rapl}:0/name", "core")
            write(f"{rapl}:0/energy_uj", 5)
            write(f"{zone}/type", "x86_pkg_temp")
            write(f"{zone}/temp", 71000)
            write(f"{zone}/trip_point_0_type", "critical")
            write(f"{zone}/trip_point_0_temp", 100000)
            write(f"{rapl}/energy_uj", 999_000_000)
            write("proc/stat", "cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3")
            write(gpu, 1000)
            write(npu, 0)

            sensors = Sensors(str(root))
            first = sensors.sample()
            self.assertEqual((first["temp_c"], first["temp_limit_c"], first["power_w"]), (71.0, 100.0, None))

            sensors._previous_time -= 2.0  # pretend two seconds passed
            write(f"{rapl}/energy_uj", 29_000_000)   # wrapped: 1 J to the end + 29 J = 30 J in 2 s
            write("proc/stat", "cpu  200 0 150 900 150 0 0 0 0 0")  # 400 ticks, 250 idle
            write(gpu, 1500)                          # idle 500 ms of 2000 ms
            write(npu, 1_000_000)                     # busy 1 s of 2 s
            second = sensors.sample()
            self.assertAlmostEqual(second["power_w"], 15.0, delta=0.2)
            self.assertAlmostEqual(second["cpu"], 37.5, delta=0.2)
            self.assertAlmostEqual(second["gpu"], 75.0, delta=1.0)
            self.assertAlmostEqual(second["npu"], 50.0, delta=1.0)
            self.assertEqual(sensors.unreadable, [])

    def test_missing_files_give_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            sensors = Sensors(tmp)
            sensors.sample()
            reading = sensors.sample()
            self.assertTrue(all(v is None for v in reading.values()))
            self.assertEqual(set(sensors.unreadable), {"power_w", "temp_c", "cpu", "gpu", "npu"})


if __name__ == "__main__":
    unittest.main()
