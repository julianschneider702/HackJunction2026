"""Anomaly scoring process. Runs in the anomalib venv (Python 3.11), started by policy_sort.py.

Anomalib is not installed in Studio's venv, so the sorter talks to this process over a local
socket (multiprocessing.connection, raw bytes + JSON, no pickling between Python versions).

Request:  send_bytes(JSON {"model": name, "shape": [n, h, w, 3]}), send_bytes(raw uint8 RGB)
Reply:    send_bytes(JSON {"scores": [...]}) or {"error": "..."}
Scores are Anomalib's normalized image scores (0..1, above 0.5 = anomalous at the trained threshold).

    ~/anomalib-env/bin/python anomaly_worker.py ADDRESS '{"wrist": "/path/model.ckpt", ...}'
Authkey comes from the environment variable SORTER_AUTHKEY.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def load(path: str, device):
    from anomalib.models import AnomalyDINO, Padim, Patchcore

    # the checkpoint knows its class; try the ones sort_defects.py trains
    errors = []
    for cls in (AnomalyDINO, Patchcore, Padim):
        try:
            m = cls.load_from_checkpoint(path, map_location="cpu")
            return m.eval().to(device)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls.__name__}: {e}")
    raise RuntimeError(f"cannot load {path}: " + " | ".join(errors))


def main():
    import numpy as np
    import torch
    from multiprocessing.connection import Listener

    address, models_json = sys.argv[1], sys.argv[2]
    authkey = os.environ["SORTER_AUTHKEY"].encode()
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    models = {}
    for name, path in json.loads(models_json).items():
        if path:
            t0 = time.time()
            models[name] = load(os.path.expanduser(path), device)
            print(f"[anomaly] {name}: {path} on {device} ({time.time() - t0:.1f}s)", flush=True)

    # warm-up so the first real request is not slow
    for m in models.values():
        with torch.no_grad():
            m(torch.zeros(1, 3, 480, 640, device=device))

    with Listener(address, family="AF_UNIX", authkey=authkey) as listener:
        print("[anomaly] ready", flush=True)
        with listener.accept() as conn:
            while True:
                try:
                    head = json.loads(conn.recv_bytes())
                except (EOFError, OSError):
                    return
                if head.get("cmd") == "quit":
                    return
                raw = conn.recv_bytes()
                try:
                    m = models[head["model"]]
                    imgs = np.frombuffer(raw, np.uint8).reshape(head["shape"])
                    x = torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2).float().div(255).to(device)
                    with torch.no_grad():
                        out = m(x)
                    scores = out.pred_score.detach().float().cpu().reshape(-1).tolist()
                    amap = getattr(out, "anomaly_map", None) if head.get("maps") else None
                    if amap is None:
                        conn.send_bytes(json.dumps({"scores": scores}).encode())
                    else:
                        # normalized maps (0..1 around the pixel threshold) as uint8, max 256 px
                        a = amap.detach().float().cpu().reshape(len(scores), *amap.shape[-2:])
                        a = torch.nn.functional.interpolate(a[:, None], size=(min(256, a.shape[-2]), min(256, a.shape[-1])),
                                                            mode="bilinear", align_corners=False)[:, 0]
                        u8 = (a.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                        conn.send_bytes(json.dumps({"scores": scores, "map_shape": list(u8.shape)}).encode())
                        conn.send_bytes(u8.tobytes())
                except Exception as e:  # noqa: BLE001
                    conn.send_bytes(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())


if __name__ == "__main__":
    main()
