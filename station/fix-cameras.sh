#!/bin/bash
# Bring the cameras back "online" in Studio after they were re-plugged.
# Writes the current USB fingerprint of each connected camera into its Studio camera entry.
python3 - <<'EOF'
import json, urllib.request
API = "http://localhost:7860/api"
projects = json.load(urllib.request.urlopen(f"{API}/projects"))
hw = {c["name"]: c["fingerprint"]
      for c in json.load(urllib.request.urlopen(f"{API}/hardware/cameras"))
      if c["driver"] == "usb_camera"}
for p in projects:
    base = f"{API}/projects/{p['id']}/cameras"
    for cam in json.load(urllib.request.urlopen(base)):
        fp = hw.get(cam["hardware_name"])
        if not fp:
            print(f"not connected: {cam['name']}")
        elif cam["fingerprint"] == fp:
            print(f"ok:            {cam['name']}")
        else:
            cam["fingerprint"] = fp
            req = urllib.request.Request(f"{base}/{cam['id']}", data=json.dumps(cam).encode(),
                                         method="PUT", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req)
            print(f"updated:       {cam['name']}")
print("Done - reload the page in the browser (F5).")
EOF
