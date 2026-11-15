#!/bin/bash
# Live load of the Intel GPU (XPU), without sudo.
# Reads the per-process counters of the xe driver. Stop with Ctrl+C.
#   station/xpu-usage.sh          live view
#   station/xpu-usage.sh --once   measure once and exit
python3 -u - "$@" <<'EOF'
import glob, time, os, sys
ONCE = "--once" in sys.argv

ENGINES = {"ccs": "Compute (training / AI)", "rcs": "Render (3D)", "vcs": "Video", "bcs": "Copy"}

def sample():
    busy, total, procs = {}, {}, {}
    for fd in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            txt = open(fd).read()
        except OSError:
            continue
        if "drm-driver:\txe" not in txt:
            continue
        pid = fd.split("/")[2]
        cid = next((l.split()[1] for l in txt.splitlines() if l.startswith("drm-client-id")), fd)
        for line in txt.splitlines():
            k, _, v = line.partition(":")
            v = v.split()
            if not v:
                continue
            if k.startswith("drm-cycles-"):
                e = k[len("drm-cycles-"):]
                busy[(cid, e)] = int(v[0])
                procs[(cid, e)] = pid
            elif k.startswith("drm-total-cycles-"):
                total[(cid, k[len("drm-total-cycles-"):])] = int(v[0])
    return busy, total, procs

def name(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
        return (cmd[:60] + "...") if len(cmd) > 60 else cmd
    except OSError:
        return "?"

b0, t0, _ = sample()
while True:
    time.sleep(1)
    b1, t1, procs = sample()
    per_engine, per_pid = {}, {}
    for key, b in b1.items():
        if key not in b0 or key not in t0 or key not in t1:
            continue
        dt = t1[key] - t0[key]
        if dt <= 0:
            continue
        pct = 100 * (b - b0[key]) / dt
        e = key[1]
        per_engine[e] = per_engine.get(e, 0) + pct
        if e == "ccs" and pct > 0.5:
            per_pid[procs[key]] = per_pid.get(procs[key], 0) + pct
    if not ONCE:
        print("\033[2J\033[H", end="")
    print("Intel Arc GPU (XPU) load" + ("" if ONCE else "   (Ctrl+C quits)") + "\n")
    for e, label in ENGINES.items():
        p = min(per_engine.get(e, 0), 100)
        print(f"{label:24s} {p:5.1f}%  " + "#" * int(p / 2.5))
    if per_pid:
        print("\nProcesses on compute:")
        for pid, p in sorted(per_pid.items(), key=lambda x: -x[1]):
            print(f"  {min(p,100):5.1f}%  pid {pid}  {name(pid)}")
    if ONCE:
        break
    b0, t0 = b1, t1
EOF
