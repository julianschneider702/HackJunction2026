#!/usr/bin/env bash
# Station laptop (Linux): start the dashboard hub and send real power, temperature and device load to it.
#   bash frontend/start_laptop.sh        # Ctrl+C stops both
# Not yet run on the station laptop. Check the readings first: python3 frontend/telemetry_linux.py --once
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8090}"

python3 hub.py --port "$PORT" &
HUB=$!
trap 'kill "$HUB" 2>/dev/null || true' EXIT
sleep 1
echo "Dashboard: http://127.0.0.1:$PORT"
echo "The power counter is root-only on most kernels, so sudo may ask for your password."
sudo python3 telemetry_linux.py --url "http://127.0.0.1:$PORT"
