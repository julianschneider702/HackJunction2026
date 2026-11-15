#!/bin/bash
# Start Physical AI Studio (backend + UI). Stop with Ctrl+C.
# Studio is expected next to this repo (../physical-ai-studio); set PHYSICAL_AI_STUDIO to use another checkout.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="${PHYSICAL_AI_STUDIO:-$REPO/../physical-ai-studio}/application"
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"
export PATH="$HOME/.local/bin:$PATH"

(cd "$APP/backend" && ./run.sh) &
BACKEND=$!
(cd "$APP/ui" && npm run start) &
UI=$!
trap 'kill $BACKEND $UI 2>/dev/null' EXIT INT TERM

echo "Studio is starting... open http://localhost:3000 in the browser"
wait
