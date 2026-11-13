#!/bin/zsh
# macOS only: start the Physical AI Studio backend (port 7860) from Terminal.app.
# Double-click it in Finder. It must run from Terminal (not from an editor/agent shell),
# otherwise macOS never grants camera access to the backend.
# Expects physical-ai-studio next to this repo, or set PHYSICAL_AI_STUDIO.
cd "${PHYSICAL_AI_STUDIO:-$(dirname "$0")/../../../physical-ai-studio}/application/backend" || exit 1
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/physicalai-studio"
export PATH="$HOME/.local/bin:$PATH"
./run.sh serve
