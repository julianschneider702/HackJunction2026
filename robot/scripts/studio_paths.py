"""Where Physical AI Studio keeps its data (mirrors backend/src/settings.py get_default_storage_dir)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def default_storage_dir() -> Path:
    if override := os.environ.get("STORAGE_DIR"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path("~/Library/Application Support/physicalai").expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        return (Path(local_app_data) if local_app_data else Path("~/AppData/Local").expanduser()) / "physicalai"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg and Path(xdg).expanduser().is_absolute():
        return Path(xdg).expanduser() / "physicalai"
    return Path("~/.local/share/physicalai").expanduser()
