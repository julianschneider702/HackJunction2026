#!/usr/bin/env python3
"""Copy the Physical AI Studio state of this machine into robot/ so it can be committed.

What it copies (stdlib only, works on macOS / Linux / Windows):
  - the Studio database (projects, robots incl. calibration, cameras, environments,
    datasets/models metadata) -> robot/studio/physicalai.db
  - a readable JSON export of robots / cameras / environments (needs the backend running)
  - the per-arm calibration JSON (LeRobot format) -> robot/calibration/
  - recorded datasets -> robot/datasets/ (skipped with --no-datasets)

Usage:
  python robot/scripts/export_studio_state.py [--storage-dir DIR] [--no-datasets]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from studio_paths import default_storage_dir

ROBOT_DIR = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:7860/api"
GITHUB_FILE_LIMIT = 95 * 1024 * 1024  # GitHub rejects files over 100 MB


def backup_db(src: Path, dst: Path) -> None:
    """Consistent snapshot even while the backend is running."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as source, sqlite3.connect(dst) as target:
        source.backup(target)


def export_api_json() -> bool:
    try:
        projects = json.load(urllib.request.urlopen(f"{API}/projects", timeout=5))
    except OSError:
        print("! backend not reachable on :7860, skipping JSON export (the .db is still exported)")
        return False
    export = []
    for project in projects:
        base = f"{API}/projects/{project['id']}"
        get = lambda path: json.load(urllib.request.urlopen(base + path, timeout=10))  # noqa: E731
        robots = get("/robots")
        export.append(
            {"project": project, "robots": robots, "cameras": get("/cameras"), "environments": get("/environments")}
        )
        for robot in robots:
            calibration = robot.get("payload", {}).get("calibration")
            if calibration:
                role = "follower" if robot["type"].endswith("Follower") else "leader"
                out = ROBOT_DIR / "calibration" / f"{role}.json"
                out.write_text(json.dumps(calibration, indent=2) + "\n")
    (ROBOT_DIR / "studio" / "project-export.json").write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n")
    return True


def copy_tree(src: Path, dst: Path) -> list[Path]:
    too_big = []
    if not src.exists():
        return too_big
    for file in src.rglob("*"):
        if not file.is_file():
            continue
        if file.stat().st_size > GITHUB_FILE_LIMIT:
            too_big.append(file)
            continue
        target = dst / file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    return too_big


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--storage-dir", type=Path, default=default_storage_dir())
    parser.add_argument("--no-datasets", action="store_true", help="do not copy recorded datasets")
    args = parser.parse_args()

    storage = args.storage_dir.expanduser()
    db = storage / "data" / "physicalai.db"
    if not db.exists():
        print(f"no Studio database at {db}")
        return 1

    backup_db(db, ROBOT_DIR / "studio" / "physicalai.db")
    print(f"database   -> {ROBOT_DIR / 'studio' / 'physicalai.db'}")
    if export_api_json():
        print(f"json       -> {ROBOT_DIR / 'studio' / 'project-export.json'}, robot/calibration/*.json")

    too_big = []
    if not args.no_datasets:
        too_big = copy_tree(storage / "datasets", ROBOT_DIR / "datasets")
        print(f"datasets   -> {ROBOT_DIR / 'datasets'}")

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_platform": sys.platform,
        "source_storage_dir": str(storage),
    }
    (ROBOT_DIR / "studio" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if too_big:
        print("\n! these files are over GitHub's 100 MB limit and were NOT copied:")
        for file in too_big:
            print(f"  {file}")
        print("  Push those datasets to the Hugging Face Hub instead (see robot/README.md).")
    print("\nNext: review `git status`, then commit robot/ on your branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
