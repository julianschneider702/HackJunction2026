#!/usr/bin/env python3
"""Load the Studio state from robot/ into Physical AI Studio on this machine.

Stop the Studio backend first. Then:
  python robot/scripts/import_studio_state.py [--storage-dir DIR] [--force]

It copies robot/studio/physicalai.db and robot/datasets/ into the Studio storage dir,
rewrites absolute dataset/model paths to this machine, and keeps a backup of an
existing database (physicalai.db.bak-<timestamp>).

After importing, open each robot and camera in the UI and fix machine-specific
values (serial port, camera device) - see robot/README.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from studio_paths import default_storage_dir

ROBOT_DIR = Path(__file__).resolve().parent.parent


def backend_running() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("http://127.0.0.1:7860/api/health", timeout=2)
    except OSError:
        return False
    return True


def rewrite_paths(db: Path, old_root: str, new_root: Path) -> int:
    changed = 0
    with sqlite3.connect(db) as conn:
        for table in ("datasets", "models"):
            rows = conn.execute(f"SELECT id, path FROM {table} WHERE path IS NOT NULL").fetchall()  # noqa: S608
            for row_id, path in rows:
                normalized = path.replace("\\", "/")
                old = old_root.replace("\\", "/").rstrip("/")
                if normalized.startswith(old + "/"):
                    new_path = str(new_root / Path(normalized[len(old) + 1 :]))
                    conn.execute(f"UPDATE {table} SET path = ? WHERE id = ?", (new_path, row_id))  # noqa: S608
                    changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--storage-dir", type=Path, default=default_storage_dir())
    parser.add_argument("--force", action="store_true", help="import even if the backend answers on :7860")
    args = parser.parse_args()

    if backend_running() and not args.force:
        print("The Studio backend is running. Stop it first (it keeps the database open).")
        return 1

    src_db = ROBOT_DIR / "studio" / "physicalai.db"
    if not src_db.exists():
        print(f"missing {src_db}; run export_studio_state.py on the source machine first")
        return 1

    storage = args.storage_dir.expanduser()
    dst_db = storage / "data" / "physicalai.db"
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    if dst_db.exists():
        backup = dst_db.with_name(f"physicalai.db.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(dst_db, backup)
        print(f"backup     -> {backup}")
    shutil.copy2(src_db, dst_db)
    print(f"database   -> {dst_db}")

    src_datasets = ROBOT_DIR / "datasets"
    if src_datasets.exists():
        for file in src_datasets.rglob("*"):
            if file.is_file() and file.name not in {".gitkeep", "README.md"}:
                target = storage / "datasets" / file.relative_to(src_datasets)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        print(f"datasets   -> {storage / 'datasets'}")

    manifest_file = ROBOT_DIR / "studio" / "manifest.json"
    if manifest_file.exists():
        old_root = json.loads(manifest_file.read_text())["source_storage_dir"]
        print(f"paths      -> {rewrite_paths(dst_db, old_root, storage)} dataset/model paths rewritten")

    print("\nDone. Start the backend, then fix serial ports and camera devices in the UI (robot/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
