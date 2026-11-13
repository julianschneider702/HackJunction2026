# robot/ - SO-101 station state for Physical AI Studio

Everything needed to move our Physical AI Studio setup (robots, calibration, cameras,
environment, recordings) between machines through this repo. Owner: Tobby.

Studio itself is **not** in this repo. It is the upstream project
[open-edge-platform/physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio),
cloned next to this repo. Only Studio's *data* lives here.

## What is in here

| Path | Content |
|---|---|
| `studio/physicalai.db` | Studio database snapshot: project `HackJunktion2026`, both arms incl. calibration, 3 cameras, environment `SO101 Teleop`, dataset/model metadata |
| `studio/project-export.json` | Same content as readable JSON (for diffs and for agents) |
| `studio/manifest.json` | When/where the snapshot was taken (used to rewrite paths on import) |
| `calibration/follower.json`, `leader.json` | Current calibration per arm (LeRobot format) |
| `calibration/matching/` | Measurements and before/after values of the leader-follower matching (see below) |
| `datasets/` | Recorded datasets copied out of Studio (empty until we record) |
| `scripts/` | Export/import and arm tools (Python stdlib, except the arm tools which need the Studio venv) |

## Hardware

| Part | Studio name | Notes |
|---|---|---|
| SO-101 follower (gripper arm, 12 V supply) | `SO101 Follower` | serial port differs per machine |
| SO-101 leader (handle + trigger, 5 V supply) | `SO101 Leader` | serial port differs per machine |
| Top-view webcam | `webcam` | 640x480 @ 30 fps |
| Gripper camera (USB2.0_CAM1) | `wrist` | 640x480 @ **15 fps** (shares a USB 2.0 hub with the webcam; at 30 fps both together exceed the bandwidth and `wrist` never gets a first frame) |
| RealSense D415 | `realsense` | opened as a plain UVC camera, so it delivers the **IR image**, not colour. 640x480 @ 30 fps |

Joint units: **normalized** on both arms (a trained policy must use the same).

## Use it on another machine (e.g. the Core Ultra laptop)

1. Clone Studio next to this repo and use the challenge branch:
   ```bash
   git clone -b junction2026 https://github.com/open-edge-platform/physical-ai-studio.git
   ```
   Our snapshot was taken with Studio `main` @ `8f132416`; its DB schema (`b2c4e6f8a012`) also exists on `junction2026`.
2. Install and start Studio as in `physical-ai-studio/JUNCTION2026.md`, Step 1
   (Intel laptop: `uv run --extra xpu physicalai-studio serve`; Mac: `--extra cpu`). Start it once so it creates its storage dir, then **stop the backend**.
3. Import our state:
   ```bash
   python robot/scripts/import_studio_state.py
   ```
   It backs up an existing database, copies ours plus `datasets/`, and rewrites dataset/model paths.
4. Start the backend and UI again. In the UI fix the machine-specific values, **without recalibrating**:
   - **Robot arms:** edit each arm and set its serial port (Linux `/dev/ttyACM*`, Windows `COM*`, macOS `/dev/cu.usbmodem*`). Use **Identify**, or check the diagnostics voltage: 12 V = follower, 5 V = leader.
   - **Cameras:** pick the matching device again for `webcam`, `wrist`, `realsense` if they show offline.
5. Open environment `SO101 Teleop`, turn on **Teleoperate**, check the follower tracks the leader.

## Calibration - do not re-run the setup wizard

The arms were calibrated, then **hand-matched 1:1**: both arms got equal-size ranges shifted by
the offset measured with both arms in the same folded rest pose (`calibration/matching/`),
so the same pose on the leader gives exactly the same pose on the follower.

- Homing offsets and the follower's min/max position limits are stored **in the servos' EEPROM**, so they travel with the arms. The ranges are stored in the Studio database (this folder).
- Re-running **Calibrate/Recalibrate** in the Studio wizard overwrites both and the matching is lost.
- If the follower's servos ever lose their limits (new servo, wizard run), restore them from `calibration/follower.json`:
  ```bash
  python robot/scripts/apply_follower_eeprom_limits.py --port <follower port> --dry-run
  python robot/scripts/apply_follower_eeprom_limits.py --port <follower port>
  ```
- To re-measure the matching, stop the backend and run `scripts/read_ticks.py --follower <port> --leader <port>` with both arms in the same pose.
- The setup wizard's 3D preview is only approximate (it shows normalized values as degrees); judge the matching with teleop, not with the preview.

## Saving new work (recordings, config changes)

On the machine where you recorded:
```bash
python robot/scripts/export_studio_state.py        # add --no-datasets to skip recordings
git add robot && git commit -m "robot: <what changed>"
```
GitHub rejects files over 100 MB; the export script skips those and lists them. Push large
datasets to the Hugging Face Hub instead (Studio's dataset export, or `huggingface-cli upload`).

## Machine notes (macOS dev Mac)

- Homebrew there runs under Rosetta, so `uv` and Node 24 are native arm64 installs in `~/.local`; the Studio venv is `~/.venvs/physicalai-studio`.
- Start the backend from **Terminal.app** (`scripts/start-studio-backend.command`). Started from Claude or an editor, macOS never grants camera access.
- `pyrealsense2-macosx` is removed from that venv: its USB device watcher segfaults the backend on camera discovery. `uv sync` reinstalls it.
- Backend log: `~/Library/Application Support/physicalai/logs/app.log`.
