# sorter/ - the inspection loop

Drives the full cycle on the station: the policy picks a brick, the script checks it for defects while it
is in the gripper, and the policy places it in the right bowl.

**Yellow bowl = defective, blue bowl = good.**

All scripts run with the Python environment of Studio's backend (`station/studio-python`), and Physical AI
Studio must be running. Close the session in the Studio web UI first, because only one process can hold
the arm and the cameras.

## Scripts

| Script | What it does |
|---|---|
| `policy_sort.py` | The main loop: policy picks, script inspects, policy places |
| `anomaly_watch.py` | Read-only: watches a run started in the Studio web UI and scores every picked brick |
| `sort_bricks.py` | Earlier variant: the script places the brick itself and tracks the bowls with the top camera |
| `calibrate.py` | Teaches poses, bowl drop points and camera regions for `sort_bricks.py` |
| `anomaly.py`, `anomaly_worker.py` | Runs the Anomalib model in its own process (Anomalib lives in a separate Python environment) |
| `vision.py` | Classical image checks: marker-stroke detector, bowl and box tracking |
| `studio.py` | Connection to Studio's runtime session: cameras, arm, policy start and stop |
| `dashboard.py` | Reports every brick to the dashboard in `frontend/` |
| `common.py`, `ui.py` | Configuration defaults, calibration file, logging, debug window |

## The main loop: `policy_sort.py`

```bash
station/studio-python sorter/policy_sort.py            # real run
station/studio-python sorter/policy_sort.py --manual   # test the inspection only: put a brick into the gripper by hand, nothing is placed
station/studio-python sorter/policy_sort.py --window   # also show the debug window with the three cameras
```

Without `--window` there is no window: the log runs in the terminal and the results appear in the dashboard.

| Input | Effect |
|---|---|
| `ENTER` | pause / continue |
| `q` + `ENTER` | stop |
| `Ctrl+C` | stop, twice = emergency hold |
| `SPACE` / `q` / `ESC` | the same in the `--window` debug window |

One cycle:

1. The policy picks a brick with the task `put the brick in the yellow bowl`. As soon as the brick is
   lifted, the script stops the policy and the arm holds still.
2. The wrist camera image goes through the anomaly model (`~/bricks-data/model_anomalydino.ckpt`).
   If it is flagged, the brick is defective.
3. Otherwise the arm carries the brick in front of the webcam and turns the wrist through eight angles.
   Every angle is checked: by a second anomaly model if one is configured (`anomaly.webcam_model`),
   until then by the stroke detector in `vision.py`. The arm then returns to where the policy was stopped.
4. The policy takes over again with `put the brick in the yellow bowl` or `put the brick in the blue bowl`
   and places the brick.

Every checked image is saved with its score in the file name under `sorter/log/policy_<time>/`, which makes
them usable as training images later. The results of a run go to `sorter/log/policy_<time>.csv`.

The bowls must stand where they stood in the training episodes. The policy does not find them elsewhere.

**Dashboard.** If the dashboard hub is running (`frontend/hub.py`, http://127.0.0.1:8090), `policy_sort.py`
reports every brick to it: decision, bowl, cycle time, score of the camera that decided, the inspected image
and its heatmap, the current phase and the duration of every anomaly check. Without the hub the loop runs
the same way.

## Watching a Studio run: `anomaly_watch.py`

```bash
station/studio-python sorter/anomaly_watch.py
```

The Studio web UI keeps control of the arm. The script only looks: for every brick the gripper picks up it
scores the wrist camera image and reports verdict, score, image and cycle time to the dashboard.

## Earlier variant: `sort_bricks.py`

Here the policy only picks. The script lifts the brick, inspects it, and places it itself. The top camera
tracks both bowls, so they may be moved during a run.

```bash
station/studio-python sorter/sort_bricks.py --test   # show the detection only, nothing moves (Studio session open)
station/studio-python sorter/calibrate.py            # teach poses and drop points once (Studio session closed)
station/studio-python sorter/sort_bricks.py          # sort
```

Keys while sorting: `SPACE` pause / continue, `q` stop (a held brick goes back into the box), `ESC` emergency hold.

`calibrate.py` keys (the follower is moved with the leader arm):

| Key | Teaches |
|---|---|
| `t` | Transit pose: high above everything, gripper closed and empty. Also stores the empty-gripper reference image |
| `k` | In the transit pose with a brick in the gripper: holding reference |
| `i` | Inspect pose: brick directly in front of the webcam |
| `h` | Home pose (optional, default is the start pose of the training episodes) |
| `y` / `b` | Centre of the yellow / blue bowl (arm out of the image) |
| `SPACE` | Afterwards: move the gripper over the bowl centre and store the drop point |
| `u` | Delete the last drop point |
| `r` | Webcam region in which the held brick appears in the inspect pose |
| `m` | Webcam region of the box |
| `q` | Quit (everything is saved immediately) |

Teach six to nine drop points with the bowl in a different place each time. Outside the covered area the
script does not place, it waits.

## Safety

- All script-driven motion stays inside the joint range of the training episodes and the taught poses.
- If the arm does not reach a target (blocked), it stops and the script pauses.
- "Gripper empty" always leads to opening above the box, so a wrongly judged brick falls back into the box.
- If the connection drops, the follower holds its position.
- Never start a run without a person at the arm.

## Settings

`sorter/config.json` overrides the defaults in `common.py` (`DEFAULT_CONFIG`):

| Section | Content |
|---|---|
| `policy_sort` | Task texts, which bowl is which, when a brick counts as lifted, time limits |
| `anomaly` | Model checkpoints and thresholds per camera, number of frames per check, whether checked images are saved |
| `inspect` | Wrist angles for the second look, settle time, frames per angle |
| `speed`, `gripper`, `pick`, `place` | Motion speeds, gripper values, pick and place limits |
| `dashboard` | Hub address, or switch reporting off |
| `vision` | Overrides for the stroke detector (`vision.py`, `DEFAULTS`) |

The stroke detector was tuned on 9 real markings and 129 images without a marking.

Operator messages of these scripts are in German.
