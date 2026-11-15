# TASKS

Shared board. Update it as you work. One line per task:
`- [ ] (julian/tobby/samuel/all) Title - done when: <acceptance criterion>`

## Backlog

- [ ] (all) Pick the idea and agree on scope - done when: one-paragraph goal is in `AGENTS.md`
- [ ] (all) Write the first version of `docs/CONTRACT.md` - done when: data model and main endpoints are listed
- [ ] (samuel) Set up repo checks and deploy pipeline - done when: a "hello world" is live at a public URL
- [ ] (julian) Backend skeleton - done when: one endpoint from the contract returns example data

## In progress

## Review

## Done

- [x] (tobby) SO-101 station set up in Physical AI Studio and shareable via `robot/` - done when: arms calibrated and matched 1:1, 3 cameras in env `SO101 Teleop`, export/import scripts + `robot/README.md` committed
- [x] (tobby) Tool to grow the dataset from the policy's own successful cycles (`robot/scripts/record_successes_gui.py`) - done when: runs a model through Studio, saves judged-correct cycles as episodes, guide in `robot/README.md`; manual check against a fake Studio server only, real arm still open
- [x] (tobby) Live dashboard for the inspection cell (`frontend/`) - done when: hub + dashboard run with `python frontend/hub.py`, show simulator data end to end, tests pass (`python -m unittest discover -s frontend/tests`).
- [x] (tobby) Real data path for the dashboard on the station laptop - done when: recording tools report every judged cycle and obey the dashboard's Stop (`robot/scripts/dashboard_report.py`), `frontend/benchmark_openvino.py` fills the device table, `frontend/start_laptop.sh` starts hub + telemetry, steps in `frontend/README.md`. Checked against a stand-in Studio server on the Mac only; arm, iGPU / NPU and laptop sensors still open.

## Requests to other areas

Format: `- [ ] (from X to Y) What you need and why`

- [ ] (from tobby to all) Confirm the dashboard part of `docs/CONTRACT.md` (events, state, control). It is a proposal until someone objects or agrees.
- [ ] (from tobby to julian) The dashboard gets real cycles from the recording tools, but no anomaly score, heatmap or model decision yet. Whatever runs the Anomalib model in the loop should send them with `frontend/hub_client.py` (`hub.cycle(anomaly_score=..., threshold=..., decision=..., heatmap=...)`, `hub.latency("anomaly", ms, device=...)`). Example in `frontend/README.md`.
- [ ] (from tobby to samuel) Run `sudo python frontend/telemetry_linux.py --once` on the station laptop and report which values are missing; it is untested on real hardware.

## Blockers and decisions

Format: `- YYYY-MM-DD HH:MM - decision or blocker - who decides`

- 2026-11-14 01:30 - new area `robot/` (owner Tobby) for Studio data; do NOT re-run the Studio calibration wizard on the arms, it undoes the leader-follower matching (see `robot/README.md`) - Tobby
