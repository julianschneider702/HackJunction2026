# TASKS

Shared board. Update it as you work. One line per task:
`- [ ] (julian/tobby/samuel/all) Title - done when: <acceptance criterion>`

## Backlog

- [ ] (all) Pick the idea and agree on scope - done when: one-paragraph goal is in `AGENTS.md`
- [ ] (all) Write the first version of `docs/CONTRACT.md` - done when: data model and main endpoints are listed
- [ ] (samuel) Set up repo checks and deploy pipeline - done when: a "hello world" is live at a public URL
- [ ] (julian) Backend skeleton - done when: one endpoint from the contract returns example data
- [ ] (tobby) Frontend skeleton - done when: one screen shows data from the backend

## In progress

## Review

## Done

- [x] (tobby) SO-101 station set up in Physical AI Studio and shareable via `robot/` - done when: arms calibrated and matched 1:1, 3 cameras in env `SO101 Teleop`, export/import scripts + `robot/README.md` committed

## Requests to other areas

Format: `- [ ] (from X to Y) What you need and why`

## Blockers and decisions

Format: `- YYYY-MM-DD HH:MM - decision or blocker - who decides`

- 2026-11-14 01:30 - new area `robot/` (owner Tobby) for Studio data; do NOT re-run the Studio calibration wizard on the arms, it undoes the leader-follower matching (see `robot/README.md`) - Tobby
