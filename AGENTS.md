# AGENTS.md

Instructions for every AI coding agent working in this repo (Claude Code, Codex, ...).
Read this file and `TASKS.md` before you start. Keep this file short and accurate.

## Project

- **Goal:** `<TBD - one paragraph once the hackathon theme is known>`
- **Stack:** `<TBD>`
- **Deadline:** 48 h hackathon. Feature freeze at hour 32 (see "Hackathon rules").

## Ownership

One owner per directory. Work only in your own area.

| Area | Path | Owner | Tool |
|---|---|---|---|
| Backend / core logic | `backend/` | Julian | Claude |
| Frontend / UI | `frontend/` | Tobby | Claude |
| Infra, tests, glue, deploy | `infra/`, `tests/` | Samuel | Codex |
| Shared contract | `docs/CONTRACT.md` | whole team | any |

Adjust the table once areas are decided. Do not edit files outside your area.
If you need a change elsewhere, add a request under "Requests to other areas" in `TASKS.md`
instead of making it yourself.

## Branches and commits

- One long-lived branch per person: `julian/<topic>`, `tobby/<topic>`, `samuel/<topic>`. Never commit directly to `main`.
- Commit small and often. Message format: `<area>: <what changed>`.
- Before merging: `git pull --rebase origin main`, run the checks below, then open a PR or merge.
- `main` must always run. If you break it, fix it first.
- Never force-push. Never rewrite shared history.
- Merge into `main` at least once per hour so integration problems show up early.

## The contract is the truth

`docs/CONTRACT.md` defines the data model and every interface between areas.
- Implement against it. Do not guess field names or formats.
- If it must change: tell the team, update the contract and its change log in the same commit, and note it in `TASKS.md`.

## How to work

1. Read `TASKS.md`. Take a task from your area (or the one assigned to you) and move it to "In progress".
2. Build the simplest end-to-end path first. Polish later.
3. Do not add new dependencies without asking the team.
4. When done: tick the task, move it to "Done", and note anything others need to know.
5. Never commit secrets. Use `.env` (ignored) and keep `.env.example` current.

## Subagents

Delegate to a subagent (if your tool supports them) instead of doing everything in your own context:

- **Tests:** after a feature works, have a subagent write tests for it.
- **Review:** before every commit, have a subagent review your diff for bugs, leftover debug code, secrets and edits outside your area.
- **Research:** when you must read many files or docs to answer one question, have a subagent do the reading and report back briefly.
- **Parallel work:** independent tasks inside your own area (e.g. several unrelated files) can run in parallel subagents.

Rules:
- Give the subagent a complete task description: goal, relevant files, constraints, how you will judge the result. It does not see this conversation.
- Subagents work only inside your own area (see Ownership) and follow every rule in this file.
- You stay responsible: read the result, check the diff, and only then commit.
- Do not use a subagent for small tasks you can finish faster yourself.

Predefined subagents (Claude Code, in `.claude/agents/`):
- `tester` - writes and runs tests. Uses a faster, cheaper model (`sonnet`).
- `reviewer` - read-only review of the uncommitted diff. Uses the same model as your session (`inherit`).

Model choice: use a cheaper/faster model for tests, research and file reading; use your strongest model for review, architecture and decisions. Do not delegate decisions to subagents. Codex users: `.claude/agents/` is Claude Code only; follow the same rules with your own tool's mechanism.

## Definition of done

- Runs locally with the commands below.
- Has at least a smoke test or a manual check noted in the task.
- Committed, rebased on `main`, `TASKS.md` updated.

## Commands

Fill in when the stack is chosen.

```
setup:   <TBD>
run:     <TBD>
test:    <TBD>
lint:    <TBD>
deploy:  <TBD>
```

## Hackathon rules

- Hour 0-2: pick one idea, write the contract, split ownership.
- Hour 2-8: thin end-to-end slice, deployed.
- Hour 8-32: features, merge hourly.
- Hour 32: feature freeze. Only bug fixes, polish, demo.
- Hour 40+: pitch and backup demo video only.
- A working demo beats an ambitious one. When in doubt, cut scope.
