# HackJunction2026

48-hour hackathon project. `<TBD - one-line description once the theme is known>`

## Team workflow

Three people, three areas, one shared contract. Rules for humans and agents are in [AGENTS.md](AGENTS.md).

| File | Purpose |
|---|---|
| `AGENTS.md` | Rules for all AI agents (ownership, branches, definition of done) |
| `CLAUDE.md` | Points Claude Code to `AGENTS.md` |
| `TASKS.md` | Shared task board, requests between areas, decisions |
| `docs/CONTRACT.md` | Data model and interfaces between areas |

## Setup

1. Clone the repo and create your branch: `git checkout -b <julian|tobby|samuel>/<topic>`
2. Copy `.env.example` to `.env` and fill in your keys.
3. Run the setup/run commands listed in `AGENTS.md` (filled in once the stack is chosen).

## Timeline

| Hours | Goal |
|---|---|
| 0-2 | Pick one idea, write the contract, split ownership |
| 2-8 | Thin end-to-end slice, deployed |
| 8-32 | Features, merge into `main` hourly |
| 32-40 | Feature freeze, polish, bug fixes, demo flow |
| 40-48 | Pitch, backup demo video, rehearsal |
