# Interface contract

Single source of truth for everything shared between areas. Implement against this, do not guess.
Change it only with team agreement, and add a line to the change log in the same commit.

## Conventions

- IDs: `<string/uuid - TBD>`
- Timestamps: ISO 8601, UTC (`2026-11-13T14:30:00Z`)
- Errors: `{ "error": { "code": "<machine_code>", "message": "<human text>" } }` with a matching HTTP status
- Auth: `<TBD>`
- Base URL (local): `<TBD>`

## Data model

### `<Entity>`

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | |
| `<field>` | `<type>` | | |

Example:

```json
{ "id": "example-1" }
```

## Endpoints / functions

### `<METHOD> <path>`

- **Purpose:** `<one line>`
- **Request:** `<fields or example JSON>`
- **Response (200):** `<fields or example JSON>`
- **Errors:** `<codes and when they occur>`

Example:

```json
// request
{}
// response
{}
```

## Change log

| Date | Who | Change |
|---|---|---|
| | | Initial version |
