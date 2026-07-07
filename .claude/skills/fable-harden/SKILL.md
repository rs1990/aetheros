---
name: fable-harden
description: Phase 7 of the Fable workflow — take working code to production-grade. Systematic audit of error handling, input validation, security, configuration, observability, and operational behavior. Use before declaring anything production-ready, before deploys, or when asked to make a repo "robust" or "production level".
---

# Fable Harden — Production Readiness

Working-on-my-machine and production-ready are separated by everything
that happens when reality misbehaves. Hardening audits the misbehavior
paths systematically.

## The audit

Walk each category against the actual code. For each gap: fix it if in
scope, or document it as a known exception — never silently skip.

### 1. Failure behavior
- Every external call (DB, HTTP, filesystem, queue) has a timeout and a
  defined failure behavior. What does the user see when the DB is down?
  (It should be an honest error, not a hang or a stack trace.)
- Partial failures don't corrupt state: multi-step writes are
  transactional or idempotent-with-retry, never "hope both succeed".
- The process handles the fatal cases deliberately: unhandled rejection /
  uncaught exception logs context and exits nonzero (supervisor restarts
  it) rather than limping on corrupt.
- Startup validates its dependencies (config present, DB reachable,
  migrations applied) and fails fast with a message naming the problem.

### 2. Input validation (server-side, at the boundary)
- Every externally reachable input — body, query, params, headers, file
  uploads, webhook payloads — validated for type, range, and size before
  use. Client-side validation counts for nothing.
- Parameterized queries only; no string-built SQL/shell/paths from input.
- Limits on everything unbounded: request body size, list lengths,
  pagination on every collection endpoint, upload size, rate limits on
  expensive/auth endpoints.

### 3. Security
- No secrets in code, logs, error responses, or git history. Env/config
  only; `.env` gitignored; rotate anything ever committed.
- AuthN and authZ on every non-public endpoint — check authZ especially
  on the endpoints added last (the classic gap: authenticated user A
  reading user B's data by changing an ID).
- Dependencies: `npm audit`/`pip-audit`/`cargo audit` clean of criticals.
- CORS locked to known origins in production; security headers on;
  cookies HttpOnly/Secure/SameSite as appropriate.
- Errors to clients are generic; details go to logs.

### 4. Configuration & environments
- One config module reads all env vars, validates them at boot, and
  documents defaults. No `process.env.X` scattered mid-codebase.
- Dev/prod differences are explicit config, not `if (NODE_ENV)` sprinkled
  logic. `.env.example` lists every required var, valueless.
- Builds are reproducible: lockfiles committed, versions pinned.

### 5. Observability
- Structured logs on every request path: what happened, for whom, how
  long, correlation/request ID. Errors logged with stack + context once
  (not at every layer).
- A `/health` (or equivalent) endpoint that checks real dependencies.
- Enough logging that the 3am incident is debuggable from logs alone —
  simulate it: "requests are failing"; can you tell why from the logs?

### 6. Data & operations
- Migrations are ordered, committed, and runnable from scratch; a fresh
  clone + documented steps produce a working system.
- Destructive operations (delete, overwrite, drop) are confirmed, soft,
  or reversible; backups exist for anything that matters.
- Graceful shutdown: drain in-flight requests, close connections on
  SIGTERM — required for zero-downtime deploys.

### 7. Performance sanity
Not optimization — absence of landmines:
- No N+1 query patterns on list endpoints; indexes on columns you filter/
  join by; no unbounded in-memory accumulation of rows/files.
- The hot path tested once with realistic data volume (1k–10k rows), not
  the 3-row seed data.

## Output contract

A hardening pass produces a written report:

```
HARDENED: <fixes applied, with files>
KNOWN EXCEPTIONS: <gap — why it's acceptable / when it must be fixed>
VERIFIED: <evidence — the failure drills actually run>
```

Drill at least the top two failure modes for real (kill the DB, send the
malformed payload) — a hardening claim without a drill is theater.
