---
name: fable-evaluate
description: Phase 5 of the Fable workflow — review the finished diff as a hostile senior engineer before shipping. Produces severity-ranked findings, each with a concrete failure scenario, verified against the actual code. Use after a feature/fix is complete, before commit or PR, or when asked to review any code.
---

# Fable Evaluate — Adversarial Self-Review

You are no longer the author. You are the reviewer whose job is to find
the bug that takes production down at 3am. Praise is worthless; only
defects and material simplifications count.

## Method

### 1. Read the diff cold, entry-point first
Re-read the full diff as if seeing it for the first time, starting from
where execution enters (route, handler, main), not file-alphabetically.
For each hunk ask: what inputs reach this line, and what happens for the
inputs the author wasn't thinking about?

### 2. Hunt in the high-yield spots
Bugs cluster. Check these deliberately, in order:

- **Boundaries**: off-by-one, `<` vs `<=`, first/last iteration, empty
  collections, zero, negative, exact-limit values.
- **Nullability**: every value that crosses a boundary (JSON parse, DB row,
  env var, optional param) — what happens when it's absent?
- **Async & ordering**: unawaited promises, race between check and use,
  concurrent writers, retries that double-apply effects.
- **Error paths**: swallowed exceptions, catch blocks that log and limp on
  with corrupt state, cleanup that doesn't run on the throw path.
- **State & lifecycle**: stale caches, listeners never removed, connections
  never closed, singletons holding request data.
- **Contract drift**: does the change silently alter a shape/status
  code/unit that some other caller depends on? Grep for the callers.
- **Security**: unvalidated external input, injection (SQL/shell/path),
  secrets in code or logs, authz checks missing on new endpoints.
- **Resource behavior at scale**: N+1 queries, unbounded lists, missing
  pagination/timeouts — will this line still work with 10,000 rows?

### 3. Verify every finding before reporting it
For each suspected defect, trace the actual failing path in the code (or
run it). Label honestly:
- **CONFIRMED** — traced the exact inputs/state to the wrong outcome.
- **PLAUSIBLE** — strong pattern match, couldn't fully trace.
A finding you can't articulate a failure scenario for is not a finding —
drop it. False positives destroy the review's credibility.

### 4. Also flag material simplifications
Duplicated logic that existing helpers already cover, dead branches,
abstractions with one caller, complexity the task never needed. Only when
the simplification is clearly better — not stylistic preference.

## Output contract

Findings ranked most-severe first:

```
1. [CRIT|HIGH|MED|LOW] file.js:42 — <one-sentence defect>
   Failure: <concrete inputs/state → wrong outcome/crash>
   Verdict: CONFIRMED | PLAUSIBLE
   Fix: <one-line direction>
```

Zero findings is a legitimate result — say so plainly rather than
inventing nits to look thorough. Never pad with style comments, praise,
or "consider maybe possibly" hedges.

## Severity calibration

- **CRIT**: data loss, security hole, crash on common path, money wrong.
- **HIGH**: wrong behavior on realistic inputs, silent corruption.
- **MED**: wrong behavior on edge inputs, resource leak, misleading errors.
- **LOW**: real but unlikely; simplifications; test gaps worth closing.

Findings loop back to `fable-implement` for fixes, then `fable-verify`
re-proves — the evaluator never edits code themselves in delegated mode.
