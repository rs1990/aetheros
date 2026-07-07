---
name: fable-debug
description: Phase 6 of the Fable workflow — the disciplined debugging loop: reproduce, isolate, hypothesize, fix at root cause, prove with a regression test. Use whenever observed behavior diverges from expected — failing test, crash, wrong output, flaky behavior — instead of tweaking code until symptoms disappear.
---

# Fable Debug — Reproduce, Isolate, Fix, Prove

Debugging is hypothesis-driven search, not code roulette. Every action in
the loop either shrinks the search space or is wasted.

## The loop

```
REPRODUCE → ISOLATE → HYPOTHESIZE → TEST HYPOTHESIS → FIX ROOT CAUSE → PROVE
```

### 1. REPRODUCE — make it fail on demand
No reliable reproduction, no debugging — you'd be shooting in the dark.
- Capture the exact failing command/input/state and the exact error text.
- Shrink the reproduction: smallest input, fewest steps, fastest cycle.
  A 2-second repro is worth an hour of setup; you'll run it 30 times.
- Flaky? Loop it (`for i in $(seq 50); do ...`) to get a failure rate —
  intermittent failures are usually ordering/timing/shared state, and the
  rate tells you when you've actually fixed it (0/200, not 'passed once').

### 2. ISOLATE — bisect the search space
Halve the problem repeatedly instead of reading everything:
- **Along the data path**: log/inspect the value at the midpoint of the
  flow. Correct there? Bug is downstream. Wrong? Upstream. Repeat.
- **Along history**: worked before? `git bisect` — mechanical and fast.
- **Along configuration**: fails in env A, works in env B? Diff the envs
  (versions, env vars, data) before diffing the code.
- Read the error message. Actually read it — the answer is in the text or
  the first stack frame that's YOUR code surprisingly often.

### 3. HYPOTHESIZE — state the cause before touching code
Write the causal chain: "Output is X because f() receives Y when Z."
The hypothesis must predict something checkable you haven't looked at yet.
If you can't state a hypothesis, you need more isolation, not more edits.

### 4. TEST THE HYPOTHESIS — one variable at a time
Make the single observation that confirms or kills it (a log line, a
debugger stop, a hardcoded input). Wrong? Good — that branch of the search
space is dead; pick the next hypothesis. NEVER stack multiple speculative
changes; after each dead hypothesis, revert to the clean failing state.

### 5. FIX THE ROOT CAUSE
The fix goes where the causal chain starts, not where the stack trace
ends. Before applying, explain why the bug ever worked / how it got in —
if you can't, you don't understand it yet. Then check for siblings: the
same author/pattern likely planted the same bug nearby (`grep` for the
pattern).

### 6. PROVE — regression test + full suite
- Add the test that encodes the reproduction: fails on the old code,
  passes on the fix. Watch it fail first if there's any doubt.
- Run the full suite. A fix that breaks two other tests is a new bug.
- For flaky bugs: rerun the loop (0 failures / 200 runs).

## Escalation rule

Three failed hypotheses on the same symptom → stop. The bug is not where
you think it is. Zoom out: re-read the whole flow from the entry point,
question the assumption you're most certain of (that's usually the wrong
one), check the "impossible" things — stale build, wrong file, wrong
process, cached dependency, case-sensitive path, PATH shadowing.

## Anti-patterns

- Changing code before reproducing the failure.
- "Fixing" by adding a null check / try-catch / retry without explaining
  why the value was null / the call threw / the race exists.
- Deleting the failing test, loosening the assertion, or marking it skip.
- Debugging by diff-staring when you could bisect in the runtime.
- Declaring victory on one green run of a flaky test.
