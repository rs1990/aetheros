---
name: fable-verify
description: Phase 4 of the Fable workflow — prove a change works by exercising real behavior and capturing evidence. Use after every implementation step, before marking any task complete, and before any commit. "Looks right" and "compiles" are not verification; observed behavior is.
---

# Fable Verify — Prove It Works

A claim of "done" without evidence is a guess. Verification converts the
guess into a fact, or into a bug report — both are progress.

## The evidence hierarchy

Strongest to weakest. Always collect the strongest level the change allows:

1. **End-to-end observation** — drive the real flow the user will use:
   hit the endpoint with curl, load the page, run the CLI with real args,
   play the audio. Capture the actual output.
2. **Automated test** — a test that fails without the change and passes
   with it. Run the FULL suite too, not just the new test (regressions
   hide in the tests you didn't think were related).
3. **Targeted execution** — run the changed function/module directly
   (REPL, script, one-off harness) with representative + edge inputs.
4. **Static checks** — typechecker, linter, build. Necessary, never
   sufficient. Level 4 alone proves nothing about behavior.

"I read the code and it's correct" is level 0 and does not count.

## Method

### 1. Verify against DONE WHEN, not against the diff
The check comes from the frame's definition of done. Verifying "my change
does what my change does" is circular; verify "the problem is gone".

### 2. Test the edges, not just the happy path
For each change, exercise at minimum:
- The nominal case.
- The empty/zero/null case (no rows, empty string, missing field).
- The boundary (limit exactly reached, first/last element, expiry moment).
- The failure path you added handling for (kill the dependency, send
  malformed input) — confirm the handling actually runs.

### 3. Verify the negative space
- What worked before must still work: run the existing suite, click
  through the adjacent flows.
- If you fixed "X is wrong", also check X's siblings — the same bug
  pattern usually has copies.

### 4. Capture evidence, verbatim
Paste the actual observation into the checklist/report:
```
- [x] Radius filter — verify: `curl ':3000/landmarks?lat=41.8&lng=-87.6&radius=500'`
      → 200, 3 rows, all within 500m (was: 14 rows unfiltered). Suite: 42 passed.
```
Trimmed real output, not a paraphrase. If verification fails, paste the
failure exactly — that's the input to the debug phase.

## Failure protocol

Verification fails → do NOT tweak-and-rerun blindly. Enter the debug loop
(`fable-debug`): the failing check is your reproduction, which is the hard
part already done. Never weaken the check to make it pass; never mark the
item done "pending a fix".

## Honesty rules

- Report outcomes exactly: "3 tests fail" not "mostly passing".
- A skipped verification is stated as skipped, with why.
- If you couldn't run something (no env, no creds), say so and list what
  WAS verified — never let the summary imply more than the evidence shows.
- The task is complete when the evidence says so, not when the code is
  written. Would a staff engineer sign off on this evidence? If not, it
  isn't done.
