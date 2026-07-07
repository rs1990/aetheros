---
name: fable-workflow
description: Master orchestrator for the Fable engineering workflow. Runs the full loop — frame, decompose, implement, verify, evaluate, debug, harden — to take any task from request to production-grade deliverable. Use when asked to "fix this repo", "make this production ready", "do this the Fable way", or for any non-trivial feature, bug, or refactor. Delegates phases to fable-* agents with model routing (opus for judgment, sonnet for execution).
---

# Fable Workflow

The workflow that turns a request into a robust, verified, production-level
deliverable. It is a loop, not a pipeline: evaluation and verification feed
back into implementation until the exit criteria hold.

## The loop

```
FRAME → DECOMPOSE → [ IMPLEMENT → VERIFY → EVALUATE ]* → DEBUG (as needed) → HARDEN → SHIP
```

Each phase has its own skill with the full method. This file tells you when
to enter each phase, when to loop back, and when you are done.

## Phase entry rules

1. **FRAME** (`fable-frame`) — always first, even for "obvious" tasks.
   Skipping framing is the #1 cause of confidently solving the wrong problem.
   Exit only with: a one-sentence problem statement, a testable definition of
   done, and the list of files that constitute ground truth.

2. **DECOMPOSE** (`fable-decompose`) — enter when the task needs 3+ steps or
   touches 3+ files. Exit with a dependency-ordered checklist in
   `tasks/todo.md` where every item is independently verifiable.
   For smaller tasks, skip straight to IMPLEMENT.

3. **IMPLEMENT** (`fable-implement`) — one checklist item at a time.
   Smallest change that fully solves the item. No speculative features.

4. **VERIFY** (`fable-verify`) — after every implement step, not at the end.
   Proof means observed behavior (test output, HTTP response, rendered page),
   never "the code looks right".

5. **EVALUATE** (`fable-evaluate`) — after the checklist is done, review the
   whole diff as a hostile senior reviewer. Findings loop back to IMPLEMENT.

6. **DEBUG** (`fable-debug`) — enter whenever observed behavior diverges from
   expected: failing test, wrong output, crash. Never patch a symptom from
   inside IMPLEMENT; switch to the debug loop, find root cause, then return.

7. **HARDEN** (`fable-harden`) — before declaring anything production-ready.
   Error handling, input validation, secrets, config, observability.

## Loop-back rules

- VERIFY fails → DEBUG (find root cause), then IMPLEMENT the real fix.
- EVALUATE finds a defect → IMPLEMENT the fix, then VERIFY again.
- Any phase reveals the frame was wrong → STOP, return to FRAME, re-plan.
  Do not push through a broken plan; sunk cost is not a reason to continue.
- Three consecutive failed fix attempts on the same symptom → stop patching,
  re-read the code from the entry point down, question every assumption.

## Delegation and model routing

When running with subagents, route by cognitive load:

| Phase | Agent | Model | Why |
|-------|-------|-------|-----|
| FRAME / DECOMPOSE | fable-architect | opus | judgment, trade-offs |
| IMPLEMENT | fable-implementer | sonnet | fast, well-specified execution |
| VERIFY | fable-verifier | sonnet | mechanical, evidence-driven |
| EVALUATE | fable-evaluator | opus | adversarial reasoning |
| DEBUG | fable-debugger | sonnet | disciplined loop, escalate to opus if stuck 3x |
| HARDEN | fable-hardener | opus | threat modeling, judgment |

Delegation contract: every subagent prompt must contain (a) the one-sentence
problem statement from FRAME, (b) the specific checklist item, (c) the
definition of done for that item, (d) instruction to report evidence, not
claims. A subagent that returns "done" without evidence has not finished.

Keep the main context clean: subagents do the reading and exploring; the main
thread holds the plan and the decisions.

## Exit criteria (definition of shipped)

- Every `tasks/todo.md` item checked, each with evidence.
- Full test suite green (run it, paste the summary).
- The primary user flow exercised end-to-end and observed working.
- EVALUATE pass produced zero unresolved CONFIRMED findings.
- HARDEN checklist complete or exceptions explicitly documented.
- Diff reviewed once more for scope creep: nothing changed that the task
  did not require.

## Anti-patterns (never do these)

- Declaring done because the code compiles.
- Fixing a symptom where the stack trace points instead of where the bug lives.
- Widening scope mid-task ("while I'm here...") without re-planning.
- Writing the summary before running the verification.
- Trusting documentation or comments over the code itself.
