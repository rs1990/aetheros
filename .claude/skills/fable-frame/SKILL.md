---
name: fable-frame
description: Phase 1 of the Fable workflow — understand the problem before touching anything. Produces a one-sentence problem statement, a testable definition of done, and a ground-truth map of the relevant code. Use at the start of every non-trivial task, or when a task feels ambiguous, or when work has gone sideways and needs re-grounding.
---

# Fable Frame — Understand Before Acting

Most failed tasks are not failed implementations; they are correct
implementations of the wrong problem. Framing is cheap insurance.

## Method

### 1. Restate the problem in one sentence
Write it down. If you cannot state it in one sentence, you do not understand
it yet. The sentence names the *user-visible outcome*, not the mechanism:
- Bad: "Refactor the auth middleware."
- Good: "Logged-in users are randomly logged out; sessions must survive
  until their real expiry."

### 2. Find ground truth in the code, not in descriptions
The request, the README, the comments, and the ticket are all *claims*.
The code and its observed behavior are *facts*. Before planning:
- Locate the entry point of the affected flow and read it top-down.
- Trace the actual data path: where does the input come from, what
  transforms it, where does it land?
- Run the thing if possible. Observed behavior beats any description.
- Note every place where reality contradicts the request's assumptions —
  these contradictions are usually where the real problem hides.

### 3. Define done as a test you could run
"Done" must be falsifiable. Write the exact check:
- "GET /api/orders returns 200 with the user's orders in <300ms"
- "npm test passes; the new test fails on main and passes on the branch"
- "The map renders pins within 2s on a cold start"
If you cannot phrase done as an observation, the task is underspecified —
surface that now, not after implementing.

### 4. Bound the blast radius
List the files/modules the change should touch. Anything outside that list
that "needs" changing later is a signal the frame was wrong — stop and
re-frame rather than silently expanding scope.

### 5. Identify constraints and invariants
- What must NOT change (public APIs, DB schemas, wire formats, behavior
  other code depends on)?
- What conventions does this codebase already use (error handling style,
  naming, test framework, directory layout)? The change must look native.

## Output contract

Framing produces exactly this, written down (in `tasks/todo.md` or the
conversation) before any code changes:

```
PROBLEM: <one sentence, user-visible outcome>
DONE WHEN: <falsifiable check(s)>
GROUND TRUTH: <files read, behavior observed, contradictions found>
BLAST RADIUS: <files/modules expected to change>
INVARIANTS: <what must not break>
```

## Red flags that mean "keep framing"

- You are about to search for a solution before reading the existing code.
- The request contains "just" or "simply" but you haven't verified it's simple.
- Two plausible interpretations of the request lead to different code.
  (Pick the one supported by the code's evidence; state the choice.)
- You cannot name where the current behavior comes from.
