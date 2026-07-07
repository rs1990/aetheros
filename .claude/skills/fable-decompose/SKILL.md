---
name: fable-decompose
description: Phase 2 of the Fable workflow — break a framed problem into a dependency-ordered, independently verifiable checklist. Use for any task needing 3+ steps or touching 3+ files, before writing code. Produces tasks/todo.md entries sized so each item can be implemented and verified in one focused pass.
---

# Fable Decompose — Task Breakdown

A good decomposition makes the rest of the work mechanical. A bad one makes
every step a renegotiation. Decompose after framing, never before.

## Method

### 1. Slice vertically, not horizontally
Each item should deliver a verifiable slice of behavior, not a layer:
- Bad: "1. Write all models. 2. Write all endpoints. 3. Write all tests."
- Good: "1. GET /landmarks returns seeded rows (model + route + test).
  2. Radius filter works (query + test). 3. ..."
Vertical slices mean every checkpoint is a working system, so failures
localize to the last small step.

### 2. Order by dependency, then by risk
- Hard prerequisites first (schema before queries, API before UI).
- Among independent items, do the riskiest/most-unknown first — if an
  approach is going to fail, fail in step 2, not step 9.
- Push pure polish (naming, docs, formatting) to the end.

### 3. Size each item for one focused pass
An item is right-sized when you can implement AND verify it without
context-switching — typically one behavior, 1–3 files, one clear check.
If an item's description needs "and", split it.

### 4. Attach a verification to every item
Each checklist line carries its own "prove it" clause:
```
- [ ] Radius filter on GET /landmarks — verify: curl with lat/lng/radius
      returns only rows within radius; test added in landmarks.test.js
```
An item without a verification is a wish, not a task.

### 5. Mark decision points explicitly
If a step depends on a choice not yet made (library, schema shape), list
the decision as its own item BEFORE the dependent steps, with the options
and a recommendation. Decisions made implicitly mid-implementation are how
scope drifts.

## Output contract

Write the plan to `tasks/todo.md`:

```
# <Task name> (<date>)
PROBLEM: <from framing>
DONE WHEN: <from framing>

- [ ] <step 1> — verify: <check>
- [ ] <step 2> — verify: <check>
...
```

Check items off as they complete. If reality diverges from the plan
(a step balloons, an assumption breaks), STOP — update the plan first,
then continue. The plan must always reflect current intent; a stale plan
is worse than none.

## Sizing heuristics

- Whole task fits in one item → skip decomposition, just implement.
- More than ~10 items → the frame is probably too big; split into phases
  and get the first phase shipped before detailing the rest.
- Any item you cannot attach a verification to → you don't understand that
  step yet; do the reading before writing the plan.

## Delegation notes

When distributing items to subagents (fable-implementer), each delegation
gets exactly one item plus the frame's PROBLEM/DONE WHEN/INVARIANTS block.
Never hand a subagent the whole plan — parallel agents editing overlapping
files produce merge chaos. Items sharing files run sequentially.
