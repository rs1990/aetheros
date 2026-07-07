---
name: fable-implement
description: Phase 3 of the Fable workflow — write the smallest change that fully solves the current checklist item, matching the codebase's existing idiom. Use when executing a planned step, applying a reviewed fix, or making any code change. Enforces root-cause fixes, minimal diffs, and native-looking code.
---

# Fable Implement — Build

Implementation is the mechanical phase. If it doesn't feel mechanical,
the frame or the plan is wrong — go back, don't improvise forward.

## Method

### 1. Re-read before you write
Open every file you're about to change and read the surrounding code first.
You are looking for:
- The idiom: how does this codebase handle errors, name things, structure
  modules, write tests? Your change must be indistinguishable from code
  the original author would write.
- Existing helpers: the function you're about to write probably half-exists.
  Search before creating (`grep` for the domain noun, check utils/lib dirs).
- The real insertion point: the first place you think of is often a symptom
  site, not the cause site.

### 2. Smallest complete change
- Solve the whole item — no TODOs, no stubs, no "phase 2 will handle it".
- But ONLY the item. No drive-by refactors, no reformatting untouched lines,
  no renaming things that work, no speculative parameters or options
  "for later". Every extra line is review surface and regression risk.
- If you notice an unrelated bug: note it in `tasks/todo.md`, don't fix it
  inline.

### 3. Root cause, not symptom
Before writing a fix, state (to yourself, explicitly) the causal chain:
"X happens because Y does Z when W." If your fix doesn't break that chain
at its origin, you're patching a symptom. Symptom patches include:
- Adding a null check where the real bug is the thing being null.
- Catching and swallowing an exception whose cause is upstream.
- Retry loops around nondeterministic failures you haven't explained.
- Special-casing one input instead of fixing the general handling.

### 4. Code quality bar
- Functions do one thing; names say what they do; no clever one-liners
  where a boring three-liner is clearer.
- Comments only for constraints the code can't express (why a lock is held,
  why an order matters, why the obvious approach fails). Never comments
  that narrate the code or justify the change.
- Validate at boundaries: external input (HTTP, files, env, user) is
  validated where it enters; internal code trusts internal callers.
- Handle the failure paths the code can actually hit — every awaited call,
  every I/O, every parse. Fail loudly with context, never silently.
- No secrets in code, ever. Env vars via config module, `.env` gitignored.

### 5. Stop conditions
STOP implementing and return to planning when:
- The change wants to touch files outside the frame's blast radius.
- You've written a workaround and called it temporary.
- You're on the third attempt at the same item with a different approach —
  the item is mis-framed; re-frame before attempt four.
- You realize mid-edit you don't know why the current code is the way it
  is. (Chesterton's Fence: understand it before replacing it.)

## Output contract

Each completed item produces:
- The diff (minimal, idiomatic, complete).
- Updated checklist: item marked done WITH its verification evidence.
- Any discovered follow-ups appended to `tasks/todo.md` as new items.

Then immediately enter VERIFY (`fable-verify`) — implementation is never
the last step for an item.
