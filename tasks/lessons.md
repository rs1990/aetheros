# Lessons

## 2026-06-10 — Parallel subagents vs session token limits
Pattern: launching 10 heavy review agents in parallel exhausted the 5-hour session
token budget mid-run; 8 of 10 died with no output because they held findings in
memory until the end.
Rules:
1. Stagger heavy agents in waves of 2-3, not 10.
2. Always instruct long-running agents to write their output file INCREMENTALLY
   (skeleton first, refine sections as they go) so partial progress survives limits.
3. After a limit reset, check which artifacts exist on disk before re-running anything.

## 2026-06-10 — Misconfigured PostToolUse hook (FIXED 2026-06-11)
Old config: agent-type PostToolUse hook for deploy monitoring. Failure modes:
(1) the evaluation agent spawned on EVERY Bash call (cost + noise) even though
`if: "Bash(git push*)"` was set; (2) hooks are one-shot — a 60s-timeout agent
can never run a 10x60s polling loop. Hooks cannot poll/loop.
Fix applied: replaced with a command hook (jq guard + `if` filter) that injects
a one-line reminder only on actual `git push`; actual deploy monitoring runs as
a normal turn (Render MCP + ScheduleWakeup), which CAN loop.
Rules:
1. Agent hooks = one bounded verification, never monitoring loops.
2. For "watch X until done": /loop, Monitor, or ScheduleWakeup in a normal turn.
3. Command hooks should self-guard on stdin JSON even when `if` is set.
