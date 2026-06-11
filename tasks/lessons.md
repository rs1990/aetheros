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

## 2026-06-10 — Misconfigured PostToolUse hook
`.claude/settings.json` PostToolUse hook uses `"if": "Bash(git push*)"` — not a
supported field. Result: a condition-evaluation agent fires on EVERY Bash call
(cost + transcript noise), and the 60s timeout can't run the intended 10x60s
deploy-monitor loop anyway. Hooks cannot poll/loop; use /loop, Monitor, or a cron
agent for deploy watching. Fix pending user sign-off.
