# Portfolio Review — All Projects (2026-06-09)

Goal: review architecture + implementation of every project, find bugs, identify
production-readiness gaps and feature improvements, clean up junk, document everything.

## Projects
- [x] AetherOS — Rust workspace (OS prototype) — PROJECT_REVIEW.md done (build FAILS, 28 bugs)
- [x] AurumAi — platform — PROJECT_REVIEW.md done (CRIT: no tenant authz; rotate Anthropic key in control-plane/.env)
- [x] bloomberg-terminal — PROJECT_REVIEW.md done (3 HIGH: live-order bypass, bot control plane, blocking LLM call)
- [x] cad-converter — FastAPI + vanilla JS (small) — PROJECT_REVIEW.md done
- [x] ClaudeCosts — single Python script + installer — PROJECT_REVIEW.md done
- [x] DocAI — Python package (ingestion/RAG/agents) — PROJECT_REVIEW.md done
- [x] OptimaLLM — Go proxy/daemon/CLI — PROJECT_REVIEW.md done (2 CRIT bugs)
- [x] SafetyEye — CV/PPE detection platform — PROJECT_REVIEW.md done (3 CRIT defects)
- [x] SitAware — FastAPI backend + Next.js frontend — PROJECT_REVIEW.md done
- [ ] slm-forge — Python ML fine-tuning tool — review agent died at session limit, RELAUNCHED

## 9/10 reviews complete. Master doc: PORTFOLIO_REVIEW.md at root.

## Per-project deliverable
`PROJECT_REVIEW.md` in each project root containing:
architecture assessment, bug list (file:line), production-readiness gaps,
feature improvement recommendations, cleanup actions.

## Steps
- [ ] Write tasks/todo.md (this file)
- [ ] Spawn parallel review agents (one per project)
- [ ] Collect results, apply safe top-level cleanup (.DS_Store, stray files)
- [ ] Write master PORTFOLIO_REVIEW.md at root
- [ ] Commit + push

## Cleanup candidates (root level)
- .DS_Store files everywhere
- test-deploy.txt (hook test artifact)
- graphify-out/ (build artifact?)
- slm-forge-backup.git/ (bare repo backup — confirm before delete)
- Misconfigured PostToolUse deployment-monitor hook (60ms timeout, fires on every Bash)
