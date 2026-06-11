# Portfolio Review — All Projects

**Date:** 2026-06-10
**Scope:** Architecture, implementation quality, bugs, security, production readiness, and feature recommendations for every project in this directory.
**Detail:** Each project has a full `PROJECT_REVIEW.md` in its root with file:line bug references. This document is the executive summary.

---

## Scorecard

| Project | What it is | Verdict | Production-ready? |
|---|---|---|---|
| bloomberg-terminal | QuantTradeAI financial terminal (FastAPI + React/Vite, Capacitor, Supabase) | Impressively engineered; good security hygiene in places, but live-order safety bypassable and authz single-tenant | No — 3 HIGH issues |
| OptimaLLM | Go LLM proxy/optimizer for Claude Code traffic | Well-structured, but request-rewrite core has 2 correctness-critical bugs | No — unsafe for real traffic |
| AurumAi | Governed multi-tenant AI platform (control-plane + Next.js + connectors) | Well-architected MVP, but core promise (tenant governance) not enforced | No — tenant authz absent |
| SafetyEye | PPE/CV safety monitoring (YOLO worker + FastAPI + Next.js) | Demo-grade breadth; detect→alert→evidence pipeline has critical defects | No — clips/alerts/tenancy broken |
| SitAware | Situational-awareness dashboard (FastAPI + Next.js, external data feeds) | Functional; integration and hardening gaps | Closest of the web apps |
| slm-forge | SLM fine-tuning studio (Gradio + training/quantize/export/RAG) | Large, feature-rich; see its PROJECT_REVIEW.md | No — needs hardening pass |
| cad-converter | 2D CAD → 3D model converter (FastAPI + Claude Vision + CadQuery + Three.js) | Solid prototype architecture; input-handling security holes | No — 2 HIGH path bugs |
| ClaudeCosts | Claude usage/cost menu-bar monitor (single Python script) | Useful tool; stale pricing, broken uninstaller, timezone bug | No — but small fix list |
| DocAI | Local/offline document intelligence (RAG over local drives) | Pre-alpha skeleton, ~25% complete; existing code high quality | No — spine missing (paused project) |
| AetherOS | Rust OS architectural prototype (5 crates) | Honest sketch; doesn't compile; flagship pipelines can't run even simulated | No — research artifact |

---

## Cross-cutting themes (fix these patterns everywhere)

1. **Authorization is the #1 systemic gap.** Three projects (bloomberg-terminal, AurumAi, SafetyEye) accept a tenant/owner identifier from the request instead of deriving it from the authenticated principal, or skip auth on mutating endpoints entirely. Pattern fix: every router dependency-injects the authenticated identity; object access always filters by it; admin actions require role checks server-side.
2. **Safety/“governance” claims not enforced at the boundary.** bloomberg-terminal's "live trading hard-disabled" is bypassable via an unauthenticated endpoint; AurumAi's RBAC is dead code; SafetyEye's RLS exists only in docs. Claims in README/CLAUDE.md must map to a specific enforced code path, ideally with a test asserting the negative case.
3. **Secrets hygiene is decent (nothing committed to git) but operationally risky.** Live keys sit in working trees: `AurumAi/control-plane/.env` (Anthropic key — ROTATE), `bloomberg-terminal/render.env` (Supabase service-role, Anthropic, Gemini, Tavily, Redis). Move to a secret manager or at minimum out of repo working trees; add pre-commit secret scanning (gitleaks) to every repo.
4. **Default credentials boot in production.** AurumAi (`dev-master-key`, `change-me-in-production`) and others start fine with placeholder secrets. Fail fast at startup when env is production-like and a secret is default/missing (bloomberg-terminal's fail-fast config does this right — copy it).
5. **No CI anywhere, and tests are thin or absent.** Only bloomberg-terminal and slm-forge/SitAware have meaningful suites; AetherOS, cad-converter, ClaudeCosts, DocAI have zero tests. Minimum bar per project: lint + typecheck + unit tests in GitHub Actions on every push.
6. **Re-serializing client payloads through partial models silently drops data.** OptimaLLM (drops `stream`, `metadata`, tool blocks) is the worst case; cad-converter's content-block assumption is the same class. Rule: mutate JSON as maps; never round-trip through a struct that models a subset of the schema.
7. **Repo hygiene:** `.DS_Store` everywhere, build artifacts committed (`tsconfig.tsbuildinfo`, `__pycache__`, `repomix-output.md` 2.9 MB), stray root files, duplicate/dead frontend stacks (AurumAi's whole `src-vite/` tree). Cleanup actions listed per project.

---

## Per-project summaries

### bloomberg-terminal (QuantTradeAI)
Strengths: fail-fast config, Fernet-encrypted creds, audit chain, leader-locked scheduler.
Top issues:
1. **HIGH** — `POST /api/alpaca/orders` (`backend/routers/alpaca.py:154`) has no auth and caller-controlled `paper: false` routes real orders to live Alpaca, bypassing the entire live_guard stack.
2. **HIGH** — Bot control plane (`backend/routers/bot.py:144-271`): any authenticated user can rewrite global bot config, swap broker creds, arm live trading; `GET /bot/config` is unauthenticated.
3. **HIGH** — `/ai/sidekick` (`backend/routers/ai.py:70-77`) calls Anthropic synchronously on the event loop (stalls all requests), accepts anonymous traffic on the platform key.
Also: founder passcode baked into public JS bundle (client-side tier gating), forgeable unsubscribe tokens, infinite 401 retry loop in `frontend/src/lib/api.ts`, unauthenticated `/ws/prices`, dual migration trees, CSP `unsafe-inline`.
Path to product: fix the three HIGHs, make authz per-user end-to-end, move LLM calls off the loop (async client + queue), add rate/cost limits, single migration tree.

### OptimaLLM
Strengths: clean Go layout, good degradation patterns, vet/tests green.
Top issues:
1. **CRIT** — `internal/proxy/orchestrator.go:1054`: re-marshal through 9-field `anthropicRequest` drops `stream`, `metadata`, `stop_sequences`, `top_p` → breaks SSE streaming through the transform path.
2. **CRIT** — `orchestrator.go:551` + `:1094`: canonicalization replaces last user message content with a string, destroying `tool_result` blocks → upstream 400s on agentic turns; auto-compact can orphan tool_use pairs.
3. **Security** — proxy token design broken (`auth.go`, `proxy.go:118`): enabling it 401s real traffic or leaks the secret upstream; not constant-time; graph server defaults to unauthenticated `0.0.0.0:7778`.
Also: dead entropy-redaction (threshold 5.6 > max 5.0), session-ID mismatch disabling 3 feedback features, stale 2h bash cache, system-prompt block arrays fail to unmarshal (optimizer no-ops on real sessions). 18 bugs total in its review.
Path to product: map-based request mutation, shadow mode (observe-only diffing) before rewrite mode, fail-open on any transform error, real traffic-shape test corpus.

### AurumAi
Strengths: coherent platform architecture, connectors + SDK + control-plane separation, good docs.
Top issues:
1. **CRIT** — tenant authz absent: `tenant_id` read from request body/query, never checked against the key (`api/routes.py:48,297,532,569`, all of `kg_routes.py`); any user-role key can upload/activate policies for any tenant (`admin_routes.py:143,188`) and list any tenant's keys. `rbac.py` is dead code; `TenantScopingMiddleware` is a no-op.
2. **Security** — live Anthropic key on disk in `control-plane/.env` (gitignored, not in history — rotate anyway); default secrets boot in prod and make the "signed immutable audit log" forgeable.
3. **HIGH** — prod frontend broken: Dockerfile copies `dist/` after `next build` (outputs `.next/`); whole `src-vite/` tree + nginx.conf are unbuildable leftovers. Next.js is the live stack — delete the Vite remnants.
Also: webhooks enqueued but never delivered (`arq_enabled=False`, no worker), OIDC non-functional, webhook SSRF, SCIM fails open on empty token, Terraform can't actually deploy (no ECS service/ALB).
Path to product: enforce tenancy at one middleware chokepoint + wire RBAC, ship the worker container, delete dead stack, finish Terraform.

### SafetyEye
Strengths: real pipeline breadth (RTSP → YOLO → events → dashboard), helm/prometheus infra thinking.
Top issues:
1. **CRIT** — timestamp domain mismatch (`worker/pipeline/ingest.py:152` stream-relative PTS vs epoch mtimes in `buffer.py:46`): evidence clips are never produced; backend receives ~1970 timestamps. FFmpeg segmenter never reconnects after RTSP drop; undrained stderr can deadlock it.
2. **CRIT** — multi-tenancy documentation-only: no RLS policies in migrations; `get_tenant_session` used by zero routers; list/get endpoints unfiltered (`violations.py:69-93`); `/ws/violations` unauthenticated.
3. **HIGH** — alert delivery lossy and content-free: in-process Redis consumer from `$` with no consumer group (`main.py:92`); outbox replays never alert; `notify.py:22-36` reads fields the payload lacks → every alert says "UNKNOWN". Shared ByteTrack state across cameras breaks dedup.
Also: render.yaml missing Redis/Celery/S3 vs compose/helm; no GPU→CPU fallback; no worker metrics; 8 loose scripts at root.
Path to product: fix timestamps to epoch at capture, per-camera trackers, Redis consumer groups + alert from outbox, real RLS + WS auth, converge the three deploy configs.

### SitAware
Strengths: tested backend, docs-driven (cost analysis, data sources, ops), docker-compose dev story.
Issues (see its review): committed `.env` at root (check + rotate anything real), copy-pasted parse helpers across 4 connectors, unused heavy geo deps (cfgrib, netCDF4, pyproj, alembic, geoalchemy2), Next 14.2.3 needs security patches.
Path to product: harden external-feed failure modes (rate limits, backoff, circuit breakers), consolidate utils, trim deps, dependency updates, deploy story beyond compose.

### slm-forge
Largest active codebase (own git repo, installer, Docker, Render deploy). Full findings in `slm-forge/PROJECT_REVIEW.md` (note: its review was interrupted by session limits twice; the on-disk review file is the authoritative record — re-run a focused review of core/backend.py + admin.py auth + hf_publisher.py token handling if anything looks thin).
Known concerns from prior audits (AUDIT_REPORT.md, SECURITY.md exist in-repo): admin surface auth, HF token handling, subprocess/GPU lifecycle, coherence of local-app vs Render-hosted product story.
Path to product: single product story (local studio first), license + telemetry decision, signed installers, CI with smoke training run on tiny model.

### cad-converter
Strengths: clean prototype layering (models/extractor/builder/transport), CadQuery correctly isolated in ProcessPoolExecutor, spec re-validated at each hop. Boots clean.
Top issues:
1. **HIGH** — path traversal via unsanitized `file.filename` (`main.py:82`, flows to `extractor.py:352`) — arbitrary file write.
2. **HIGH** — unvalidated `session_id` path segments (`main.py:104,121,149,168`) — validate as UUID.
3. **MED** — 20 MB cap enforced after full read into memory (`main.py:75-77`); DXF W/D/H heuristic unreliable (`extractor.py:261-264`); content-block 0 assumed text (`extractor.py:171`).
Also: CadQuery not installed in current env (`/api/build` fails); frontend feature-row index desync after remove+add (`app.js:308-322`); CORS wide open; no tests/CI/Docker.
Path to product: fix the two HIGHs (an afternoon), add upload streaming, pin deps + Dockerfile, golden-file tests for extractor, then it's a credible niche SaaS.

### ClaudeCosts
Single-file menu-bar cost monitor. Defensive JSONL parsing is good.
Top issues:
1. **HIGH** — pricing table stale (`claude_usage_monitor.py:71-82,120-126`): no Fable 5/Opus 4.8 (silently costed at Sonnet fallback), Opus 4.6/4.7 at legacy $15/$75 vs current $5/$25 tier, Haiku 4.5 at 3.5 rates, three dead reverse-named keys.
2. **HIGH** — uninstall.sh removes a path install.sh never creates, leaves the PreToolUse hook in `~/.claude/settings.json` → monitor resurrects next session.
3. **MED-HIGH** — UTC day-bucketing (`:218`) vs local `date.today()` filter (`:676`): evening usage lands on "tomorrow", Today reads ~$0. Also AppKit UI mutated from background thread (`:654-671`) — crash risk.
Path to product: fix the three, fetch pricing from a remote manifest (or models endpoint) instead of hardcoding, package for PyPI/homebrew, add tests for the JSONL parser.

### DocAI
Pre-alpha, ~25% of PROJECT_BRIEF.md, intentionally paused ("hardware procurement"). What exists (~3,400 LOC) is high quality: clean dataclass contracts, per-file error isolation, atomic writes, WAL SQLite, genuinely local-only (no cloud calls, verified).
Blockers: `docai.indexing.{embedder,faiss_index,sqlite_store}` never written → retrieval/agents unimportable; no `__init__.py` anywhere (breaks packaging); zero tests; 3 of 9 parsers; three competing file-state manifests; scanner hashes every file every scan defeating the mtime fast path.
Path forward: write the indexing spine (embedder/FAISS/store ≈ the remaining 75%'s core), make it a real package, one manifest store, then a CLI.

### AetherOS
Research prototype, honestly documented as such. Does not compile on stable (obsolete `#![feature]` gates in `aether-core/src/lib.rs:8` + missing `Box` import in `memory_pressure.rs` — both one-line fixes, after which nightly is unnecessary).
Flagship pipelines can't run even simulated: driver synthesis can never emit `DriverReady` (confidence 0.40 always < compiler's 0.55 gate, `aether-hal/src/synthesis.rs:72`); JanitorAgent unconditionally evicts 256 vectors per sweep (threshold const never used, `janitor.rs:50`) plus ABBA deadlock risk; SPSC ring bus shared by multiple producers is UB (`ipc.rs:222`); gossip declares all live peers Dead in 30s. 28 file:line bugs in its review.
Path forward: it's a portfolio/architecture piece — fix the build, add a runnable simulation demo with the three pipelines actually completing, zero→some tests. Don't productize.

---

## Release priority recommendation

If the goal is shipping something public soonest, effort-to-credible-product ranking:

1. **ClaudeCosts** — days. Fix pricing/uninstall/timezone, package, ship to homebrew/PyPI. Real audience (Claude Code users) and clear distribution.
2. **cad-converter** — 1–2 weeks. Two security fixes + packaging + tests; niche but differentiated (Claude Vision → parametric CAD).
3. **SitAware** — 2–4 weeks of hardening; closest full web app.
4. **OptimaLLM** — fix the 2 CRITs, run in shadow mode against your own traffic for a week, then it's a genuinely interesting public tool.
5. **bloomberg-terminal** — strongest product, but financial domain: fix the 3 HIGHs, then a real security pass before any public user touches it.
6. **slm-forge / SafetyEye / AurumAi** — months; B2B-shaped, need the tenancy/reliability work before anyone pays.
7. **DocAI / AetherOS** — paused/portfolio; not release candidates.

## Workspace cleanup performed / proposed

Done:
- Removed root `test-deploy.txt`, root `.DS_Store`, all nested `.DS_Store`.
- Moved root QuantTradeAI artifacts (`COMPLETE_DOCUMENTATION.md`, `generate_docs.py`, `scratch/` product docs) into `bloomberg-terminal/docs/product/`.
- Per-project `PROJECT_REVIEW.md` written (10 projects), `tasks/todo.md` updated.

Proposed (need your confirmation — destructive):
- Delete `slm-forge-backup.git/` (bare backup repo; slm-forge has its own live .git).
- Delete root `graphify-out/` (generated OptimaLLM graph artifact; regenerable).
- Delete `bloomberg-terminal/repomix-output.md` (2.9 MB generated dump) and `test-results/`.

Tooling fix needed:
- `.claude/settings.json` PostToolUse hook uses unsupported `"if": "Bash(git push*)"` — an evaluation agent fires on EVERY Bash command (cost + noise), and its 60s timeout can't run the intended 10×60s deploy-monitor loop anyway. Replace with a Stop-gap: remove the hook and use `/loop` or a Monitor-based check after pushes, or a command-type hook that exits 0 unless `$TOOL_INPUT` matches `git push`.
