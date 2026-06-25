# Portfolio Data — rs1990

Generated: 2026-06-24  
Source: Local code analysis of all 11 repos.

---

## quanttradeai

**GitHub URL:** https://github.com/rs1990/quanttradeai  
**Primary Language:** Python (backend, 52 services) + TypeScript (frontend, React 18)  
**Tech Stack:**
- Frontend: React 18.3, TypeScript 5.7, Vite 6.0, Zustand 5.0, React Query 5.62, Tailwind 3.4, Lightweight Charts 4.2, Monaco Editor 4.7, Axios 1.7
- Backend: FastAPI 0.115, Uvicorn 0.30, Python 3.11, Pydantic 2.10+, Supabase GoTrue (auth), asyncpg, redis-py 5.1, APScheduler 3.10, Celery 5.4, Anthropic 0.87, google-generativeai 0.8, OpenAI 1.55, structlog 24.4, Sentry 2.19, slowapi 0.1
- Database: PostgreSQL 15 (Supabase), Upstash Redis (TLS), Polygon.io (optional premium)
- Data Sources: Yahoo Finance, Alpaca Market Data, SEC EDGAR, CoinGecko, FRED, Tavily, StockTwits, Reddit API, FINRA RegSho, Finnhub
- Deployment: Vercel (frontend), Render (backend), Supabase (DB + auth)

**What it actually does:**  
Production-grade financial analytics platform combining Bloomberg Terminal features (real-time charting, 35+ technical indicators, options analytics, screener) with AI research (multi-provider streaming from Claude/Gemini/GPT-4), portfolio optimization (Monte Carlo, efficient frontier, backtesting), institutional intelligence (13-F filings, insider trades, Congress trades, dark pool flow), and accountability infrastructure (hash-linked audit chain, trust-weighted consensus pricing, outcome ledger). Targets $49–$199/mo vs Bloomberg $2,000/mo. 282 features across 30 tabs.

**Architecture:**
```
Tier 1 — Frontend (React 18 + Vite)
├─ 30 pages (Terminal, Screener, Options, Portfolio, Research, Social, Risk, etc.)
├─ Zustand stores (terminal, auth, alert, watchlist)
├─ React Query + Axios (server-state caching)
├─ Web Workers (indicator math off main thread via Comlink)
├─ BroadcastChannel (cross-tab sync of symbol/period/interval)
└─ Lightweight Charts (WebGL rendering, <100ms for 50k candles)

Tier 2 — Backend (FastAPI + Python 3.11)
├─ 51 routers (market, technicals, options, screener, portfolio, ai, earnings,
│   institutional, audit, alerts, bot, supply_chain, geopolitical, etc.)
├─ 52 services (market_data, technical, options, cache, llm, audit_chain,
│   trust_graph, learning, execution, live_guard, bot_engine, etc.)
├─ Circuit breaker (rate limit resilience; Yahoo Finance + Alpaca fallback)
├─ Streaming responses (SSE for AI analysis, token-by-token)
├─ APScheduler (alert checks every 1 min, daily policy updates)
└─ Celery (async compute: Monte Carlo, ARIMA/ETS forecasting)

Tier 3 — Database (PostgreSQL + Supabase)
├─ auth.users (Supabase Auth managed)
├─ User data (watchlists, portfolios, alerts, api_keys — RLS protected)
├─ Shared data (messages, iv_history, ai_artifacts)
├─ Audit tables (audit_events — SHA-256 hash-linked, immutable)
├─ Trust tables (trust_snapshots, source_accuracy)
├─ Trading tables (trade_orders, bot_risk_state, live_broker_creds)
└─ Migrations 001–015 (bot config, trade ledger, live creds, policy state)

Tier 4 — Caching (Redis via Upstash)
├─ quote:{symbol} (6h TTL)
├─ chart:{symbol}:{period}:{interval} (1h TTL)
├─ iv_surface:{symbol}:{date} (4h TTL)
├─ options_chain:{symbol}:{exp_date} (15m TTL)
└─ Rate limiting (SlowAPI + Redis backend)
```

**Key Technical Features:**
1. Real-time Charts — Candlestick/Line/Area; 1m–1mo intervals; 1D–Max periods; crosshair sync across tabs via BroadcastChannel
2. 35+ Technical Indicators — RSI, MACD, Bollinger, Ichimoku, Parabolic SAR, VWAP, Volume Profile, Keltner, Donchian, Supertrend, HMA, DEMA, TEMA, StochRSI, CCI, ADX/DI, ROC, MFI, CMF, ATR, Williams %R, EMA/SMA variants; offloaded to Web Worker via Comlink RPC
3. Pine Script v5 Editor — Custom indicator authoring in Monaco; transpiler to sandboxed JS; supports ta.ema/sma/rsi/atr/cci/roc/mfi/cmf, plot(), math.*, if/for/var
4. Options Analytics — IV Surface (2D heatmap), Greeks (Δ, Γ, Θ, ν, ρ), Black-Scholes, 8 multi-leg strategies (iron condor, butterfly, calendar, etc.), payoff diagrams
5. Screener — 805 liquid US equities, 20+ filters (RSI, P/E, P/B, market cap, RVOL, dividend, EV/EBITDA, SMA200 proximity), quant scores (V/G/Q/M, 1–10 scale), presets, virtualized 800+ row table, CSV export
6. AI Research — Streaming reports (executive summary, bull/bear case, peer comp, technical, risk/reward) + 10-K summaries + earnings transcripts; multi-provider (Claude default, Gemini/GPT-4 fallback), BYO API key, token-by-token streaming, artifact registry with version tracking
7. Portfolio Management — MPT optimizer (efficient frontier, 3,000 Monte Carlo sims), backtester (4 strategies: RSI, MACD, Bollinger, SMA cross), risk metrics (Sharpe, Sortino, Beta, Alpha, max drawdown, VaR, CVaR, correlation matrix)
8. Institutional Intelligence — 13-F filings, insider trades (Form 4), short interest (FINRA RegSho 30/60/90d), Congress trades (Senate + House), dark pool short volume breakdown
9. Self-Learning Trading Bot — Paper-proven gate (≥50 trades, ≥56d span, +mean alpha), news-aware (Claude reads market news before each trade), 3-axis learned sizing (slot × symbol × VIX regime), quarter-Kelly sizing, real-money HARD-DISABLED by default (master switch + encrypted creds + arm + live_guard all required)
10. Audit Chain + Trust Graph — SHA-256 hash-linked immutable event log, PageRank trust scoring per data source, trust-weighted consensus pricing, data provenance UI panel

**Data Flow:**
1. User searches symbol → React useQuery → GET /api/market/quote/{symbol}
2. Cache miss → tries Alpaca (15-min delayed) → yfinance fallback → cache SET 6h TTL → Pydantic serialized JSON
3. AI analysis triggered → backend fetches context (quote, technicals, news via Tavily) → SSE stream to Anthropic Claude 3.5 → token-by-token to frontend via React Suspense → on completion, artifact_registry.save() + audit_chain.insert_event() (HMAC-chained)
4. Background alerts: APScheduler every 1 min → fetches active alerts → checks current quote vs target → WebSocket push + Discord/Telegram webhook
5. Trading bot: bot proposes trade → services/execution.py (single chokepoint) → pre-checks (master switch, encrypted creds, armed, live_guard: drawdown <15%, daily loss cap, no PDT, fat-finger) → submit_order() → trade_ledger.py (idempotent) → learning.py updates policy (reward = benchmark-relative alpha, losses 2×)

**Key Files:**
- `backend/main.py` — FastAPI app factory + lifespan events (cache init, scheduler startup)
- `backend/config.py` — Pydantic Settings; all 30+ env vars typed + validated at startup
- `backend/services/market_data.py` — Quote fetching + circuit breaker; Alpaca → yfinance fallback chain
- `backend/services/technical.py` — All 35+ indicator calculations (pure math, no I/O)
- `backend/services/cache.py` — Redis connection pool + in-memory fallback; TTL-based expiry
- `backend/services/llm/anthropic.py` — Streaming Claude 3.5; guardrails + symbol validation
- `backend/services/audit_chain.py` — SHA-256 hash-linked immutable log (WORM table)
- `backend/services/execution.py` — Single guarded chokepoint for paper + live orders
- `frontend/src/App.tsx` — Root component; QueryClient, RouterProvider, BroadcastChannel init
- `frontend/src/pages/Terminal.tsx` — Primary charting interface; Lightweight Charts + indicator overlays

**Real Metrics / Numbers from Code:**
- 282 features across 30 tabs
- 51 API routers, 52 backend services
- 42 unique API endpoints (per README; routers aggregate by domain)
- 805 symbols in screener universe with quant scores
- 35+ technical indicators (overlays + sub-charts)
- 25+ AI-powered features
- 3 LLM providers (Claude 3.5 default, Gemini 2.0, GPT-4 Turbo fallback) — all streaming
- 12 data sources (Yahoo Finance, Alpaca, SEC EDGAR, CoinGecko, FRED, Tavily, StockTwits, Reddit, FINRA, Finnhub, Senate, House trades)
- 310 backend tests passing (README says 242; CLAUDE.md notes 310/310 as of 2026-05-18)
- 0 TypeScript errors
- P99 latency <2s quotes, <500ms technicals, <1s per AI chunk
- 3,000 Monte Carlo simulations for portfolio optimizer

**External Integrations / APIs Used:**
- Anthropic Claude (primary AI), Google Gemini (fallback), OpenAI GPT-4 (final fallback)
- Yahoo Finance (free, 500K req/day), Alpaca (free tier, 15-min delayed)
- SEC EDGAR (10-K, 8-K, Form 4, daily), FRED (GDP, CPI, NFP, unemployment, yield curve)
- StockTwits, Reddit API (r/wallstreetbets), Tavily (web search), Finnhub (IPO calendar)
- FINRA RegSho (short volume), Congressional Disclosures (Senate + House)
- CoinGecko (crypto, 1-min updates), Polygon.io (premium optional, $250/mo)
- Resend (transactional email), Web Push API (VAPID browser notifications)
- Supabase (PostgreSQL + auth), Upstash Redis (TLS)
- Polygon L2 blockchain (optional on-chain audit anchoring, ~0.5 MATIC/thousands of anchors)

**What Makes It Distinctive:**
1. Fail-closed real-money trading: 5 independent guards must all pass before live order (master switch + encrypted creds + arm/disarm + live_guard + paper-proven gate). Single execution chokepoint. No silent downgrade path.
2. Self-learning bot with capital preservation: reward = benchmark-relative alpha (losses weighted 2×), quarter-Kelly per-VIX regime, news-aware via Claude before each trade, $500 capital hard constant (never auto-scaled), exploration only in paper
3. SHA-256 hash-linked audit chain + PageRank trust graph: tamper-evident compliance log, consensus pricing = trust-weighted median quote
4. Web Worker indicator engine: all 35+ indicator math offloaded via Comlink RPC — zero UI jank on 50k candles
5. BroadcastChannel cross-tab sync: symbol/period/interval changes broadcast to other tabs via browser-native pub/sub
6. Circuit breaker rate limit resilience: semaphore + error counter + open/pause state; prevents cascading failures on 429s
7. Fernet-encrypted API keys at rest: AES-128-CBC + HMAC-SHA256; never logged; startup fails in production if ENCRYPTION_KEY unset
8. Institutional data integration: 13-F, Form 4 insider trades, Congress trades, FINRA short interest, dark pool flow — all from free government sources

**Honest Status:**  
Production-ready MVP. Core features fully tested (310/310 tests, 0 TS errors, deployed on Vercel + Render + Supabase). Two bugs resolved post-sprint (institutional page slowness, web research timeout). One known deferred: vol surface crash guard (mitigation added, root fix pending). Pre-launch blockers requiring user action: Stripe payment setup, Polygon.io license, Resend domain, legal ToS/Privacy Policy, Supabase migrations 010–015 must run before bot/live endpoints activate, ~9-week paper-proving window needed before go-live authorization.

**README Claims:**
- "282 features across 30 tabs" ✓
- "42 API endpoints" ✓ (51 router files, 42 unique endpoint paths)
- "12 data sources integrated" ✓
- "3 AI providers (Claude, Gemini, GPT-4) with streaming" ✓
- "35+ technical indicators" ✓
- "805 equities with quant scores" ✓
- "242/242 backend tests passing" (README; actual: 310/310 per CLAUDE.md post-sprint)
- "0 TypeScript errors" ✓
- "P99 <2s quotes, <100ms charts" ✓
- "$49–$199/mo vs Bloomberg $2,000/mo" ✓ (pricing positioned, not yet live)

---

## SafetyEye

**GitHub URL:** https://github.com/rs1990/SafetyEye  
**Primary Language:** Python (backend/worker) + TypeScript/React (frontend)  
**Tech Stack:**
- Backend: FastAPI 0.115+, SQLAlchemy 2.0+, asyncpg, Pydantic 2.7+, Alembic, Redis 5.0+, Celery 5.4+, boto3 1.35+, prometheus-fastapi-instrumentator 7.0+, anthropic 0.40+, python-jose, bcrypt <4.1, passlib, geoalchemy2 0.15+, shapely 2.0+
- Worker: ultralytics 8.3+ (YOLOv8+ByteTrack), opencv-python-headless 4.10+, PyAV 12.0+, redis, httpx, boto3, prometheus-client, psutil, cryptography 43+, pydantic
- Frontend: Next.js 14.2+, React 18.3+, TypeScript 5.5+, Tailwind 3.4+, Recharts 2.12+, Radix UI, TanStack Query 5.51+
- Infra: PostgreSQL 16 + TimescaleDB + PostGIS, Redis 7, MinIO (dev) / S3 (prod), FFmpeg, MediaMTX (RTSP/HLS gateway), Prometheus, Kubernetes/Helm (K3s/EKS/GKE), Docker Compose

**What it actually does:**  
Enterprise real-time computer vision safety monitoring for 100+ concurrent camera feeds in heavy manufacturing, automotive, and industrial facilities. Edge GPU workers ingest RTSP streams, run YOLOv8+ByteTrack inference, evaluate PPE/zone/dwell/proximity/ergonomic rules, extract violation clips from rolling FFmpeg segment buffer, route to Claude Vision or local SLM for failure analysis, publish to FastAPI control plane (Postgres+TimescaleDB+Redis), consumed by Next.js dashboard. Includes OSHA compliance automation (Form 300/300A, DART/TRIR), ePTW, muster/evacuation, 9-system integrations hub, and mobile PWA.

**Architecture:**
```
Edge GPU Worker:
├─ worker/pipeline/ingest.py — RTSPDecoder (PyAV) + FFmpeg segment writer (1-sec .ts, 60-sec rolling ring)
├─ worker/pipeline/inference.py — YOLOv8m TensorRT FP16 + ByteTrack multi-object tracking
├─ worker/pipeline/rules.py — Rule evaluator: zone/PPE/dwell/proximity/ergonomic, per-track state
├─ worker/pipeline/clipper.py — SegmentBuffer ring, FFmpeg -c copy clip extraction, optional face blur, S3/MinIO
├─ worker/pipeline/analyzer.py — LLM/SLM failure classification (Claude Sonnet Vision or local Qwen2-VL/MiniCPM-V, bounded pool)
├─ worker/pipeline/publisher.py — Violation POST + Redis Streams XADDs
└─ worker/pipeline/outbox.py — SQLite WAL offline store-and-forward + server-side dedup on event_id

Backend Control Plane (FastAPI):
├─ 30+ routers (violations, audit, compliance, briefings, assistant, actions, observations,
│   ergo, ptw, muster, integrations, ota, devices, cameras, zones, rules, analytics,
│   admin, billing, onboarding, wearables, contractors, workers)
├─ PostgreSQL 16 + TimescaleDB (violations hypertable on detected_at, continuous aggregates)
├─ PostGIS (zone polygon storage + spatial queries)
├─ Redis Streams (event log, consumer groups for notification fan-out)
├─ Celery beat (audit aggregation, clip purge, model re-eval, notification escalation, briefings)
└─ Prometheus instrumentation

Frontend (Next.js 14 App Router):
├─ 20+ pages: overview, live (multi-camera HLS grid), violations, audit, heatmap, cameras,
│   risk (pSIF), actions (Kanban), observations, briefings, compliance (OSHA), ergo,
│   analytics, ptw (Kanban), muster, ops (OEE), contractor, induction, admin
├─ WebSocket for live violations + per-camera detections
├─ HLS/WebRTC via MediaMTX for video preview
└─ Mobile PWA (4 pages: violation triage, briefing sign-off, actions, inductions)
```

**Key Technical Features:**
1. RTSP ingest + NVIDIA NVDEC GPU decode (NVDEC) or PyAV fallback with FFmpeg rolling segment buffer (1-sec .ts files, 60-sec ring for zero-gap clip extraction)
2. YOLOv8m TensorRT FP16 + ByteTrack multi-object tracking at 15 fps per camera (100 cameras × 15 fps = 1,500 fps cluster throughput on 3–4 RTX 4090s)
3. Zone/PPE/dwell/proximity/ergonomic rule engine (JSONB config, hot-reload via Redis pub-sub, no clip re-encode via stream copy)
4. Dual-stage analysis: fast rule eval (15 fps) → async LLM/SLM failure classification (Claude Sonnet Vision or local Qwen2-VL-7B/MiniCPM-V-2.6, bounded thread pool, per-violation dedup via merge-window)
5. Multi-tenancy with Stripe billing: free (1–5 cameras), starter ($5K/yr, 5–20), pro ($20K/yr, 20–100), enterprise (100+, custom); usage tracking per org
6. OSHA compliance: auto-map violations to 29 CFR standards, export Form 300/300A, DART/TRIR calculation, NEP risk assessment, ITA e-submission
7. Ergonomics: RULA/REBA per-frame scoring, exposure hour tracking, MSD risk categorization, auto-action creation
8. pSIF composite risk score (0–100, weighted violations + severity + open actions + compliance trends + area factors), 7-day forecasting via linear trend analysis
9. ePTW (electronic permit-to-work): 6 permit types, approval chain, hazard assessment JSONB, automatic expiry, clip integration
10. Integrations hub: 9 systems (SAP S/4HANA, IBM Maximo, Intelex, Cority, Power BI, Tableau, Slack, Microsoft Teams, generic webhooks) with event routing, credential encryption, retry logic
11. Evacuation/muster: camera-assisted headcount via live detections, expected vs. actual comparison, sign-off audit trail
12. Ed25519 chain-of-custody signing on violation clips (worker-side), server-side verify — auditable evidence trail

**Data Flow:**
1. RTSP stream (IP camera 1280×720 15 fps) → parallel FFmpeg segmenter (1-sec .ts to /tmpfs) + PyAV decoder thread (bounded frame queue, drop-oldest)
2. Per-frame YOLOv8 detection (person + PPE classes: hard_hat, hi_vis_vest, safety_glasses, gloves) + ByteTrack tracking
3. Per-track rule evaluation (zone containment via Shapely polygon, PPE presence/absence, dwell timer, proximity distance) → Violation event when rule predicate fails ≥ min_duration_s
4. Clip extraction: schedules `ffmpeg -f concat -c copy` from segment ring window; optional face blur (OpenCV Haar cascade + Gaussian); S3/MinIO upload
5. Optional: async LLM/SLM failure classification → JSON (failure_type, sub_type, severity_assessment, description, recommended_action)
6. Durable SQLite outbox (WAL, at-least-once) + POST /api/v1/violations to backend
7. Backend: violation → Postgres (TimescaleDB hypertable) → Redis Streams XADD → WebSocket broadcast → Celery notify task → rollup aggregation

**Key Files:**
- `backend/app/main.py` — FastAPI setup, lifespan hooks, Alembic migrations, violation Redis consumer, CORS
- `backend/app/api/violations.py` — POST ingest (device-authed), GET violations (list/filter/paginate), PATCH reviews (confirm/dismiss), clip verify (Ed25519)
- `worker/main.py` — Worker entrypoint, heartbeat loop, per-camera ingest threads, shared inference orchestration
- `worker/pipeline/ingest.py` — RTSPDecoder (PyAV) + FFmpeg segment writer subprocess
- `worker/pipeline/inference.py` — Detector (shared YOLOv8 + ByteTrack; thread-unsafe shared predictor — known bug)
- `worker/pipeline/rules.py` — RuleEvaluator (zone/PPE/dwell logic, merge-window dedup, unbounded state dict — known bug)
- `worker/pipeline/clipper.py` — SegmentBuffer ring, clip extraction (time-domain bug known), face blur, S3 upload
- `frontend/app/(root pages)/` — 20+ Next.js pages (overview, live, violations, audit, heatmap, cameras, risk, actions, observations, compliance, ergo, briefings, analytics, ptw, muster, ops, contractor, induction, admin)
- `backend/app/services/ai_assistant.py` — Claude AI safety chat with tool use (query violations, risk, observations, trends)
- `backend/app/services/osha_ita.py` — OSHA ITA e-submission, Form 300/300A export, DART/TRIR calculation

**Real Metrics / Numbers from Code:**
- Camera concurrency target: 100 feeds @ 1280×720, 15 fps inference
- Throughput: YOLOv8m TensorRT FP16 on RTX 4090: 480 fps batched (b=8) → 3–4 GPUs needed; A100 80GB: 950 fps → 2 sufficient
- Alert latency SLA: ≤10s (typical 1–3s with edge inference)
- Clip extraction: <500ms stream copy (no re-encode), configurable pre/post windows (default 5s/15s)
- Model: YOLOv8m fine-tuned on construction-site-safety-v30 dataset (Roboflow), 25 classes
- 30+ API routers
- 20+ dashboard pages + 4 mobile PWA pages
- Clip retention: 30 days per plant (configurable)
- Stripe tiers: free (1–5 cams) → starter $5K/yr → pro $20K/yr → enterprise

**External Integrations / APIs Used:**
- Anthropic Claude Sonnet Vision (LLM-based failure analysis, optional per-rule routing)
- Local SLM via OpenAI-compatible endpoint (Qwen2-VL-7B, MiniCPM-V-2.6, llama.cpp, Ollama, vLLM)
- OSHA ITA e-submission API (29 CFR standards mapping)
- SAP S/4HANA, IBM Maximo, Intelex, Cority, Power BI, Tableau, Slack, Microsoft Teams, generic webhooks
- MediaMTX (RTSP ingest + HLS/WebRTC gateway)
- AWS S3 (clip storage), MinIO (dev equivalent)
- Redis Streams (event log), PostgreSQL + TimescaleDB + PostGIS
- Stripe (billing), Prometheus (metrics)

**What Makes It Distinctive:**
1. Dual-model inference pipeline: real-time deterministic rules (YOLOv8+ByteTrack 15 fps) + async generative failure classification (Claude/SLM, deduped by merge-window) — separates "did rule fire" (fast) from "why did it fail" (expensive LLM)
2. Rolling segment buffer: independent FFmpeg writer vs. inference consumer; stream-copy clip extraction (no re-encode); zero video gap
3. Ed25519 chain-of-custody signing on clips (worker-side verify on server) — auditable legal evidence
4. Durable outbox (SQLite WAL) for offline resilience + server-side idempotency (event_id dedup) — survives worker outages
5. Per-plant privacy toggle (blur_faces) baked into pipeline, not post-hoc
6. pSIF composite risk score (0–100) + 7-day forecasting from historical violation trends
7. RULA/REBA ergonomic posture scoring from per-frame skeleton detections
8. Hard-stop webhooks for machine interlock (safety violation → signal machinery to shut down)
9. OSHA standard mapping on rules + auto-export Form 300/300A + ITA e-submission — full compliance automation

**Honest Status:**  
Prototype/MVP with extensive feature breadth. Docs claim "12 production-ready phases complete" but PROJECT_REVIEW.md lists 17 bugs in the hot path: timestamp-domain mismatch breaks clips (#1), shared ByteTrack state across cameras (#2), segment writer never reconnects on RTSP drop (#3), inference thread leaks on camera removal (#4), merge-window state never resets (#5), no GPU fallback (#6), publisher failures orphan clips (#7). Additionally 7 security issues: no tenant isolation RLS (#1), unauthenticated WebSockets (#2), CORS wildcard (#3), dev credentials baked in (#4). Alert delivery not guaranteed (no durable consumer group). Multi-camera correctness broken. **Not suitable for production as a safety product** without hardening the detect→alert→evidence spine first.

**README Claims:**
- "Enterprise-grade real-time CV safety monitoring for heavy manufacturing, automotive, and industrial facilities" ✓ Architecture supports this; correctness bugs block production use
- "100+ concurrent cameras" — design target; correctness bugs break multi-camera in current code
- "AI-powered failure analysis" ✓ Claude Vision integration working
- "OSHA compliance automation" ✓ Form 300/300A, DART/TRIR, ITA e-submission implemented
- "12 production-ready phases complete" — feature breadth real; production readiness overstated given PROJECT_REVIEW bugs
- "Mobile PWA, wearable integration, integration hub (SAP, Maximo, Slack, Teams, Power BI, etc.)" ✓ All implemented
- "Edge inference, cloud aggregation, multi-tenancy, Stripe billing" ✓ All present; multi-tenancy isolation not enforced at DB level

---

## slm-forge

**GitHub URL:** https://github.com/rs1990/slm-forge  
**Primary Language:** Python 3.10+  
**Tech Stack:**
- UI: Gradio 4.44+
- Training (Apple Silicon): MLX-LM 0.20+
- Training (NVIDIA): PyTorch 2.2+ CUDA, Transformers 4.40+, PEFT 0.10+, TRL 0.8+
- Export: llama-cpp-python 0.3+ (GGUF), coremltools 8.0+ (Core ML), Optimum 1.20+ (ONNX), ai-edge-torch 0.2+ (TFLite), each in isolated virtual envs
- Data: Datasets 2.20+, Hugging Face Hub 0.24+
- RAG: sentence-transformers 3.0+
- Agents: Anthropic 0.30+, APScheduler 3.10+, PyGithub 2.1+
- Misc: fsnotify (file watching), catalog.yaml (27 curated models weekly auto-refresh)

**What it actually does:**  
Local fine-tuning workbench for small language models (135M–7B parameters). Takes raw data (JSONL/CSV/HF Hub/images), trains LoRA/QLoRA/DPO adapters on edge hardware (Apple Silicon via MLX, NVIDIA via CUDA, ARM fallback), quantizes, then exports to 5+ edge formats (GGUF, MLX, Core ML, ONNX, TFLite). Ships trained models to phones, Raspberry Pis, and Jetson Nanos without cloud infrastructure. One app replaces Axolotl + llama.cpp + separate quantizer + separate converter.

**Architecture:**
```
app.py — Gradio orchestrator, 12 tabs, per-session unlock gate, system banner hardware probe
core/
├─ backend.py — BackendProfile, detects MLX/CUDA/MPS/CPU, caches best engine
├─ trainer.py — MLX subprocess wrapper, loss/memory/tokens-per-sec parser, TrainHandle
├─ trainer_cuda.py — CUDA subprocess builder (accelerate launch cmd), single/multi-GPU
├─ _cuda_train_script.py — subprocess entry for CUDA (SFTTrainer/DPOTrainer via TRL)
├─ trainer_classify.py — classification-head trainer (bypass LoRA for tabular/tagging)
├─ exporter/__init__.py — 6 modular exporters (GGUF, MLX, CoreML, ONNX, TFLite, Ollama)
├─ distiller.py — knowledge distillation (local or HF Inference API teacher)
├─ dataset_detector.py — modality scoring (keyword heuristics → LLM fallback)
├─ dataset_normalizer.py — maps columns to canonical chat format (system/user/assistant/DPO)
├─ quantizer.py — 4/8-bit weight quantization, TurboQuant KV-cache compression
├─ rag_*.py — sentence-transformer indexing + retrieval for RAG context
├─ run_registry.py — JSON metadata + metrics logging for reproducibility
├─ resilience.py — pre-flight RAM/disk checks, atomic writes, OOM auto-shrink retry
├─ system_probe.py + system_tiers.py — RAM/VRAM detection, tier S/A/B/C/D, model recommendation
├─ session_auth.py — per-session unlock (SLMFORGE_PASSWORD env var, secrets.compare_digest)
├─ agents.py — Claude API triage agents for auto-issue-filing (optional)
└─ scheduler.py — APScheduler background jobs (hourly triage, weekly digest)
catalog.yaml — 27 curated edge models with train/inference RAM budgets
```

**Key Technical Features:**
1. Hardware-adaptive backend routing: detects Apple Silicon (MLX/Metal primary), NVIDIA CUDA (multi-GPU via accelerate), PyTorch MPS, CPU-only; auto-recommends model based on available RAM
2. Pre-flight OOM prevention: every catalog model tagged with train_ram_gb + inference_ram_gb_q4; trainer refuses jobs that won't fit — no silent OOM
3. Multi-format edge export: same LoRA adapter exports to GGUF (Q4_K_M/Q5_K_M/Q8_0/F16), MLX (iOS/macOS), Core ML (App Store), ONNX (cross-platform), TFLite (Android), Ollama Modelfile, HF Hub publish
4. LoRA + QLoRA + DoRA + PiSSA + LoRA+ training modes; SFT or DPO preference alignment; 4-bit/8-bit auto-enabled when VRAM tight
5. Knowledge distillation: local teacher or HF Inference API teacher → synthetic data → small student model
6. Classification-head trainer: bypass generative LoRA, train tiny classifier for document/email triage
7. Modality auto-detection: JSONL/CSV/Parquet/HF Hub → keyword heuristics + LLM fallback (Claude) → modality scoring + column mapping for text/code/vision/DPO/reasoning/time-series
8. Subprocess event streaming: loss/memory/tokens-per-sec parsed from stdout into queue; UI displays live metrics without polling

**Data Flow:**
1. User uploads dataset (JSONL/CSV/Parquet/folder) via Dataset tab
2. `dataset_detector.py` scores modality via keyword heuristics → optional Claude LLM if ambiguous
3. `dataset_normalizer.py` maps columns to canonical format (system/user/assistant/rejected, or image+text for vision)
4. Optional packing: short examples grouped into longer sequences for token efficiency
5. Training:
   - MLX path: subprocess launches `mlx_lm.lora` with YAML config, streams `Iter N: Train loss X, Val loss Y` → parsed to TrainEvent objects → UI queue
   - CUDA path: `_cuda_train_script.py` subprocess (TRL SFTTrainer or DPOTrainer) → same event stream
6. LoRA adapter saved to `runs/<timestamp>/adapter/`
7. Quantization (optional): 4-bit/8-bit weight quant → `runs/<timestamp>/quantized/`
8. Export: each format converter loads base model + adapter, fuses (optional), exports to target binary
9. Validation: Benchmark tab loads exported model, reports peak RAM + tokens/sec on device profile (Pi 4/5, Jetson Orin, M2 MacBook, ESP32)
10. Chat: Inference tab tests model (MLX or GGUF backend) with optional RAG retrieval

**Key Files:**
- `app.py` (295 LOC) — Gradio app entry, system banner, telemetry consent, per-session unlock, 12 tabs
- `core/backend.py` (94 LOC) — BackendProfile dataclass, detects + caches best training engine
- `core/trainer.py` (370 LOC) — MLX subprocess wrapper, loss/memory parser, TrainHandle with escalating SIGTERM→SIGKILL
- `core/trainer_cuda.py` (180 LOC) — CUDA subprocess builder, `accelerate launch` cmd, single/multi-GPU + quantization overrides
- `core/_cuda_train_script.py` (320 LOC) — subprocess entry, SFTTrainer/DPOTrainer, emits progress lines
- `core/dataset_detector.py` (240 LOC) — modality scoring (keyword heuristics → LLM fallback), returns (modality, suggested_column_map)
- `core/quantizer.py` (160 LOC) — 4/8-bit weight quantization config, TurboQuant emit
- `core/exporter/__init__.py` (95 LOC) — modular exporter registry, format availability checks, isolated venvs
- `catalog.yaml` — 27 curated edge-friendly models with hardcoded RAM budgets, weekly auto-refresh
- `core/system_tiers.py` (87 LOC) — maps detected hardware → tier letter S/A/B/C/D, auto-recommends starter model

**Real Metrics / Numbers from Code:**
- 27 catalog models spanning 39M–7B parameters (SmolLM2-135M up to Phi-3.5-mini)
- 6 export formats: GGUF (Q4_K_M, Q5_K_M, Q8_0, F16), MLX, Core ML, ONNX, TFLite, Ollama Modelfile
- 5 LoRA variants: LoRA, QLoRA, DoRA, PiSSA, LoRA+
- 2 training modes: SFT + DPO (no RLHF/PPO)
- 4 modalities: text, code, vision (mlx-vlm), time-series
- 4 hardware tiers: S (32+ GB workstation), A (16-32 GB pro laptop), B (7-16 GB entry laptop), C (<7 GB / CPU-only)
- TurboQuant: 5–6× KV-cache compression (experimental, Apple Silicon only)
- 101 unit tests in CI (0.20s on CPU); CUDA e2e validator runs on every push
- "Smoke test: ~5 minutes from raw dataset to quantized edge artifact on most hardware" (README claim)

**External Integrations / APIs Used:**
- Hugging Face Hub (dataset pull, model catalog, publish adapter, HF Inference API for distillation teachers)
- Anthropic Claude API (optional: agents for issue triage, modality detection fallback, requires ANTHROPIC_API_KEY)
- GitHub API (optional: auto crash-report filing, requires GITHUB_TOKEN)
- APScheduler (hourly/weekly background triage jobs)
- llama.cpp (GGUF inference reference), Ollama (Modelfile export target)
- Keyring (secure HF token storage)
- Sentence-transformers (RAG embedding backend)
- MLX-community (catalog weekly refresh source)

**What Makes It Distinctive:**
1. Single-machine end-to-end pipeline: no cloud, no multi-node — fine-tune → quantize → export all in one Gradio UI
2. Apple Silicon native priority: MLX-LM is primary trainer (Metal/unified memory); CUDA is secondary — most benchmarks on M1/M2
3. Hardware-tier classification: probes actual RAM/VRAM at startup, maps to coarse tier (S/A/B/C/D), refuses training that won't fit instead of OOM-ing mid-run
4. Multi-format export from one adapter: same LoRA output → 5 binary targets (GGUF + MLX + Core ML + ONNX + TFLite) — no separate toolchain per platform
5. Curated model catalog: 27 hand-picked models with RAM contracts, auto-refreshes from mlx-community weekly
6. CI-enforced subprocess contracts: `validate_cuda_e2e.py` runs full CUDA path on every commit; catches regex/formatter regressions (previously dropped val_loss silently)
7. Per-session unlock for server mode: browse as anon, unlock with password for compute-heavy ops; no cloud account ever required
8. Subprocess event streaming: live loss/memory/tokens-per-sec without polling — regex parses stdout into TrainEvent queue

**Honest Status:**  
Production-ready on Apple Silicon (MLX path fully battle-tested). CUDA path is beta: CPU-mode e2e validated in CI on every push; live GPU validation pending self-hosted runner. All SECURITY.md audit fixes merged (env whitelist, secret masking, image sandbox, server auth, sanitized crash reports, opt-in trust_remote_code). 101 unit tests passing. Remaining items (subprocess watchdog, hardcoded model IDs, admin-port auth, launcher duplication) are polish, not blockers. Known gaps: single-machine only (no multi-node), LoRA/QLoRA only (no full fine-tune), TurboQuant experimental.

**README Claims:**
- "Fine-tune small open-weight models (135M–7B parameters) on whatever hardware you have — Apple Silicon, NVIDIA CUDA, or CPU" ✓
- "Export to 5 edge formats: GGUF, MLX, Core ML, ONNX, TFLite, plus Ollama Modelfile & HF Hub publish" ✓
- "Pre-flight RAM checks refuse to start training that won't fit" ✓
- "Multi-format pipeline in one app — no juggling Axolotl + llama.cpp + separate quantizer + separate converter" ✓
- "Catalog-driven: 27 curated edge models auto-refreshed weekly from mlx-community, RAM-tagged against your machine" ✓
- "LoRA / QLoRA / DoRA / PiSSA / LoRA+" ✓
- "DPO for preference alignment in addition to SFT" ✓
- "Smoke test: ~5 minutes from raw dataset to quantized edge artifact on most hardware" — plausible on Apple Silicon; CUDA path beta
- "167 unit tests + 5-min UI smoke run recommended before release" — README says 167; code has 101 passing

---

## OptimaLLM

**GitHub URL:** https://github.com/rs1990/OptimaLLM  
**Primary Language:** Go 1.26  
**Tech Stack:**
- Core: Go 1.26
- Storage: BadgerDB v4.9.1 (knowledge graph + persistent cache), Ristretto v2.2.0 (in-memory LRU cache)
- Hashing: xxhash v2.3.0 (cache keys), google/uuid v1.6.0 (node IDs)
- File watching: fsnotify v1.9.0
- Config: yaml.v3
- Optional: Python 3.9+ (tree-sitter code indexing, pdfminer PDF extraction, pytesseract OCR), Ollama (embedding-based semantic search)
- Frontend: Three.js WebGL (3D knowledge graph viz, embedded HTML)

**What it actually does:**  
Local reverse proxy (127.0.0.1:7777) that intercepts Claude Code CLI requests to Anthropic's API. Applies a 13-step automatic optimization pipeline per request: intent classification, model routing, context pruning, text deduplication, tool-result compression, image/PDF stripping, bash-result caching, knowledge-graph context injection. Reduces token usage 15–90% depending on workload (measured 67% on a 635-request session). Zero workflow changes required — Claude Code forwards transparently through the proxy. Builds a persistent local knowledge graph from indexed code + extracted conversation facts.

**Architecture:**
```
cmd/daemon — HTTP proxy entry point (:7777), single-instance guard (flock)
cmd/optimallm — Management CLI (status, stats, graph index/serve/backup/restore, reindex)
internal/proxy/
├─ orchestrator.go (2,245 LOC) — full 13-step transform pipeline, request/response transform
├─ proxy.go — server setup, reverse proxy director, response modifier
├─ logger.go — TokenLogger, SSE stream parsing, token extraction, starvation detection
├─ auth.go — proxy bearer token management
└─ batches.go — batch API deduplication
internal/classifier — Haiku-based intent classification (7 modes: CONVERSATION/CODE_EXECUTION/
   CODE_ANALYSIS/QUERY_GRAPH/WRITING/PLANNING/Vision), 60s cache, confidence escalation
internal/router — model routing table per intent (Haiku→Sonnet→Opus by capability tier), OpenRouter vendor-prefixing
internal/memory — 4-tier memory manager:
   Tier 0: noise stripping (hedges, temporal refs, pleasantries)
   Tier 1: session working memory, 2h TTL
   Tier 2: project facts, LRU-bounded 50 entries, persistent
   Tier 3: BadgerDB knowledge graph
internal/graph — BadgerDB backend + hybrid query (0.38 embedding + 0.33 BM25 + 0.14 recency
   + 0.10 provenance + 0.05 hit-count), fsnotify file watcher, Three.js 3D viz server,
   optional Ollama embedder, remote HTTP backend support
internal/guardrails — pre-flight destructive command detection + post-response starvation phrase detection
ingest/ingest.py — Python subprocess for code indexing (tree-sitter AST, 13 languages) + PDF/image OCR
shim/install.sh — one-command installer (build, launchd/systemd service registration)
```

**Key Technical Features:**
1. Text deduplication: SHA256-keyed exact-match replacement of repeated message blocks (≥40 chars) with "[previously included in turn N]" references; 55–80% savings on Q&A-heavy sessions
2. Tool-result compression: bash output cache (2h TTL, key: SHA256(cmd+cwd)) + aggressive truncation to last 2 turns; deterministic commands (ls, git log, cat) reuse cached output; 50–75% savings on repeated bash
3. Image stripping + OCR ingestion: base64 payload always stripped (saves 25K–125K tokens/image); Python-async OCR indexes into graph when Python available
4. Bash result caching: SHA256(command+cwd) keyed, 2h TTL; excludes non-deterministic (date, curl, npm test); cache hits bypass Anthropic round-trip entirely
5. Auto project indexing: first visit to directory triggers background goroutine indexing all code files (tree-sitter, 13 languages) into BadgerDB; queryable within 10–30s, zero API calls
6. Tier-0 noise stripping: removes hedges ("I think", "as an AI"), temporal refs, pleasantries; ~15–25% output-token reduction
7. Intent classification (7 modes): Haiku 60s-cached classification determines cheapest capable model; Haiku for CONVERSATION, Sonnet for CODE, Opus for PLANNING/Vision
8. Starvation detection + self-healing cascade: detects 12 starvation phrases; on next turn, auto-escalates model tier (Haiku→Sonnet→Opus) without user intervention

**Data Flow:**
```
User: claude "refactor auth module"
→ launch.sh sets ANTHROPIC_BASE_URL=http://127.0.0.1:7777
→ Claude Code CLI: POST /v1/messages (JSON + SSE streaming)
→ Proxy daemon on :7777
→ Parse request: load/create session memory (from session registry or root header)
→ Classify intent: Haiku side-call (60s cache) → mode (e.g., CODE_EXECUTION)
→ 13-step transform pipeline:
   1. Dedup old messages (SHA256 blocks)
   2. Compress tool results (bash cache lookup, 2-turn window)
   3. Intercept images (strip base64, OCR if Python available)
   4. Intercept PDFs (normalize blocks, async graph ingest)
   5. Cache bash deterministic results
   6. Detect project + auto-index if new directory
   7. Strip Tier-0 noise (hedges, temporal)
   8. Apply classified intent (cached)
   9. Route to cheapest capable model (Haiku/Sonnet/Opus)
   10. Cap max_tokens per intent (CONVERSATION→512, QUERY_GRAPH→256, WRITING→2048)
   11. Inject terse-output system directive
   12. Query graph → inject ≤15 ranked context nodes (hybrid score)
   13. Apply Anthropic prompt-cache control header
→ Rewrite JSON (model, system, messages, max_tokens, thinking budget)
→ Forward to Anthropic API (or OpenRouter) — preserve stream:true
→ Tee SSE response back to Claude Code + parse token counts
→ Post-response async: extract facts → BadgerDB, record hit-counts, detect starvation,
   account savings (6 counters), check daily budget (warn 80%/90%, block 100%)
→ Log per-request TSV: session, model, intent, input_tokens, output_tokens, duration, savings
```

**Key Files:**
- `cmd/daemon/main.go:17–65` — Entry point, port flag, daemon lock, config reload signal handler
- `internal/proxy/orchestrator.go:1–150` — Session + classification + full 13-step pipeline (2,245 LOC)
- `internal/proxy/proxy.go:54–115` — Server setup, reverse proxy director, response modifier
- `internal/proxy/logger.go:40–100` — TokenLogger, SSE stream parsing, token extraction, starvation detection
- `internal/classifier/classifier.go:39–77` — Classify function, Haiku side-call, 7-mode enum, confidence + escalation
- `internal/router/router.go:40–70` — Routing table (mode → model decision), OpenRouter prefixing
- `internal/memory/memory.go:79–150` — 4-tier Manager, Tier-0 stripping, Tier-1 working memory, Tier-2 project facts
- `internal/graph/store.go:1–100` — BadgerDB backend, node/relation schema, provenance enum (EXTRACTED/INFERRED/AMBIGUOUS/SYNTHESIZED)
- `internal/graph/query.go` — Hybrid ranking (5 signals), top-15 node selection
- `shim/install.sh:1–60` — One-command install: build binaries, detect OS/arch, register launchd/systemd

**Real Metrics / Numbers from Code:**
- Measured session: 635 requests, 295K baseline → 98K actual (~67% reduction)
- Workload-dependent range: 15–90% (short queries 10–25%, image-heavy 70–90%, code-analysis 40–65%, conversation 55–80%, planning 25–50%)
- Classifier: Haiku call adds 300ms–2s latency per turn; 60s cache hit rate ~89% measured
- Graph performance: index latency 10–30s first visit (background); query injection adds 300–1.5K input tokens
- Transform pipeline overhead: <200ms; SSE tee + parse adds <50ms
- Max-tokens cap: QUERY_GRAPH→256, CONVERSATION→512, WRITING→2048
- Extended thinking budget: 8,192 tokens added to PLANNING requests
- Starvation cascade: 12 detection phrases; measured 2 auto-escalation events in 635-request session
- Supported models: Anthropic (Haiku 4.5, Sonnet 4.6, Opus 4.7); OpenRouter (DeepSeek, OpenAI, others)
- Knowledge graph tested: up to 2,000 nodes; 3D viz loads <5s on modern browser

**External Integrations / APIs Used:**
- Anthropic API `https://api.anthropic.com/v1/messages` (default upstream)
- OpenRouter `https://openrouter.ai/api/v1/messages` (via OPTIMALLM_UPSTREAM_BASE env var; auto vendor-prefixes models)
- Ollama (optional local embedding server, nomic-embed-text, for semantic dedup + hybrid graph ranking)
- Python subprocess: tree-sitter-languages (code AST), pdfminer.six (PDF), pytesseract (OCR), pillow — opt-in, auto-disabled if unavailable
- BadgerDB (embedded, no external DB), fsnotify (file watching)
- launchd (macOS) / systemd user service (Linux) for auto-start

**What Makes It Distinctive:**
1. 100% local (except Anthropic/OpenRouter API calls): all indexing, caching, classification, memory, graph on-machine — data never leaves
2. Streaming-aware proxy: preserves SSE chunking (FlushInterval: -1); doesn't buffer full response before forwarding
3. 4-tier memory model: Tier-0 (noise/per-request) → Tier-1 (session 2h, auto-compact at 8K tokens) → Tier-2 (project LRU 50 entries) → Tier-3 (BadgerDB persistent graph)
4. Hybrid graph ranking: combines 5 signals (embedding cosine 0.38, BM25 0.33, recency 0.14, provenance 0.10, hit-count 0.05); provenance gates CODE_EXECUTION to EXTRACTED nodes only
5. Starvation detection + deferred LLM cascade: 12 phrases detected; escalates model tier on next turn without disrupting current SSE stream
6. Provenance-tagged knowledge graph: EXTRACTED (AST), INFERRED (LLM), AMBIGUOUS (potential secret), SYNTHESIZED (cross-session promoted)
7. Tool-result-aware bash caching: distinguishes deterministic (ls, git log) from non-deterministic (date, curl, npm test) — safe to replay from cache
8. OpenRouter transparent support: single OPTIMALLM_UPSTREAM_BASE env var switches entire routing

**Honest Status:**  
Prototype/MVP with serious production-blocking bugs. Unit tests pass (classifier, graph, memory, guardrails, router all green), but PROJECT_REVIEW.md identifies 3 critical bugs: (1) request re-marshaling drops `stream` field → breaks SSE; (2) destructively replaces tool_results messages → causes upstream 400s in agentic loops; (3) canonicalizes prompt to single phrase, losing 90% of user intent. Additional HIGH bugs: auto-compact breaks tool_use/tool_result pairing; budget/graph-only responses return plain JSON to streaming clients. No shadow mode, unbounded memory growth (caches never evicted), session-ID mismatch breaks 3 feedback features. Security: broken proxy-token design (forwards proxy secret upstream), unauthenticated /ingest + /graph/* control surface, graph viz defaults 0.0.0.0:7778. README claims accurate *if* CRIT bugs are fixed; current code will break streaming and corrupt agentic conversations.

**README Claims:**
- "A transparent reverse proxy that optimizes every Claude Code request before it reaches Anthropic's API" ✓ Accurate architecture
- "Token usage drops 15–90% depending on workload" ✓ Measured 67% on one 635-request session; range realistic
- "One measured session: 635 requests, 295K baseline → 98K actual (~67% reduction)" ✓ Exact metric from code
- "Zero workflow changes required — Claude Code works exactly as before" ✗ Currently fails on agentic (tool-heavy) workflows due to CRIT bugs #1 & #2
- "100% local — no third-party servers beyond Anthropic's standard API" ✓
- "2 minutes installation; includes auto-start on login" ✓ shim/install.sh does this
- "13-step optimization pipeline" ✓ All 13 steps implemented and listed

---

## SitAware

**GitHub URL:** https://github.com/rs1990/SitAware  
**Primary Language:** Python (backend) + TypeScript (frontend)  
**Tech Stack:**
- Backend: FastAPI 0.115.0 + uvicorn, asyncpg 0.29.0 + SQLAlchemy 2.0.36, Redis 5.1.1, httpx 0.27.2, APScheduler 3.10.4, Anthropic 0.39.0 (claude-haiku-4-5 + claude-sonnet-4-6), slowapi 0.1.9
- Frontend: Next.js 14.2.3 + React 18 + TypeScript 5, MapLibre GL 4.3.2 + react-map-gl 7.1.7, SWR 2.2.5, Tailwind CSS 3.4.1
- Infra: PostgreSQL 16 + PostGIS 3.4 (Docker), Redis 7-alpine, LibreTranslate (self-hosted AGPL), Docker Compose

**What it actually does:**  
Global situational-awareness interactive map aggregating ~25 open data sources (USGS earthquakes, GDACS multi-hazard, NASA FIRMS wildfire, ACLED conflict, ReliefWeb humanitarian, aviation/maritime positions, weather alerts, ocean currents) into a unified PostGIS incident table. Viewport-filtered GeoJSON served to a Next.js + MapLibre frontend. Two-stage Claude pipeline (Haiku structured extraction → Sonnet narrative) generates regional summaries under hard cost budgets (45 summaries/day, 490K tokens/day, 5 per session).

**Architecture:**
```
Backend (FastAPI):
├─ 11 ingestion connectors (disaster, conflict, aviation, maritime, weather,
│   ocean, population, news + base.py abstraction)
├─ 6 API routers: /sources, /incidents, /tiles, /poptiles, /summary, /news
├─ APScheduler in-process cron (cadence per source: 5min earthquakes → 24hr UCDP)
├─ Redis caching (ephemeral position snapshots 15min–1h, AI summaries 6h,
│   session/daily token counters)
└─ PostGIS incidents table (unified schema), source_runs_log audit trail

AI Pipeline:
├─ POST /summary → Haiku structured extraction → Sonnet narrative
├─ Cached by SHA-256(geohash_L5 + sorted incident IDs + days window)
└─ Layered degradation: AI → Haiku-only → extractive → cached stale → plain incidents

Frontend (Next.js 14):
├─ SitMap (MapLibre GL) — viewport, layer rendering, event handlers
├─ LayerPanel — 8 category toggles (disaster, conflict, accident, aviation,
│   maritime, weather, ocean, population)
├─ TimeWindow — 7/15/30 day selector
├─ SummaryModal — AI regional summary display
├─ SourcesBar — per-source health status
└─ Client-side persistence: localStorage for layer state, time window, theme, sidebar
```

**Key Technical Features:**
1. Viewport-based GeoJSON tile queries with PostGIS bbox filtering (no full-table scans)
2. Two-stage AI summary pipeline: Haiku structured extraction → Sonnet narrative; cached by geohash + incident hash + days window — exploits spatial/temporal locality for cache reuse
3. Hard token/cost budgeting: Redis atomic reserve/rollback, session caps (5 calls), daily ceiling (490K tokens), layered degradation (AI → Haiku-only → extractive → empty) — user never sees raw "budget exceeded" error
4. Hash-based upsert deduplication with xmax=0 insert detection (PostgreSQL internal visibility to distinguish true inserts from no-op updates)
5. Unified incident schema across 25 heterogeneous sources (normalizes earthquakes/wildfire/conflicts/aviation/ships into common geometry/severity/confidence)
6. Machine translation pipeline: auto-detect non-English, queue to LibreTranslate (self-hosted), store original lang + English with confidence downgrade (0.9)
7. Per-connector Redis snapshot caching: aviation/maritime write to Redis without DB (ephemeral positions); durable events only in PostGIS
8. Geohash-level cache keys: summaries reused across similar viewport positions without recomputation

**Data Flow:**
1. Frontend viewport change → GET /api/v1/tiles/{layer}/{z}/{x}/{y}
2. Redis cache check → miss → PostGIS bbox query (ST_Intersects) → GeoJSON response → cache SET
3. User clicks "Summarize region" → POST /api/v1/summary {bbox, days}
4. Redis: atomic reserve(tokens_needed) → check session cap (5) + daily ceiling (490K) → proceed or degrade
5. Fetch ≤80 incidents for bbox (200 char truncated descriptions), sort by severity
6. Stage 1: Haiku structured extraction → JSON (categories, severity, affected_areas, key_incidents)
7. Stage 2: Sonnet narrative → human-readable regional summary
8. Cache result under SHA-256(geohash_L5 + sorted_incident_ids + days); Redis.adjust() reconciles actual vs forecast tokens
9. Return to frontend → SummaryModal display

**Key Files:**
- `backend/app/main.py` — FastAPI entry, router setup, health endpoint, APScheduler lifespan
- `backend/app/ingestion/base.py` — BaseConnector abstract class, IngestionResult dataclass, fetch→normalize→translate→upsert pipeline
- `backend/app/db/crud.py` — PostGIS upsert_incidents() with hash-based change detection, bulk insert/update
- `backend/app/cache/redis_client.py` — atomic token reserve/rollback, session/daily counter management
- `backend/app/api/summary.py` — POST /summary, two-stage Claude pipeline, budget enforcement, caching
- `backend/app/api/tiles.py` — GET /tiles/{layer}/{z}/{x}/{y}, viewport GeoJSON, Redis caching
- `backend/app/config.py` — Pydantic Settings; all DB/Redis/Anthropic/rate limit/TTL env vars
- `backend/app/scheduler.py` — register_jobs() wires all 11 connectors to APScheduler with per-source cadence
- `frontend/src/components/SitMap.tsx` — MapLibre GL viewport, layer rendering, event handlers
- `frontend/src/app/page.tsx` — Next.js root, component composition, localStorage persistence

**Real Metrics / Numbers from Code:**
- 25 data sources integrated
- 6 API routers, ~16+ endpoints total
- 80 incidents max per region summary (enforced in code)
- 200 char description truncation per incident
- 2 AI models: Haiku (extraction), Sonnet (narrative)
- 4,000 max tokens per AI call
- 490,000 daily token ceiling (with 10K safety margin)
- 45 AI summaries per day max
- 5 AI summaries per session max
- 8 incident categories (disaster, conflict, accident, aviation, maritime, weather, ocean, population)
- 11 active connectors
- 13+ frontend components, 7+ hooks
- 2,601 backend LOC (ingestion module)
- Scheduling cadence: 5min (earthquakes) → 24hr (UCDP)

**External Integrations / APIs Used:**
- Anthropic: claude-haiku-4-5-20251001 (extraction), claude-sonnet-4-6 (narrative)
- USGS (GeoJSON), GDACS (GeoJSON/RSS), NASA FIRMS (CSV/GeoJSON), Global Volcanism Program (JSON), PTWC (RSS/XML)
- Copernicus EMS (GeoJSON/XML)
- ACLED (CSV/JSON), ReliefWeb (JSON), UCDP GED (JSON), ACAPS (CSV)
- OpenSky (REST JSON; Basic auth deprecated), ADS-B Exchange (JSON free tier)
- VesselFinder (REST JSON free tier)
- NOAA NWS (GeoJSON), Open-Meteo (JSON), GOES satellite imagery (PNG tiles)
- CMEMS (NetCDF), NOAA CoastWatch/ERDDAP (NetCDF/CSV), HYCOM (NetCDF)
- GeoNames cities15000 (TSV), GHS-POP R2023A (GeoTIFF)
- LibreTranslate (self-hosted AGPL, non-English translation)
- Reuters/AP RSS (defunct — feeds discontinued)
- PostgreSQL + PostGIS, Redis (Upstash or local), slowapi (rate limiting)

**What Makes It Distinctive:**
1. Atomic token budgeting: Redis reserve() checks cap, rollback() on failure, adjust() reconciles actual vs forecast — enforces budget without race conditions
2. Hash-based upsert with xmax=0 insert detection: uses PostgreSQL internal visibility to distinguish true inserts vs no-op updates (prevents duplicate-count inflation)
3. Geohash-level caching: summaries cached by SHA-256(geohash_L5 + sorted incident IDs + days), exploits spatial locality for cache reuse
4. Layered degradation without errors: AI → Haiku-only → extractive bullets → cached stale → plain incidents; transparent to user
5. Per-connector snapshot caching: aviation/maritime write ephemeral positions to Redis only; PostGIS only stores durable events
6. Bidirectional translation pipeline: non-English auto-detected, queued to LibreTranslate (self-hosted), stored with original lang + confidence downgrade
7. Docker-first architecture: entire stack (Postgres, Redis, LibreTranslate, backend) in single docker-compose.yml; `docker-compose up + npm run dev`

**Honest Status:**  
Prototype/MVP v0.4.0. Core architecture and pipelines (PostGIS schema, Redis token budgeting, AI summary pipeline, APScheduler connectors, MapLibre frontend) are correctly implemented. However, 12+ known bugs including permanently broken data layers: weather layer empty (cache key mismatch — OpenMeteo writes `tile:weather:openmeteo:current`, endpoint reads `tile:weather:global`); ocean layer incomplete (CMEMS writes `tile:ocean:current:cmems`, endpoint reads `tile:ocean:currents`). Dead connectors: Reuters/AP RSS feeds discontinued (2 of 5 news sources dead), ACLED uses retired legacy auth, OpenSky Basic auth deprecated. UCDP connector sets all records to expired immediately on ingest (date_end bug). News endpoint has no rate limit + no token budget → cost DoS surface. No Alembic migrations (schema only via docker-entrypoint SQL). No incident TTL/expiry job (table grows unbounded). Suitable for local/private use; not production-ready.

**README Claims:**  
No README.md found. Documentation in PROJECT_REVIEW.md, DESIGN.md, DATA_SOURCES.md, COST_ANALYSIS.md, OPERATIONS.md, DEVLOG.md. PROJECT_REVIEW.md states: "SitAware is a global situational-awareness map: a FastAPI backend ingests ~25 open data sources... serves viewport-filtered GeoJSON to a Next.js 14 + MapLibre frontend." All descriptions above are sourced from these internal docs.

---

## AurumAi

**GitHub URL:** https://github.com/rs1990/AurumAi  
**Primary Language:** Python (backend) + TypeScript (frontend)  
**Tech Stack:**
- Backend: FastAPI 0.111+, Pydantic 2.7+, SQLAlchemy 2.0+ (asyncio, asyncpg), Alembic, Anthropic SDK 0.26+, Redis 5.0+, APScheduler 3.10+, Prometheus client 0.20+, OpenTelemetry 1.23+, arq 0.25+ (background jobs), boto3 1.34+ (S3/AWS), cryptography 42+, PyJWT 2.8+, python3-saml 1.16+
- Frontend: Next.js 15.3+, React 19, Radix UI, Tailwind CSS 3.4+, TypeScript 5

**What it actually does:**  
Enterprise AI execution control plane — a governance layer between users and LLMs. Decomposes user intent into deterministic task graphs, routes tasks to SLMs (temperature=0, structured work) or Claude (reasoning/generative), enforces policy pre/post execution, generates HMAC-SHA256-signed immutable audit trails, tracks cost per team, and captures human decisions for continuous improvement (supermemory). Designed for organizations that need auditable, cost-bounded, policy-enforced AI without autonomous agents.

**Architecture:**
```
Request Path:
Middleware stack (OIDC, API key auth, idempotency, rate limit, tenant scoping)
→ Domain routers (invoke, admin, KG, decisions, code execution, health)
→ Control plane:
   ├─ decomposer.py — user input → task graph (DAG)
   ├─ task_graph.py — Task/TaskGraph classes, DAG execution
   ├─ policy_engine.py — pre/post policy enforcement (YAML-driven)
   └─ context_resolver.py — minimal context from KG + supermemory
→ Model router:
   ├─ SLM pipeline (intent classifier, extractor, validator, formatter; all temp=0)
   └─ General LLM (Claude) for reasoning tasks + confidence fallback
→ Observability:
   ├─ audit_log.py — HMAC-SHA256 signing + immutable logging
   ├─ outcome/decision/contextual snapshot memory (90-day retention, per-team)
   ├─ cost accounting per invocation/team
   ├─ trust signals + drift detection (hourly scheduled job)
   └─ webhooks (5 event types, arq worker, retry schedule)

Database (PostgreSQL + SQLAlchemy asyncio):
├─ Invocation, Task, Policy, KGNode, Outcome, Decision, CostRecord, 30+ tables
└─ 1 Alembic migration

Frontend (Next.js 15 App Router):
└─ API proxy catch-all + React 19 UI
```

**Key Technical Features:**
1. Single `/invoke` endpoint + SSE streaming; deterministic task graph with pre/post policy enforcement
2. Model role separation: SLMs (temp=0) for structured tasks (classify/extract/validate/format), Claude for reasoning/generation — separates determinism from intelligence
3. HMAC-SHA256-signed immutable audit logs — cryptographic audit trail per invocation
4. Outcome + Decision + Contextual snapshot supermemory (3 types, 90-day retention, per-team, bounded to 100 records per team/task-type)
5. Knowledge Graph: manual ingestion, keyword matching, optional embeddings; minimal context resolution (KG + Supermemory) to reduce tokens 60–75% vs full RAG
6. Per-tenant API keys + basic RBAC (admin/user/reviewer)
7. Cost attribution per invocation/team with per-model breakdown
8. Webhook events (5 types: invocation.completed, invocation.failed, policy.violation, drift.alert, cost.threshold) with retry schedule via arq
9. Piston sandboxed code execution (70+ languages) as a task type
10. Policy-as-YAML: versioned policies with rollback + impact dry-run; pre/post enforcement per team

**Data Flow:**
1. User submits `{tenant_id, team_id, user_id, user_input, context}` → POST /invoke
2. Auth middleware: API key or OIDC → tenant scoping (note: currently no-op bug)
3. Intent Classifier SLM identifies task type (classify/extract/validate/format/reason)
4. Context Resolver queries KG + Supermemory → minimal context assembly
5. Pre-execution policy check (block if team not allowed per YAML policy)
6. Model Router: SLM (temp=0, low cost, deterministic) or LLM (reasoning, confidence fallback)
7. Task Graph executes tasks in dependency order
8. Post-execution policy validation (output schema, confidence threshold check)
9. Cost Accounting: per-model breakdown calculated
10. Audit Log: HMAC-SHA256 signs full trace → stored immutably
11. Outcome Memory: approved results recorded for future context
12. Response: `{final_response, task_graph, confidence_summary, trust_signal, cost_usd, audit_signature}`

**Key Files:**
- `control-plane/main.py` — FastAPI app init + CORS + middleware stack
- `api/routes.py` — `/invoke`, `/invoke/stream`, `/invocations` list/get
- `control_plane/decomposer.py` — user input → task graph
- `control_plane/task_graph.py` — Task/TaskGraph classes, DAG execution
- `control_plane/policy_engine.py` — pre/post policy enforcement
- `models/router.py` — SLM vs LLM selection + confidence fallback
- `observability/audit_log.py` — HMAC-SHA256 signing + immutable logging
- `api/middleware.py` — auth (API key + OIDC), tenant scoping, idempotency, rate limiting
- `database/models.py` — SQLAlchemy ORM (Invocation, Task, Policy, KGNode, Outcome, Decision, CostRecord, etc.)
- `frontend/src/app/` — Next.js 15 App Router with API proxy catch-all

**Real Metrics / Numbers from Code:**
- 19 test modules
- 62+ passing tests (per PROJECT_REVIEW)
- 4 SLM task types: classify, extract, validate, format
- 3 supermemory types: outcome, decision, contextual snapshot
- 9 core API endpoints (/invoke, /invoke/stream, /invocations, /admin/*, /kg/*, /execute, /health)
- 5 webhook event types
- Piston supports 70+ languages for sandboxed code execution
- 2 client SDKs (Python, JavaScript)
- 1 Alembic migration shipped
- Supermemory bounded: 100 records per team/task-type; 90-day retention

**External Integrations / APIs Used:**
- Anthropic Claude (primary LLM — haiku-4-5 / sonnet-4-6; settings drift between README and actual defaults)
- Piston code sandbox (Docker, port 2000)
- Redis (caching, rate limit), PostgreSQL (audit, supermemory)
- AWS S3 (audit export, cold archive), AWS KMS / Azure Key Vault (BYOK encryption)
- OIDC/OAuth 2.0 (Okta, Azure AD), SAML/SCIM (identity provisioning)
- Slack, GitHub, Confluence connectors (CLI scripts calling `/api/v1/kg/ingest`)
- Model-swap support: Anthropic → OpenAI-compatible → Ollama

**What Makes It Distinctive:**
1. Policy-as-YAML config (not code): versioned policies with rollback + impact dry-run; pre/post enforcement per team
2. Supermemory bounded design: 100 records per team/task-type prevents unbounded growth; 3 distinct memory types for different learning signals
3. Confidence-mismatch detection: flags when model is confident but human disagrees — drift signal for policy tuning
4. No autonomous agents by design: all tasks explicit, deterministic, user-triggered — auditability over capability
5. SLM determinism enforced (temperature=0) for structured tasks — reproducible outputs for compliance
6. Minimal context resolution (KG + Supermemory) targets 60–75% token reduction vs naive RAG
7. Drift detection scheduled job (hourly) watching decision mismatch rates
8. Model registry + evaluation harness for benchmarking SLM vs LLM routing decisions

**Honest Status:**  
MVP+1 (April 2026). Core API, task graph, SLM pipeline, policy engine, audit signing, knowledge graph, supermemory, cost accounting, trust signals, human decision capture all shipped. NOT production-ready due to critical security gaps:
- **Critical**: Tenant isolation unenforced (cross-tenant IDOR on all routes; TenantScopingMiddleware is no-op), admin RBAC missing on policy endpoints, OIDC broken (no discovery, claims unmapped, doesn't bypass APIKeyMiddleware), idempotency caches streaming responses (breaks SSE)
- **High**: Frontend Dockerfile broken (copies non-existent /app/dist), default secrets boot in prod (`master_api_key="dev-master-key"`), webhook delivery disabled by default (arq off), CORS empty list in prod
- **Medium**: Code execution endpoint open to any role, Terraform incomplete, Helm defaults unsafe (networkPolicy off, tag=latest)

**README Claims:**
- "Enterprise AI execution fabric with governance, auditability, and multi-tenant access control" — governance + auditability mostly working; multi-tenant isolation NOT enforced
- "FastAPI, PostgreSQL, Redis, Piston, SLM + LLM routing" ✓ All present
- "Policy Engine (YAML-driven, pre/post checks)" ✓ Functional
- "Audit Logging (immutable, HMAC-SHA256)" ✓ Functional
- "Cost Attribution per Team" ✓ Tracked per invocation; no dashboard UI yet
- "Streaming API: SSE" ✓ Implemented but idempotency middleware breaks it for caching
- "Code Execution: Piston sandbox 70+ languages" ✓ Working; not in compose by default
- "Webhook Events" ✓ Defined but delivery off unless arq worker running
- "SDKs: Python, JavaScript" ✓ Both exist but minimal
- "Anthropic Claude (claude-3-5-sonnet)" — README drift; settings.py defaults to "claude-haiku-4-5" / "claude-sonnet-4-6"

---

## supply-chain-intel

**GitHub URL:** https://github.com/rs1990/supply-chain-intel  
**Primary Language:** Python (backend) + TypeScript (frontend)  
**Tech Stack:**
- Backend: FastAPI 0.111.0, SQLAlchemy 2.0.30, Uvicorn 0.29.0, APScheduler 3.10.4, pandas 2.2.2, scikit-learn 1.4.2, numpy 1.26.4, Pydantic 2.7.1, psycopg2 2.9.9
- Frontend: React 19.2.6, TypeScript 6.0.2, Vite 8.0.12, Tailwind 4.3.1, Recharts 2.12.7, React Router 6.28.2
- Database: SQLAlchemy ORM (SQLite dev, PostgreSQL prod via Render)
- Deployment: Render (free tier web service + PostgreSQL)

**What it actually does:**  
PACCAR supply chain intelligence platform ingesting from 8 data sources (SAP, Ariba, CDK, Manhattan, Snowflake, TMS/Freight, Warranty, Demand) to compute daily KPIs (fill rate, OTD, production attainment, freight on-time) and weekly aggregates (inventory turns, forecast MAPE, costs). Detects warranty claim anomalies via z-score analysis and forecasts demand 14 days forward.

**Architecture:**
```
Backend Core:
├─ backend/main.py — FastAPI app with lifespan (init DB, run ingestion, compute metrics,
│   start scheduler), CORS middleware, SPA serving
├─ backend/config.py — Pydantic Settings for all 9 integrations (SAP, Ariba, By,
│   Manhattan, CDK, PACCAR Connect, Snowflake, SMTP, Slack)
├─ backend/database.py — SQLAlchemy engine, SessionLocal, DeclarativeBase,
│   init_db(), get_db() dependency
└─ backend/models.py — 13 tables: InventorySnapshot, SupplierOrder, ProductionOutput,
    DealerInventory, FreightShipment, WarrantyClaim, DemandActual, DailyMetric,
    WeeklyMetric, ConnectorLog

Ingestion Layer:
├─ backend/ingestion/pipeline.py — run_ingestion(db, since) orchestrates 7 connector
│   methods per data source; upserts via 7 functions (_upsert_inventory,
│   _upsert_supplier_orders, _upsert_production_output, _upsert_dealer_inventory,
│   _upsert_freight_shipments, _upsert_warranty_claims, _upsert_demand_actuals);
│   logs success/error/duration per connector
├─ backend/connectors/base.py — BaseConnector ABC with 7 abstract methods:
│   pull_inventory(), pull_supplier_orders(), pull_production_output(),
│   pull_dealer_inventory(), pull_freight_shipments(), pull_warranty_claims(),
│   pull_demand_actuals(), plus health_check()
├─ backend/connectors/mock.py — MockConnector generates realistic synthetic PACCAR
│   data (25 parts across 5 DCs, 4 plants, 10 suppliers, 8 dealers, 7 carriers)
│   with stochastic variance and 25% systemic spike on E-21403 turbocharger
└─ backend/connectors/sap.py — SAPConnector with OAuth2 token mgmt calling OData v4
    APIs (materialstock, purchase orders, production orders); freight/warranty/demand
    endpoints stubbed returning []

Metrics Engine:
├─ backend/metrics/daily.py — compute_daily_metrics(db, for_date) computes 13 KPIs:
│   fill_rate_pct, open_backorders, critical_backorders, supplier_otd_pct,
│   production_attainment_pct, freight_exception_count, freight_on_time_pct,
│   warranty_claims_count, warranty_cost, inventory_value, active_pos,
│   po_overdue_count; plus _build_alerts() for 5 thresholds
│   (fill <95%, critical backorders, OTD <90%, attainment <95%, exceptions >5)
└─ backend/metrics/weekly.py — compute_weekly_metrics(db, week_start) calculates
    12 aggregates: inventory_turns (COGS/avg_inv*52), forecast_mape, avg_fill_rate,
    avg_supplier_otd, avg_production_attainment, total_freight_cost,
    total_warranty_cost, total_po_value, excess_inventory_value,
    short_inventory_parts, top_warranty_part; _compute_mape() compares weekly
    demand to prior week moving average

ML Analytics:
├─ backend/ml/forecast.py — forecast_demand(db, horizon_days=14) per part/region:
│   fits 60-day linear trend + blends 40% trend / 60% 7-day MA,
│   applies 0.3 weekend factor
└─ backend/ml/anomaly.py — detect_warranty_anomalies(db, lookback_days=60,
    z_threshold=2.0) groups claims by part per day, compares 7-day recent vs
    60-day history z-score, returns ranked anomalies with severity
    "warning" (z>=2) or "critical" (z>=3)

API Layer:
├─ backend/api/routes_metrics.py — 10 endpoints:
│   GET /api/metrics/daily (30d history)
│   GET /api/metrics/daily/today (single day + alerts)
│   POST /api/metrics/daily/compute (manual trigger)
│   GET /api/metrics/weekly (12 weeks)
│   POST /api/metrics/weekly/compute
│   GET /api/metrics/anomalies
│   GET /api/metrics/forecast
│   GET /api/metrics/inventory/summary (by location)
│   GET /api/metrics/suppliers/performance (30d OTD)
│   GET /api/metrics/freight/summary (carrier OTD/cost)
│   GET /api/metrics/warranty/summary (top 15 parts)
└─ backend/api/routes_ingestion.py — 3 endpoints:
    POST /api/ingest/run (manual pull)
    GET /api/ingest/logs (connector run history)
    POST /api/upload/inventory (CSV bulk load)

Scheduler:
└─ backend/scheduler.py — APScheduler BackgroundScheduler:
    pull data every 15min, daily metrics 6am, weekly metrics Monday 7am

Frontend:
├─ frontend/src/App.tsx — React Router container, 3 pages (Daily, Weekly, Inventory),
│   dark theme sidebar nav
├─ frontend/src/pages/Daily.tsx — KPI cards (fill rate, OTD, attainment, on-time %)
│   + 30d trend line charts (recharts) + warranty anomalies table with z-score,
│   refresh button, alerts display
├─ frontend/src/pages/Weekly.tsx — weekly aggregates dashboard
├─ frontend/src/pages/Inventory.tsx — inventory positioning view
├─ frontend/src/api/client.ts — TypeScript interfaces (DailyMetric, WeeklyMetric,
│   Anomaly, SupplierPerf, FreightSummary, WarrantySummary) + api object with
│   11 methods (getToday, getDailyHistory, getWeeklyHistory, getAnomalies,
│   getSupplierPerf, getFreightSummary, getWarrantySummary, getInventorySummary,
│   triggerIngestion, computeDaily, computeWeekly)
└─ frontend/src/components/MetricCard.tsx — Reusable card: status color
    (green/yellow/red) based on threshold + direction
```

**Key Technical Features:**
1. Multi-source data ingestion via BaseConnector ABC: 8 configured APIs (SAP, Ariba, By, Manhattan, CDK, PACCAR Connect, Snowflake, TMS) with OAuth2, fallback to realistic mock for dev
2. Daily KPI dashboard: 12 supply chain metrics + 5 configurable alert thresholds computed at 6am daily via APScheduler
3. Weekly analytics: inventory turns (COGS/avg_inv×52), forecast accuracy (MAPE), cost aggregates
4. Warranty anomaly detection: z-score statistical spike detection; seed data has intentional 25% systemic E-21403 turbocharger spike for demo
5. 14-day demand forecasting: blended linear trend (40%) + 7-day MA (60%) with 0.3 weekend factor, per part/region
6. Supplier/carrier performance tracking: 30d OTD% by supplier/carrier with on-time/late counts
7. CSV bulk upload endpoint: inventory snapshot ingestion via HTTP multipart (validated required cols: part_number, location, qty_on_hand, snapshot_date)
8. Real-time inventory positioning: total value, by-location breakdown, in-stock counts

**Data Flow:**
1. Ingest: Every 15min, `run_ingestion()` calls each connector's 7 pull methods, writes ~950 synthetic records per run (200 supplier orders, 125 inventory snapshots/DC, 80 freight shipments, 60 warranty claims, 375 demand records, 120 production, 120 dealer inventory)
2. Normalize: Records upserted to 8 raw tables with conflict resolution
3. Compute: Daily 6am computes 13 KPIs from last 7–30d data window → DailyMetric table + ConnectorLog
4. Aggregate: Weekly Monday 7am aggregates daily metrics + top-level supplier/freight/warranty performance → WeeklyMetric table
5. Detect: Anomaly detection runs on-demand, z-scores 60d warranty history per part vs recent 7d
6. Forecast: On-demand demand forecast fits trend + MA, returns 14d forward per part/region
7. Serve: Frontend fetches metrics via typed client, displays KPIs + Recharts time-series + anomalies table + performance tables

**Key Files:**
- `backend/main.py` — App entry point, lifespan, route registration
- `backend/models.py` — All 13 SQLAlchemy table definitions
- `backend/ingestion/pipeline.py` — Orchestration logic, 7-method upsert patterns
- `backend/connectors/mock.py` — 900-line synthetic PACCAR-realistic data generator with seeded stochastic variance
- `backend/metrics/daily.py` — KPI computation with alerting logic and 5 threshold checks
- `backend/metrics/weekly.py` — Weekly aggregates + inventory turns calculation
- `backend/ml/anomaly.py` — Warranty z-score detection algorithm
- `backend/api/routes_metrics.py` — 10 metric endpoints
- `frontend/src/pages/Daily.tsx` — Main dashboard component (KPI cards + charts + anomalies)
- `frontend/src/api/client.ts` — Type-safe fetch client with 11 API methods

**Real Metrics / Numbers from Code:**
- 11 API endpoints (10 metrics + 1 manual compute, 2 ingest, 1 CSV upload)
- 13 database tables (8 raw + DailyMetric + WeeklyMetric + ConnectorLog)
- 8 configured data connectors (7 real systems + 1 mock fallback)
- ~950 synthetic records generated per ingestion run
- 13 daily KPIs, 12 weekly aggregates
- 5 alerting thresholds (fill <95%, OTD <90%, attainment <95%, on-time >92%, exceptions <5)
- 14-day demand forecast horizon per part/region
- 60-day anomaly lookback window, z-threshold 2.0 warning / 3.0 critical
- SAP OAuth2 OData v4 (3 endpoints working: materialstock, purchase orders, production orders)

**External Integrations / APIs Used:**
- SAP S/4HANA (OData v4, OAuth2 token management)
- Ariba APIs, CDK Dealer portal, Manhattan Associates WMS, By freight ops
- PACCAR Connect, Snowflake data warehouse
- SMTP (alert emails), Slack webhooks (configured in settings, no routes wired)
- Recharts (React charting), Tailwind 4.3.1 (dark theme)

**What Makes It Distinctive:**
1. Stochastic mock data with seeded (42) realistic variance: 12% PO late rate, 8% freight late rate, deliberate E-21403 spike for anomaly detection demo
2. Blended forecasting: linear trend (40%) + moving average (60%) + weekend factor (0.3x) — not naive single-method
3. BaseConnector ABC: maps heterogeneous enterprise APIs to uniform 7-method interface; swap any connector without touching pipeline
4. APScheduler concurrency safety: background scheduler tasks + FastAPI lifespan manage 15-min/daily/weekly jobs independently

**Honest Status:**  
MVP/Prototype with core pipeline functional and deployable on Render. Incomplete: no test suite (0 test files), SAP connector half-stubbed (freight/warranty/demand return []), Slack alerting wired in config but no routes, PDF/Excel export not implemented (reports dir empty), no auth/RBAC (public API), no CI/CD pipeline.

**README Claims:**  
No project-level README. Frontend README is generic Vite template. No explicit metric targets in code; thresholds hardcoded in alerts. KPI targets: fill >95%, OTD >90%, attainment >95%, on-time >92%, freight exceptions <5.

---

## SovereignGrid

**GitHub URL:** https://github.com/rs1990/SovereignGrid  
**Primary Language:** Python  
**Tech Stack:**
- FastAPI ≥0.115, Uvicorn[standard] ≥0.32, Pydantic ≥2.9
- APScheduler ≥3.10 (requeue sweep + settlement cron)
- Supabase ≥2.10 (optional PostgreSQL; in-memory MemoryStore fallback)
- eth-account ≥0.13 (EIP-191 Ethereum wallet signatures)
- websockets ≥13.0 (provider daemon WS client)
- httpx ≥0.27 (daemon HTTP client to local inference engines)

**What it actually does:**  
Decentralized AI compute grid: developers connect local GPU hardware running open-source inference engines (Ollama/vLLM/Aphrodite) via persistent WebSocket daemons into a federated pool. A central FastAPI orchestrator routes consumer inference requests across nodes with smart dispatch, automatic failover, sanitization sandbox, and on-chain-style compute credits tracked per-model in an append-only ledger.

**Architecture:**
```
Orchestrator (FastAPI gateway at orchestrator/):
├─ main.py — WebSocket hub (/ws/provider), consumer API endpoints
│   (POST /v1/chat/completions, POST /v1/jobs, GET /v1/jobs/{id},
│   GET /v1/models, GET /health), lifespan lifecycle management
├─ dispatch.py — Smart load-balanced job routing to least-loaded node,
│   result dispatch via asyncio futures, requeue logic for failed jobs,
│   3-layer failover convergence
├─ registry.py — In-process live node registry keyed by wallet DID,
│   model-to-nodes index for fast routing lookup, least-loaded node picker
│   (min active_jobs), in-process snapshot for health endpoint
├─ auth.py — EIP-191 wallet signature nonce challenge/verify,
│   guild membership verification (guild_members table lookup),
│   admin admission signing utilities
├─ db.py — Dual-mode state layer:
│   MemoryStore — in-process dicts (dev/demo, no DB required)
│   SupabaseStore — PostgreSQL via Supabase client (production)
│   Both implement identical interface: upsert_node, get_node,
│   list_nodes, create_job, update_job_status, get_job,
│   list_jobs_by_status, append_ledger, settle_ledger
├─ models.py — Pydantic schemas: Handshake (wallet, sig, models,
│   max_concurrency, engine), JobAssignment (job_id, payload,
│   deadline_seconds), JobResult (job_id, ok, output, tokens_in,
│   tokens_out), Heartbeat (node_id, active_jobs),
│   CompletionRequest/Response (OpenAI-compatible), JobStatus
├─ scheduler.py — APScheduler AsyncIOScheduler wiring:
│   requeue_stale_jobs every 15s (configurable GRID_REQUEUE_INTERVAL)
│   settle_ledger hourly cron
├─ ledger.py — Append-only compute proof: record_inference(job_id,
│   node_id, model, tokens_in, tokens_out) → looks up model rate →
│   calculates credits → appends inference_ledger row
├─ sanitize.py — Denylist regex sandbox: 10 patterns blocking shell
│   injection, fork bombs, reverse shells, file:// paths, dangerous
│   system calls; scans payload before any node sees it
└─ config.py — Settings from env vars: SUPABASE_URL, SUPABASE_KEY,
    GRID_JOB_LEASE_SECONDS (300), GRID_JOB_MAX_ATTEMPTS (3),
    GRID_REQUEUE_INTERVAL (15), GRID_DISPATCH_TIMEOUT (300)

Provider daemon (daemon/provider_daemon.py):
├─ Wallet-signed authentication via EIP-191
│   (private key never leaves local machine — BYOK design)
├─ Connects to orchestrator WS /ws/provider
├─ Receives nonce challenge → signs "grid-auth:<nonce>" with eth-account
├─ Sends Handshake (wallet, sig, models[], max_concurrency, engine)
├─ Async job handler with concurrency semaphore (max_concurrency slots)
├─ Engine bridge:
│   Ollama: GET /api/tags (list models), POST /api/chat (inference)
│   vLLM: GET /v1/models, POST /v1/chat/completions (OpenAI-compat)
│   Aphrodite: GET /v1/models, POST /v1/chat/completions (port 2242)
├─ Heartbeat loop every 30s (ping with active_jobs count)
├─ Exponential backoff reconnect: 1s → 2s → 4s → ... → 60s cap
├─ Token counting from engine responses
└─ Job result serialization: {output, tokens_in, tokens_out}

Supabase PostgreSQL schema (supabase/schema.sql):
├─ nodes — provider DIDs (did:grid:<wallet>), wallet address, models[],
│   max_concurrency, engine type, balance_credits, status (online/offline)
├─ guild_admins — admin wallet registry
├─ guild_members — member wallet + admin signature + revocation support
├─ job_queue — status state machine (queued→running→succeeded/failed/dead),
│   lease_expires_at (300s TTL), attempts counter, payload JSONB, result JSONB
├─ inference_ledger — append-only rows: job_id, node_id, model,
│   tokens_in, tokens_out, credits_earned, settled boolean
├─ model_rates — prefix-pattern pricing table:
│   70B models: 10 credits/1k tokens
│   8B models: 1.5 credits/1k tokens
│   7B models: 1.2 credits/1k tokens
│   4B models: 0.8 credits/1k tokens
└─ settle_ledger() PL/pgSQL function — atomic settlement: aggregates
    unsettled ledger rows per node, updates nodes.balance_credits,
    flips settled=true in single transaction
```

**Key Technical Features:**
1. Wallet-native auth: EIP-191 signatures, no passwords; guild-admin-signed membership; revocation support via guild_members table
2. Smart dispatch: least-loaded node picker (min active_jobs), concurrent job tracking per node, multi-model routing index
3. 3-layer failover: WS disconnect → immediate requeue; dispatch timeout (300s) → requeue; APScheduler stale-lease sweep (15s) → requeue or mark dead after max_attempts=3
4. Compute credits: per-model tokenomic ledger priced by parameter count (70B pays ~6.7× an 8B per generated token)
5. Atomic settlement: PL/pgSQL `settle_ledger()` runs hourly, aggregates unsettled rows per node — one short transaction instead of write-lock per inference
6. Sanitization sandbox: 10-pattern regex denylist scans payloads before any provider node sees them
7. Dual-mode store: MemoryStore for dev/demo (no DB needed), SupabaseStore for production — single env var swap
8. BYOK daemon: provider private key never leaves local machine; only wallet address + signature transmitted; nonce consumption prevents replay

**Data Flow:**
1. Provider onboarding: daemon connects → orchestrator sends nonce → daemon signs `grid-auth:<nonce>` with wallet key → sends Handshake (wallet, sig, models[], max_concurrency, engine) → orchestrator verifies sig + guild_members row → NodeConnection registered in registry with DID `did:grid:<wallet>`
2. Consumer job submission: POST /v1/chat/completions → sanitize_payload (regex denylist) → create job in job_queue (status=queued) → registry.pick() selects least-loaded node serving the model → JobAssignment sent over WS with deadline_seconds → asyncio Future stored in conn.in_flight[job_id]
3. Daemon execution: receives JobAssignment → acquires semaphore slot → calls local engine (Ollama /api/chat or vLLM /v1/chat/completions) → extracts output + tokens → sends JobResult frame → releases semaphore
4. Result handling: orchestrator receives JobResult → resolves Future → if ok=True: mark_done(succeeded) + credit_inference() appends inference_ledger row; if ok=False or timeout: _requeue_or_kill() → requeue if attempts < max, else mark dead
5. Settlement (hourly): settle_ledger() PL/pgSQL atomically aggregates unsettled ledger rows per node → updates nodes.balance_credits → flips settled=true

**Key Files:**
- `orchestrator/main.py` — WebSocket gateway, consumer API (POST /v1/chat/completions, POST /v1/jobs, GET /v1/jobs/{id}, GET /v1/models, GET /health)
- `orchestrator/dispatch.py` — Smart routing, result futures, 3-layer failover/requeue logic
- `orchestrator/registry.py` — Live node registry, model-to-nodes index, least-loaded picker
- `orchestrator/auth.py` — EIP-191 wallet signature nonce challenge/verify, guild membership checks
- `orchestrator/db.py` — Dual MemoryStore/SupabaseStore abstraction, job CRUD, ledger append
- `daemon/provider_daemon.py` — Local engine bridge (Ollama/vLLM/Aphrodite), WS client, wallet auth, concurrency semaphore, heartbeat
- `supabase/schema.sql` — 6 tables + settle_ledger() PL/pgSQL function
- `orchestrator/scheduler.py` — APScheduler jobs: requeue_stale_jobs (15s), settle_ledger (hourly)
- `orchestrator/sanitize.py` — 10-pattern denylist sandbox
- `orchestrator/ledger.py` — Compute-proof crediting, model-rate lookup, token-to-credit conversion

**Real Metrics / Numbers from Code:**
- Consumer max_tokens: 1–32,768 (default 1,024)
- Provider max_concurrency: 1–64 per node
- Job lease TTL: 300s (configurable GRID_JOB_LEASE_SECONDS)
- Job max attempts: 3 (configurable GRID_JOB_MAX_ATTEMPTS)
- Requeue sweep interval: 15s (configurable GRID_REQUEUE_INTERVAL)
- Dispatch timeout: 300s (configurable GRID_DISPATCH_TIMEOUT)
- Nonce TTL: 60s (fixed)
- Heartbeat interval: 30s (daemon-side)
- Daemon reconnect backoff: 1s–60s exponential
- Payload size cap: 200K characters
- Model rates: 70B=10 credits/1k tokens, 8B=1.5, 7B=1.2, 4B=0.8 (seeded in schema; editable)

**External Integrations / APIs Used:**
- Ollama: GET /api/tags (list models), POST /api/chat (inference)
- vLLM: GET /v1/models, POST /v1/chat/completions (OpenAI-compatible)
- Aphrodite: GET /v1/models, POST /v1/chat/completions (port 2242, OpenAI-compatible)
- Supabase (PostgreSQL): RPC call to settle_ledger(), CRUD on all tables
- eth-account: Account.from_key(), sign_message(), recover_message() (EIP-191)
- APScheduler: AsyncIOScheduler, IntervalTrigger (requeue), CronTrigger (settle)

**What Makes It Distinctive:**
1. Wallet-first auth without blockchain: EIP-191 signatures + Supabase as guild registry; no smart contracts needed, guild admins control membership offline
2. Hybrid store: single interface (MemoryStore vs SupabaseStore) allows dev without DB; production seamlessly swaps to Postgres via one env var
3. Atomic settlement via PL/pgSQL: append-only ledger + hourly batch aggregation avoids hot-path write contention; perfect for high-concurrency grids
4. Three-layer failover convergence: all failure modes (disconnect, timeout, stale lease) requeue into the same path; consumer retries across nodes transparently
5. BYOK daemon design: private key never leaves provider machine; only wallet address + signature transmitted; nonce consumption prevents replay attacks
6. Model-size-aware crediting: rate table multiplies tokens_out by model-specific credit_per_1k — larger models naturally pay out more per token

**Honest Status:**  
Prototype → MVP. Fully functional end-to-end (WS handshake, job dispatch, result handling, settlement). Known skeleton scope (documented in README): no SSE streaming relay (returns whole results), consumer auth is bearer-key hash not metered billing, token counts trusted from daemon (no cross-check), single orchestrator instance (registry in-process; horizontal scale needs sticky routing or shared dispatch bus). No TODOs/stubs in implemented code — all endpoints complete with clean error paths.

**README Claims:**
- "Capability handshake...one-time nonce...signature...recovered address must be guild-admitted" ✓
- "Smart dispatch...least-loaded online node" ✓
- "Three layers of failure handling all converging on same requeue path" ✓
- "Compute credits...70B pays ~7× an 8B per generated token" ✓ (exact: 10/1.5 = 6.7×)
- "Hourly settle_ledger() batch — one short transaction per hour instead of write-lock per inference" ✓
- "Sanitization sandbox before anything reaches a provider machine" ✓
- "No streaming, bearer-key hash consumer auth, trusted daemon-reported token counts, single orchestrator instance" ✓ (all documented as known limitations)

---

## DocAI

**GitHub URL:** https://github.com/rs1990/DocAI  
**Primary Language:** Python 3.10+  
**Tech Stack:**
- Core: python-dotenv, pyyaml, click, rich, tqdm
- Parsing: pdfminer.six, python-docx, python-pptx, pandas, openpyxl, Pillow, pytesseract, beautifulsoup4, lxml, chardet, python-magic
- Text: nltk, tiktoken, unicodedata2
- Embeddings: sentence-transformers 2.6.1+, FlagEmbedding 1.2.5+, transformers 4.38.0+, torch 2.2.0+, huggingface-hub
- Vector: faiss-cpu 1.7.4+ (or faiss-gpu)
- LLM: ollama 0.1.8+ via HTTP localhost:11434 (Mistral-7B)
- DB: SQLite (built-in WAL mode)
- Retrieval: rank-bm25 0.2.2+
- Dashboard: streamlit 1.32.0+, plotly 5.20.0+, pyvis 0.3.2+, networkx 3.2.1+
- Scheduling: apscheduler 3.10.4+
- Fine-tuning: peft 0.9.0+, accelerate 0.27.0+, datasets 2.17.0+
- Testing: pytest 8.0.0+, pytest-cov 4.1.0+, pytest-mock 3.12.0+

**What it actually does:**  
Local-only, fully offline document intelligence system. Scans local and network drives, extracts and indexes all document content, provides RAG-based AI chat over documents via a local LLM (Mistral-7B via Ollama), runs 5 AI agents (Researcher, Extractor, Meeting Clerk, Compliance, Code Generator), and exposes a Streamlit dashboard with document graph visualization — all without any cloud dependency. Status: ~25% complete (ingestion + processing layer working; indexing, retrieval, LLM, agents, dashboard all missing or stub).

**Architecture:**
```
docai/core.py — Canonical data contracts:
  Enums: FileType (pdf/docx/pptx/xlsx/csv/txt/html/image/matlab)
         LinkType (explicit_mention/name_based/folder/timestamp_id/semantic/matlab_shallow)
         IssueSeverity (low/medium/high/critical)
  Dataclasses: FileRecord, ParsedDocument, Chunk, SearchResult, Citation,
               DocumentLink, AgentResponse, DevCostEntry, DecisionRecord, IssueRecord

docai/ingestion/
├─ scanner.py — Recursive directory scanner; supports local paths + SMB/NFS/NAS
│   as OS-mounted paths + UNC prefix detection; batching; retry with exponential
│   backoff; 50ms rate limiting per file on network drives; symlink handling;
│   extension→FileType mapping; yields FileRecord objects
├─ file_router.py — Dispatches FileRecord by filetype to appropriate parser;
│   lazy imports to avoid hard deps on unused libraries; per-file exception
│   handling (returns ParsedDocument.parse_error, never halts pipeline)
├─ safe_copy.py — Non-destructive workspace copy; content-addressed layout
│   <workspace>/<hash[:8]>/<filename>; SQLite manifest prevents redundant I/O;
│   original files never modified
├─ change_detector.py — Incremental detection via SHA-256 + mtime manifest;
│   ChangeSet tracks new/modified/deleted files across runs
├─ parsers/pdf_parser.py — pdfminer.six native text extraction + pdf2image +
│   pytesseract OCR fallback for scanned PDFs [WORKING]
├─ parsers/docx_parser.py — python-docx text extraction [WORKING]
├─ parsers/pptx_parser.py — python-pptx slide text extraction [WORKING]
├─ parsers/txt_parser.py — [MISSING — router dispatches to it, ImportError]
├─ parsers/html_parser.py — [MISSING — router dispatches to it, ImportError]
├─ parsers/xlsx_parser.py — [MISSING — router dispatches to it, ImportError]
├─ parsers/csv_parser.py — [MISSING — router dispatches to it, ImportError]
├─ parsers/image_ocr_parser.py — [MISSING — router dispatches to it, ImportError]
└─ parsers/matlab_parser.py — [MISSING — router dispatches to it, ImportError]

docai/processing/
├─ normalizer.py — Unicode normalization (NFC), zero-width/control char removal,
│   OCR noise filtering, whitespace collapse, quote/dash normalization,
│   HTML entity decode [WORKING]
├─ chunker.py — Sentence-aware sliding-window chunker; respects sentence
│   boundaries via nltk; 512–2000 token range (tiktoken); configurable overlap
│   (default 64 tokens); produces Chunk objects with token counts [WORKING]
├─ jsonl_writer.py — Atomic JSONL persistence per document; tmp+rename pattern
│   (os.replace) prevents partial reads [WORKING]
└─ metadata_extractor.py — Language detection (langdetect + ASCII fallback),
    reading time estimate, file size, character count [WORKING]

docai/indexing/ (SCHEMA ONLY — no Python implementation)
├─ schema.sql — Full SQLite DDL: source_files, chunks, embeddings
│   (FAISS vector_id→chunk_id mapping), document_links, summaries,
│   manifest, decisions, issues, dev_costs, automation_runs, test_results
│   (11 tables total)
├─ sqlite_store.py — [MISSING — imports broken in retriever.py]
├─ embedder.py — [MISSING — imports broken in retriever.py]
├─ faiss_index.py — [MISSING — imports broken in retriever.py]
└─ index_manager.py — [MISSING — imports broken in retriever.py]

docai/retrieval/ (DEAD CODE — imports nonexistent indexing modules)
├─ retriever.py — Implements hybrid ANN (FAISS) + BM25 with minmax normalization;
│   imports faiss_index/sqlite_store/embedder → all missing → ImportError at load
├─ reranker.py — Optional cross-encoder reranker (SentenceTransformers);
│   no-op if disabled [CONDITIONALLY WORKING if indexing existed]
└─ citation_builder.py — Citation deduplication from SearchResult; excerpt
    extraction; page/slide hint inference; inline/footnote formatting
    [BUG B5: enum str() returns "filetype.pdf" never matches "pdf"]

docai/llm/ (ENTIRELY EMPTY)
├─ __init__.py — 0 bytes
├─ model_loader.py — [MISSING]
├─ inference.py — [MISSING]
├─ prompt_templates.py — [MISSING]
├─ rag_pipeline.py — [MISSING]
└─ finetuning/ — [MISSING: lora_trainer.py, dataset_builder.py, eval_harness.py]

docai/agents/ (BASE CLASS ONLY)
├─ base_agent.py — Reason-act loop with Tool descriptors; [TOOL: name(args)]
│   regex parser; _call_llm() makes raw HTTP to http://localhost:11434/api/generate
│   (hardcoded, no retry, 120s timeout); confidence scoring; citation attachment
│   [BUG B11: regex [^)]* truncates tool args containing closing parens]
├─ document_researcher.py — [MISSING]
├─ data_extractor.py — [MISSING]
├─ meeting_clerk.py — [MISSING]
├─ compliance_checker.py — [MISSING]
└─ code_generator.py — [MISSING]

docai/linking/ (SCAFFOLDED — docstring only)
├─ __init__.py — docstring only
├─ link_types.py — [MISSING: 6 link type implementations]
├─ link_detector.py — [MISSING]
├─ matlab_linker.py — [MISSING]
└─ graph_builder.py — [MISSING]

docai/automation/ (SCAFFOLDED — docstring only)
├─ __init__.py — docstring only
├─ engine.py — [MISSING]
├─ scheduler.py — [MISSING]
└─ actions.py — [MISSING: YAML task executor for config/settings.yaml automations]

docai/tracking/ (EMPTY — schema exists, no CRUD)
├─ decision_log.py — [MISSING]
├─ bug_tracker.py — [MISSING]
└─ cost_tracker.py — [MISSING]

docai/dashboard/ (ENTIRELY MISSING)
├─ app.py — [MISSING]
├─ pages/home.py, search.py, chat.py, agents.py, graph.py, issues.py, cost.py — [MISSING]
└─ components/citation_card.py, document_viewer.py, graph_widget.py — [MISSING]

docai/cli/ (MISSING ENTIRE DIRECTORY)
├─ main.py — [MISSING: entry point referenced in pyproject.toml as 'docai.cli.main:cli']
└─ cmd_run.py, cmd_chat.py, cmd_dashboard.py, cmd_export.py, ... — [MISSING]

config/settings.yaml — 214-line YAML config (fully written): scanning paths,
  workspace, parsing options, chunking (512–2000 tokens, 64 overlap), embeddings
  (BAAI/bge-large-en-v1.5 1024-dim, batch 16), FAISS (HNSW M=32 ef=200),
  SQLite (WAL mode), LLM/Ollama (Mistral-7B, temp 0.2, top_p 0.95,
  max_new_tokens 1024, context_window 4096), RAG (top_k=8, rerank_top_k=4),
  agents (5 defined, max_iterations=10, confidence_threshold=0.6), linking
  (6 link types, max_links_per_file=50), automations (YAML tasks), scheduler,
  dashboard, cost tracking
```

**Key Technical Features:**
1. Directory scanner with network drive support: SMB/NFS/NAS via UNC prefix detection, 50ms per-file rate limiting, exponential backoff retry, symlink handling
2. PDF parser with OCR fallback: pdfminer.six for native text + pdf2image + pytesseract for scanned PDFs
3. Sentence-aware chunker: nltk sentence boundaries, 512–2000 token range via tiktoken, configurable overlap (default 64 tokens)
4. Unicode normalizer: NFC, zero-width/control char removal, OCR noise, HTML entities, quote/dash normalization
5. Content-addressed safe copy: workspace layout `<hash[:8]>/<filename>` prevents collision, SQLite manifest prevents redundant I/O
6. Incremental change detection: SHA-256 + mtime manifest → ChangeSet (new/modified/deleted) for efficient re-indexing
7. Base agent framework: reason-act loop, Tool descriptors, `[TOOL: name(args)]` regex parser, Ollama HTTP client, confidence scoring
8. Atomic JSONL writes: tmp file + os.replace() ensures readers never see partial chunks

**Data Flow (as implemented — pipeline stops at chunking):**
1. Scan: `scanner.py` recursively yields FileRecord (path, hash, mtime, filetype)
2. Copy: `SafeCopy.copy_if_needed()` mirrors to content-addressed workspace, stores manifest
3. Change Detection: `ChangeDetector.get_changes()` compares hash/mtime → ChangeSet (new/modified/deleted)
4. Route & Parse: `FileRouter.route()` dispatches by filetype → parser (PDF/DOCX/PPTX working; TXT/HTML/XLSX/CSV/image/MATLAB crash with ImportError)
5. Normalize: `normalizer.normalize()` cleans text
6. Chunk: `chunker.chunk_document()` → sentence-aware Chunk objects with token counts
7. Serialize: `jsonl_writer.write_chunks()` → atomic JSONL per document
8. **(BROKEN: embedder.py missing)** → would batch-embed chunks into BAAI/bge-large-en-v1.5
9. **(BROKEN: faiss_index.py missing)** → would build HNSW FAISS index
10. **(BROKEN: sqlite_store.py missing)** → would persist metadata and vector_id mappings
11. **(MISSING: LLM/RAG layer)** → retriever.py + rag_pipeline.py would synthesize answers
12. **(MISSING: Agents)** → 5 concrete agent classes don't exist
13. **(MISSING: Dashboard)** → Streamlit app not implemented
14. **(MISSING: Automations)** → YAML scheduler not implemented

**Key Files:**
- `docai/core.py` — Canonical data contracts (6 enums, 11 dataclasses — all pipeline types)
- `docai/ingestion/scanner.py` — Recursive file discovery with network drive detection and retry
- `docai/ingestion/parsers/pdf_parser.py` — PDF + pytesseract OCR fallback
- `docai/processing/chunker.py` — Sentence-aware tokenized chunking
- `docai/processing/normalizer.py` — Unicode/whitespace/OCR noise normalization
- `docai/ingestion/safe_copy.py` — Content-addressed workspace copy with manifest
- `docai/agents/base_agent.py` — Reason-act loop with Ollama HTTP client
- `config/settings.yaml` — All runtime config (214 lines, fully specified)
- `docai/indexing/schema.sql` — Full SQLite DDL (11 tables)
- `requirements.txt` — 75 pinned dependencies

**Real Metrics / Numbers from Code:**
- 3 of 9 parsers working (PDF, DOCX, PPTX; TXT/HTML/XLSX/CSV/image/MATLAB missing)
- Network rate limiting: 50ms between file reads on network paths (configurable)
- Embedding model: BAAI/bge-large-en-v1.5, 1024-dim, batch size 16 (CPU)
- FAISS: HNSW (M=32, ef_construction=200) or IVF-FLAT (nlist=1024, nprobe=64)
- Chunk range: 512–2000 tokens, 64-token overlap
- LLM target: Mistral-7B via Ollama (temp 0.2, top_p 0.95, max_new_tokens 1024, context_window 4096)
- RAG: top_k=8 chunks → rerank_top_k=4 via optional cross-encoder
- 5 agents defined in config: researcher, extractor, meeting_clerk, compliance, code_generator; max_iterations=10, confidence_threshold=0.6
- 6 document link types defined: explicit mention, name-based, folder-based, timestamp/ID, semantic similarity, MATLAB shallow
- 11 SQLite tables in schema
- 14 test categories planned; 0 tests implemented
- 8 CLI commands planned (run, chat, dashboard, export, cost, issues, test, doctor); 0 implemented
- 7 dashboard pages planned (home, search, chat, agents, graph, issues, cost); 0 implemented

**External Integrations / APIs Used:**
- Ollama HTTP API: `http://localhost:11434/api/generate` (hardcoded in base_agent.py)
- HuggingFace sentence-transformers (BGE model from HF Hub, local inference)
- FAISS (in-memory vector search, planned)
- SQLite (WAL mode, local)
- Streamlit (planned dashboard, host 0.0.0.0:8501)
- APScheduler (cron-style weekly triggers, planned)
- PEFT + accelerate + datasets (LoRA fine-tuning, planned)

**What Makes It Distinctive:**
1. Content-addressed safe copy: `<hash[:8]>/<filename>` layout prevents file collision + enables dedup without touching originals
2. Per-file error isolation: all parser exceptions caught at file_router layer; pipeline never halts on single bad file
3. Atomic JSONL writes: tmp + os.replace() ensures readers never see partial chunk data
4. Rate-limited network scanning: 50ms/file on network paths (configurable) prevents overwhelming NAS/SMB shares
5. Lazy parser imports: prevents hard deps on unused libraries (graceful degradation if e.g. python-docx not installed)
6. Reason-act agent base: tool-call parsing via regex (not structured output); confidence scoring + citation attachment per step

**Honest Status:**  
Pre-alpha skeleton (~25% complete). Code compiles without syntax errors but is not installable (broken pyproject.toml build backend, entry point references nonexistent CLI module) or runnable end-to-end. Working: ingestion (3 of 9 parsers), processing (normalizer/chunker/JSONL), core data contracts, config, safe copy, change detector, scanner, base agent framework, citation builder, schema.sql. Broken/missing: indexing layer (embedder, FAISS, SQLiteStore), 6 parsers, LLM layer (entirely empty), CLI (doesn't exist), dashboard (doesn't exist), all 5 concrete agents, all linking system, all automations, all tracking CRUD. PROJECT_BRIEF.md notes "PAUSED — Hardware procurement required before build resumes" (as of 2026-03-24).

**README Claims:**  
From PROJECT_BRIEF.md: "DocAI is a fully local, offline, open-source Document Intelligence System. Scans local folders + SMB/NFS/NAS, parses PDF/DOCX/PPTX/XLSX/CSV/TXT/HTML/images (OCR)/MATLAB (shallow), builds semantic search index (FAISS + BGE embeddings), provides RAG-based AI chat using local LLM (Mistral-7B), runs 5 AI agents (Researcher, Extractor, Meeting Clerk, Compliance, Code Generator), exposes visual dashboard with document graph, performs automated weekly re-indexing, includes full CLI, engineering decision logs, and bug tracking — all without any cloud dependency." Actual coverage: ~25% of spec implemented. Paused.

---

## ClaudeCosts

**GitHub URL:** https://github.com/rs1990/ClaudeCosts  
**Primary Language:** Python 3.9+ + Bash  
**Tech Stack:**
- rumps ≥0.4.0 — macOS menu bar app framework
- pyobjc-framework-Cocoa ≥10.0 — Cocoa bindings (NSView, NSImage, NSMenuItem)
- matplotlib ≥3.5.0 — headless PNG chart rendering (Agg backend)
- Python 3.9+, macOS 12+

**What it actually does:**  
macOS menu bar app that reads Claude Code JSONL session logs from `~/.claude/projects/`, computes token-based costs from hardcoded pricing, and displays Today/Month/All-Time totals + per-model breakdown + inline 14-day token/cost chart as a native NSMenu dropdown. Auto-refreshes every 15 minutes. Installed via LaunchAgent (auto-start at login) + PreToolUse hook (fires on every Claude tool call).

**Architecture:**
```
claude_usage_monitor.py (717 lines total):

Constants & Config (lines 32–117):
├─ CLAUDE_DIR (~/.claude/), LOG_PATH, SETTINGS_PATH (~/.claude/settings.json)
├─ REFRESH_INTERVAL = 900s (15 min default)
├─ MAX_MODEL_SLOTS = 10 (max models shown in dropdown)
├─ GRAPH_W = 370px, GRAPH_H = 130px, GRAPH_DPI = 144 (Retina 2×)
├─ STATUS_URL = "https://status.anthropic.com/api/v2/status.json"
├─ STATUS_TTL = 300s (5 min status cache)
└─ _STATUS_ICONS dict: maps none/minor/major/critical/unknown → emoji

Pricing & Model Config (lines 69–117):
├─ PRICING dict (lines 71–81):
│   claude-opus-4-6: $15/$75 per million input/output tokens
│   claude-opus-4-7: $15/$75 per million input/output tokens
│   claude-sonnet-4-6: $3/$15 per million input/output tokens
│   claude-sonnet-4-5: $3/$15 per million input/output tokens
│   claude-haiku-4-5: $0.80/$4 per million input/output tokens
│   cache_write: $3.75, cache_read: $0.30 (Sonnet rates)
│   deprecated entries: claude-opus-3-5, claude-sonnet-3-5, claude-haiku-3-5
├─ _DEFAULT_PRICING — fallback (Sonnet rates) for unknown models, silent
├─ AVAILABLE_MODELS — 5-item list for Session Config dropdown
└─ REFRESH_OPTIONS — {label: seconds} for interval picker

Utilities (lines 120–181):
├─ _pricing(model) — dict lookup with fuzzy substring match, fallback to DEFAULT
├─ _calc_cost(model, it, ot, cw, cr) — multiplies 4 token types by per-model rates
├─ _fmt_tok(n) — formats as M/K/raw (e.g., "1.2M", "450K", "312")
├─ _fmt_cost(c) — formats USD 2–4 decimal places
├─ _short_model(model) — strips "claude-" prefix + "-20251001" suffix, titlecase
├─ _load_settings() — reads ~/.claude/settings.json, empty dict if missing
├─ _save_settings(settings) — writes settings dict to JSON (non-atomic write bug)
└─ _get_setting(settings, *keys, default=None) — nested dict path lookup

Data Parsing (lines 185–287):
├─ parse_usage() (lines 185–242):
│   - Scans all *.jsonl under ~/.claude/projects/ (graceful if dir missing)
│   - Reads line-by-line, skips blank/malformed JSON
│   - Filters to "type": "assistant" entries only
│   - Extracts usage object (input_tokens, output_tokens,
│     cache_creation_input_tokens, cache_read_input_tokens)
│   - Deduplicates by UUID (uses filepath:line_no if UUID missing)
│   - Parses ISO 8601 timestamp → local date string
│   - Returns: {date_str: {model: {input, output, cache_write, cache_read, cost, calls}}}
├─ _aggregate(data, prefix="") (lines 245–258):
│   - Sums all models across matching date prefixes
│   - Returns {input, output, cache_write, cache_read, cost, calls, by_model: {...}}
└─ _get_daily_series(data, days=14) (lines 261–287):
    - Rolls up daily totals for last N days (or all-time if days=None)
    - Returns list of {date, label, input, output, cost, calls} dicts

Graph Rendering (lines 292–383):
├─ _render_graph_png(data, days=14) (lines 292–370):
│   - Uses Agg (headless) backend; renders at 144 DPI (2× Retina)
│   - GridSpec: stacked token bars (input=blue, output=green) over cost line chart
│   - Dark theme (#1c1c1e bg, #8e8e93 text); token Y-axis in K; cost in $
│   - Returns raw PNG bytes or None if matplotlib missing
└─ _make_ns_image(png_bytes) (lines 378–383):
    - Wraps PNG bytes in NSImage, sizes to GRAPH_W × GRAPH_H logical pixels

Menu Bar App (lines 388–712):
ClaudeUsageApp(rumps.App):
├─ __init__ (390–401) — initializes title, data dict, graph_days (14), starts timer
├─ _setup_menu() (405–513) — builds nested NSMenu:
│   ├─ Status section: header + status label + open-status-page button
│   ├─ Inline graph placeholder (replaced with NSImageView in _install_inline_graph)
│   ├─ Time-range selectors: 7 / 14 / 30 days / all-time checkmark items
│   ├─ Today section: cost label, token label, API calls label
│   ├─ This Month section: cost label, token label, API calls label
│   ├─ All Time section: cost label, token label, API calls label
│   ├─ Per-model table: 10 rows (MAX_MODEL_SLOTS), sorted by cost desc
│   └─ ⚙️ Session Config submenu:
│       model dropdown (5 models), refresh interval (5/15/30 min, 1hr),
│       verbose tools toggle, auto-compact toggle, edit-settings link
├─ _install_inline_graph() (517–543) — replaces placeholder MenuItem with
│   native NSMenuItem whose view is an NSImageView (no file I/O)
├─ _update_inline_graph() (545–556) — calls _render_graph_png → NSImage → push to NSImageView
├─ _sync_config_state() (560–581) — updates checkmarks on menu items from current config
├─ _on_set_graph_range(sender) (585–590) — updates _graph_days, syncs UI, re-renders
├─ _on_set_model(sender) (592–600) — writes selected model to settings.json
├─ _on_set_refresh(sender) (602–611) — stops old timer, starts new with new interval
├─ _toggle_verbose(sender) (613–623) — toggles settings.json["env"]["CLAUDE_VERBOSE_TOOLS"]
├─ _toggle_autocompact(sender) (625–629) — toggles settings.json["autoCompact"] boolean
├─ _open_settings_file(sender) (631–632) — opens settings.json in TextEdit
├─ _open_status_page(sender) (634–635) — opens status.anthropic.com in browser
├─ _on_timer(_) (639–640) — 15-min timer callback → _do_refresh
├─ _manual_refresh(_) (642–643) — "Refresh Now" button → _do_refresh
├─ _do_refresh() (645–654) — re-parse usage, update display, update graph,
│   spawn daemon thread for status fetch
├─ _refresh_status() (656–661) — fetches status JSON, updates label + prepends icon
├─ _update_title_with_status(indicator) (663–671) — prepends ⚠️/🔴 to title
│   (strips old icon first to avoid accumulation)
└─ _update_display(data) (675–711) — aggregates Today/Month/All from parsed data,
    formats all labels, updates menu bar title: "⚡ $cost | tokens tok",
    sorts models by cost desc, fills MAX_MODEL_SLOTS model rows

install.sh (139 lines):
├─ Detects Python 3 installation
├─ Installs rumps/pyobjc/matplotlib via pip
├─ Writes LaunchAgent plist to ~/Library/LaunchAgents/com.claude.usage-monitor.plist
├─ Injects PreToolUse command hook into ~/.claude/settings.json
└─ Launches app immediately via launchctl

uninstall.sh (30 lines):
├─ Unloads LaunchAgent, removes plist
├─ Removes script
└─ DOES NOT remove PreToolUse hook from settings.json (known bug)
```

**Key Technical Features:**
1. Menu bar title shows today's cost + token count at a glance (updates every 15 min)
2. Dropdown menu breaks down usage by Today / This Month / All Time (cost, tokens, API calls)
3. Per-model breakdown sorted by cost descending (top 10 models via MAX_MODEL_SLOTS)
4. Inline 14-day matplotlib chart: stacked token bars (input/output) + cost line, rendered as PNG bytes → NSImage → NSImageView in NSMenu (zero file I/O, degrades gracefully without matplotlib)
5. Auto-refresh every 15 min (configurable 5min/15min/30min/1hr via Session Config submenu)
6. Claude API status polled from status.anthropic.com every 5 min; title flashes ⚠️/🔴 on incidents
7. Session Config submenu: model selection, refresh interval, verbose tools toggle, auto-compact toggle (all write to ~/.claude/settings.json)
8. Graph time-range selector: 7 / 14 / 30 days / all-time
9. Auto-start via LaunchAgent at login + PreToolUse hook fires on every Claude session
10. UUID-based deduplication: subagent JSONL files overlap with main session file; dedup prevents double-counting

**Data Flow:**
1. Input: `~/.claude/projects/*.jsonl` (Claude Code internal session logs)
2. Parse: `parse_usage()` scans all JSONL, extracts `assistant` type entries, pulls usage fields (input/output/cache tokens, timestamp, model)
3. Deduplicate: UUID seen-set across all files (fallback: filepath:line_no)
4. Bucket: groups by UTC date string + model into nested dict
5. Price: `_calc_cost()` multiplies 4 token types by per-model rates from PRICING dict
6. Aggregate: `_aggregate()` sums by date prefix (Today=YYYY-MM-DD, Month=YYYY-MM, All="")
7. Display: `_update_display()` formats labels, pushes to menu items; `_update_inline_graph()` renders 14-day chart
8. Status: parallel daemon thread fetches `status.anthropic.com/api/v2/status.json`, updates status label + title icon
9. Output: macOS menu bar icon with dropdown (NSMenu + inline NSImageView)

**Key Files:**
- `claude_usage_monitor.py` (717 lines) — entire app: parsing, pricing, UI, LaunchAgent
- `install.sh` (139 lines) — Python detection, pip install, LaunchAgent plist write, PreToolUse hook injection
- `uninstall.sh` (30 lines) — LaunchAgent unload + remove (does NOT remove PreToolUse hook)
- `requirements.txt` — rumps, pyobjc-framework-Cocoa, matplotlib
- `README.md` — feature overview, install/uninstall instructions, pricing info, troubleshooting
- `PROJECT_REVIEW.md` (112 lines) — audit: 15 bugs logged (high/medium/low severity)

**Real Metrics / Numbers from Code:**
- 10 model slots displayed (MAX_MODEL_SLOTS)
- 5 models in AVAILABLE_MODELS dropdown
- 9 entries in PRICING dict (5 current-gen + 3 deprecated 3.5 + 1 duplicate Haiku)
- 14-day default graph window; 370×130px logical size; rendered at 144 DPI (Retina)
- 15-minute default refresh interval; 300s status cache TTL
- 718 total lines of Python (main script)

**External Integrations / APIs Used:**
- Input: reads `~/.claude/projects/*.jsonl` (Claude Code internal session logs — local files only)
- Config: reads/writes `~/.claude/settings.json` (Claude Code configuration)
- External HTTP: `status.anthropic.com/api/v2/status.json` (one GET every 5 min; this is the only external call)
- LaunchAgent: `~/Library/LaunchAgents/com.claude.usage-monitor.plist` (macOS auto-start)
- PreToolUse hook: injects into settings.json to spawn monitor on every Claude tool call

**What Makes It Distinctive:**
1. Zero-copy inline graph: renders matplotlib to PNG bytes in-memory → wraps in NSImage → embeds in NSImageView within NSMenu — no temp files, degrades gracefully without matplotlib
2. UUID deduplication across JSONL: subagent session files overlap with main session; seen-set prevents double-counting costs
3. LaunchAgent + PreToolUse belt-and-suspenders: two independent startup mechanisms; either alone sufficient but both ensure survival across logout/login and session restarts
4. Live repo checkout in plist: LaunchAgent points directly at repo source path — pragmatic for developer tool (updates automatically on git pull)

**Honest Status:**  
Prototype/MVP with known production bugs. Critical: stale/wrong pricing (missing Opus 4.8, Fable 5; wrong Haiku 4.5 rates causing 3× Opus cost overstatement); uninstall.sh doesn't actually uninstall (leaves PreToolUse hook in settings.json); UI mutations from background thread (undefined Cocoa behavior). Medium: timezone day-bucketing bug (UTC timestamps vs local `date.today()` → evening usage bleeds to tomorrow for US users); crash if `~/.claude/` doesn't exist; pip install hits PEP 668 on modern Python. No tests, no CI, no packaging.

**README Claims:**
- "A macOS menu bar app that reads your Claude Code session logs and shows live token usage and cost estimates — broken down by today, this month, and all time." ✓
- "Menu bar title shows today's cost and token count at a glance" ✓
- "Per-model breakdown sorted by cost" ✓
- "Auto-refreshes every 15 minutes" ✓
- "Claude API status pulled from status.anthropic.com — title bar flashes ⚠️/🔴 on incidents" ✓
- "No data leaves your machine. No API calls are made." ✗ (one GET to status.anthropic.com; usage cost calculations are local-only)
- "Installer copies the script to ~/.claude/" ✗ (it does not; plist points to repo path)

---

## cad-converter

**GitHub URL:** https://github.com/rs1990/cad-converter  
**Primary Language:** Python (backend) + Vanilla JavaScript (frontend)  
**Tech Stack:**
- Backend: FastAPI 0.115+, uvicorn 0.32+, Anthropic SDK 0.40+ (claude-sonnet-4-6 Vision), CadQuery (missing from requirements.txt — conda/pip via start.sh), ezdxf (missing from requirements.txt), trimesh 4.4+, pypdfium2 4.17+, pydantic 2.0+, python-dotenv 1.0+, python-multipart 0.0.12+, numpy 1.24+
- Frontend: Three.js 0.170.0 (CDN, no build step), STLLoader, OrbitControls (ES modules via importmap)
- Build/Deploy: Bash start.sh with conda/pip fallback for CadQuery (complex install path)

**What it actually does:**  
Multi-step 2D CAD → 3D solid converter. Extracts structured part specifications from raster images (PNG/JPG), PDFs (via Claude Sonnet Vision API), or DXF/DWG files (heuristic ezdxf parsing), allows user clarification and editing of dimensions through an interactive 4-step wizard, builds 3D solids using CadQuery (OCCT kernel) with parametric features (holes, pockets, fillets, etc.), exports to STL/STEP/OBJ + per-face metadata JSON. Web UI drives the flow: upload → clarify → edit spec → 3D preview with clickable face inspector.

**Architecture:**
```
Backend (FastAPI):
├─ backend/main.py — FastAPI app with lifespan, ProcessPoolExecutor (2 workers)
│   for CadQuery (OpenCASCADE kernel is thread-unsafe), session lifecycle mgmt,
│   24h TTL cleanup loop, static frontend serving
│   Endpoints:
│     POST /api/extract — upload file, extract PartSpec via Claude Vision or DXF
│     POST /api/clarify — apply Q&A answers to existing PartSpec
│     POST /api/build — build 3D model from finalized PartSpec
│     GET /api/model/{id}/{fmt} — download STL/STEP/OBJ file
│     GET /api/model/{id}/faces — per-face metadata JSON
├─ backend/models.py — Complete Pydantic schema with discriminated unions:
│   Discriminated by "type" field:
│   BaseShape variants: box, cylinder, l_shape, t_shape, polygon
│   Feature variants: hole, cbore (counterbore), csk (countersink), slot,
│                     fillet, chamfer, pocket
│   HolePattern variants: grid, bolt_circle
│   Top-level: PartSpec (base_shape, features[], hole_patterns[], units, material)
│   API contracts: ExtractRequest, ExtractResponse (spec + clarifications[]),
│                  ClarifyRequest, BuildRequest, BuildResponse
├─ backend/extractor.py — Input routing and extraction:
│   extract_from_image() — Claude Sonnet Vision API call; returns JSON PartSpec
│                          + ClarificationQuestion list for ambiguous dims
│   extract_from_dxf() — ezdxf heuristics: reads ENTITIES layer, detects
│                        LINE/CIRCLE/ARC, infers bounding box dimensions
│   apply_clarification_answers() — dot-path mutation via regex parser
│                                   (e.g., "base_shape.dims.height" → nested update)
│   parse_upload() — file type dispatch based on MIME/extension
│   PDF handling — pypdfium2 renders first page to PNG → passed to extract_from_image
└─ backend/builder.py — CadQuery solid construction and export:
    build_part(spec: PartSpec) → CadQuery Workplane object:
      _build_base() — dispatches to box/cylinder/l_shape/t_shape/polygon builder
      _apply_feature() — dispatches to hole/cbore/csk/slot/fillet/chamfer/pocket
      _apply_hole_pattern() — dispatches to grid/bolt_circle pattern
    export_all(result, session_dir) — CadQuery → STL, STEP; trimesh STL→OBJ
    export_face_info() — per-face metadata: direction classification via normal
      vector (_classify_normal: top/bottom/front/back/left/right/angled),
      surface area, bounding box, associated feature names
    Session storage: UPLOADS_DIR/<uuid>/ contains uploaded file, spec.json,
      model.stl, model.step, model.obj, faces.json

Frontend (Vanilla JS + Three.js, no build step):
├─ frontend/index.html — Semantic HTML5; 4-card sections (upload → clarify →
│   spec editor → 3D preview); ES module importmap for Three.js 0.170 CDN;
│   drag-drop upload zone; form fieldsets for part info/base shape/features/patterns
├─ frontend/app.js (~860 LOC) — pure vanilla JS state machine:
│   State: sessionId, currentSpec, pendingClarifications, threeScene,
│           partMesh, facesData, selectedFaceDir
│   Lifecycle functions:
│     handleFile() — validates file type/size, shows upload feedback
│     uploadAndExtract() — POST /api/extract, handles clarify or spec branch
│     renderClarifyForm() — dynamically builds Q&A form from ClarificationQuestion[]
│     submitClarifications() — POST /api/clarify, re-renders spec editor
│     renderSpecForm() — dynamically generates fieldsets from PartSpec JSON
│     renderFeaturesList() — builds feature rows with inline DOM splicing
│     renderPatternsList() — builds hole pattern rows
│     collectSpec() — serializes form DOM → PartSpec JSON
│     buildModel() — POST /api/build, progress feedback
│     initThreePreview() — sets up Three.js scene (camera, lighting, controls)
│     loadSTLPreview() — loads STL via STLLoader, centers mesh
│     loadFaceData() — fetches faces.json, builds face→direction lookup
│     onFaceClick() — Raycaster face picking, highlights selected face,
│                     shows info panel (area, normal, bbox, features)
└─ frontend/styles.css (~200+ LOC) — dark theme (CSS vars: --bg #0d0d0f,
    --surface, --border, --accent, --text, --text-muted); Flexbox card layout;
    form styling; loading overlay; error box; Three.js canvas container
    with ResizeObserver for responsive viewport
```

**Key Technical Features:**
1. Multi-format input: PNG/JPG/BMP/TIFF/WebP via Claude Vision; PDF (first page via pypdfium2 → PNG); DXF/DWG via ezdxf heuristics
2. AI-powered dimension extraction: Claude Sonnet 4.6 Vision API analyzes drawing images, returns JSON PartSpec + ClarificationQuestion list for ambiguous dimensions
3. Interactive specification wizard: Q&A clarification flow for ambiguous hand-drawn sketches, full spec editor form with dynamic fieldset generation
4. Parametric 3D solids: CadQuery builder (OpenCASCADE OCCT kernel) supports 5 base shapes + 7 feature types + 2 hole patterns
5. Multi-format export: STL (mesh), STEP (CAD assembly), OBJ (via trimesh re-export), per-face metadata JSON
6. Interactive 3D preview: Three.js with OrbitControls, STLLoader, face raycasting + highlight, surface inspector (area, normal, bounding box, associated features)
7. Session persistence: file-based session storage `UPLOADS_DIR/<uuid>/`, auto-cleanup at 24h TTL
8. ProcessPoolExecutor for CadQuery: explicit workaround for thread-unsafe OpenCASCADE kernel (documented in builder.py)

**Data Flow:**
1. User uploads file (PNG/PDF/DXF) → POST /api/extract → file type detected
2. Image/PDF path: pypdfium2 renders PDF first page → Claude Sonnet Vision API → returns PartSpec JSON + ClarificationQuestion[]
3. DXF path: ezdxf reads ENTITIES layer → LINE/CIRCLE/ARC heuristics → bounding box inference → PartSpec
4. Frontend: if clarifications → Q&A form → collect answers → POST /api/clarify (apply_clarification_answers dot-path mutation) → re-render spec editor
5. User edits spec form → collectSpec() serializes DOM → POST /api/build
6. Backend: build_and_export() runs in ProcessPoolExecutor; CadQuery builds solid; exports STL/STEP; trimesh converts STL→OBJ; generates face metadata JSON
7. Frontend: Three.js loads STL; fetches faces.json; renders 3D with face inspector on click

**Key Files:**
- `backend/main.py` — FastAPI app, ProcessPoolExecutor, session routing, cleanup loop, 5 endpoints
- `backend/extractor.py` — Claude Vision integration, DXF heuristics, dot-path clarification mutation
- `backend/builder.py` — CadQuery builder (5 shapes, 7 features, 2 patterns), exports, face metadata
- `backend/models.py` — Complete Pydantic schema with discriminated unions (10 shape/feature types)
- `frontend/app.js` (~860 LOC) — vanilla JS state machine, form rendering, Three.js integration
- `frontend/index.html` — Semantic HTML, 4-step form structure, Three.js importmap
- `frontend/styles.css` — Dark theme variables, Flexbox layout
- `backend/requirements.txt` — 10 packages (MISSING: cadquery, ezdxf — critical gap)
- `start.sh` — Conda/pip CadQuery installer, server bootstrap
- `.env.example` — ANTHROPIC_API_KEY, UPLOADS_DIR, SESSION_TTL_HOURS

**Real Metrics / Numbers from Code:**
- Supported input formats: PNG, JPG, JPEG, BMP, TIFF, TIF, WebP, PDF (1st page only), DXF, DWG (10 formats)
- Supported base shapes: 5 (box, cylinder, L-extrusion, T-extrusion, polygon)
- Supported features: 7 types (hole, counterbore, countersink, slot, fillet, chamfer, pocket)
- Hole patterns: 2 (grid, bolt-circle)
- Export formats: 3 (STL, STEP, OBJ)
- Max upload size: 20 MB
- Session TTL: 24 hours (configurable via SESSION_TTL_HOURS)
- ProcessPoolExecutor workers: 2 concurrent builds
- Three.js version: 0.170.0 (CDN, no bundler)
- AI model: claude-sonnet-4-6 (hardcoded in extractor.py)

**External Integrations / APIs Used:**
- Anthropic: `client.messages.create()` for Claude Sonnet 4-6 Vision API (image analysis)
- ezdxf: DXF/DWG file parsing (conditional import, not in requirements.txt — will crash)
- CadQuery/OpenCASCADE (OCP): OCCT-based parametric solid modeling (not in requirements.txt — conda/pip via start.sh)
- trimesh: OBJ export via STL reload (`trimesh.load(stl_path)`)
- pypdfium2: PDF first-page → PNG conversion
- FastAPI CORS: wide-open middleware (`allow_origins=["*"]`)
- python-dotenv: env var loading

**What Makes It Distinctive:**
1. ProcessPoolExecutor for CadQuery: explicit workaround for thread-unsafe OpenCASCADE kernel — documented in code comment
2. Dot-path mutation for spec updates: `apply_clarification_answers()` uses regex to parse paths like `"base_shape.dims.height"` and update nested Pydantic model fields dynamically
3. Pydantic discriminated unions: `type` field discriminator for BaseShape, Feature, HolePattern — clean polymorphism without isinstance checks
4. Face-centric 3D UI: per-face metadata (normal vector direction classification, area, bbox, feature associations) enables interactive surface inspector in Three.js
5. No frontend build step: pure ES modules + CDN importmap; Three.js loaded from CDN; works directly from filesystem

**Honest Status:**  
Prototype/MVP — not production-ready. **Critical missing deps:** `cadquery` and `ezdxf` not in `requirements.txt`; `pip install -r requirements.txt` yields non-functional app (confirmed: `import cadquery` → ModuleNotFoundError: No module named 'OCP'`). Zero tests. No CI. No Docker (CadQuery install fragile — conda on macOS vs pip on Linux). Security issues: path traversal (file.filename, session_id), no auth, CORS wide-open, unthrottled Claude Vision calls (cost abuse vector), `.env` not in `.gitignore`. Known bugs: DXF dimension heuristic unreliable, feature/pattern index desync after remove+re-add, deprecated FastAPI `@app.on_event()` hooks, no dimension validation (0/negative values silently fail at build time).

**README Claims:**  
No README.md exists. PROJECT_REVIEW.md is an audit document that explicitly flags production gaps: no tests, no CI, no Docker, no deployment manifest, critical missing dependencies.
