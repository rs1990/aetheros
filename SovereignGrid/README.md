# SovereignGrid

A decentralized, sovereign AI computing grid: a cooperative pool of developers
(a DAO or Guild) connect their local GPU hardware running open-source models
(Llama 3.1, Mistral, Phi-3, …) into a single unified asynchronous API gateway
for heavy background agent workloads.

```
 consumer agents                         provider guild
      │                                       │
      ▼                                       ▼
 POST /v1/chat/completions          ┌──────────────────┐
 POST /v1/jobs (async)              │ provider daemon   │── Ollama  :11434
      │                             │ (wallet-signed)   │── vLLM    :8000
      ▼                             └────────┬─────────┘
┌─────────────────────┐    persistent WS     │
│  FastAPI gateway     │◄────────────────────┘
│  · smart dispatch    │
│  · sanitize sandbox  │        ┌──────────────────────┐
│  · APScheduler       │───────►│ Supabase (PostgreSQL) │
│    requeue + settle  │        │ nodes · guild · jobs  │
└─────────────────────┘        │ inference_ledger      │
                               └──────────────────────┘
```

## Components

| Path | Role |
|---|---|
| `orchestrator/` | FastAPI gateway: WebSocket hub, smart dispatch, sanitization, APScheduler requeue/settlement |
| `daemon/` | Provider daemon: binds strictly to a local engine, wallet-signature auth, reconnect with backoff |
| `supabase/schema.sql` | DDL for `nodes`, `guild_admins`, `guild_members`, `job_queue`, `inference_ledger`, `model_rates` + `settle_ledger()` |

## How it works

**Capability handshake.** On WS connect the gateway issues a one-time nonce.
The daemon signs `grid-auth:<nonce>` with its wallet key (EIP-191), and
broadcasts its active models and `max_concurrency`. The recovered address must
be guild-admitted: a Guild Administrator's signature over
`guild-admit:<wallet>` is stored in `guild_members` and re-verified on every
connect — spoofed rows without a valid admin signature are rejected.

**Smart dispatch.** Consumer requests are matched to the least-loaded online
node serving the requested model. Each node runs at most `max_concurrency`
jobs.

**Failure handling.** Three layers, all converging on the same requeue path:
1. WS disconnect → every in-flight job on that socket is requeued immediately
   and waiting consumers fail over to the next node.
2. Dispatch timeout → requeue.
3. APScheduler sweep (15 s) → any `running` job whose lease expired (missed
   disconnect, network partition) is requeued. After `max_attempts` a job goes
   `dead` instead of looping forever.

**Compute credits.** Each completed inference appends one `inference_ledger`
row priced by the model-size rate table (70B pays ~7× an 8B per generated
token). Node balances are only updated by the hourly `settle_ledger()` batch
(APScheduler cron or Supabase pg_cron) — one short transaction per hour
instead of a write-lock per inference.

**Sanitization sandbox.** The gateway rejects prompts matching host-execution
payload patterns (pipe-to-shell, fork bombs, reverse shells, `file:///etc/…`)
before anything reaches a provider machine. Heuristic by design; the guild
trust layer is the primary boundary.

## Quick start (dev, no database)

```bash
# 1. Gateway (in-memory store when SUPABASE_URL is unset)
cd orchestrator && pip install -r requirements.txt
uvicorn main:app --port 8080

# 2. Admit yourself to the guild (dev shortcut: with the memory store,
#    insert an admin + member via a tiny script, or run against Supabase)
python - <<'EOF'
from eth_account import Account
acct = Account.create()
print("wallet:", acct.address)
print("key   :", acct.key.hex())
EOF

# 3. Provider daemon (needs a local Ollama with at least one model pulled)
cd daemon && pip install -r requirements.txt
GRID_WALLET_KEY=0x... python provider_daemon.py \
    --gateway ws://127.0.0.1:8080/ws/provider --engine ollama

# 4. Consume
curl -s localhost:8080/v1/chat/completions \
  -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"hello"}]}'
```

## Production setup

1. Create a Supabase project, run `supabase/schema.sql` in the SQL editor.
2. Insert guild admins into `guild_admins`; each admin signs members with
   `auth.sign_admission()` (see `orchestrator/auth.py`) and inserts the row.
3. Set `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` for the orchestrator. The
   service-role key stays server-side only.
4. Optionally move settlement into the database:
   `select cron.schedule('settle-grid-ledger', '0 * * * *', $$select settle_ledger()$$);`

## Known limitations (skeleton scope)

- No streaming (SSE) relay yet — results return whole.
- Consumer auth is a bearer-key hash, not metered billing.
- "Compute Proof" trusts daemon-reported token counts; cross-checking via
  spot re-execution or output-length bounds is future work.
- Single orchestrator instance (registry is in-process). Horizontal scale
  needs sticky WS routing or a shared dispatch bus.
