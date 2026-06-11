-- SovereignGrid — Supabase (PostgreSQL) schema
-- Immutable state machine: node DIDs, guild trust, job queue, tokenomic ledger.

-- ─── Nodes ────────────────────────────────────────────────────────────────────
-- One row per provider node. The DID is derived from the node's wallet address;
-- a node may reconnect from anywhere but always authenticates by signature.
create table if not exists nodes (
    id              uuid primary key default gen_random_uuid(),
    did             text not null unique,           -- did:grid:<wallet_address>
    wallet_address  text not null unique,           -- 0x… checksummed
    display_name    text,
    models          jsonb not null default '[]',    -- ["llama3.1:8b", "phi3"] (last handshake)
    max_concurrency int  not null default 1,        -- hardware capacity constraint
    engine          text not null default 'ollama', -- ollama | vllm | aphrodite
    status          text not null default 'offline' -- offline | online | draining
                    check (status in ('offline', 'online', 'draining')),
    balance_credits numeric(20, 6) not null default 0,  -- settled compute credits
    first_seen_at   timestamptz not null default now(),
    last_seen_at    timestamptz not null default now()
);

create index if not exists nodes_status_idx on nodes (status);

-- ─── Guild membership (DID whitelisting) ─────────────────────────────────────
-- A node is admitted only if a recognized Guild Administrator signed its wallet
-- address. The orchestrator verifies admin_signature against guild_admins.
create table if not exists guild_admins (
    wallet_address text primary key,                -- admin wallet (checksummed)
    label          text not null,
    added_at       timestamptz not null default now()
);

create table if not exists guild_members (
    wallet_address  text primary key,               -- member (provider) wallet
    admin_address   text not null references guild_admins (wallet_address),
    admin_signature text not null,                  -- admin's sig over "guild-admit:<member_wallet>"
    admitted_at     timestamptz not null default now(),
    revoked_at      timestamptz                     -- non-null = banned/expelled
);

create index if not exists guild_members_active_idx
    on guild_members (wallet_address) where revoked_at is null;

-- ─── Job queue ────────────────────────────────────────────────────────────────
-- The orchestrator is the only writer. APScheduler requeues stale rows:
-- a job stuck in 'running' past lease_expires_at is returned to 'queued'.
create table if not exists job_queue (
    id              uuid primary key default gen_random_uuid(),
    consumer_key    text not null,                  -- API key hash of the requester
    model           text not null,
    payload         jsonb not null,                 -- sanitized prompt payload
    status          text not null default 'queued'
                    check (status in ('queued', 'running', 'succeeded', 'failed', 'dead')),
    assigned_node   uuid references nodes (id),
    attempts        int not null default 0,
    max_attempts    int not null default 3,
    lease_expires_at timestamptz,                   -- running-job lease; stale ⇒ requeue
    result          jsonb,
    error           text,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists job_queue_dispatch_idx
    on job_queue (status, model, created_at) where status = 'queued';
create index if not exists job_queue_lease_idx
    on job_queue (lease_expires_at) where status = 'running';

-- ─── Inference ledger (Compute Proof) ────────────────────────────────────────
-- One append-only row per completed inference. Credits are computed at insert
-- from the model-size multiplier; balances are settled in hourly batches to
-- avoid write-locking nodes.balance on the hot path.
create table if not exists model_rates (
    model_pattern text primary key,    -- prefix match, e.g. 'llama3.1:70b'
    param_b       numeric not null,    -- parameter count in billions
    credit_per_1k numeric not null     -- credits per 1K generated tokens
);

create table if not exists inference_ledger (
    id            bigint generated always as identity primary key,
    job_id        uuid not null references job_queue (id),
    node_id       uuid not null references nodes (id),
    model         text not null,
    tokens_in     int  not null default 0,
    tokens_out    int  not null default 0,
    credits       numeric(20, 6) not null,
    settled       boolean not null default false,
    created_at    timestamptz not null default now()
);

create index if not exists ledger_unsettled_idx
    on inference_ledger (node_id) where settled = false;

-- ─── Settlement ──────────────────────────────────────────────────────────────
-- Called hourly by the orchestrator's APScheduler (or a Supabase cron job).
-- Aggregates unsettled ledger rows per node in one statement, then flips the
-- settled flag — a single short transaction instead of a write per inference.
create or replace function settle_ledger()
returns table (node_id uuid, credited numeric) as $$
begin
    return query
    with pending as (
        select l.id, l.node_id as nid, l.credits
        from inference_ledger l
        where l.settled = false
        for update skip locked
    ),
    totals as (
        select nid, sum(credits) as total
        from pending
        group by nid
    ),
    bump as (
        update nodes n
        set balance_credits = n.balance_credits + t.total
        from totals t
        where n.id = t.nid
        returning n.id
    )
    update inference_ledger l
    set settled = true
    from pending p
    where l.id = p.id
    returning p.nid, p.credits;
end;
$$ language plpgsql;

-- Optional: run settlement from Supabase itself (pg_cron) instead of the
-- orchestrator scheduler. Enable the pg_cron extension first.
-- select cron.schedule('settle-grid-ledger', '0 * * * *', $$select settle_ledger()$$);

-- updated_at maintenance
create or replace function touch_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists job_queue_touch on job_queue;
create trigger job_queue_touch
    before update on job_queue
    for each row execute function touch_updated_at();

-- Seed rates (edit to taste). Larger models pay out more per generated token.
insert into model_rates (model_pattern, param_b, credit_per_1k) values
    ('llama3.1:405b', 405, 40.0),
    ('llama3.1:70b',  70,  10.0),
    ('llama3.1:8b',   8,   1.5),
    ('mistral:7b',    7,   1.2),
    ('phi3',          4,   0.8)
on conflict (model_pattern) do nothing;
