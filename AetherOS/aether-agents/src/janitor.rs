//! JanitorAgent — automated cleanup of stale system state.
//!
//! Runs on a configurable schedule (default: every 5 minutes) and cleans:
//!
//!   1. USDF VectorStore — evict entries scoring below `EVICTION_SCORE_THRESHOLD`
//!      using the `AccessRegistry` + `RetentionScorer`.
//!
//!   2. Synthesized driver shims — remove source strings for shims whose
//!      device has been dead for more than `SHIM_TTL`.
//!
//!   3. Stale RDMA imports — unmap cluster regions whose owning node has
//!      status `Dead` in the gossip membership table.
//!
//!   4. Persona Overlay pages — evict agent-weight pages not accessed within
//!      `PERSONA_IDLE_TTL`.
//!
//!   5. Log vectors — USDF entries tagged `kind=log` older than `LOG_TTL`.
//!
//! After each sweep the agent emits a `CleanupReport` summary via the bus
//! and writes it to the USDF under `/system/janitor/last_report`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossbeam::channel::Receiver;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use aether_core::agent::AgentId;
use aether_core::ipc::{BusMessage, MessagePayload};
use aether_cluster::gossip::NodeStatus;
use aether_cluster::rdma::ClusterMemoryMap;
use aether_usdf::retention::AccessRegistry;
use aether_usdf::vector::{SemanticNamespace, VectorStore};
use aether_usdf::vfs::VfsBridge;

use crate::bus::AgentMailbox;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

const SWEEP_INTERVAL:        Duration = Duration::from_secs(300);  // 5 minutes
const SHIM_TTL:              Duration = Duration::from_secs(3600); // 1 hour
const PERSONA_IDLE_TTL:      Duration = Duration::from_secs(7200); // 2 hours
const LOG_TTL:               Duration = Duration::from_secs(86400 * 3); // 3 days
const MAX_EVICT_PER_SWEEP:   usize    = 256;
const EVICTION_SCORE_THRESHOLD: f64  = 0.01; // anything below this is swept

// ---------------------------------------------------------------------------
// CleanupReport
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupReport {
    pub sweep_id:          u64,
    pub timestamp_secs:    u64,
    pub duration_ms:       u64,
    pub usdf_evicted:      usize,
    pub shims_removed:     usize,
    pub rdma_unmapped:     usize,
    pub persona_pages_freed: usize,
    pub log_vectors_purged: usize,
    pub bytes_recovered:   u64,
}

impl CleanupReport {
    pub fn empty(sweep_id: u64) -> Self {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Self {
            sweep_id,
            timestamp_secs:    ts,
            duration_ms:       0,
            usdf_evicted:      0,
            shims_removed:     0,
            rdma_unmapped:     0,
            persona_pages_freed: 0,
            log_vectors_purged: 0,
            bytes_recovered:   0,
        }
    }
}

// ---------------------------------------------------------------------------
// JanitorConfig — runtime-tunable parameters
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct JanitorConfig {
    pub sweep_interval:     Duration,
    pub shim_ttl:           Duration,
    pub persona_idle_ttl:   Duration,
    pub log_ttl:            Duration,
    pub max_evict_per_sweep: usize,
}

impl Default for JanitorConfig {
    fn default() -> Self {
        Self {
            sweep_interval:      SWEEP_INTERVAL,
            shim_ttl:            SHIM_TTL,
            persona_idle_ttl:    PERSONA_IDLE_TTL,
            log_ttl:             LOG_TTL,
            max_evict_per_sweep: MAX_EVICT_PER_SWEEP,
        }
    }
}

// ---------------------------------------------------------------------------
// JanitorAgent
// ---------------------------------------------------------------------------

pub struct JanitorAgent {
    config:         JanitorConfig,
    sweep_id:       u64,
    last_sweep:     Option<Instant>,

    // Shared state handles.
    store:          Arc<Mutex<VectorStore>>,
    registry:       Arc<Mutex<AccessRegistry>>,
    bridge:         Arc<Mutex<VfsBridge>>,
    cluster_map:    Arc<Mutex<ClusterMemoryMap>>,

    /// Synthesized driver shim source strings, keyed by device name.
    /// Shared with SynthesisAgent's shim_cache.
    shim_cache:     Arc<Mutex<HashMap<String, (String, Instant)>>>,

    /// Gossip membership: node_id → status string.
    dead_nodes:     Arc<Mutex<Vec<[u8; 16]>>>,
}

impl JanitorAgent {
    pub fn new(
        store:       Arc<Mutex<VectorStore>>,
        registry:    Arc<Mutex<AccessRegistry>>,
        bridge:      Arc<Mutex<VfsBridge>>,
        cluster_map: Arc<Mutex<ClusterMemoryMap>>,
        shim_cache:  Arc<Mutex<HashMap<String, (String, Instant)>>>,
        dead_nodes:  Arc<Mutex<Vec<[u8; 16]>>>,
    ) -> Self {
        Self {
            config:      JanitorConfig::default(),
            sweep_id:    0,
            last_sweep:  None,
            store,
            registry,
            bridge,
            cluster_map,
            shim_cache,
            dead_nodes,
        }
    }

    pub fn with_config(mut self, cfg: JanitorConfig) -> Self {
        self.config = cfg;
        self
    }

    /// Main loop.
    pub fn run(
        mut self,
        _id:     AgentId,
        rx:      Receiver<BusMessage>,
        mailbox: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[Janitor] started (sweep every {:?})", self.config.sweep_interval);

        loop {
            // Drain inbound messages.
            while let Ok(msg) = rx.try_recv() {
                // opcode 0x40 = ForceSweepNow
                if let MessagePayload::AgentControl { opcode: 0x40, .. } = msg.payload {
                    info!("[Janitor] forced sweep requested");
                    self.run_sweep(&mailbox);
                }
                // opcode 0x41 = UpdateJanitorConfig (arg encodes new interval in seconds)
                if let MessagePayload::AgentControl { opcode: 0x41, arg, .. } = msg.payload {
                    if arg > 0 {
                        self.config.sweep_interval = Duration::from_secs(arg);
                        info!("[Janitor] sweep interval updated to {} s", arg);
                    }
                }
            }

            if self.last_sweep.map_or(true, |t| t.elapsed() >= self.config.sweep_interval) {
                self.run_sweep(&mailbox);
            }

            thread::sleep(Duration::from_secs(30));
        }
    }

    // ---------------------------------------------------------------------------
    // Sweep orchestration
    // ---------------------------------------------------------------------------

    fn run_sweep(&mut self, mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>) {
        self.sweep_id += 1;
        let start = Instant::now();
        let mut report = CleanupReport::empty(self.sweep_id);
        info!("[Janitor] sweep #{} starting", self.sweep_id);

        report.usdf_evicted        = self.sweep_usdf();
        report.log_vectors_purged  = self.purge_log_vectors();
        report.shims_removed       = self.sweep_shim_cache();
        report.rdma_unmapped       = self.sweep_dead_rdma();
        report.persona_pages_freed = self.sweep_persona_overlay();
        report.duration_ms         = start.elapsed().as_millis() as u64;

        // Estimate bytes recovered (rough: average 4 KiB per USDF entry, 8 KiB per shim).
        report.bytes_recovered =
            (report.usdf_evicted + report.log_vectors_purged) as u64 * 4096
            + report.shims_removed as u64 * 8192
            + report.rdma_unmapped as u64 * 1024 * 1024; // RDMA regions are typically MiB-scale

        info!(
            "[Janitor] sweep #{} done in {} ms — usdf={} logs={} shims={} rdma={} persona={} freed={}KiB",
            self.sweep_id,
            report.duration_ms,
            report.usdf_evicted,
            report.log_vectors_purged,
            report.shims_removed,
            report.rdma_unmapped,
            report.persona_pages_freed,
            report.bytes_recovered / 1024,
        );

        self.persist_report(&report);
        self.broadcast_report(&report, mailbox);
        self.last_sweep = Some(Instant::now());
    }

    // ---------------------------------------------------------------------------
    // Sweep passes
    // ---------------------------------------------------------------------------

    /// Evict low-scoring USDF vector entries.
    fn sweep_usdf(&self) -> usize {
        let (mut reg, mut store) = match (self.registry.lock(), self.store.lock()) {
            (Ok(r), Ok(s)) => (r, s),
            _ => return 0,
        };
        let candidates = reg.eviction_candidates(self.config.max_evict_per_sweep);
        let count = candidates.len();
        for id in candidates {
            store.remove(id);
            reg.on_remove(id);
        }
        if count > 0 { debug!("[Janitor] usdf: evicted {count} entries"); }
        count
    }

    /// Purge log vectors (tagged `kind=log`) older than `log_ttl`.
    fn purge_log_vectors(&self) -> usize {
        let (mut reg, mut store) = match (self.registry.lock(), self.store.lock()) {
            (Ok(r), Ok(s)) => (r, s),
            _ => return 0,
        };
        let aged = reg.aged_candidates(self.config.log_ttl);
        let mut purged = 0usize;
        for id in aged {
            if let Some(v) = store.get(id) {
                if v.meta.get("kind").map(|k| k == "log").unwrap_or(false) {
                    let _ = store.remove(id);
                    reg.on_remove(id);
                    purged += 1;
                }
            }
        }
        if purged > 0 { debug!("[Janitor] logs: purged {purged} log vectors"); }
        purged
    }

    /// Remove synthesized shim sources older than `shim_ttl`.
    fn sweep_shim_cache(&self) -> usize {
        let mut cache = match self.shim_cache.lock() {
            Ok(c) => c,
            _     => return 0,
        };
        let before = cache.len();
        cache.retain(|_, (_, ts)| ts.elapsed() < self.config.shim_ttl);
        let removed = before - cache.len();
        if removed > 0 { debug!("[Janitor] shims: removed {removed} stale shim sources"); }
        removed
    }

    /// Unmap RDMA regions whose owning nodes are dead.
    fn sweep_dead_rdma(&self) -> usize {
        let dead = match self.dead_nodes.lock() {
            Ok(d) => d.clone(),
            _     => return 0,
        };
        if dead.is_empty() { return 0; }

        let mut cmap = match self.cluster_map.lock() {
            Ok(m) => m,
            _     => return 0,
        };

        let stale_ids: Vec<_> = cmap
            .imported_regions()
            .filter(|r| dead.contains(&r.owner.0))
            .map(|r| r.id)
            .collect();

        let count = stale_ids.len();
        for id in stale_ids {
            cmap.unmap(id);
        }
        if count > 0 { debug!("[Janitor] rdma: unmapped {count} dead-node regions"); }
        count
    }

    /// Sweep persona overlay: evict agent-weight entries idle for `persona_idle_ttl`.
    /// In production: sends a `MemoryControl::EvictPersonaPage` to the kernel.
    fn sweep_persona_overlay(&self) -> usize {
        // Access tracking for the Persona Overlay is managed by the kernel's
        // page-table accessed bits.  Here we simulate counting idle pages
        // and reporting them; the kernel decommits on the next scan.
        // Production: read /proc/aether/persona/idle_pages.
        debug!("[Janitor] persona: scanning idle pages (stub)");
        0 // stub — real impl queries kernel page-walk results
    }

    // ---------------------------------------------------------------------------
    // Reporting
    // ---------------------------------------------------------------------------

    fn persist_report(&self, report: &CleanupReport) {
        let json = serde_json::to_string_pretty(report).unwrap_or_default();
        let embedding = vec![0.0f32; 768]; // placeholder
        let meta = HashMap::from([
            ("kind".to_string(),     "janitor_report".to_string()),
            ("sweep_id".to_string(), report.sweep_id.to_string()),
        ]);

        if let (Ok(mut store), Ok(mut reg), Ok(mut bridge)) = (
            self.store.lock(),
            self.registry.lock(),
            self.bridge.lock(),
        ) {
            let id = store.insert(
                SemanticNamespace::SYSTEM,
                embedding,
                Some(json.into_bytes()),
                meta,
            );
            let path = format!("/system/janitor/sweep_{}", report.sweep_id);
            bridge.bind(&path, id, SemanticNamespace::SYSTEM);
            reg.on_insert(id, 512, Some(path));
        }
    }

    fn broadcast_report(
        &self,
        report:  &CleanupReport,
        mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        let msg = BusMessage {
            origin:  AgentId::JANITOR,
            dest:    AgentId::BROADCAST,
            seq:     0,
            payload: MessagePayload::AgentControl {
                opcode: 0x42, // CleanupComplete
                target: AgentId::KERNEL,
                arg:    report.bytes_recovered,
            },
        };
        if let Ok(map) = mailbox.lock() {
            for mb in map.values() { let _ = mb.tx.send(msg); }
        }
    }
}
