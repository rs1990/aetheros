//! MemoryOptimizerAgent — real-time CPU and GPU memory optimization.
//!
//! Responsibilities:
//!   1. Poll `MemoryMonitor` every 100 ms for per-region pressure levels.
//!   2. Drive `PoolRegistry::respond_to_pressure` on GPU pools.
//!   3. Drive `AccessRegistry::eviction_candidates` on the USDF store.
//!   4. Under OOM: broadcast `AgentControl::OffloadToCluster` so the
//!      OrchestratorAgent migrates tasks to cluster nodes.
//!   5. Cool-down Hot→Warm tier demotion every 30 s.
//!
//! The agent also listens for `ResourceAdded` events from the Orchestrator
//! so it can register new GPU pools without a restart.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::Receiver;
use tracing::{info, warn};

use aether_core::agent::AgentId;
use aether_core::ipc::{BusMessage, MessagePayload};
use aether_core::memory_pressure::{
    EvictionRegion, EvictionRequest, MemoryMonitor, MemorySnapshot, PressureLevel,
};
use aether_hal::memory_pool::PoolRegistry;
use aether_usdf::retention::AccessRegistry;
use aether_usdf::vector::VectorStore;

use crate::bus::AgentMailbox;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

const POLL_INTERVAL:   Duration = Duration::from_millis(100);
const COOLDOWN_EVERY:  Duration = Duration::from_secs(30);
const HOT_IDLE_SECS:   f64     = 10.0;  // Hot→Warm demotion threshold
const MAX_EVICT_BATCH: usize   = 64;     // USDF entries per eviction pass

// ---------------------------------------------------------------------------
// Simulated system memory sampler
// ---------------------------------------------------------------------------

/// In production this reads `/proc/meminfo` or kernel-exported counters.
/// Here we parse `/proc/meminfo` on Linux or return synthetic values elsewhere.
fn sample_system_ram() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let parse = |key: &str| -> u64 {
                content.lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0) * 1024 // kB → bytes
            };
            let total     = parse("MemTotal:");
            let available = parse("MemAvailable:");
            return (total.saturating_sub(available), total);
        }
    }
    // macOS / fallback: return synthetic plausible values.
    (4 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024) // 4 GiB used / 16 GiB total
}

// ---------------------------------------------------------------------------
// MemoryOptimizerAgent
// ---------------------------------------------------------------------------

pub struct MemoryOptimizerAgent {
    monitor:  MemoryMonitor,
    pools:    PoolRegistry,
    /// Shared access registry from the USDF layer.
    registry: Arc<Mutex<AccessRegistry>>,
    store:    Arc<Mutex<VectorStore>>,
    last_cooldown: Instant,
}

impl MemoryOptimizerAgent {
    pub fn new(
        registry: Arc<Mutex<AccessRegistry>>,
        store:    Arc<Mutex<VectorStore>>,
    ) -> Self {
        // Build initial snapshot from system memory.
        let (ram_used, ram_total) = sample_system_ram();
        let mut snapshot = MemorySnapshot::new(
            ram_total,
            128 * 1024 * 1024 * 1024, // 128 GiB weight-cache (physical limit)
            128 * 1024 * 1024 * 1024, // 128 GiB cluster-mapped
            64  * 1024 * 1024 * 1024, // 64 GiB persona overlay
        );
        snapshot.standard_ram.used_bytes = ram_used;
        snapshot.refresh_all();

        let monitor = MemoryMonitor::new(snapshot);

        // Register default pools aligned to the physical memory map.
        let mut pools = PoolRegistry::new();
        pools.register("cpu_heap",    0x0000_0000_0100_0000, ram_total);
        pools.register("weight_cache",0x0000_0020_0000_0000, 128 * 1024 * 1024 * 1024);
        pools.register("ane_vram",    0x0000_0080_0000_0000, 16  * 1024 * 1024 * 1024);
        pools.register("intel_npu",   0x0000_0090_0000_0000, 16  * 1024 * 1024 * 1024);
        pools.register("nvidia_vram", 0x0000_00A0_0000_0000, 32  * 1024 * 1024 * 1024);

        Self { monitor, pools, registry, store, last_cooldown: Instant::now() }
    }

    /// Main loop — runs on its own thread.
    pub fn run(
        mut self,
        _id:     AgentId,
        rx:      Receiver<BusMessage>,
        mailbox: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[MemoryOptimizer] started");
        let mut last_tick = Instant::now();

        loop {
            // Drain inbound messages (non-blocking).
            while let Ok(msg) = rx.try_recv() {
                self.handle_message(msg, &mailbox);
            }

            let delta_ns = last_tick.elapsed().as_nanos() as u64;
            last_tick = Instant::now();

            // Refresh CPU RAM sample.
            let (used, _) = sample_system_ram();
            self.monitor.snapshot.standard_ram.used_bytes = used;
            self.monitor.tick(delta_ns);

            // Act on current pressure level.
            self.respond_to_pressure(&mailbox);

            // Periodic tier cool-down.
            if self.last_cooldown.elapsed() >= COOLDOWN_EVERY {
                for pool_name in ["ane_vram", "intel_npu", "nvidia_vram", "weight_cache"] {
                    if let Some(pool) = self.pools.get_mut(pool_name) {
                        pool.cool_down(HOT_IDLE_SECS);
                    }
                }
                self.last_cooldown = Instant::now();
            }

            thread::sleep(POLL_INTERVAL);
        }
    }

    fn handle_message(
        &mut self,
        msg:     BusMessage,
        _mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        // When a new compute unit is added, register its memory pool.
        if let MessagePayload::AgentControl { opcode: 0x10, arg: unit_id, .. } = msg.payload {
            // In production: query HAL for VRAM size of the new unit.
            // Stub: register a 8 GiB pool for unit_id.
            let name = Box::leak(format!("unit_{unit_id}").into_boxed_str());
            let base  = 0x0000_00C0_0000_0000 + unit_id * 0x0002_0000_0000;
            self.pools.register(name, base, 8 * 1024 * 1024 * 1024);
            info!("[MemoryOptimizer] registered pool for unit {unit_id}");
        }
    }

    fn respond_to_pressure(&mut self, mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>) {
        let level = self.monitor.snapshot.system_level;
        match level {
            PressureLevel::Normal => {}

            PressureLevel::Elevated => {
                // Evict a small batch of cold USDF entries.
                self.evict_usdf(16);
                let report = self.pools.global_usage_report();
                for (name, pct, used, free) in &report {
                    if *pct > 70 {
                        self.pools.respond_to_pressure(256 * 1024 * 1024); // 256 MiB
                        break;
                    }
                }
            }

            PressureLevel::Critical => {
                warn!("[MemoryOptimizer] CRITICAL pressure — aggressive eviction");
                self.evict_usdf(MAX_EVICT_BATCH);
                self.pools.respond_to_pressure(1024 * 1024 * 1024); // 1 GiB
            }

            PressureLevel::Oom => {
                warn!("[MemoryOptimizer] OOM — force-evict + offload to cluster");
                self.evict_usdf(MAX_EVICT_BATCH * 4);
                self.pools.respond_to_pressure(u64::MAX);
                // Notify Orchestrator to shed tasks to cluster.
                let msg = BusMessage {
                    origin:  AgentId::MEMORY_OPTIMIZER,
                    dest:    AgentId::ORCHESTRATOR,
                    seq:     0,
                    payload: MessagePayload::AgentControl {
                        opcode: 0x20, // OffloadToCluster
                        target: AgentId::SCHEDULER,
                        arg:    0,
                    },
                };
                if let Ok(map) = mailbox.lock() {
                    if let Some(mb) = map.get(&AgentId::ORCHESTRATOR.0) {
                        let _ = mb.tx.send(msg);
                    }
                }
            }
        }
    }

    fn evict_usdf(&self, batch: usize) {
        if let (Ok(mut reg), Ok(mut store)) = (self.registry.lock(), self.store.lock()) {
            let candidates = reg.eviction_candidates(batch);
            for id in candidates {
                store.remove(id);
                reg.on_remove(id);
            }
            if batch > 0 {
                info!("[MemoryOptimizer] evicted up to {batch} USDF entries");
            }
        }
    }

    pub fn pressure_level(&self) -> PressureLevel {
        self.monitor.snapshot.system_level
    }

    pub fn pool_report(&self) -> Vec<(&'static str, u8, u64, u64)> {
        self.pools.global_usage_report()
    }
}
