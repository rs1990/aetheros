//! Memory pressure monitoring and callback dispatch.
//!
//! The kernel samples physical memory usage every `SAMPLE_INTERVAL_NS`
//! nanoseconds.  Each `MemoryRegion` has independent watermarks.  When a
//! region crosses a watermark the `PressureLevel` escalates and all
//! registered `PressureCallback`s are fired in priority order.
//!
//! Four levels:
//!   Normal   (<70%)  — no action
//!   Elevated (70-85%) — flush soft-pinned USDF cache
//!   Critical (85-95%) — evict KV-cache pages, unload cold model weights
//!   Oom      (>95%)  — force-evict all non-essential pages, offload to cluster

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Pressure levels
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PressureLevel {
    Normal   = 0,
    Elevated = 1,
    Critical = 2,
    Oom      = 3,
}

impl PressureLevel {
    pub fn from_usage_pct(pct: u8) -> Self {
        match pct {
            0..=69  => Self::Normal,
            70..=84 => Self::Elevated,
            85..=94 => Self::Critical,
            _       => Self::Oom,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal   => "normal",
            Self::Elevated => "elevated",
            Self::Critical => "critical",
            Self::Oom      => "oom",
        }
    }
}

// ---------------------------------------------------------------------------
// Per-region watermarks
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct RegionWatermarks {
    /// Total bytes in this region.
    pub total_bytes:    u64,
    /// Bytes currently allocated / dirty.
    pub used_bytes:     u64,
    /// Bytes pinned (cannot be evicted under any pressure level).
    pub pinned_bytes:   u64,
    /// Current computed pressure level.
    pub level:          PressureLevel,
}

impl RegionWatermarks {
    pub const fn new(total_bytes: u64) -> Self {
        Self { total_bytes, used_bytes: 0, pinned_bytes: 0, level: PressureLevel::Normal }
    }

    pub fn usage_pct(&self) -> u8 {
        if self.total_bytes == 0 {
            return 0;
        }
        ((self.used_bytes * 100) / self.total_bytes).min(100) as u8
    }

    pub fn evictable_bytes(&self) -> u64 {
        self.used_bytes.saturating_sub(self.pinned_bytes)
    }

    pub fn refresh(&mut self) {
        self.level = PressureLevel::from_usage_pct(self.usage_pct());
    }
}

// ---------------------------------------------------------------------------
// System-wide memory snapshot
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct MemorySnapshot {
    pub standard_ram:   RegionWatermarks,
    pub weight_cache:   RegionWatermarks,
    pub cluster_mapped: RegionWatermarks,
    pub persona:        RegionWatermarks,
    /// Combined worst-case pressure across all regions.
    pub system_level:   PressureLevel,
}

impl MemorySnapshot {
    pub fn new(
        ram_total:     u64,
        wc_total:      u64,
        cluster_total: u64,
        persona_total: u64,
    ) -> Self {
        Self {
            standard_ram:   RegionWatermarks::new(ram_total),
            weight_cache:   RegionWatermarks::new(wc_total),
            cluster_mapped: RegionWatermarks::new(cluster_total),
            persona:        RegionWatermarks::new(persona_total),
            system_level:   PressureLevel::Normal,
        }
    }

    pub fn refresh_all(&mut self) {
        self.standard_ram.refresh();
        self.weight_cache.refresh();
        self.cluster_mapped.refresh();
        self.persona.refresh();
        self.system_level = [
            self.standard_ram.level,
            self.weight_cache.level,
            self.cluster_mapped.level,
        ]
        .iter()
        .copied()
        .max()
        .unwrap_or(PressureLevel::Normal);
    }
}

// ---------------------------------------------------------------------------
// Eviction request — emitted by the monitor toward USDF / agents
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct EvictionRequest {
    pub region:        EvictionRegion,
    pub target_bytes:  u64, // how much to free
    pub pressure:      PressureLevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionRegion {
    StandardRam,
    WeightCache,
    ClusterMapped,
    Persona,
}

// ---------------------------------------------------------------------------
// PressureCallback trait (no_std-compatible function pointer style)
// ---------------------------------------------------------------------------

/// Implement this for any subsystem that can shed load under memory pressure.
pub trait PressureCallback: Send + Sync {
    fn priority(&self) -> u8; // higher = called first
    fn on_pressure(&mut self, req: EvictionRequest) -> u64; // returns bytes freed
}

// ---------------------------------------------------------------------------
// MemoryMonitor
// ---------------------------------------------------------------------------

pub const SAMPLE_INTERVAL_NS: u64 = 100_000_000; // 100 ms

pub struct MemoryMonitor {
    pub snapshot:  MemorySnapshot,
    prev_level:    PressureLevel,
    #[cfg(feature = "alloc")]
    callbacks:     Vec<Box<dyn PressureCallback>>,
    tick_accum_ns: u64,
}

impl MemoryMonitor {
    pub fn new(snapshot: MemorySnapshot) -> Self {
        Self {
            prev_level:    PressureLevel::Normal,
            snapshot,
            #[cfg(feature = "alloc")]
            callbacks:     Vec::new(),
            tick_accum_ns: 0,
        }
    }

    #[cfg(feature = "alloc")]
    pub fn register_callback(&mut self, cb: Box<dyn PressureCallback>) {
        self.callbacks.push(cb);
        self.callbacks.sort_unstable_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Called by the kernel tick handler.  `delta_ns` = time since last call.
    pub fn tick(&mut self, delta_ns: u64) {
        self.tick_accum_ns += delta_ns;
        if self.tick_accum_ns < SAMPLE_INTERVAL_NS {
            return;
        }
        self.tick_accum_ns = 0;
        self.snapshot.refresh_all();

        if self.snapshot.system_level > self.prev_level {
            self.on_pressure_escalate();
        }
        self.prev_level = self.snapshot.system_level;
    }

    fn on_pressure_escalate(&mut self) {
        let level = self.snapshot.system_level;
        // Compute target: free enough to drop back to Elevated.
        let target = self.compute_target_bytes(level);
        if target == 0 {
            return;
        }
        let req = EvictionRequest {
            region:       EvictionRegion::WeightCache, // KV-cache first
            target_bytes: target,
            pressure:     level,
        };
        #[cfg(feature = "alloc")]
        {
            let mut freed = 0u64;
            for cb in &mut self.callbacks {
                if freed >= target {
                    break;
                }
                freed += cb.on_pressure(req);
            }
        }
    }

    fn compute_target_bytes(&self, level: PressureLevel) -> u64 {
        let wc = &self.snapshot.weight_cache;
        match level {
            PressureLevel::Normal   => 0,
            PressureLevel::Elevated => wc.evictable_bytes() / 4, // free 25%
            PressureLevel::Critical => wc.evictable_bytes() / 2, // free 50%
            PressureLevel::Oom      => wc.evictable_bytes(),      // free all evictable
        }
    }
}
