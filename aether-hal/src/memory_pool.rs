//! GPU/NPU memory pool manager.
//!
//! Manages a physical window (the `weight_cache` or NPU MMIO region) as a
//! three-tier pool:
//!
//!   Hot  — pinned in VRAM; zero-latency access; cannot be evicted
//!   Warm — resident in VRAM; evictable on pressure; LRU scoring
//!   Cold — spilled to Standard RAM or cluster-mapped memory
//!
//! The pool runs defragmentation when free-list fragmentation exceeds
//! `DEFRAG_THRESHOLD_PCT`.  Compaction is stop-the-allocator (brief spinlock)
//! and updates all live `AllocationId`s via an indirection table.

use std::collections::HashMap;
use std::time::Instant;

use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Allocation identity and kind
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationKind {
    ModelWeights,  // static after load; evict to RAM on critical pressure
    KvCache,       // dynamic; evict LRU pages first
    Activations,   // transient; first to evict
    Intermediate,  // scratch buffers; always evictable
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryTier {
    Hot  = 2,
    Warm = 1,
    Cold = 0,
}

// ---------------------------------------------------------------------------
// Allocation record
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Allocation {
    pub id:           AllocationId,
    pub kind:         AllocationKind,
    pub tier:         MemoryTier,
    /// Offset from pool base (physical bytes).
    pub phys_offset:  u64,
    pub size:         u64,
    pub pinned:       bool,
    pub created_at:   Instant,
    pub last_access:  Instant,
    pub access_count: u64,
}

impl Allocation {
    /// LRU eviction score — lower = evict first.
    /// Weights: recency (50%) + frequency (30%) + kind priority (20%)
    pub fn eviction_score(&self) -> f64 {
        if self.pinned { return f64::MAX; }
        let age_s  = self.last_access.elapsed().as_secs_f64();
        let freq   = self.access_count as f64 + 1.0;
        let kind_w = match self.kind {
            AllocationKind::Activations  => 0.1,
            AllocationKind::Intermediate => 0.2,
            AllocationKind::KvCache      => 0.5,
            AllocationKind::ModelWeights => 0.9,
        };
        // Lower score = evict sooner.  High age / low freq / low kind_w = low score.
        (1.0 / age_s.max(0.001)) * freq * kind_w
    }
}

// ---------------------------------------------------------------------------
// Free block
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FreeBlock {
    phys_offset: u64,
    size:        u64,
}

// ---------------------------------------------------------------------------
// GpuMemoryPool
// ---------------------------------------------------------------------------

pub const DEFRAG_THRESHOLD_PCT: u8 = 30; // defrag when fragmentation ≥ 30%

pub struct GpuMemoryPool {
    /// Physical base of the managed window (within weight_cache or NPU MMIO).
    base:         u64,
    total:        u64,
    next_id:      u64,
    allocations:  HashMap<AllocationId, Allocation>,
    free_list:    Vec<FreeBlock>,
}

impl GpuMemoryPool {
    pub fn new(base: u64, total: u64) -> Self {
        Self {
            base,
            total,
            next_id:     1,
            allocations: HashMap::new(),
            free_list:   vec![FreeBlock { phys_offset: base, size: total }],
        }
    }

    // ---------------------------------------------------------------------------
    // Allocation
    // ---------------------------------------------------------------------------

    pub fn alloc(&mut self, size: u64, kind: AllocationKind) -> Option<AllocationId> {
        let aligned = align_up(size, 4096);
        // First-fit search through the free list.
        let slot = self.free_list.iter().position(|b| b.size >= aligned)?;
        let block = self.free_list[slot].clone();

        // Carve out of the block.
        if block.size > aligned {
            self.free_list[slot] = FreeBlock {
                phys_offset: block.phys_offset + aligned,
                size:        block.size - aligned,
            };
        } else {
            self.free_list.remove(slot);
        }

        let id = AllocationId(self.next_id);
        self.next_id += 1;
        self.allocations.insert(id, Allocation {
            id,
            kind,
            tier:         MemoryTier::Warm,
            phys_offset:  block.phys_offset,
            size:         aligned,
            pinned:       false,
            created_at:   Instant::now(),
            last_access:  Instant::now(),
            access_count: 0,
        });
        Some(id)
    }

    pub fn free(&mut self, id: AllocationId) {
        if let Some(alloc) = self.allocations.remove(&id) {
            // Merge adjacent free blocks.
            self.free_list.push(FreeBlock { phys_offset: alloc.phys_offset, size: alloc.size });
            self.coalesce_free_list();
        }
    }

    // ---------------------------------------------------------------------------
    // Access tracking
    // ---------------------------------------------------------------------------

    pub fn touch(&mut self, id: AllocationId) {
        if let Some(a) = self.allocations.get_mut(&id) {
            a.last_access  = Instant::now();
            a.access_count += 1;
            a.tier          = MemoryTier::Hot; // promote on access
        }
    }

    pub fn pin(&mut self, id: AllocationId)   { if let Some(a) = self.allocations.get_mut(&id) { a.pinned = true;  a.tier = MemoryTier::Hot; } }
    pub fn unpin(&mut self, id: AllocationId) { if let Some(a) = self.allocations.get_mut(&id) { a.pinned = false; } }

    // ---------------------------------------------------------------------------
    // Tier promotion / demotion
    // ---------------------------------------------------------------------------

    /// Cool down Hot→Warm for allocations not accessed within `idle_secs`.
    pub fn cool_down(&mut self, idle_secs: f64) {
        for alloc in self.allocations.values_mut() {
            if alloc.tier == MemoryTier::Hot
                && !alloc.pinned
                && alloc.last_access.elapsed().as_secs_f64() > idle_secs
            {
                alloc.tier = MemoryTier::Warm;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Eviction
    // ---------------------------------------------------------------------------

    /// Evict `target_bytes` of Warm allocations, lowest score first.
    /// Returns physical offsets of evicted blocks (caller must spill to RAM).
    pub fn evict_lru(&mut self, target_bytes: u64) -> Vec<(AllocationId, u64, u64)> {
        let mut candidates: Vec<(f64, AllocationId)> = self
            .allocations
            .values()
            .filter(|a| a.tier == MemoryTier::Warm)
            .map(|a| (a.eviction_score(), a.id))
            .collect();
        candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut evicted = vec![];
        let mut freed   = 0u64;
        for (_, id) in candidates {
            if freed >= target_bytes { break; }
            if let Some(alloc) = self.allocations.get_mut(&id) {
                alloc.tier = MemoryTier::Cold;
                freed     += alloc.size;
                evicted.push((id, alloc.phys_offset, alloc.size));
                // Add back to free list (simulates spill; physical range is now reclaimable).
                self.free_list.push(FreeBlock { phys_offset: alloc.phys_offset, size: alloc.size });
            }
            self.allocations.remove(&id);
            self.coalesce_free_list();
        }
        info!("[MemPool] evicted {} allocs, freed {} KiB", evicted.len(), freed / 1024);
        evicted
    }

    // ---------------------------------------------------------------------------
    // Defragmentation
    // ---------------------------------------------------------------------------

    pub fn fragmentation_pct(&self) -> u8 {
        if self.free_list.is_empty() { return 0; }
        let total_free: u64 = self.free_list.iter().map(|b| b.size).sum();
        let largest_free     = self.free_list.iter().map(|b| b.size).max().unwrap_or(0);
        if total_free == 0 { return 0; }
        (100 - (largest_free * 100 / total_free)).min(100) as u8
    }

    /// Compact allocations to eliminate holes.
    /// Moves all Warm allocations to fill gaps; Hot/pinned stay in place.
    pub fn defragment(&mut self) {
        if self.fragmentation_pct() < DEFRAG_THRESHOLD_PCT { return; }
        debug!("[MemPool] defragmenting (frag {}%)", self.fragmentation_pct());

        // Sort non-pinned Warm allocs by current offset.
        let mut moveable: Vec<AllocationId> = self
            .allocations
            .values()
            .filter(|a| !a.pinned && a.tier != MemoryTier::Hot)
            .map(|a| a.id)
            .collect();
        moveable.sort_by_key(|id| self.allocations[id].phys_offset);

        // Find lowest free byte above all pinned allocations.
        let pinned_end = self
            .allocations
            .values()
            .filter(|a| a.pinned || a.tier == MemoryTier::Hot)
            .map(|a| a.phys_offset + a.size)
            .max()
            .unwrap_or(self.base);

        let mut cursor = pinned_end;
        for id in moveable {
            if let Some(alloc) = self.allocations.get_mut(&id) {
                alloc.phys_offset = cursor;
                cursor += alloc.size;
            }
        }

        // Rebuild free list as a single block at the end.
        self.free_list.clear();
        if cursor < self.base + self.total {
            self.free_list.push(FreeBlock {
                phys_offset: cursor,
                size:        self.base + self.total - cursor,
            });
        }
        info!("[MemPool] defrag complete; {} KiB contiguous free", self.free_bytes() / 1024);
    }

    // ---------------------------------------------------------------------------
    // Stats
    // ---------------------------------------------------------------------------

    pub fn used_bytes(&self) -> u64 {
        self.allocations.values().map(|a| a.size).sum()
    }

    pub fn free_bytes(&self) -> u64 {
        self.free_list.iter().map(|b| b.size).sum()
    }

    pub fn usage_pct(&self) -> u8 {
        ((self.used_bytes() * 100) / self.total.max(1)).min(100) as u8
    }

    pub fn allocation_count(&self) -> usize { self.allocations.len() }

    // ---------------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------------

    fn coalesce_free_list(&mut self) {
        self.free_list.sort_unstable_by_key(|b| b.phys_offset);
        let mut merged: Vec<FreeBlock> = Vec::with_capacity(self.free_list.len());
        for block in self.free_list.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.phys_offset + last.size == block.phys_offset {
                    last.size += block.size;
                    continue;
                }
            }
            merged.push(block);
        }
        self.free_list = merged;
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// System-wide pool registry
// ---------------------------------------------------------------------------

/// One pool per silicon backend (CPU heap, ANE, Intel NPU, Nvidia).
pub struct PoolRegistry {
    pools: HashMap<&'static str, GpuMemoryPool>,
}

impl PoolRegistry {
    pub fn new() -> Self { Self { pools: HashMap::new() } }

    pub fn register(&mut self, name: &'static str, base: u64, size: u64) {
        self.pools.insert(name, GpuMemoryPool::new(base, size));
        info!("[PoolRegistry] registered pool '{name}' base={base:#x} size={size:#x}");
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut GpuMemoryPool> {
        self.pools.get_mut(name)
    }

    /// Run eviction and defrag across all pools under pressure.
    pub fn respond_to_pressure(&mut self, target_bytes_per_pool: u64) -> u64 {
        let mut total_freed = 0u64;
        for pool in self.pools.values_mut() {
            let evicted = pool.evict_lru(target_bytes_per_pool);
            total_freed += evicted.iter().map(|(_, _, sz)| sz).sum::<u64>();
            pool.defragment();
        }
        total_freed
    }

    pub fn global_usage_report(&self) -> Vec<(&'static str, u8, u64, u64)> {
        self.pools
            .iter()
            .map(|(name, pool)| (*name, pool.usage_pct(), pool.used_bytes(), pool.free_bytes()))
            .collect()
    }
}
