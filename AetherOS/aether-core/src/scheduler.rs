//! Neural Intent Scheduler — treats KV-Cache pages as a first-class resource.
//!
//! Priority tiers (highest → lowest):
//!   RealTime   — interrupt handlers, DMA completion callbacks
//!   NpuInfer   — inference requests; gets Weight-Cache pages pinned
//!   KvCache    — KV-Cache maintenance (eviction, prefill, prefix-share)
//!   System     — agent housekeeping, driver synthesis
//!   Background — idle USDF indexing, cluster gossip rebalance
//!
//! New hardware is added at runtime via `add_compute_unit` — no reboot.

use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TaskPriority {
    Background = 0,
    System     = 1,
    KvCache    = 2,
    NpuInfer   = 3,
    RealTime   = 4,
}

// ---------------------------------------------------------------------------
// Compute unit descriptor (CPU core, NPU engine, GPU SM)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputeUnitKind {
    CpuCore,
    AppleAne,
    IntelNpu,
    NvidiaGpu,
    ClusterNode, // remote compute via RDMA
}

#[derive(Clone, Copy, Debug)]
pub struct ComputeUnit {
    pub id:    u32,
    pub kind:  ComputeUnitKind,
    /// TOPS (integer-equivalent throughput × 10 for fixed-point representation).
    pub tops_x10: u32,
    /// True while in the active pool; toggled by `add_compute_unit` / hot-unplug.
    pub online: bool,
}

// ---------------------------------------------------------------------------
// Task descriptor
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TaskDescriptor {
    pub id:       u64,
    pub priority: TaskPriority,
    /// Weight-Cache pages required (0 = no KV-cache dependency).
    pub kv_pages: u32,
    /// Preferred compute unit kind (None = any).
    pub affinity: Option<ComputeUnitKind>,
    /// Opaque function pointer index (resolved by the agent that created it).
    pub fn_ptr:   usize,
    pub arg:      u64,
}

// ---------------------------------------------------------------------------
// SchedulerHandle — returned to callers that enqueue tasks
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerHandle(pub u64);

// ---------------------------------------------------------------------------
// NeuralIntentScheduler
// ---------------------------------------------------------------------------

pub struct NeuralIntentScheduler {
    next_task_id: AtomicU64,
    #[cfg(feature = "alloc")]
    units: Vec<ComputeUnit>,
    #[cfg(feature = "alloc")]
    /// Five priority queues (indexed by TaskPriority as usize).
    queues: [Vec<TaskDescriptor>; 5],
    /// Monotonic nanosecond clock (updated by the kernel tick handler).
    pub clock_ns: AtomicU64,
}

impl NeuralIntentScheduler {
    #[cfg(feature = "alloc")]
    pub fn new() -> Self {
        Self {
            next_task_id: AtomicU64::new(1),
            units:        Vec::new(),
            queues:       core::array::from_fn(|_| Vec::new()),
            clock_ns:     AtomicU64::new(0),
        }
    }

    /// Register a compute unit at boot or during hot-plug.  No reboot needed.
    #[cfg(feature = "alloc")]
    pub fn add_compute_unit(&mut self, unit: ComputeUnit) {
        if let Some(existing) = self.units.iter_mut().find(|u| u.id == unit.id) {
            existing.online = true;
        } else {
            self.units.push(unit);
        }
    }

    /// Remove a compute unit (hot-unplug without reboot).
    #[cfg(feature = "alloc")]
    pub fn remove_compute_unit(&mut self, id: u32) {
        if let Some(u) = self.units.iter_mut().find(|u| u.id == id) {
            u.online = false;
        }
    }

    /// Enqueue a task.  Returns its handle.
    #[cfg(feature = "alloc")]
    pub fn enqueue(&mut self, mut task: TaskDescriptor) -> SchedulerHandle {
        let id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        task.id = id;
        self.queues[task.priority as usize].push(task);
        SchedulerHandle(id)
    }

    /// Dequeue the highest-priority ready task, respecting KV-Cache budget.
    ///
    /// `free_kv_pages` — pages currently available in the Weight-Cache region.
    #[cfg(feature = "alloc")]
    pub fn dequeue(&mut self, free_kv_pages: u32) -> Option<TaskDescriptor> {
        for queue in self.queues.iter_mut().rev() {
            if let Some(pos) = queue.iter().position(|t| t.kv_pages <= free_kv_pages) {
                return Some(queue.remove(pos));
            }
        }
        None
    }

    pub fn advance_clock(&self, delta_ns: u64) {
        self.clock_ns.fetch_add(delta_ns, Ordering::Relaxed);
    }
}
