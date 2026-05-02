//! Zero-copy inter-agent IPC bus.
//!
//! Design goals:
//!   • Single-producer / multi-consumer (SPMC) per agent channel.
//!   • Zero-copy: messages carry a raw pointer + length; ownership is
//!     transferred atomically via a generation counter.
//!   • Lock-free hot path: only `AtomicUsize` operations on the ring indices.
//!   • Bounded ring buffers → deterministic memory usage inside the kernel.

use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, sync::Arc};

use crate::agent::AgentId;

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

/// Discriminant for zero-copy payload variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PayloadKind {
    RawSlice       = 0, // ptr + len; no allocation
    SemanticRef    = 1, // index into USDF vector store
    HardwareEvent  = 2, // plug/unplug, MMIO IRQ
    AgentControl   = 3, // spawn / kill / migrate
    ClusterSync    = 4, // RDMA invalidation or gossip delta
}

/// Hardware-event sub-type carried inside HardwareEvent payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HardwareEventKind {
    DeviceAttached  = 0,
    DeviceDetached  = 1,
    MmioIrq         = 2,
    NpuJobComplete  = 3,
    ClusterNodeJoin = 4,
    ClusterNodeLeave= 5,
}

/// Lightweight handle to a semantic vector (no pointer dereference needed
/// on the receiver side; the USDF layer resolves it on demand).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticRef {
    pub namespace: u32,
    pub vector_id: u64,
    pub generation: u32,
}

/// Raw zero-copy slice: the *sender* retains the allocation; the receiver
/// MUST NOT outlive the sender's buffer.  Used for DMA-mapped I/O frames.
#[derive(Clone, Copy, Debug)]
pub struct RawSliceHandle {
    pub ptr: *const u8,
    pub len: usize,
    /// Physical address, for DMA-capable recipients.
    pub phys_addr: u64,
}

// SAFETY: AetherOS Ring-0 agents run in a single-address-space kernel where
// pointer provenance is enforced by the memory-map immutability rules.
unsafe impl Send for RawSliceHandle {}
unsafe impl Sync for RawSliceHandle {}

/// Tagged-union payload – kept to 24 bytes to fit in one cache line
/// alongside the AgentId header.
#[derive(Clone, Copy, Debug)]
pub enum MessagePayload {
    RawSlice(RawSliceHandle),
    SemanticRef(SemanticRef),
    HardwareEvent { kind: HardwareEventKind, device_id: u32, data: u64 },
    AgentControl  { opcode: u8, target: AgentId, arg: u64 },
    ClusterSync   { node_id: u64, epoch: u64 },
}

/// Envelope that traverses the bus.  Total size = 40 bytes (fits two per
/// 64-byte cache line with header).
#[derive(Clone, Copy, Debug)]
pub struct BusMessage {
    pub origin:    AgentId,
    pub dest:      AgentId,   // `AgentId::BROADCAST` for multicast
    pub seq:       u64,
    pub payload:   MessagePayload,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusError {
    /// Ring buffer is full; back-pressure the producer.
    QueueFull,
    /// Ring buffer is empty; no message ready.
    QueueEmpty,
    /// Destination agent not registered.
    UnknownDest,
}

// ---------------------------------------------------------------------------
// Lock-free ring buffer
// ---------------------------------------------------------------------------

const BUS_RING_SIZE: usize = 1024; // must be power-of-two

/// Single-producer / single-consumer lock-free ring buffer.
///
/// Use one `RingChannel` per ordered pair (sender, receiver).  The `AgentBus`
/// multiplexes these channels across all registered agents.
pub struct RingChannel {
    head: AtomicUsize, // written by producer
    _pad0: [u8; 56],
    tail: AtomicUsize, // written by consumer
    _pad1: [u8; 56],
    slots: UnsafeCell<[MaybeUninit<BusMessage>; BUS_RING_SIZE]>,
}

// SAFETY: We guarantee only one producer and one consumer touch head/tail.
unsafe impl Send for RingChannel {}
unsafe impl Sync for RingChannel {}

impl RingChannel {
    pub const fn new() -> Self {
        Self {
            head:  AtomicUsize::new(0),
            _pad0: [0u8; 56],
            tail:  AtomicUsize::new(0),
            _pad1: [0u8; 56],
            slots: UnsafeCell::new(unsafe {
                MaybeUninit::<[MaybeUninit<BusMessage>; BUS_RING_SIZE]>::uninit().assume_init()
            }),
        }
    }

    /// Enqueue a message.  Returns `BusError::QueueFull` if no slot is free.
    pub fn send(&self, msg: BusMessage) -> Result<(), BusError> {
        let head = self.head.load(Ordering::Relaxed);
        let next = (head + 1) & (BUS_RING_SIZE - 1);
        if next == self.tail.load(Ordering::Acquire) {
            return Err(BusError::QueueFull);
        }
        // SAFETY: head is exclusively owned by the producer.
        unsafe {
            (*self.slots.get())[head].write(msg);
        }
        self.head.store(next, Ordering::Release);
        Ok(())
    }

    /// Dequeue a message.  Returns `BusError::QueueEmpty` if none ready.
    pub fn recv(&self) -> Result<BusMessage, BusError> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return Err(BusError::QueueEmpty);
        }
        // SAFETY: tail is exclusively owned by the consumer.
        let msg = unsafe { (*self.slots.get())[tail].assume_init_read() };
        self.tail.store((tail + 1) & (BUS_RING_SIZE - 1), Ordering::Release);
        Ok(msg)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Acquire) == self.head.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Multi-agent bus
// ---------------------------------------------------------------------------

/// Maximum concurrently registered agents on the internal bus.
pub const MAX_AGENTS: usize = 64;

/// An entry in the bus routing table.
struct BusEntry {
    id:      AgentId,
    /// Inbound ring for this agent (messages addressed to it).
    inbound: RingChannel,
}

/// The kernel-internal agent bus.  All Ring-0 agents share one instance
/// embedded in the static kernel BSS.
///
/// Routing is O(N) over registered agents; N ≤ 64 in typical deployments,
/// so a simple linear scan is acceptable and cache-hot.
pub struct AgentBus {
    entries: [Option<BusEntry>; MAX_AGENTS],
    count:   AtomicUsize,
    seq:     AtomicUsize,
}

impl AgentBus {
    pub const fn new() -> Self {
        const NONE_ENTRY: Option<BusEntry> = None;
        Self {
            entries: [NONE_ENTRY; MAX_AGENTS],
            count:   AtomicUsize::new(0),
            seq:     AtomicUsize::new(0),
        }
    }

    /// Register an agent and allocate its inbound ring.
    pub fn register(&mut self, id: AgentId) -> bool {
        let n = self.count.load(Ordering::Relaxed);
        if n >= MAX_AGENTS {
            return false;
        }
        self.entries[n] = Some(BusEntry { id, inbound: RingChannel::new() });
        self.count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Dispatch a message to its destination (or all agents on broadcast).
    pub fn send(&self, mut msg: BusMessage) -> Result<(), BusError> {
        msg.seq = self.seq.fetch_add(1, Ordering::Relaxed) as u64;
        let n = self.count.load(Ordering::Relaxed);
        if msg.dest == AgentId::BROADCAST {
            for entry in self.entries[..n].iter().flatten() {
                // Best-effort broadcast: skip full rings rather than block.
                let _ = entry.inbound.send(msg);
            }
            return Ok(());
        }
        for entry in self.entries[..n].iter().flatten() {
            if entry.id == msg.dest {
                return entry.inbound.send(msg);
            }
        }
        Err(BusError::UnknownDest)
    }

    /// Pull the next message addressed to `id`.
    pub fn recv(&self, id: AgentId) -> Result<BusMessage, BusError> {
        let n = self.count.load(Ordering::Relaxed);
        for entry in self.entries[..n].iter().flatten() {
            if entry.id == id {
                return entry.inbound.recv();
            }
        }
        Err(BusError::UnknownDest)
    }
}
