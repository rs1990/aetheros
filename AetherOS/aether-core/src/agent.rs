//! SystemAgent trait — the contract every Ring-0 agent must satisfy.

use crate::ipc::{AgentBus, BusMessage};

// ---------------------------------------------------------------------------
// Agent identity
// ---------------------------------------------------------------------------

/// Compact 8-byte identifier.  Upper 4 bytes = class; lower 4 bytes = instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct AgentId(pub u64);

impl AgentId {
    pub const BROADCAST:        Self = Self(0xFFFF_FFFF_FFFF_FFFF);
    pub const KERNEL:           Self = Self(0x0000_0000_0000_0001);
    pub const SYNTHESIS:        Self = Self(0x0001_0000_0000_0001);
    pub const ORCHESTRATOR:     Self = Self(0x0002_0000_0000_0001);
    pub const CURATOR:          Self = Self(0x0003_0000_0000_0001);
    pub const SCHEDULER:        Self = Self(0x0004_0000_0000_0001);
    pub const MEMORY_OPTIMIZER: Self = Self(0x0005_0000_0000_0001);
    pub const LIBRARY_UPDATER:  Self = Self(0x0006_0000_0000_0001);
    pub const JANITOR:          Self = Self(0x0007_0000_0000_0001);

    #[inline]
    pub const fn class(self) -> u32 {
        (self.0 >> 32) as u32
    }
    #[inline]
    pub const fn instance(self) -> u32 {
        self.0 as u32
    }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AgentCapability: u32 {
        /// May issue MMIO read/write.
        const MMIO_ACCESS     = 1 << 0;
        /// May allocate pages from the Weight-Cache region.
        const KV_CACHE_ALLOC  = 1 << 1;
        /// May register new drivers with the HAL.
        const DRIVER_REGISTER = 1 << 2;
        /// May spawn child agents.
        const AGENT_SPAWN     = 1 << 3;
        /// May map/unmap cluster memory.
        const CLUSTER_MAP     = 1 << 4;
        /// May read/write the encrypted Persona Overlay.
        const PERSONA_RW      = 1 << 5;
        /// May send cluster-scope broadcast messages.
        const CLUSTER_BCAST   = 1 << 6;
    }
}

// ---------------------------------------------------------------------------
// SystemAgent trait
// ---------------------------------------------------------------------------

/// The trait every Ring-0 agent must implement.
///
/// Agents are event-driven: the kernel calls `on_message` in a tight loop
/// for each inbound message on the agent's ring channel.  `tick` is called
/// at a fixed rate (configurable per-agent) for time-driven housekeeping.
pub trait SystemAgent: Send + Sync {
    /// Stable identifier; must be unique within the bus.
    fn id(&self) -> AgentId;

    /// Declare the capabilities this agent requires at registration time.
    fn capabilities(&self) -> AgentCapability;

    /// Called exactly once after the bus registers the agent.
    fn on_start(&mut self, bus: &AgentBus);

    /// Process a single inbound message.  Must not block.
    fn on_message(&mut self, msg: BusMessage, bus: &AgentBus);

    /// Periodic heartbeat.  `tick_ns` = nanoseconds since last tick.
    fn tick(&mut self, tick_ns: u64, bus: &AgentBus);

    /// Called before the agent is deregistered (graceful shutdown only).
    fn on_stop(&mut self, bus: &AgentBus);
}
