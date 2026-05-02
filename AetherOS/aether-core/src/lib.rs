//! aether-core — no_std microkernel primitives for AetherOS.
//!
//! Provides the memory map, zero-copy IPC bus, neural-intent scheduler,
//! and the foundational SystemAgent trait consumed by all Ring-0 agents.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![feature(allocator_api, const_option)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod agent;
pub mod ipc;
pub mod memory;
pub mod memory_pressure;
pub mod scheduler;

pub use agent::{AgentCapability, AgentId, SystemAgent};
pub use ipc::{AgentBus, BusError, BusMessage, MessagePayload};
pub use memory::{MemoryRegion, PhysicalMemoryMap, RegionFlags, MEMORY_MAP};
pub use memory_pressure::{
    EvictionRegion, EvictionRequest, MemoryMonitor, MemorySnapshot, PressureCallback,
    PressureLevel, RegionWatermarks,
};
pub use scheduler::{NeuralIntentScheduler, SchedulerHandle, TaskDescriptor, TaskPriority};
