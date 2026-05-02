//! aether-hal — Neural Hardware Abstraction Layer.
//!
//! Three layers:
//!   discovery  — PCI/USB enumeration + `DriverDiscovery` trait
//!   npu        — Unified `NeuralDevice` trait; JIT-patch context per silicon
//!   synthesis  — `AgentCompiler` trait; MMIO pattern → Rust driver shim

pub mod discovery;
pub mod memory_pool;
pub mod npu;
pub mod synthesis;

pub use discovery::{DeviceClass, DeviceProfile, DriverDiscovery, PciId, UsbId};
pub use memory_pool::{AllocationId, AllocationKind, GpuMemoryPool, MemoryTier, PoolRegistry};
pub use npu::{JitPatchContext, NeuralDevice, NpuBackend, NpuJob, NpuJobResult};
pub use synthesis::{AgentCompiler, CompileError, DriverShim, MmioPattern};
