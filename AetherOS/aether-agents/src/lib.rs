//! aether-agents — Ring-0 system agent swarm.
//!
//! All agents share the `AgentBus` (zero-copy MPSC rings from aether-core).
//! Three specialised agents handle the full device plug-in pipeline:
//!
//!   SynthesisAgent    — driver generation (HAL miss path)
//!   OrchestratorAgent — resource ledger; notifies Scheduler of new units
//!   CuratorAgent      — maps new storage into the USDF
//!
//! Module `bus` re-exports the higher-level `std`-backed bus used by the
//! user-space agent runner that wraps the no_std `AgentBus`.

pub mod bus;
pub mod curator;
pub mod janitor;
pub mod library_updater;
pub mod memory_optimizer;
pub mod orchestrator;
pub mod synthesis;

pub use bus::{AgentRuntime, RuntimeError};
pub use curator::CuratorAgent;
pub use janitor::{CleanupReport, JanitorAgent, JanitorConfig};
pub use library_updater::{DepEntry, LibraryUpdaterAgent, UpdateRecord, UpgradeKind};
pub use memory_optimizer::MemoryOptimizerAgent;
pub use orchestrator::OrchestratorAgent;
pub use synthesis::SynthesisAgent;
