//! aether-cluster — zero-conf node discovery, gossip protocol, and RDMA memory.

pub mod gossip;
pub mod rdma;

pub use gossip::{ClusterMembership, GossipState, NodeId, NodeState};
pub use rdma::{ClusterMemoryMap, RdmaError, RdmaRegion, RdmaRegionId};
