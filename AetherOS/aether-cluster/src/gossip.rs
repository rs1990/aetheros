//! SWIM-inspired gossip protocol for zero-conf AetherOS cluster membership.
//!
//! Key properties:
//!   • Zero-configuration: nodes announce themselves via UDP multicast on
//!     the link-local address `ff02::aether` (port 7777).
//!   • Failure detection: each node piggybacks a random subset of its
//!     membership table on every heartbeat (indirect probing).
//!   • Convergence: O(log N) rounds for full cluster awareness.
//!   • Metadata carried per node: TOPS, memory tier capacities, RDMA caps.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use rand::seq::SliceRandom;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Node identity
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 16]); // UUID bytes

impl NodeId {
    pub fn new() -> Self {
        Self(*Uuid::new_v4().as_bytes())
    }

    pub fn to_uuid(&self) -> Uuid {
        Uuid::from_bytes(self.0)
    }
}

impl Default for NodeId {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Node metadata
// ---------------------------------------------------------------------------

/// Hardware capabilities advertised by each cluster member.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Total TOPS available (sum of all NPU/GPU units × 0.1).
    pub tops_x10:       u32,
    /// Weight-Cache size in MiB.
    pub weight_cache_mb: u32,
    /// Standard RAM in MiB.
    pub ram_mb:          u32,
    /// RDMA-capable NICs present.
    pub rdma:            bool,
    /// AetherOS kernel version.
    pub kernel_version:  [u8; 3],
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            tops_x10:        0,
            weight_cache_mb: 0,
            ram_mb:          0,
            rdma:            false,
            kernel_version:  [0, 1, 0],
        }
    }
}

// ---------------------------------------------------------------------------
// Node state (per membership entry)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Alive,
    Suspect,   // failed direct probe; indirect probing in progress
    Dead,      // confirmed unreachable; will be garbage-collected
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeState {
    pub id:          NodeId,
    pub addr:        SocketAddr,
    pub status:      NodeStatus,
    pub incarnation: u64, // incremented by the node to refute Suspect
    pub caps:        NodeCapabilities,
    #[serde(skip)]
    pub last_ack:    Option<Instant>,
}

impl NodeState {
    pub fn new(id: NodeId, addr: SocketAddr, caps: NodeCapabilities) -> Self {
        Self { id, addr, status: NodeStatus::Alive, incarnation: 0, caps, last_ack: Some(Instant::now()) }
    }

    pub fn is_alive(&self) -> bool { self.status == NodeStatus::Alive }
}

// ---------------------------------------------------------------------------
// Gossip message
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Periodic heartbeat + membership delta.
    Heartbeat {
        from:  NodeId,
        epoch: u64,
        /// Subset of membership (piggybacked gossip).
        delta: Vec<NodeState>,
    },
    /// Join announcement — broadcast on startup.
    Join {
        state: NodeState,
    },
    /// Ack to a direct probe.
    Ack {
        from: NodeId,
        seq:  u64,
    },
    /// Indirect probe request: "please probe `target` for me".
    IndirectProbe {
        requester: NodeId,
        target:    NodeId,
        seq:       u64,
    },
}

// ---------------------------------------------------------------------------
// GossipState — the local membership table
// ---------------------------------------------------------------------------

pub const GOSSIP_FANOUT: usize = 3;          // nodes to piggyback on each HB
pub const SUSPECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const DEAD_GC_TIMEOUT:  Duration = Duration::from_secs(30);

pub struct GossipState {
    pub local:   NodeState,
    members:     HashMap<NodeId, NodeState>,
    pub epoch:   u64,
}

impl GossipState {
    pub fn new(local: NodeState) -> Self {
        let mut members = HashMap::new();
        members.insert(local.id, local.clone());
        Self { local, members, epoch: 0 }
    }

    /// Merge an inbound membership delta.
    pub fn merge(&mut self, delta: &[NodeState]) {
        for remote in delta {
            match self.members.get_mut(&remote.id) {
                Some(existing) => {
                    // Higher incarnation always wins; same incarnation favours
                    // Alive over Suspect over Dead.
                    if remote.incarnation > existing.incarnation
                        || (remote.incarnation == existing.incarnation
                            && remote.status == NodeStatus::Alive
                            && existing.status != NodeStatus::Alive)
                    {
                        *existing = remote.clone();
                        existing.last_ack = Some(Instant::now());
                    }
                }
                None => {
                    let mut state = remote.clone();
                    state.last_ack = Some(Instant::now());
                    self.members.insert(remote.id, state);
                    info!("[Gossip] new member {} at {}", remote.id.to_uuid(), remote.addr);
                }
            }
        }
    }

    /// Produce a heartbeat message with a random delta subset.
    pub fn make_heartbeat(&mut self) -> GossipMessage {
        self.epoch += 1;
        let mut rng = rand::thread_rng();
        let alive: Vec<NodeState> = self
            .members
            .values()
            .filter(|n| n.id != self.local.id)
            .cloned()
            .collect();
        let sample: Vec<NodeState> = alive
            .choose_multiple(&mut rng, GOSSIP_FANOUT.min(alive.len()))
            .cloned()
            .collect();
        GossipMessage::Heartbeat { from: self.local.id, epoch: self.epoch, delta: sample }
    }

    /// Mark a node as Suspect if its last ACK is overdue.
    pub fn run_failure_detection(&mut self) {
        let now = Instant::now();
        for state in self.members.values_mut() {
            if state.id == self.local.id { continue; }
            if let Some(last) = state.last_ack {
                if state.status == NodeStatus::Alive && now.duration_since(last) > SUSPECT_TIMEOUT {
                    state.status = NodeStatus::Suspect;
                    warn!("[Gossip] node {} suspect", state.id.to_uuid());
                } else if state.status == NodeStatus::Suspect && now.duration_since(last) > DEAD_GC_TIMEOUT {
                    state.status = NodeStatus::Dead;
                    warn!("[Gossip] node {} dead", state.id.to_uuid());
                }
            }
        }
        // GC dead nodes after timeout
        self.members.retain(|_, s| s.status != NodeStatus::Dead
            || s.last_ack.map_or(true, |t| now.duration_since(t) < DEAD_GC_TIMEOUT));
    }

    pub fn alive_members(&self) -> impl Iterator<Item = &NodeState> {
        self.members.values().filter(|s| s.is_alive())
    }

    pub fn member_count(&self) -> usize { self.members.len() }
}

// ---------------------------------------------------------------------------
// ClusterMembership — public façade used by the agents layer
// ---------------------------------------------------------------------------

pub struct ClusterMembership {
    state: GossipState,
}

impl ClusterMembership {
    pub fn new(local: NodeState) -> Self {
        Self { state: GossipState::new(local) }
    }

    pub fn handle_message(&mut self, msg: GossipMessage) {
        match msg {
            GossipMessage::Join { state } => {
                self.state.merge(&[state]);
            }
            GossipMessage::Heartbeat { delta, .. } => {
                self.state.merge(&delta);
            }
            GossipMessage::Ack { from, .. } => {
                if let Some(s) = self.state.members.get_mut(&from) {
                    s.status   = NodeStatus::Alive;
                    s.last_ack = Some(Instant::now());
                }
            }
            GossipMessage::IndirectProbe { .. } => {
                debug!("[Gossip] indirect probe received (not yet implemented)");
            }
        }
    }

    pub fn tick(&mut self) {
        self.state.run_failure_detection();
    }

    pub fn heartbeat(&mut self) -> GossipMessage {
        self.state.make_heartbeat()
    }

    pub fn alive_count(&self) -> usize {
        self.state.alive_members().count()
    }

    pub fn local_id(&self) -> NodeId { self.state.local.id }
}
