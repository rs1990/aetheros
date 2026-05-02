//! RDMA region management for AetherOS cluster memory.
//!
//! Each node exposes slices of its `cluster_mapped` physical range
//! (0x0000_0040_0000_0000 base, 128 GiB) as named RDMA regions.
//! Remote nodes can map these into their own cluster_mapped window
//! via a simple registration protocol.
//!
//! The `ClusterMemoryMap` maintains the local ledger of:
//!   • Exported regions (local memory shared with peers).
//!   • Imported regions (peer memory mapped into local cluster window).

use std::collections::HashMap;
use std::net::SocketAddr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use aether_core::memory::MEMORY_MAP;
use crate::gossip::NodeId;

// ---------------------------------------------------------------------------
// Region descriptor
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RdmaRegionId(pub u64);

impl RdmaRegionId {
    pub fn new(node: NodeId, local_idx: u32) -> Self {
        let node_hi = u64::from_le_bytes(node.0[..8].try_into().unwrap());
        Self(node_hi ^ (local_idx as u64 * 0x0001_0000_0000))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RdmaRegion {
    pub id:          RdmaRegionId,
    pub owner:       NodeId,
    pub owner_addr:  SocketAddr,
    /// Physical base address on the owning node (within cluster_mapped range).
    pub remote_phys: u64,
    pub size:        usize,
    /// Local physical address where this region is mapped in our cluster window.
    /// `None` = not yet mapped.
    pub local_phys:  Option<u64>,
    /// Access key — must be presented on every RDMA read/write.
    pub rkey:        u32,
    pub flags:       RdmaFlags,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RdmaFlags: u32 {
        const READ  = 1 << 0;
        const WRITE = 1 << 1;
        /// Region contains KV-Cache pages; scheduler gives priority.
        const KV_CACHE = 1 << 2;
        /// Region contains model weights (read-only export).
        const WEIGHTS  = 1 << 3;
    }
}

// ---------------------------------------------------------------------------
// ClusterMemoryMap
// ---------------------------------------------------------------------------

/// Allocator for the local cluster-mapped physical window.
///
/// Physical range: `MEMORY_MAP.cluster_mapped` (128 GiB default).
/// We allocate sub-regions with a simple bump allocator; a real
/// implementation would use a buddy allocator with reclaim.
pub struct ClusterMemoryMap {
    base:     u64,
    size:     u64,
    bump:     u64,
    exported: HashMap<RdmaRegionId, RdmaRegion>,
    imported: HashMap<RdmaRegionId, RdmaRegion>,
    local_id: NodeId,
    next_local_idx: u32,
}

impl ClusterMemoryMap {
    pub fn new(local_id: NodeId) -> Self {
        let base = MEMORY_MAP.cluster_mapped.base;
        let size = MEMORY_MAP.cluster_mapped.size;
        Self {
            base,
            size,
            bump: base,
            exported: HashMap::new(),
            imported: HashMap::new(),
            local_id,
            next_local_idx: 0,
        }
    }

    // ---------------------------------------------------------------------------
    // Export: share local memory with the cluster
    // ---------------------------------------------------------------------------

    /// Carve `size` bytes from the cluster window and export it.
    pub fn export(
        &mut self,
        size:         usize,
        flags:        RdmaFlags,
        local_addr:   SocketAddr,
    ) -> Result<RdmaRegion, RdmaError> {
        let aligned_size = align_up(size as u64, 4096);
        if self.bump + aligned_size > self.base + self.size {
            return Err(RdmaError::OutOfRegions);
        }
        let idx = self.next_local_idx;
        self.next_local_idx += 1;

        let region = RdmaRegion {
            id:          RdmaRegionId::new(self.local_id, idx),
            owner:       self.local_id,
            owner_addr:  local_addr,
            remote_phys: self.bump,
            size,
            local_phys:  Some(self.bump),
            rkey:        simple_rkey(self.local_id, idx),
            flags,
        };
        self.bump += aligned_size;
        self.exported.insert(region.id, region.clone());
        Ok(region)
    }

    // ---------------------------------------------------------------------------
    // Import: map a peer's region into local cluster window
    // ---------------------------------------------------------------------------

    pub fn import(&mut self, region: RdmaRegion) -> Result<u64, RdmaError> {
        if self.imported.contains_key(&region.id) {
            return Err(RdmaError::AlreadyMapped(region.id));
        }
        let aligned = align_up(region.size as u64, 4096);
        if self.bump + aligned > self.base + self.size {
            return Err(RdmaError::OutOfRegions);
        }
        let local_phys = self.bump;
        self.bump += aligned;
        let mut r = region;
        r.local_phys = Some(local_phys);
        self.imported.insert(r.id, r);
        Ok(local_phys)
    }

    pub fn unmap(&mut self, id: RdmaRegionId) {
        // In production: reclaim bump range and update page tables.
        self.imported.remove(&id);
    }

    pub fn revoke_export(&mut self, id: RdmaRegionId) {
        self.exported.remove(&id);
    }

    // ---------------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------------

    pub fn get_import(&self, id: RdmaRegionId) -> Option<&RdmaRegion> {
        self.imported.get(&id)
    }

    pub fn exported_regions(&self) -> impl Iterator<Item = &RdmaRegion> {
        self.exported.values()
    }

    pub fn imported_regions(&self) -> impl Iterator<Item = &RdmaRegion> {
        self.imported.values()
    }

    pub fn used_bytes(&self) -> u64 { self.bump - self.base }
    pub fn free_bytes(&self) -> u64 { self.base + self.size - self.bump }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

fn simple_rkey(node: NodeId, idx: u32) -> u32 {
    let h = u32::from_le_bytes(node.0[..4].try_into().unwrap());
    h.wrapping_mul(0x9e37_79b9).wrapping_add(idx)
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum RdmaError {
    #[error("cluster-mapped window exhausted")]
    OutOfRegions,
    #[error("region {0:?} is already mapped locally")]
    AlreadyMapped(RdmaRegionId),
    #[error("invalid rkey")]
    InvalidRkey,
    #[error("remote node unreachable: {0}")]
    NodeUnreachable(String),
}
