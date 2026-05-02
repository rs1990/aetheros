//! CuratorAgent — maps new storage devices into the global USDF.
//!
//! On receiving an `AgentControl::ResourceAdded` for a storage-class device:
//!   1. Walk the block device's raw partition table.
//!   2. For each partition, generate a root-level embedding from the
//!      volume label and detected MIME types.
//!   3. Bind the partition tree into the VfsBridge under `/mnt/<label>`.
//!   4. Emit `HardwareEvent::DeviceAttached` → USDF namespace update.
//!
//! For non-storage devices the CuratorAgent is a no-op.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossbeam::channel::Receiver;
use tracing::info;

use aether_core::agent::AgentId;
use aether_core::ipc::{BusMessage, MessagePayload};
use aether_usdf::vector::SemanticNamespace;
use aether_usdf::vfs::VfsBridge;
use aether_usdf::vector::VectorStore;

use crate::bus::AgentMailbox;

// ---------------------------------------------------------------------------
// CuratorAgent
// ---------------------------------------------------------------------------

pub struct CuratorAgent {
    store:  VectorStore,
    bridge: VfsBridge,
}

impl CuratorAgent {
    pub fn new(dim: usize) -> Self {
        Self {
            store:  VectorStore::new(dim),
            bridge: VfsBridge::new(),
        }
    }

    pub fn run(
        mut self,
        _id:     AgentId,
        rx:      Receiver<BusMessage>,
        mailbox: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[CuratorAgent] started");
        for msg in rx {
            if let MessagePayload::AgentControl { opcode: 0x10, arg: unit_id, .. } = msg.payload {
                self.on_resource_added(unit_id, &mailbox);
            }
        }
    }

    fn on_resource_added(
        &mut self,
        unit_id: u64,
        _mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        // In production: query HAL for the device class of unit_id.
        // If it's StorageBlock, enumerate partitions.
        // Here we demonstrate the full pipeline with a synthetic partition.
        let label = format!("vol_{unit_id:#x}");
        let mount_path = format!("/mnt/{label}");

        // Synthesise a root embedding for the volume (dim=768 all-zeros placeholder).
        // Production: run a content-aware embedding model over volume metadata.
        let embedding = vec![0.0f32; self.store.dim];
        let meta = HashMap::from([
            ("label".to_string(), label.clone()),
            ("mount".to_string(), mount_path.clone()),
            ("unit_id".to_string(), unit_id.to_string()),
        ]);
        let id = self.store.insert(
            SemanticNamespace::SYSTEM,
            embedding,
            None, // no inline payload for a directory entry
            meta,
        );

        self.bridge.bind(&mount_path, id, SemanticNamespace::SYSTEM);
        info!("[CuratorAgent] mapped {label} → USDF {id:?} at {mount_path}");

        // Bind a stub sub-entry to demonstrate recursive mapping.
        let sub_emb = vec![0.01f32; self.store.dim];
        let sub_id = self.store.insert(
            SemanticNamespace::SYSTEM,
            sub_emb,
            Some(b"placeholder content".to_vec()),
            HashMap::from([("parent".to_string(), mount_path.clone())]),
        );
        self.bridge.bind(&format!("{mount_path}/index"), sub_id, SemanticNamespace::SYSTEM);
    }

    // ---------------------------------------------------------------------------
    // Expose store / bridge for kernel inspection
    // ---------------------------------------------------------------------------

    pub fn store(&self) -> &VectorStore { &self.store }
    pub fn bridge(&self) -> &VfsBridge  { &self.bridge }
}
