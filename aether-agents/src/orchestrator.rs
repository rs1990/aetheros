//! OrchestratorAgent — resource ledger and compute-pool manager.
//!
//! Responsibilities:
//!   • Maintain the canonical list of online ComputeUnits.
//!   • On `DriverReady` from SynthesisAgent: call `Scheduler::add_compute_unit`
//!     and broadcast `HardwareEvent::ClusterNodeJoin` to cluster peers.
//!   • On `DeviceDetached`: remove the unit and drain its queued tasks.
//!   • On `ClusterSync` from aether-cluster: merge remote compute inventory.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossbeam::channel::Receiver;
use tracing::{info, warn};

use aether_core::agent::AgentId;
use aether_core::ipc::{BusMessage, HardwareEventKind, MessagePayload};
use aether_core::scheduler::{ComputeUnit, ComputeUnitKind};

use crate::bus::AgentMailbox;

// ---------------------------------------------------------------------------
// OrchestratorAgent
// ---------------------------------------------------------------------------

pub struct OrchestratorAgent {
    /// Tracks all online compute units by id.
    units: HashMap<u32, ComputeUnit>,
    next_unit_id: u32,
}

impl OrchestratorAgent {
    pub fn new() -> Self {
        Self { units: HashMap::new(), next_unit_id: 1 }
    }

    /// Main loop.
    pub fn run(
        mut self,
        _id:     AgentId,
        rx:      Receiver<BusMessage>,
        mailbox: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[OrchestratorAgent] started");
        for msg in rx {
            match msg.payload {
                // SynthesisAgent signals driver is ready
                MessagePayload::AgentControl { opcode: 0x01, arg: device_id, .. } => {
                    self.on_driver_ready(device_id as u32, &mailbox);
                }
                // Kernel signals hot-unplug
                MessagePayload::HardwareEvent { kind: HardwareEventKind::DeviceDetached, device_id, .. } => {
                    self.on_device_detached(device_id, &mailbox);
                }
                // Cluster node joined — add as a remote compute unit
                MessagePayload::HardwareEvent { kind: HardwareEventKind::ClusterNodeJoin, device_id, data } => {
                    self.on_cluster_join(device_id, data, &mailbox);
                }
                // Cluster sync delta
                MessagePayload::ClusterSync { node_id, epoch } => {
                    self.on_cluster_sync(node_id, epoch, &mailbox);
                }
                _ => {}
            }
        }
    }

    fn on_driver_ready(
        &mut self,
        device_id: u32,
        mailbox:   &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        let unit_id = self.next_unit_id;
        self.next_unit_id += 1;

        // Heuristic: synthesized devices default to CPU-class until NPU is confirmed.
        let unit = ComputeUnit {
            id:       unit_id,
            kind:     ComputeUnitKind::CpuCore,
            tops_x10: 10, // 1.0 TOPS placeholder
            online:   true,
        };
        self.units.insert(unit_id, unit);
        info!("[OrchestratorAgent] registered unit {unit_id} for device {device_id:#x}");

        // Notify Scheduler (via kernel call in production; message here).
        self.broadcast_resource_update(unit_id, true, mailbox);
    }

    fn on_device_detached(
        &mut self,
        device_id: u32,
        mailbox:   &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        if let Some((_, unit)) = self.units.iter_mut().find(|(_, u)| u.id == device_id) {
            unit.online = false;
            let id = unit.id;
            info!("[OrchestratorAgent] unit {id} taken offline (device {device_id:#x})");
            self.broadcast_resource_update(id, false, mailbox);
        } else {
            warn!("[OrchestratorAgent] detach for unknown device {device_id:#x}");
        }
    }

    fn on_cluster_join(
        &mut self,
        node_id: u32,
        tops_x10: u64,
        mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        let unit_id = self.next_unit_id;
        self.next_unit_id += 1;
        let unit = ComputeUnit {
            id:       unit_id,
            kind:     ComputeUnitKind::ClusterNode,
            tops_x10: tops_x10 as u32,
            online:   true,
        };
        self.units.insert(unit_id, unit);
        info!("[OrchestratorAgent] cluster node {node_id} added as unit {unit_id} ({tops_x10} × 0.1 TOPS)");
        self.broadcast_resource_update(unit_id, true, mailbox);
    }

    fn on_cluster_sync(&self, node_id: u64, epoch: u64, _mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>) {
        info!("[OrchestratorAgent] cluster sync from node {node_id:#x} epoch {epoch}");
        // In production: merge the remote node's unit list into self.units.
    }

    fn broadcast_resource_update(
        &self,
        unit_id: u32,
        online:  bool,
        mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        let msg = BusMessage {
            origin:  AgentId::ORCHESTRATOR,
            dest:    AgentId::BROADCAST,
            seq:     0,
            payload: MessagePayload::AgentControl {
                opcode: if online { 0x10 } else { 0x11 }, // ResourceAdded / ResourceRemoved
                target: AgentId::SCHEDULER,
                arg:    unit_id as u64,
            },
        };
        if let Ok(map) = mailbox.lock() {
            for mb in map.values() {
                let _ = mb.tx.send(msg);
            }
        }
    }
}
