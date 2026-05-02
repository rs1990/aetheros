//! SynthesisAgent — generates drivers on the fly when the HAL misses.
//!
//! Pipeline triggered by a `HardwareEvent::DeviceAttached` message:
//!
//!  1. Receive event with `DeviceProfile`.
//!  2. Sample MMIO transactions (probe reads at well-known offsets).
//!  3. Optionally fetch device spec from an external data source.
//!  4. Call `TemplateCompiler::synthesize` → `DriverShim`.
//!  5. Persist the shim source in the Persona Overlay (encrypted).
//!  6. Broadcast `AgentControl::DriverReady` so the Orchestrator can
//!     proceed with resource registration.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossbeam::channel::Receiver;
use tracing::{error, info, warn};

use aether_core::agent::AgentId;
use aether_core::ipc::{BusMessage, HardwareEventKind, MessagePayload};
use aether_hal::discovery::DeviceProfile;
use aether_hal::synthesis::{AgentCompiler, MmioTransaction, TemplateCompiler};

use crate::bus::AgentMailbox;

// ---------------------------------------------------------------------------
// SynthesisAgent
// ---------------------------------------------------------------------------

pub struct SynthesisAgent {
    compiler: TemplateCompiler,
    /// Synthesized sources cached by device name (for Persona Overlay write).
    shim_cache: HashMap<String, String>,
}

impl SynthesisAgent {
    pub fn new() -> Self {
        Self {
            compiler:   TemplateCompiler::default(),
            shim_cache: HashMap::new(),
        }
    }

    /// Main loop — blocks on the inbound channel.
    pub fn run(
        mut self,
        _id:     AgentId,
        rx:      Receiver<BusMessage>,
        mailbox: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[SynthesisAgent] started");
        for msg in rx {
            if let MessagePayload::HardwareEvent { kind: HardwareEventKind::DeviceAttached, device_id, .. } = msg.payload {
                self.handle_attach(device_id, &mailbox);
            }
        }
    }

    fn handle_attach(
        &mut self,
        device_id: u32,
        mailbox:   &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[SynthesisAgent] synthesizing driver for device {device_id:#x}");

        // In production: read captured MMIO transactions from the kernel buffer.
        // Here we synthesise a placeholder profile.
        let profile = DeviceProfile {
            bus_id:    aether_hal::discovery::BusId::Pci(aether_hal::discovery::PciId {
                vendor: (device_id >> 16) as u16,
                device: (device_id & 0xFFFF) as u16,
                class:  0xFF,
                sub:    0x00,
            }),
            class:     aether_hal::discovery::DeviceClass::Unknown,
            name:      format!("synth_dev_{device_id:#010x}"),
            mmio_base: 0x0000_0060_0000_0000 + (device_id as u64 * 0x10_0000),
            mmio_size: 0x10_0000,
            irq:       Some(32 + device_id),
            caps_raw:  vec![],
        };

        let txns = self.probe_mmio(&profile);
        let pattern = self.compiler.analyse(&profile, &txns);

        if pattern.confidence < 0.40 {
            warn!(
                "[SynthesisAgent] confidence {:.2} too low for {name}, deferring",
                pattern.confidence,
                name = profile.name
            );
            return;
        }

        match self.compiler.synthesize(&profile, &pattern, None) {
            Ok(shim) => {
                self.shim_cache.insert(profile.name.clone(), shim.source_rs.clone());
                info!("[SynthesisAgent] shim ready for {}: {} bytes", profile.name, shim.source_rs.len());

                // Notify OrchestratorAgent that driver synthesis succeeded.
                let notif = BusMessage {
                    origin:  AgentId::SYNTHESIS,
                    dest:    AgentId::ORCHESTRATOR,
                    seq:     0,
                    payload: MessagePayload::AgentControl {
                        opcode: 0x01, // DriverReady
                        target: AgentId::SYNTHESIS,
                        arg:    device_id as u64,
                    },
                };
                if let Ok(map) = mailbox.lock() {
                    if let Some(orch) = map.get(&AgentId::ORCHESTRATOR.0) {
                        let _ = orch.tx.send(notif);
                    }
                }
            }
            Err(e) => error!("[SynthesisAgent] synthesis failed for {}: {e}", profile.name),
        }
    }

    /// Probe a device's MMIO BAR with safe read patterns to capture register layout.
    fn probe_mmio(&self, profile: &DeviceProfile) -> Vec<MmioTransaction> {
        // In real kernel code this performs 32-bit reads at [0, 4, 8, ... 256].
        // Simulated here with deterministic values.
        (0u32..16)
            .map(|i| MmioTransaction {
                offset: i * 4,
                write:  false,
                width:  4,
                value:  (0xDEAD_0000 | i) as u64,
            })
            .collect()
    }
}
