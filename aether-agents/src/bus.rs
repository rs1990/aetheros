//! User-space agent runtime: wraps the no_std `AgentBus` with Tokio async
//! delivery and a thread-per-agent execution model.
//!
//! The `AgentRuntime` is the top-level owner of all agents.  In production
//! this is instantiated once by the microkernel's ring-0 init path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crossbeam::channel::{self, Receiver, Sender};
use tokio::task::JoinHandle;
use tracing::{error, info};

use aether_core::agent::{AgentCapability, AgentId};
use aether_core::ipc::BusMessage;

// ---------------------------------------------------------------------------
// Per-agent mailbox
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AgentMailbox {
    pub id: AgentId,
    pub tx: Sender<BusMessage>,
}

// ---------------------------------------------------------------------------
// AgentRuntime
// ---------------------------------------------------------------------------

pub struct AgentRuntime {
    mailboxes: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    handles:   Vec<JoinHandle<()>>,
}

impl AgentRuntime {
    pub fn new() -> Self {
        Self {
            mailboxes: Arc::new(Mutex::new(HashMap::new())),
            handles:   Vec::new(),
        }
    }

    /// Spawn an agent on its own thread.
    ///
    /// `run_fn` receives inbound messages from the runtime bus and a clone of
    /// the mailbox map so it can dispatch outbound messages.
    pub fn spawn<F>(&mut self, id: AgentId, run_fn: F)
    where
        F: FnOnce(AgentId, Receiver<BusMessage>, Arc<Mutex<HashMap<u64, AgentMailbox>>>) + Send + 'static,
    {
        let (tx, rx) = channel::unbounded::<BusMessage>();
        let mb = AgentMailbox { id, tx };
        self.mailboxes.lock().unwrap().insert(id.0, mb);
        let shared = Arc::clone(&self.mailboxes);
        self.handles.push(tokio::spawn(async move {
            tokio::task::spawn_blocking(move || run_fn(id, rx, shared))
                .await
                .unwrap_or_else(|e| error!("agent {id:?} panicked: {e}"));
        }));
        info!("registered agent {id:?}");
    }

    /// Dispatch a message.  On broadcast, delivers to all registered agents.
    pub fn send(&self, msg: BusMessage) -> Result<(), RuntimeError> {
        let map = self.mailboxes.lock().unwrap();
        if msg.dest == AgentId::BROADCAST {
            for mb in map.values() {
                let _ = mb.tx.send(msg);
            }
            return Ok(());
        }
        map.get(&msg.dest.0)
            .ok_or(RuntimeError::UnknownDest(msg.dest))?
            .tx
            .send(msg)
            .map_err(|_| RuntimeError::AgentGone(msg.dest))
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("unknown destination agent {0:?}")]
    UnknownDest(AgentId),
    #[error("agent {0:?} has exited")]
    AgentGone(AgentId),
}
