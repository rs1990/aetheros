//! LibraryUpdaterAgent — automatic dependency freshness enforcement.
//!
//! Pipeline (runs every `UPDATE_INTERVAL`):
//!
//!   1. Read workspace `Cargo.toml` to enumerate all dependency names + pinned versions.
//!   2. For each dependency, query crates.io (`/api/v1/crates/{name}`) for the latest
//!      stable version.
//!   3. Compute a semver-compatible upgrade set (minor/patch bumps only; major bumps
//!      require explicit agent approval via the bus).
//!   4. Write an updated `Cargo.toml` with bumped versions.
//!   5. Execute `cargo update` to regenerate `Cargo.lock`.
//!   6. Run `cargo check` to validate the workspace compiles.
//!   7. Broadcast `AgentControl::LibraryUpdated` with a JSON payload listing changes.
//!   8. On compile failure: revert `Cargo.toml` and broadcast `UpdateFailed`.
//!
//! Security note: crates.io queries are read-only.  The compiled Cargo.lock
//! is the authoritative integrity manifest; the agent never fetches pre-built
//! binaries.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::Receiver;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use aether_core::agent::AgentId;
use aether_core::ipc::{BusMessage, MessagePayload};

use crate::bus::AgentMailbox;

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const UPDATE_INTERVAL: Duration = Duration::from_secs(3600 * 6); // every 6 hours

// ---------------------------------------------------------------------------
// Cargo.toml dependency model (minimal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepEntry {
    pub name:    String,
    pub current: String, // semver string as declared
    pub latest:  Option<String>,
    pub upgrade: UpgradeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpgradeKind {
    None,
    Patch,
    Minor,
    /// Major bumps require manual approval.
    MajorPending,
}

// ---------------------------------------------------------------------------
// crates.io API response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CratesIoResponse {
    #[serde(rename = "crate")]
    krate: CrateInfo,
}

#[derive(Deserialize)]
struct CrateInfo {
    newest_version: String,
}

// ---------------------------------------------------------------------------
// Semver helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemVer { major: u32, minor: u32, patch: u32 }

impl SemVer {
    fn parse(s: &str) -> Option<Self> {
        // Strip leading `^`, `~`, `>=`, etc.
        let s = s.trim_start_matches(|c: char| !c.is_ascii_digit());
        let parts: Vec<u32> = s.split('.').filter_map(|p| p.parse().ok()).collect();
        Some(Self {
            major: parts.first().copied().unwrap_or(0),
            minor: parts.get(1).copied().unwrap_or(0),
            patch: parts.get(2).copied().unwrap_or(0),
        })
    }

    fn upgrade_kind(&self, other: &SemVer) -> UpgradeKind {
        if other.major > self.major { return UpgradeKind::MajorPending; }
        if other.minor > self.minor { return UpgradeKind::Minor; }
        if other.patch > self.patch { return UpgradeKind::Patch; }
        UpgradeKind::None
    }

    fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// LibraryUpdaterAgent
// ---------------------------------------------------------------------------

pub struct LibraryUpdaterAgent {
    workspace_root: PathBuf,
    last_run:       Option<Instant>,
    history:        Vec<UpdateRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateRecord {
    pub timestamp_secs: u64,
    pub upgrades:       Vec<DepEntry>,
    pub success:        bool,
    pub error:          Option<String>,
}

impl LibraryUpdaterAgent {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            last_run:       None,
            history:        Vec::new(),
        }
    }

    /// Main loop.
    pub fn run(
        mut self,
        _id:     AgentId,
        rx:      Receiver<BusMessage>,
        mailbox: Arc<Mutex<HashMap<u64, AgentMailbox>>>,
    ) {
        info!("[LibraryUpdater] started; workspace = {}", self.workspace_root.display());

        // Run once at startup.
        self.run_update_cycle(&mailbox);

        loop {
            // Drain messages.
            while let Ok(msg) = rx.try_recv() {
                // opcode 0x30 = ForceUpdateNow
                if let MessagePayload::AgentControl { opcode: 0x30, .. } = msg.payload {
                    info!("[LibraryUpdater] forced update requested");
                    self.run_update_cycle(&mailbox);
                }
            }

            // Scheduled interval.
            if self.last_run.map_or(true, |t| t.elapsed() >= UPDATE_INTERVAL) {
                self.run_update_cycle(&mailbox);
            }

            thread::sleep(Duration::from_secs(60));
        }
    }

    // ---------------------------------------------------------------------------
    // Core update pipeline
    // ---------------------------------------------------------------------------

    fn run_update_cycle(&mut self, mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>) {
        self.last_run = Some(Instant::now());
        info!("[LibraryUpdater] starting update cycle");

        let cargo_path = self.workspace_root.join("Cargo.toml");
        let cargo_orig = match fs::read_to_string(&cargo_path) {
            Ok(s)  => s,
            Err(e) => { error!("[LibraryUpdater] cannot read Cargo.toml: {e}"); return; }
        };

        let deps = self.parse_workspace_deps(&cargo_orig);
        if deps.is_empty() {
            info!("[LibraryUpdater] no workspace deps found");
            return;
        }

        let mut checked = self.check_crates_io(deps);
        let upgrades: Vec<&DepEntry> = checked.iter()
            .filter(|d| d.upgrade == UpgradeKind::Patch || d.upgrade == UpgradeKind::Minor)
            .collect();

        if upgrades.is_empty() {
            info!("[LibraryUpdater] all dependencies are current");
            self.record(checked, true, None);
            return;
        }

        info!("[LibraryUpdater] {} upgrade(s) available", upgrades.len());
        for dep in &upgrades {
            info!(
                "  {} {} → {}",
                dep.name,
                dep.current,
                dep.latest.as_deref().unwrap_or("?")
            );
        }

        // Warn about major-pending items but skip them.
        for dep in checked.iter().filter(|d| d.upgrade == UpgradeKind::MajorPending) {
            warn!(
                "[LibraryUpdater] MAJOR bump for {} {} → {} — requires manual approval",
                dep.name,
                dep.current,
                dep.latest.as_deref().unwrap_or("?")
            );
        }

        // Apply: rewrite Cargo.toml.
        let new_cargo = self.apply_upgrades(&cargo_orig, &checked);
        if let Err(e) = fs::write(&cargo_path, &new_cargo) {
            error!("[LibraryUpdater] failed to write Cargo.toml: {e}");
            return;
        }

        // cargo update → regenerate Cargo.lock
        if !self.run_cargo(&["update"]) {
            warn!("[LibraryUpdater] `cargo update` failed; reverting");
            let _ = fs::write(&cargo_path, &cargo_orig);
            self.record(checked, false, Some("cargo update failed".into()));
            self.broadcast_failure(mailbox, "cargo update failed");
            return;
        }

        // cargo check — ensure workspace still compiles
        if !self.run_cargo(&["check", "--workspace", "--quiet"]) {
            warn!("[LibraryUpdater] `cargo check` failed after upgrade; reverting");
            let _ = fs::write(&cargo_path, &cargo_orig);
            self.run_cargo(&["update"]); // re-sync lock to original
            self.record(checked, false, Some("cargo check failed".into()));
            self.broadcast_failure(mailbox, "cargo check failed after upgrade");
            return;
        }

        info!("[LibraryUpdater] upgrade cycle complete — {} packages updated", upgrades.len());
        self.record(checked.clone(), true, None);
        self.broadcast_success(mailbox, &checked);
    }

    // ---------------------------------------------------------------------------
    // Cargo.toml parsing (simplified TOML key-value scan)
    // ---------------------------------------------------------------------------

    fn parse_workspace_deps(&self, toml: &str) -> Vec<DepEntry> {
        let mut in_workspace_deps = false;
        let mut deps = Vec::new();

        for line in toml.lines() {
            let trimmed = line.trim();
            if trimmed == "[workspace.dependencies]" {
                in_workspace_deps = true;
                continue;
            }
            if trimmed.starts_with('[') {
                in_workspace_deps = false;
            }
            if !in_workspace_deps { continue; }
            // Match: `name = { version = "x.y.z", ... }` or `name = "x.y.z"`
            if let Some((name, rest)) = trimmed.split_once('=') {
                let name    = name.trim().to_string();
                let rest    = rest.trim();
                let version = extract_version(rest);
                if let Some(v) = version {
                    deps.push(DepEntry {
                        name,
                        current: v,
                        latest:  None,
                        upgrade: UpgradeKind::None,
                    });
                }
            }
        }
        deps
    }

    // ---------------------------------------------------------------------------
    // crates.io version check
    // ---------------------------------------------------------------------------

    fn check_crates_io(&self, mut deps: Vec<DepEntry>) -> Vec<DepEntry> {
        for dep in &mut deps {
            match fetch_latest_version(&dep.name) {
                Ok(latest) => {
                    let cur = SemVer::parse(&dep.current);
                    let lat = SemVer::parse(&latest);
                    dep.upgrade = match (cur, lat.as_ref()) {
                        (Some(c), Some(l)) => c.upgrade_kind(l),
                        _                  => UpgradeKind::None,
                    };
                    dep.latest = Some(latest);
                }
                Err(e) => {
                    warn!("[LibraryUpdater] crates.io fetch failed for {}: {e}", dep.name);
                }
            }
        }
        deps
    }

    // ---------------------------------------------------------------------------
    // Rewrite Cargo.toml with bumped versions
    // ---------------------------------------------------------------------------

    fn apply_upgrades(&self, original: &str, deps: &[DepEntry]) -> String {
        let mut result = original.to_string();
        for dep in deps.iter().filter(|d| {
            d.upgrade == UpgradeKind::Patch || d.upgrade == UpgradeKind::Minor
        }) {
            if let Some(new_ver) = &dep.latest {
                // Replace `"<old_ver>"` with `"<new_ver>"` for this dep's line.
                // We match the specific line to avoid false replacements.
                let old_pat = format!("\"{}\"", dep.current);
                let new_pat = format!("\"{}\"", new_ver);
                // Only replace within the dep's line context.
                if let Some(pos) = result.find(&format!("{} =", dep.name)) {
                    let end = result[pos..].find('\n').map(|i| pos + i).unwrap_or(result.len());
                    let line = &result[pos..end];
                    if line.contains(&old_pat) {
                        let new_line = line.replacen(&old_pat, &new_pat, 1);
                        result = format!("{}{}{}", &result[..pos], new_line, &result[end..]);
                    }
                }
            }
        }
        result
    }

    // ---------------------------------------------------------------------------
    // Run cargo subprocess
    // ---------------------------------------------------------------------------

    fn run_cargo(&self, args: &[&str]) -> bool {
        match Command::new("cargo")
            .args(args)
            .current_dir(&self.workspace_root)
            .output()
        {
            Ok(out) => {
                if !out.status.success() {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    warn!("[LibraryUpdater] cargo {} failed:\n{}", args.join(" "), stderr);
                    false
                } else {
                    true
                }
            }
            Err(e) => {
                error!("[LibraryUpdater] failed to spawn cargo: {e}");
                false
            }
        }
    }

    // ---------------------------------------------------------------------------
    // History + broadcast
    // ---------------------------------------------------------------------------

    fn record(&mut self, deps: Vec<DepEntry>, success: bool, error: Option<String>) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        self.history.push(UpdateRecord {
            timestamp_secs: ts,
            upgrades:       deps,
            success,
            error,
        });
        if self.history.len() > 100 {
            self.history.drain(..50); // keep last 50
        }
    }

    fn broadcast_success(
        &self,
        mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>,
        deps:    &[DepEntry],
    ) {
        let count = deps.iter().filter(|d| d.upgrade != UpgradeKind::None).count() as u64;
        let msg = BusMessage {
            origin:  AgentId::LIBRARY_UPDATER,
            dest:    AgentId::BROADCAST,
            seq:     0,
            payload: MessagePayload::AgentControl {
                opcode: 0x31, // LibraryUpdated
                target: AgentId::KERNEL,
                arg:    count,
            },
        };
        if let Ok(map) = mailbox.lock() {
            for mb in map.values() { let _ = mb.tx.send(msg); }
        }
    }

    fn broadcast_failure(&self, mailbox: &Arc<Mutex<HashMap<u64, AgentMailbox>>>, _reason: &str) {
        let msg = BusMessage {
            origin:  AgentId::LIBRARY_UPDATER,
            dest:    AgentId::BROADCAST,
            seq:     0,
            payload: MessagePayload::AgentControl {
                opcode: 0x32, // LibraryUpdateFailed
                target: AgentId::KERNEL,
                arg:    0,
            },
        };
        if let Ok(map) = mailbox.lock() {
            for mb in map.values() { let _ = mb.tx.send(msg); }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_version(s: &str) -> Option<String> {
    // Handles: `"1.2.3"` or `{ version = "1.2.3", ... }`
    let inner = if s.contains("version") {
        s.split("version").nth(1)?
    } else {
        s
    };
    let start = inner.find('"')? + 1;
    let end   = inner[start..].find('"')? + start;
    Some(inner[start..end].to_string())
}

/// Fetch latest stable version from crates.io.
fn fetch_latest_version(name: &str) -> Result<String, String> {
    let url = format!("https://crates.io/api/v1/crates/{name}");
    // Use ureq (sync) or fall back gracefully.
    #[cfg(feature = "network")]
    {
        let resp: CratesIoResponse = ureq::get(&url)
            .set("User-Agent", "AetherOS/LibraryUpdater (contact@aetheros.dev)")
            .call()
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;
        return Ok(resp.krate.newest_version);
    }
    // Without network feature: simulate no-op (real build adds ureq).
    Err("network feature disabled".to_string())
}
