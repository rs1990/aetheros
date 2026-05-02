//! Driver synthesis: MMIO pattern analysis → compiled Rust driver shim.
//!
//! The `AgentCompiler` trait is the contract between the Synthesis Agent
//! and the HAL.  A concrete implementation would:
//!   1. Receive a `DeviceProfile` with raw MMIO capture.
//!   2. Run `MmioPattern::classify` to fingerprint register layout.
//!   3. Call an LLM or template engine to produce a `DriverShim` source.
//!   4. JIT-compile the shim (via a bundled Rust frontend / cranelift backend).
//!   5. Return a boxed `dyn DriverDiscovery` that the HalRegistry can use.

use crate::discovery::{DeviceProfile, DriverDiscovery, DriverError};

// ---------------------------------------------------------------------------
// MMIO pattern capture
// ---------------------------------------------------------------------------

/// A captured MMIO transaction: offset from BAR, direction, value.
#[derive(Clone, Copy, Debug)]
pub struct MmioTransaction {
    pub offset: u32,
    pub write:  bool,
    pub width:  u8,  // 1, 2, 4, or 8 bytes
    pub value:  u64,
}

/// Classification of a device's MMIO register layout, derived by analysing
/// a sequence of `MmioTransaction` captures.
#[derive(Clone, Debug)]
pub struct MmioPattern {
    /// Detected register stride (0 = irregular).
    pub stride:       u32,
    /// Offsets that behave as command/status registers.
    pub csr_offsets:  Vec<u32>,
    /// Offsets that behave as data fifos.
    pub fifo_offsets: Vec<u32>,
    /// Offsets that look like DMA descriptor rings.
    pub dma_offsets:  Vec<u32>,
    /// Vendor-specific magic values seen in early writes.
    pub magic_seqs:   Vec<u64>,
    /// Confidence 0.0–1.0 that the classification is correct.
    pub confidence:   f32,
}

impl MmioPattern {
    /// Classify a transaction log.  Simple heuristics — real impl uses
    /// a small ONNX model stored in the weight-cache region.
    pub fn classify(txns: &[MmioTransaction]) -> Self {
        let mut csr  = vec![];
        let mut fifo = vec![];
        let mut dma  = vec![];
        let mut magic = vec![];

        // Heuristic: writes to offset 0 with large values → magic/init
        // Writes to offset 4/8 with monotone increase → DMA ring head
        // Reads from offset 0 with changing value → CSR status poll
        for t in txns {
            if t.offset == 0 && t.write && t.value > 0xFFFF {
                magic.push(t.value);
            } else if t.write && t.offset % 4 == 0 && t.offset < 64 {
                csr.push(t.offset);
            } else if t.write && t.offset >= 0x100 {
                dma.push(t.offset);
            } else if !t.write {
                fifo.push(t.offset);
            }
        }

        csr.dedup();
        fifo.dedup();
        dma.dedup();

        let confidence = if txns.len() > 16 { 0.75 } else { 0.40 };

        Self {
            stride:      4,
            csr_offsets:  csr,
            fifo_offsets: fifo,
            dma_offsets:  dma,
            magic_seqs:   magic,
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// DriverShim — compiled output
// ---------------------------------------------------------------------------

/// An in-memory compiled driver shim, ready to be linked into the HAL.
pub struct DriverShim {
    /// Rust source (for audit / storage in the Persona Overlay).
    pub source_rs: String,
    /// Compiled ELF object bytes (produced by the JIT compiler).
    pub obj_bytes: Vec<u8>,
    /// Entry points the kernel will link.
    pub init_sym:     String,
    pub poll_sym:     String,
    pub shutdown_sym: String,
}

// ---------------------------------------------------------------------------
// AgentCompiler trait
// ---------------------------------------------------------------------------

/// The Synthesis Agent implements this trait to produce a live
/// `DriverDiscovery` from observed MMIO behaviour and fetched specs.
///
/// # Pipeline
/// ```text
///  DeviceProfile + MmioCapture
///       │
///       ▼
///  AgentCompiler::analyse  ──►  MmioPattern
///       │
///       ▼
///  AgentCompiler::synthesize  ──►  DriverShim (source + obj)
///       │
///       ▼
///  AgentCompiler::link  ──►  Box<dyn DriverDiscovery>
///       │
///       ▼
///  HalRegistry::register_driver
/// ```
pub trait AgentCompiler: Send + Sync {
    /// Phase 1: classify the MMIO behaviour of the device.
    fn analyse(&self, profile: &DeviceProfile, txns: &[MmioTransaction]) -> MmioPattern;

    /// Phase 2: generate Rust source and compiled object from the pattern.
    ///
    /// `spec_json` — device spec fetched from an external data source
    ///               (may be empty if unavailable; compiler falls back to
    ///               pure MMIO heuristics).
    fn synthesize(
        &self,
        profile: &DeviceProfile,
        pattern: &MmioPattern,
        spec_json: Option<&str>,
    ) -> Result<DriverShim, CompileError>;

    /// Phase 3: link the compiled shim into a live `DriverDiscovery`.
    ///
    /// # Safety
    /// Caller must guarantee `shim.obj_bytes` was produced by this compiler.
    unsafe fn link(&self, shim: DriverShim) -> Result<Box<dyn DriverDiscovery>, CompileError>;
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("MMIO pattern confidence too low ({0:.2}); cannot synthesize")]
    LowConfidence(f32),
    #[error("spec fetch failed: {0}")]
    SpecFetch(String),
    #[error("codegen error: {0}")]
    CodeGen(String),
    #[error("link error: {0}")]
    Link(String),
    #[error("driver init failed: {0}")]
    Init(#[from] DriverError),
}

// ---------------------------------------------------------------------------
// Template-based compiler (reference implementation)
// ---------------------------------------------------------------------------

/// Generates a minimal MMIO driver from templates without a full JIT backend.
/// Suitable for simple PIO devices; complex DMA engines require `JitCompiler`.
pub struct TemplateCompiler {
    pub min_confidence: f32,
}

impl Default for TemplateCompiler {
    fn default() -> Self {
        Self { min_confidence: 0.55 }
    }
}

impl AgentCompiler for TemplateCompiler {
    fn analyse(&self, _profile: &DeviceProfile, txns: &[MmioTransaction]) -> MmioPattern {
        MmioPattern::classify(txns)
    }

    fn synthesize(
        &self,
        profile: &DeviceProfile,
        pattern: &MmioPattern,
        spec_json: Option<&str>,
    ) -> Result<DriverShim, CompileError> {
        if pattern.confidence < self.min_confidence {
            return Err(CompileError::LowConfidence(pattern.confidence));
        }

        let source = generate_source_template(profile, pattern, spec_json);
        // In a real implementation, cranelift or rustc would compile `source`.
        // Here we emit a placeholder ELF stub so the pipeline is complete.
        let obj_bytes = vec![0x7f, b'E', b'L', b'F']; // ELF magic placeholder

        Ok(DriverShim {
            source_rs:    source,
            obj_bytes,
            init_sym:     format!("{}_init", sanitize_name(&profile.name)),
            poll_sym:     format!("{}_poll", sanitize_name(&profile.name)),
            shutdown_sym: format!("{}_shutdown", sanitize_name(&profile.name)),
        })
    }

    unsafe fn link(&self, shim: DriverShim) -> Result<Box<dyn DriverDiscovery>, CompileError> {
        // Real: dlopen-equivalent over the compiled ELF object.
        // Stub returns an error indicating JIT linking is not yet wired.
        Err(CompileError::Link(format!(
            "JIT linking not available for shim '{}'; deploy full JitCompiler",
            shim.init_sym
        )))
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}

fn generate_source_template(
    profile: &DeviceProfile,
    pattern: &MmioPattern,
    spec_json: Option<&str>,
) -> String {
    let spec_comment = spec_json
        .map(|s| format!("// spec: {} bytes\n", s.len()))
        .unwrap_or_default();
    format!(
        r#"// Auto-synthesized by AetherOS TemplateCompiler
// Device: {name}  MMIO base: {mmio:#x}  confidence: {conf:.2}
{spec_comment}
use aether_hal::discovery::{{DriverDiscovery, DeviceProfile, DeviceHandle, DriverError, PollResult}};

pub struct SynthDriver {{ mmio_base: u64 }}

impl DriverDiscovery for SynthDriver {{
    fn probe(&self, p: &DeviceProfile) -> Option<u8> {{
        // Generated probe: match on MMIO base address
        if p.mmio_base == {mmio:#x} {{ Some(10) }} else {{ None }}
    }}

    unsafe fn init(&self, _p: &DeviceProfile, mmio: *mut u8) -> Result<Box<dyn DeviceHandle>, DriverError> {{
        // Generated init: write magic sequence then enable device
        {magic_inits}
        Ok(Box::new(SynthHandle {{ mmio_base: {mmio:#x} }}))
    }}

    fn shutdown(&self, _h: &mut dyn DeviceHandle) {{}}
}}

struct SynthHandle {{ mmio_base: u64 }}
impl DeviceHandle for SynthHandle {{
    fn device_id(&self) -> aether_hal::discovery::BusId {{ todo!() }}
    fn class(&self) -> aether_hal::discovery::DeviceClass {{ aether_hal::discovery::DeviceClass::Unknown }}
    fn poll(&mut self) -> PollResult {{ PollResult::NoPending }}
}}
"#,
        name    = profile.name,
        mmio    = profile.mmio_base,
        conf    = pattern.confidence,
        magic_inits = pattern.magic_seqs.iter().enumerate().map(|(i, v)| {
            format!("        unsafe {{ (mmio as *mut u32).add({i}).write_volatile({v:#x} as u32); }}")
        }).collect::<Vec<_>>().join("\n"),
    )
}
