//! Unified `NeuralDevice` trait and per-silicon JIT-patch context.
//!
//! The HAL presents all NPU/GPU backends through a single `NeuralDevice`
//! interface.  Backends translate the abstract job description into
//! silicon-specific command streams at submission time.  A `JitPatchContext`
//! holds the runtime-generated dispatch stubs that get linked into the
//! kernel's function-pointer tables — no reboot required on hot-add.

use aether_core::memory::NpuSilicon;

// ---------------------------------------------------------------------------
// NpuJob — silicon-agnostic inference request
// ---------------------------------------------------------------------------

/// Tensor layout descriptor (row-major, NHWC).
#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub ptr:      *const u8, // weight-cache physical pointer
    pub shape:    [u32; 4],  // [N, H, W, C]
    pub dtype:    DataType,
}

// SAFETY: Weight-cache pages are pinned and kernel-owned.
unsafe impl Send for TensorDesc {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    F32, F16, BF16, Int8, Int4,
}

/// Submitted to a `NeuralDevice` to schedule an inference pass.
#[derive(Debug)]
pub struct NpuJob {
    pub id:          u64,
    pub model_token: u64,    // opaque handle into the Weight-Cache region
    pub inputs:      Vec<TensorDesc>,
    pub output_ptr:  *mut u8, // caller-allocated buffer in weight-cache
    pub output_len:  usize,
    /// KV-Cache slice (base + len) that this job may extend.
    pub kv_cache_phys: Option<(u64, usize)>,
}

unsafe impl Send for NpuJob {}

#[derive(Debug)]
pub struct NpuJobResult {
    pub job_id:    u64,
    pub success:   bool,
    pub tokens_ns: u64, // wall time for the job
    pub error:     Option<String>,
}

// ---------------------------------------------------------------------------
// NeuralDevice trait
// ---------------------------------------------------------------------------

/// Uniform interface for Apple ANE, Intel NPU, and Nvidia GPU.
pub trait NeuralDevice: Send + Sync {
    fn silicon(&self) -> NpuSilicon;

    /// True operating TOPS (updated after each calibration tick).
    fn measured_tops(&self) -> f32;

    /// Load a model into the weight-cache region.  Returns a model token.
    fn load_model(&mut self, weights_phys: u64, weights_len: usize) -> Result<u64, NpuError>;

    /// Evict a model from the weight-cache (e.g., under memory pressure).
    fn evict_model(&mut self, token: u64);

    /// Submit an inference job.  Non-blocking; completion signalled via IRQ.
    fn submit(&mut self, job: NpuJob) -> Result<(), NpuError>;

    /// Poll completed jobs.  Returns a vec of results.
    fn poll_completions(&mut self) -> Vec<NpuJobResult>;

    /// Called when the Scheduler adds/removes this device from the active pool.
    fn set_online(&mut self, online: bool);
}

// ---------------------------------------------------------------------------
// JIT-patch context
// ---------------------------------------------------------------------------

/// Holds the runtime-generated dispatch stubs for a single silicon backend.
///
/// On hot-add the HAL:
///   1. Instantiates the appropriate `NpuBackend` (ANE / Intel / CUDA).
///   2. Calls `JitPatchContext::patch` to write dispatch trampolines into
///      the kernel's function-pointer table (rwx page, then locked R/X).
///   3. The scheduler's `add_compute_unit` is called so new tasks can flow
///      to the device immediately.
pub struct JitPatchContext {
    pub silicon:   NpuSilicon,
    /// Virtual address of the kernel dispatch table entry for this device.
    pub table_ptr: *mut usize,
    /// Generated stub: compiled from template IR at runtime.
    pub stub_code: Vec<u8>,
}

impl JitPatchContext {
    /// Write the generated stub into the dispatch table.
    ///
    /// # Safety
    /// `table_ptr` must point to a writable kernel page that the caller has
    /// temporarily marked RW.  The caller is responsible for flushing the
    /// I-cache and re-sealing the page as R/X after this call.
    pub unsafe fn patch(&self) {
        let fn_ptr = self.stub_code.as_ptr() as usize;
        core::ptr::write_volatile(self.table_ptr, fn_ptr);
        // Compiler fence: ensure the store is ordered before I-cache flush.
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }
}

// SAFETY: The dispatch table and stub are in kernel-owned memory.
unsafe impl Send for JitPatchContext {}
unsafe impl Sync for JitPatchContext {}

// ---------------------------------------------------------------------------
// NpuBackend — factory for NeuralDevice instances
// ---------------------------------------------------------------------------

/// Builder pattern: instantiates the correct `NeuralDevice` for detected silicon.
pub struct NpuBackend;

impl NpuBackend {
    /// Returns the appropriate `NeuralDevice` for the given silicon type.
    ///
    /// In production each match arm links the vendor-supplied Rust module.
    /// Stubs are returned for unrecognised silicon (they produce `NpuError::Unsupported`).
    pub fn create(silicon: NpuSilicon, mmio_base: u64) -> Box<dyn NeuralDevice> {
        match silicon {
            NpuSilicon::AppleAne  => Box::new(AneDevice::new(mmio_base)),
            NpuSilicon::IntelNpu  => Box::new(IntelNpuDevice::new(mmio_base)),
            NpuSilicon::NvidiaGpu => Box::new(NvidiaDevice::new(mmio_base)),
            NpuSilicon::Reserved  => Box::new(StubDevice),
        }
    }
}

// ---------------------------------------------------------------------------
// Apple ANE device (stub — real implementation links vendor ANE library)
// ---------------------------------------------------------------------------

struct AneDevice { mmio: u64, online: bool }
impl AneDevice {
    fn new(mmio: u64) -> Self { Self { mmio, online: false } }
}
impl NeuralDevice for AneDevice {
    fn silicon(&self) -> NpuSilicon { NpuSilicon::AppleAne }
    fn measured_tops(&self) -> f32  { 38.0 } // M2 Ultra ANE
    fn load_model(&mut self, _wp: u64, _wl: usize) -> Result<u64, NpuError> { Ok(0) }
    fn evict_model(&mut self, _t: u64) {}
    fn submit(&mut self, _j: NpuJob)   -> Result<(), NpuError> { Ok(()) }
    fn poll_completions(&mut self)      -> Vec<NpuJobResult>    { vec![] }
    fn set_online(&mut self, v: bool)  { self.online = v; }
}

struct IntelNpuDevice { mmio: u64, online: bool }
impl IntelNpuDevice {
    fn new(mmio: u64) -> Self { Self { mmio, online: false } }
}
impl NeuralDevice for IntelNpuDevice {
    fn silicon(&self) -> NpuSilicon { NpuSilicon::IntelNpu }
    fn measured_tops(&self) -> f32  { 11.5 } // Core Ultra NPU tile
    fn load_model(&mut self, _wp: u64, _wl: usize) -> Result<u64, NpuError> { Ok(0) }
    fn evict_model(&mut self, _t: u64) {}
    fn submit(&mut self, _j: NpuJob)   -> Result<(), NpuError> { Ok(()) }
    fn poll_completions(&mut self)      -> Vec<NpuJobResult>    { vec![] }
    fn set_online(&mut self, v: bool)  { self.online = v; }
}

struct NvidiaDevice { mmio: u64, online: bool }
impl NvidiaDevice {
    fn new(mmio: u64) -> Self { Self { mmio, online: false } }
}
impl NeuralDevice for NvidiaDevice {
    fn silicon(&self) -> NpuSilicon { NpuSilicon::NvidiaGpu }
    fn measured_tops(&self) -> f32  { 312.0 } // H100 SXM
    fn load_model(&mut self, _wp: u64, _wl: usize) -> Result<u64, NpuError> { Ok(0) }
    fn evict_model(&mut self, _t: u64) {}
    fn submit(&mut self, _j: NpuJob)   -> Result<(), NpuError> { Ok(()) }
    fn poll_completions(&mut self)      -> Vec<NpuJobResult>    { vec![] }
    fn set_online(&mut self, v: bool)  { self.online = v; }
}

struct StubDevice;
impl NeuralDevice for StubDevice {
    fn silicon(&self) -> NpuSilicon { NpuSilicon::Reserved }
    fn measured_tops(&self) -> f32  { 0.0 }
    fn load_model(&mut self, _: u64, _: usize) -> Result<u64, NpuError> {
        Err(NpuError::Unsupported)
    }
    fn evict_model(&mut self, _: u64) {}
    fn submit(&mut self, _: NpuJob)   -> Result<(), NpuError> { Err(NpuError::Unsupported) }
    fn poll_completions(&mut self)     -> Vec<NpuJobResult>    { vec![] }
    fn set_online(&mut self, _: bool) {}
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum NpuError {
    #[error("silicon backend not supported")]
    Unsupported,
    #[error("command queue full")]
    QueueFull,
    #[error("model token {0} not loaded")]
    ModelNotLoaded(u64),
    #[error("MMIO error at {0:#x}")]
    Mmio(u64),
}
