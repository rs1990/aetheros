//! Device discovery: PCI/USB enumeration and the `DriverDiscovery` trait.
//!
//! On hardware enumeration the kernel:
//!   1. Reads PCI config space / USB descriptors.
//!   2. Calls `DriverDiscovery::probe` for every registered HAL.
//!   3. On miss, emits `HardwareEvent::DeviceAttached` to trigger the
//!      Synthesis Agent.

use hashbrown::HashMap;

// ---------------------------------------------------------------------------
// Device IDs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PciId {
    pub vendor: u16,
    pub device: u16,
    pub class:  u8,  // PCI base class
    pub sub:    u8,  // PCI sub-class
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UsbId {
    pub vid:   u16,
    pub pid:   u16,
    pub class: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BusId {
    Pci(PciId),
    Usb(UsbId),
}

// ---------------------------------------------------------------------------
// Device class
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    Npu,
    Gpu,
    StorageBlock,
    NetworkNic,
    Sensor,
    Accelerator,
    Unknown,
}

// ---------------------------------------------------------------------------
// DeviceProfile — full description used by the Synthesis Agent
// ---------------------------------------------------------------------------

/// Everything the Synthesis Agent needs to attempt driver generation.
#[derive(Clone, Debug)]
pub struct DeviceProfile {
    pub bus_id:       BusId,
    pub class:        DeviceClass,
    /// Human-readable name from ACPI / USB string descriptor.
    pub name:         String,
    /// Base address of MMIO BAR (PCI) or device-mapped region (USB).
    pub mmio_base:    u64,
    pub mmio_size:    u64,
    /// Interrupt line/vector.
    pub irq:          Option<u32>,
    /// Raw capability bytes from PCI extended config space.
    pub caps_raw:     Vec<u8>,
}

// ---------------------------------------------------------------------------
// DriverDiscovery trait
// ---------------------------------------------------------------------------

/// Implemented by every first-party HAL module.
///
/// When the kernel enumerates a device it iterates the registered
/// `DriverDiscovery` implementations in priority order.  The first one that
/// returns `Some` for `probe` wins and its `init` is called.
pub trait DriverDiscovery: Send + Sync {
    /// Return `Some(priority)` if this HAL can handle the given device.
    /// Higher priority wins a tie (e.g., vendor-specific > generic).
    fn probe(&self, profile: &DeviceProfile) -> Option<u8>;

    /// Initialise the device and return an opaque handle.
    ///
    /// `mmio` is a mutable reference to the kernel-mapped MMIO window for
    /// this device's BAR.
    ///
    /// # Safety
    /// Caller ensures `mmio` points to a valid, exclusively-owned MMIO range.
    unsafe fn init(&self, profile: &DeviceProfile, mmio: *mut u8) -> Result<Box<dyn DeviceHandle>, DriverError>;

    /// Called on graceful removal (hot-unplug).
    fn shutdown(&self, handle: &mut dyn DeviceHandle);
}

// ---------------------------------------------------------------------------
// DeviceHandle — returned by init, kept by the kernel
// ---------------------------------------------------------------------------

pub trait DeviceHandle: Send + Sync {
    fn device_id(&self) -> BusId;
    fn class(&self)     -> DeviceClass;
    /// Poll for pending completions; called from the kernel interrupt handler.
    fn poll(&mut self) -> PollResult;
}

#[derive(Debug)]
pub enum PollResult {
    NoPending,
    Completed(u32),  // number of completions processed
    Error(DriverError),
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("MMIO mapping failed at {0:#x}")]
    MmioMapFailed(u64),
    #[error("device did not respond within timeout")]
    Timeout,
    #[error("driver initialisation failed: {0}")]
    Init(String),
    #[error("synthesis error: {0}")]
    Synthesis(String),
}

// ---------------------------------------------------------------------------
// HAL registry
// ---------------------------------------------------------------------------

/// Kernel-global registry of `DriverDiscovery` implementations.
pub struct HalRegistry {
    drivers: Vec<Box<dyn DriverDiscovery>>,
    handles: HashMap<u64, Box<dyn DeviceHandle>>, // keyed by device serial
}

impl HalRegistry {
    pub fn new() -> Self {
        Self { drivers: Vec::new(), handles: HashMap::new() }
    }

    pub fn register_driver(&mut self, drv: Box<dyn DriverDiscovery>) {
        self.drivers.push(drv);
    }

    /// Returns the winning driver for `profile`, or `None` on miss.
    pub fn best_driver(&self, profile: &DeviceProfile) -> Option<&dyn DriverDiscovery> {
        self.drivers
            .iter()
            .filter_map(|d| d.probe(profile).map(|p| (p, d.as_ref())))
            .max_by_key(|(p, _)| *p)
            .map(|(_, d)| d)
    }
}
