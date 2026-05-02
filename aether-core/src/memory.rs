//! Physical memory layout for AetherOS.
//!
//! Three first-class memory types:
//!   Standard RAM      — general userspace / kernel heap
//!   Local Weight-Cache — KV-cache and model weight pages (non-evictable by default)
//!   Cluster-Mapped    — RDMA-backed pages owned by the swarm layer
//!
//! The Persona Overlay is an encrypted, copy-on-write region that holds
//! agent weights and per-user state.  All other kernel sections are
//! IMMUTABLE after the boot integrity check.

use bitflags::bitflags;

// ---------------------------------------------------------------------------
// Region flags
// ---------------------------------------------------------------------------

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct RegionFlags: u32 {
        const READABLE   = 1 << 0;
        const WRITABLE   = 1 << 1;
        const EXECUTABLE = 1 << 2;
        const ENCRYPTED  = 1 << 3;
        const CACHED     = 1 << 4;
        const MMIO       = 1 << 5;
        const DMA        = 1 << 6;
        /// Kernel enforces no writes after boot integrity check.
        const IMMUTABLE  = 1 << 7;
        /// Pages belong to the KV-Cache memory tier; scheduler-pinned.
        const KV_CACHE   = 1 << 8;
        /// Pages are RDMA-mapped from a remote cluster node.
        const CLUSTER    = 1 << 9;
        /// Copy-on-write (used by Persona Overlay).
        const COW        = 1 << 10;
    }
}

// ---------------------------------------------------------------------------
// Region descriptor
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRegion {
    pub base:  u64,
    pub size:  u64,
    pub flags: RegionFlags,
}

impl MemoryRegion {
    #[inline]
    pub const fn end(&self) -> u64 {
        self.base + self.size
    }

    #[inline]
    pub const fn contains_phys(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.end()
    }
}

// ---------------------------------------------------------------------------
// NPU silicon identifiers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NpuSilicon {
    AppleAne  = 0,
    IntelNpu  = 1,
    NvidiaGpu = 2,
    Reserved  = 3,
}

// ---------------------------------------------------------------------------
// Full physical memory map
// ---------------------------------------------------------------------------

pub struct PhysicalMemoryMap {
    pub standard_ram:    MemoryRegion,
    pub weight_cache:    MemoryRegion,
    pub cluster_mapped:  MemoryRegion,
    pub persona_overlay: MemoryRegion,
    pub mmio:            MemoryRegion,
    pub npu:             [(NpuSilicon, MemoryRegion); 3],
}

/// Canonical AetherOS 64-bit physical address layout.
///
/// ```text
/// 0x0000_0000_0001_0000  kernel code  (15 MB, immutable)
/// 0x0000_0000_0100_0000  Standard RAM (127 GB max)
/// 0x0000_0020_0000_0000  Weight-Cache (128 GB, KV-cache pinned)
/// 0x0000_0040_0000_0000  Cluster-RDMA (128 GB, DMA-mapped)
/// 0x0000_0060_0000_0000  MMIO         (64 GB)
/// 0x0000_0070_0000_0000  Persona      (64 GB, encrypted COW)
/// 0x0000_0080_0000_0000  Apple ANE    (16 GB MMIO)
/// 0x0000_0090_0000_0000  Intel NPU    (16 GB MMIO)
/// 0x0000_00A0_0000_0000  Nvidia GPU   (32 GB DMA+MMIO)
/// ```
pub const MEMORY_MAP: PhysicalMemoryMap = PhysicalMemoryMap {
    standard_ram: MemoryRegion {
        base:  0x0000_0000_0100_0000,
        size:  0x0000_001F_FF00_0000, // ~127 GB
        flags: RegionFlags::READABLE.union(RegionFlags::WRITABLE).union(RegionFlags::CACHED),
    },
    weight_cache: MemoryRegion {
        base:  0x0000_0020_0000_0000,
        size:  0x0000_0020_0000_0000, // 128 GB
        flags: RegionFlags::READABLE
            .union(RegionFlags::WRITABLE)
            .union(RegionFlags::CACHED)
            .union(RegionFlags::KV_CACHE),
    },
    cluster_mapped: MemoryRegion {
        base:  0x0000_0040_0000_0000,
        size:  0x0000_0020_0000_0000, // 128 GB
        flags: RegionFlags::READABLE
            .union(RegionFlags::WRITABLE)
            .union(RegionFlags::DMA)
            .union(RegionFlags::CLUSTER),
    },
    mmio: MemoryRegion {
        base:  0x0000_0060_0000_0000,
        size:  0x0000_0010_0000_0000, // 64 GB
        flags: RegionFlags::READABLE.union(RegionFlags::WRITABLE).union(RegionFlags::MMIO),
    },
    persona_overlay: MemoryRegion {
        base:  0x0000_0070_0000_0000,
        size:  0x0000_0010_0000_0000, // 64 GB
        flags: RegionFlags::READABLE
            .union(RegionFlags::WRITABLE)
            .union(RegionFlags::ENCRYPTED)
            .union(RegionFlags::COW),
    },
    npu: [
        (NpuSilicon::AppleAne, MemoryRegion {
            base:  0x0000_0080_0000_0000,
            size:  0x0000_0004_0000_0000, // 16 GB
            flags: RegionFlags::READABLE.union(RegionFlags::WRITABLE).union(RegionFlags::MMIO),
        }),
        (NpuSilicon::IntelNpu, MemoryRegion {
            base:  0x0000_0090_0000_0000,
            size:  0x0000_0004_0000_0000, // 16 GB
            flags: RegionFlags::READABLE.union(RegionFlags::WRITABLE).union(RegionFlags::MMIO),
        }),
        (NpuSilicon::NvidiaGpu, MemoryRegion {
            base:  0x0000_00A0_0000_0000,
            size:  0x0000_0008_0000_0000, // 32 GB
            flags: RegionFlags::READABLE
                .union(RegionFlags::WRITABLE)
                .union(RegionFlags::MMIO)
                .union(RegionFlags::DMA),
        }),
    ],
};
