# AetherOS

A Rust-based, AI-native microkernel OS architecture optimized for local NPUs,
autonomous hardware integration, and cluster-scale distributed compute.

> **Current status: Architectural prototype / research codebase.**
> This is not yet a bootable operating system. See [Project Status](#project-status) for a full honest assessment.

---

## Vision

AetherOS is designed around three ideas that diverge from conventional OS design:

1. **The OS is managed by AI agents, not configuration files.**
   A swarm of Ring-0 System Agents handle device discovery, driver synthesis,
   resource scheduling, and storage indexing autonomously.

2. **Data is addressed by meaning, not location.**
   The Unified Semantic Data Fabric (USDF) stores everything as embedding
   vectors. A legacy VFS layer presents this as a normal filesystem to
   existing apps.

3. **Hardware is a runtime resource, not a boot-time constant.**
   New CPUs, NPUs, GPUs, and cluster nodes are integrated into the active
   compute pool without reboots via the Neural Intent Scheduler.

---

## Crate Architecture

```
aetheros/
├── aether-core        # no_std microkernel primitives
│   ├── memory         # Physical memory map (Standard RAM / Weight-Cache / Cluster-Mapped / Persona)
│   ├── ipc            # Zero-copy lock-free agent bus (ring buffers, BusMessage)
│   ├── agent          # SystemAgent trait, AgentId, AgentCapability bitflags
│   ├── scheduler      # NeuralIntentScheduler — KV-Cache as first-class resource
│   └── memory_pressure# Pressure monitor (Normal/Elevated/Critical/OOM watermarks)
│
├── aether-hal         # Neural Hardware Abstraction Layer
│   ├── discovery      # DriverDiscovery trait, PCI/USB enumeration, HalRegistry
│   ├── npu            # NeuralDevice trait, JitPatchContext, NpuBackend (ANE/Intel/CUDA)
│   ├── synthesis      # AgentCompiler trait, MmioPattern classifier, TemplateCompiler
│   └── memory_pool    # GPU/NPU memory pool (Hot/Warm/Cold tiers, LRU eviction, defrag)
│
├── aether-usdf        # Unified Semantic Data Fabric
│   ├── vector         # VectorStore, EmbeddingIndex (cosine ANN), SemanticNamespace
│   ├── vfs            # VfsBridge — path trie → VectorId; semantic_open(query, k)
│   └── retention      # AccessRegistry, RetentionPolicy, eviction scoring
│
├── aether-agents      # Ring-0 system agent swarm
│   ├── bus            # AgentRuntime (Tokio-backed), AgentMailbox
│   ├── synthesis      # SynthesisAgent — HAL miss → MMIO probe → DriverShim
│   ├── orchestrator   # OrchestratorAgent — compute ledger, hot-plug, cluster join
│   ├── curator        # CuratorAgent — maps storage devices into USDF
│   ├── memory_optimizer# MemoryOptimizerAgent — CPU+GPU pressure response, pool eviction
│   ├── library_updater # LibraryUpdaterAgent — crates.io version checks, auto cargo update
│   └── janitor        # JanitorAgent — periodic sweep: USDF, shims, RDMA, persona overlay
│
└── aether-cluster     # Cluster swarm layer
    ├── gossip          # SWIM-inspired membership (NodeId, heartbeat, failure detection)
    └── rdma            # RDMA region registry, ClusterMemoryMap bump allocator
```

---

## Physical Memory Layout

```
Physical Address          Region               Size      Flags
─────────────────────────────────────────────────────────────────────
0x0000_0000_0001_0000    Kernel code          15 MB     IMMUTABLE
0x0000_0000_0100_0000    Standard RAM         ~127 GB   RW CACHED
0x0000_0020_0000_0000    Weight-Cache         128 GB    RW CACHED KV_CACHE
0x0000_0040_0000_0000    Cluster-Mapped       128 GB    RW DMA CLUSTER
0x0000_0060_0000_0000    MMIO                 64 GB     RW MMIO
0x0000_0070_0000_0000    Persona Overlay      64 GB     RW ENCRYPTED COW
0x0000_0080_0000_0000    Apple ANE MMIO       16 GB     RW MMIO
0x0000_0090_0000_0000    Intel NPU MMIO       16 GB     RW MMIO
0x0000_00A0_0000_0000    Nvidia GPU DMA       32 GB     RW DMA MMIO
```

---

## Device Plug-in Pipeline

When a new device is connected the following autonomous pipeline runs:

```
Hardware IRQ (new PCI/USB ID)
       │
       ▼
Kernel enumerates device → DeviceProfile
       │
       ├─► HalRegistry.best_driver() ──► found ──► driver.init() ──► DeviceHandle
       │
       └─► miss ──► SynthesisAgent
                         │  1. probe_mmio() — safe reads at known offsets
                         │  2. MmioPattern::classify() — fingerprint registers
                         │  3. fetch spec from external data source (optional)
                         │  4. TemplateCompiler::synthesize() → DriverShim (Rust source + ELF stub)
                         │  5. Persist shim source to Persona Overlay
                         ▼
              AgentControl::DriverReady ──► OrchestratorAgent
                         │  1. Register new ComputeUnit in Scheduler
                         │  2. Broadcast ResourceAdded to all agents
                         ▼
              (if StorageBlock) ──► CuratorAgent
                         │  1. Walk partition table
                         │  2. Generate root embedding for volume
                         │  3. VectorStore.insert() + VfsBridge.bind("/mnt/<label>")
```

---

## Memory Optimization System

### CPU + GPU Memory

| Component | What it does |
|---|---|
| `MemoryMonitor` | Samples all regions every 100 ms; escalates `Normal→Elevated→Critical→OOM` |
| `GpuMemoryPool` | Three-tier (Hot/Warm/Cold) allocator per NPU; LRU eviction with kind weighting |
| `PoolRegistry` | One pool per silicon: cpu_heap, weight_cache, ane_vram, intel_npu, nvidia_vram |
| `MemoryOptimizerAgent` | Responds to pressure: evict USDF entries → GPU LRU → OOM offload to cluster |

### Pressure Response

```
Normal   (< 70%)  — no action
Elevated (70-85%) — evict 16 USDF entries, free 256 MiB from GPU pools
Critical (85-95%) — evict 64 USDF entries, free 1 GiB from GPU pools
OOM      (> 95%)  — evict all evictable, broadcast OffloadToCluster to Orchestrator
```

---

## Automated Cleanup (JanitorAgent)

Sweeps every 5 minutes:

| Target | Policy |
|---|---|
| USDF vector entries | LRU × frequency × size scoring; evict lowest-scored batch |
| Log vectors (`kind=log`) | Purge after 3 days |
| Synthesized shim sources | Remove after 1 hour (compiled object kept) |
| RDMA import regions | Unmap regions owned by gossip-Dead nodes |
| Persona overlay pages | Evict pages idle > 2 hours |

Each sweep writes a `CleanupReport` JSON to `/system/janitor/sweep_N` in the USDF.

---

## Library Auto-Update (LibraryUpdaterAgent)

Runs every 6 hours:

1. Parses workspace `Cargo.toml` for all dependency versions.
2. Queries `crates.io/api/v1/crates/{name}` for latest stable versions.
3. Applies **patch and minor** bumps automatically.
4. Flags **major** bumps as pending — requires manual approval.
5. Runs `cargo update` then `cargo check --workspace`.
6. Reverts on compile failure; broadcasts `LibraryUpdateFailed`.

Enable live network queries: `cargo build --features aether-agents/network`

---

## Project Status

### What is implemented

| Area | Status | Notes |
|---|---|---|
| Memory map constants | ✅ Complete | 64-bit layout, all regions defined |
| Zero-copy agent bus | ✅ Complete | Lock-free SPSC ring, 1024 slots, cache-line padded |
| SystemAgent trait | ✅ Complete | on_start / on_message / tick / on_stop |
| Neural Intent Scheduler | ✅ Complete | 5 priority tiers, KV-cache budget, hot-plug |
| Memory pressure monitor | ✅ Complete | 4 levels, per-region watermarks, callback dispatch |
| DriverDiscovery trait | ✅ Complete | probe/init/shutdown, HalRegistry priority routing |
| NeuralDevice trait | ✅ Complete | ANE/Intel NPU/Nvidia stubs; JitPatchContext |
| AgentCompiler trait | ✅ Complete | MMIO pattern classifier, TemplateCompiler |
| GPU memory pool | ✅ Complete | 3-tier, LRU eviction, defrag, PoolRegistry |
| USDF VectorStore | ✅ Complete | Cosine ANN, per-namespace index |
| VFS bridge | ✅ Complete | Path trie → VectorId, semantic_open, readdir |
| Retention / access tracking | ✅ Complete | 4 composable policies, eviction scoring |
| SynthesisAgent | ✅ Complete | Full MMIO probe → shim pipeline |
| OrchestratorAgent | ✅ Complete | Resource ledger, hot-plug, cluster join |
| CuratorAgent | ✅ Complete | Partition walk → USDF binding |
| MemoryOptimizerAgent | ✅ Complete | CPU+GPU pressure response, /proc/meminfo |
| LibraryUpdaterAgent | ✅ Complete | crates.io checks, cargo update, rollback |
| JanitorAgent | ✅ Complete | 5-pass sweep, cleanup report, USDF persistence |
| Gossip protocol | ✅ Complete | SWIM heartbeat, failure detection, GC |
| RDMA region manager | ✅ Complete | Bump allocator, export/import/unmap |

### What is NOT implemented (not a real OS yet)

| Missing piece | Impact |
|---|---|
| **Bootloader** | Cannot boot on any hardware |
| **x86_64/ARM64 startup** | No GDT, IDT, paging setup |
| **Real hardware drivers** | All HAL implementations are stubs |
| **Process isolation** | No address spaces, no userspace |
| **System calls** | No kernel/user boundary |
| **Real file I/O** | VFS bridge is in-memory only |
| **Network stack** | No TCP/IP (gossip has no real transport) |
| **Display / graphics** | No framebuffer, no display server |
| **POSIX / Linux ABI** | Cannot run any existing Linux app |
| **JIT compiler** | TemplateCompiler produces source; linking is a stub |

### Can I run Chrome, Office, or normal apps?

**No — not in the current state, and not without years of additional work.**

Running standard applications requires:
- A full POSIX-compatible kernel (Linux took 30+ years to build)
- A complete userspace (libc, dynamic linker, display server)
- Windows app support requires Wine on top of the above

### Realistic paths forward

**Option A — Userspace prototype (weeks)**
Run AetherOS agents as a Linux process. The USDF, memory optimizer, and agent bus
all work today as libraries. Validates the AI-native model without kernel work.

**Option B — Build on Redox OS (months)**
[Redox OS](https://www.redox-os.org/) is a working Rust microkernel that can already
run some apps. Port the AetherOS agent layer and USDF on top of it.

**Option C — Full bare-metal kernel (years)**
The architecture is sound. The distance from here to a bootable OS with app support
is enormous and requires a dedicated team.

---

## Building

```bash
# Requires Rust nightly (nightly features used in aether-core)
rustup install nightly
rustup override set nightly

cargo build --workspace

# With live crates.io update checks:
cargo build --workspace --features aether-agents/network
```

---

## Contributing

This is an open architectural research project. Areas where contributions are most valuable:

- Bootloader integration (`bootboot` or `limine`)
- Real PCI enumeration (replace HAL stubs)
- smoltcp integration for gossip transport
- HNSW index replacing the brute-force EmbeddingIndex
- Cranelift backend for AgentCompiler JIT linking

---

## License

MIT
