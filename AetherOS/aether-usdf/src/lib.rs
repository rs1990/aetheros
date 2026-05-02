//! aether-usdf — Unified Semantic Data Fabric.
//!
//! Two sub-systems:
//!   vector  — semantic embedding index (the actual data store)
//!   vfs     — POSIX-compatible VFS bridge that maps path strings to
//!             vector entries so legacy applications see a normal filesystem

pub mod retention;
pub mod vector;
pub mod vfs;

pub use retention::{
    AccessRecord, AccessRegistry, FrequencyPolicy, IdleTimePolicy,
    MaxAgePolicy, RetentionPolicy, RetentionScorer, SizeWeightedPolicy,
};
pub use vector::{
    EmbeddingIndex, SemanticNamespace, SemanticVector, VectorId, VectorStore,
};
pub use vfs::{VfsBridge, VfsError, VfsNode, VfsNodeKind};
