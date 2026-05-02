//! Semantic vector store — the core of the USDF.
//!
//! Architecture:
//!   • Data is stored as fixed-dimension float32 embeddings.
//!   • An HNSW-inspired approximate-nearest-neighbour index sits on top.
//!   • Each vector lives in a `SemanticNamespace`; namespaces map to
//!     logical "drives" in the VFS bridge.
//!   • The store is backed by pages in the Weight-Cache memory region
//!     (`RegionFlags::KV_CACHE`) so the scheduler keeps them hot.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// 64-bit stable identifier for a vector entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(pub u64);

/// Logical partition (think: separate index per "drive" or "mount point").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticNamespace(pub u32);

impl SemanticNamespace {
    pub const SYSTEM:  Self = Self(0);
    pub const USER:    Self = Self(1);
    pub const CLUSTER: Self = Self(2);
}

/// A single embedding entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticVector {
    pub id:        VectorId,
    pub ns:        SemanticNamespace,
    /// Embedding components (dimension fixed at index creation time).
    pub embedding: Vec<f32>,
    /// Optional raw bytes for the associated data payload.
    /// `None` if the data is stored externally (e.g., a large file in cluster mem).
    pub payload:   Option<Vec<u8>>,
    /// Arbitrary string metadata (MIME type, origin path, timestamp, etc.).
    pub meta:      HashMap<String, String>,
}

impl SemanticVector {
    /// Cosine similarity in [−1, 1].
    pub fn cosine_similarity(&self, other: &[f32]) -> f32 {
        assert_eq!(self.embedding.len(), other.len(), "dimension mismatch");
        let dot: f32 = self.embedding.iter().zip(other).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingIndex — lightweight flat index (replace with HNSW for prod)
// ---------------------------------------------------------------------------

/// Approximate nearest-neighbour index over one namespace.
///
/// Current implementation: brute-force cosine scan.
/// Drop-in replacement: HNSW layer that stores graph edges in the
/// weight-cache region alongside the embeddings.
pub struct EmbeddingIndex {
    pub ns:        SemanticNamespace,
    pub dim:       usize,
    vectors:       Vec<(VectorId, Vec<f32>)>,
}

impl EmbeddingIndex {
    pub fn new(ns: SemanticNamespace, dim: usize) -> Self {
        Self { ns, dim, vectors: Vec::new() }
    }

    pub fn insert(&mut self, id: VectorId, embedding: Vec<f32>) {
        assert_eq!(embedding.len(), self.dim);
        if let Some(pos) = self.vectors.iter().position(|(v, _)| *v == id) {
            self.vectors[pos].1 = embedding;
        } else {
            self.vectors.push((id, embedding));
        }
    }

    pub fn remove(&mut self, id: VectorId) {
        self.vectors.retain(|(v, _)| *v != id);
    }

    /// Return up to `k` nearest neighbours by cosine similarity.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<(VectorId, f32)> {
        assert_eq!(query.len(), self.dim);
        let mut scores: Vec<(VectorId, f32)> = self
            .vectors
            .iter()
            .map(|(id, emb)| {
                let dot: f32 = emb.iter().zip(query).map(|(a, b)| a * b).sum();
                let na: f32  = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32  = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) };
                (*id, sim)
            })
            .collect();
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
}

// ---------------------------------------------------------------------------
// VectorStore — the global USDF store
// ---------------------------------------------------------------------------

/// Central store: holds both the vector metadata and the embedding indices.
pub struct VectorStore {
    /// id → full SemanticVector (metadata + payload)
    entries:  HashMap<VectorId, SemanticVector>,
    /// namespace → ANN index
    indices:  HashMap<SemanticNamespace, EmbeddingIndex>,
    next_id:  u64,
    pub dim:  usize,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        let mut indices = HashMap::new();
        for ns in [SemanticNamespace::SYSTEM, SemanticNamespace::USER, SemanticNamespace::CLUSTER] {
            indices.insert(ns, EmbeddingIndex::new(ns, dim));
        }
        Self { entries: HashMap::new(), indices, next_id: 1, dim }
    }

    /// Insert a new vector; returns its assigned `VectorId`.
    pub fn insert(
        &mut self,
        ns:        SemanticNamespace,
        embedding: Vec<f32>,
        payload:   Option<Vec<u8>>,
        meta:      HashMap<String, String>,
    ) -> VectorId {
        let id = VectorId(self.next_id);
        self.next_id += 1;
        let idx = self.indices.entry(ns).or_insert_with(|| EmbeddingIndex::new(ns, self.dim));
        idx.insert(id, embedding.clone());
        self.entries.insert(id, SemanticVector { id, ns, embedding, payload, meta });
        id
    }

    pub fn get(&self, id: VectorId) -> Option<&SemanticVector> {
        self.entries.get(&id)
    }

    pub fn remove(&mut self, id: VectorId) {
        if let Some(v) = self.entries.remove(&id) {
            if let Some(idx) = self.indices.get_mut(&v.ns) {
                idx.remove(id);
            }
        }
    }

    /// Semantic search across a namespace.
    pub fn query(&self, ns: SemanticNamespace, query: &[f32], k: usize) -> Vec<(VectorId, f32)> {
        self.indices.get(&ns).map_or(vec![], |idx| idx.query(query, k))
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}
