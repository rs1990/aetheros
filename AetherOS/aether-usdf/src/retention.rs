//! Retention policy engine for the USDF vector store.
//!
//! Every vector entry accumulates an `AccessRecord`.  The `RetentionScorer`
//! computes an eviction score (lower = evict first) from:
//!   • Age since last access (recency)
//!   • Total access count (frequency)
//!   • Data size (large cold entries evicted first)
//!   • Policy pin overrides
//!
//! Policies compose additively: multiple policies can be registered and
//! their scores are multiplied — a vector that scores low on *both* recency
//! and frequency is a strong eviction candidate.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::vector::VectorId;

// ---------------------------------------------------------------------------
// Access record
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AccessRecord {
    pub vector_id:    VectorId,
    pub created_at:   Instant,
    pub last_access:  Instant,
    pub access_count: u64,
    /// Bytes of the associated payload (0 if none).
    pub payload_bytes: usize,
    /// True if an agent has explicitly pinned this entry.
    pub pinned:        bool,
    /// Optional logical path (for VFS-accessible entries).
    pub path:          Option<String>,
}

impl AccessRecord {
    pub fn new(id: VectorId, payload_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            vector_id:    id,
            created_at:   now,
            last_access:  now,
            access_count: 0,
            payload_bytes,
            pinned:       false,
            path:         None,
        }
    }

    pub fn touch(&mut self) {
        self.last_access  = Instant::now();
        self.access_count += 1;
    }

    pub fn age(&self)         -> Duration { self.created_at.elapsed() }
    pub fn idle_time(&self)   -> Duration { self.last_access.elapsed() }
}

// ---------------------------------------------------------------------------
// RetentionPolicy trait
// ---------------------------------------------------------------------------

/// A scoring function; lower scores are evicted first.
/// Returning `f64::MAX` means "never evict this entry under this policy."
pub trait RetentionPolicy: Send + Sync {
    fn name(&self) -> &'static str;
    fn score(&self, record: &AccessRecord) -> f64;
}

// ---------------------------------------------------------------------------
// Built-in policies
// ---------------------------------------------------------------------------

/// Evict entries idle for more than `max_idle`.
pub struct IdleTimePolicy {
    pub max_idle: Duration,
}

impl RetentionPolicy for IdleTimePolicy {
    fn name(&self) -> &'static str { "idle_time" }

    fn score(&self, rec: &AccessRecord) -> f64 {
        if rec.pinned { return f64::MAX; }
        let ratio = rec.idle_time().as_secs_f64() / self.max_idle.as_secs_f64();
        1.0 / ratio.max(0.001) // high idle → low score → evict
    }
}

/// Evict large payloads first when under storage pressure.
pub struct SizeWeightedPolicy {
    pub size_threshold_bytes: usize,
}

impl RetentionPolicy for SizeWeightedPolicy {
    fn name(&self) -> &'static str { "size_weighted" }

    fn score(&self, rec: &AccessRecord) -> f64 {
        if rec.pinned { return f64::MAX; }
        if rec.payload_bytes < self.size_threshold_bytes { return f64::MAX; }
        // Score inversely proportional to size: larger = lower score.
        1.0 / (rec.payload_bytes as f64 + 1.0)
    }
}

/// Preserve frequently-accessed entries.
pub struct FrequencyPolicy {
    /// Accesses below this count are scored for potential eviction.
    pub min_accesses: u64,
}

impl RetentionPolicy for FrequencyPolicy {
    fn name(&self) -> &'static str { "frequency" }

    fn score(&self, rec: &AccessRecord) -> f64 {
        if rec.pinned { return f64::MAX; }
        if rec.access_count >= self.min_accesses { return f64::MAX; }
        1.0 / (rec.access_count as f64 + 1.0)
    }
}

/// Hard age limit: evict anything older than `max_age` regardless of access.
pub struct MaxAgePolicy {
    pub max_age: Duration,
}

impl RetentionPolicy for MaxAgePolicy {
    fn name(&self) -> &'static str { "max_age" }

    fn score(&self, rec: &AccessRecord) -> f64 {
        if rec.pinned                         { return f64::MAX; }
        if rec.age() < self.max_age           { return f64::MAX; }
        // Force-evict: return very low score.
        0.0001
    }
}

// ---------------------------------------------------------------------------
// RetentionScorer — composite scorer
// ---------------------------------------------------------------------------

pub struct RetentionScorer {
    policies: Vec<Box<dyn RetentionPolicy>>,
}

impl RetentionScorer {
    pub fn new() -> Self { Self { policies: Vec::new() } }

    pub fn with_defaults() -> Self {
        let mut s = Self::new();
        s.add(Box::new(IdleTimePolicy    { max_idle:             Duration::from_secs(3600) }));
        s.add(Box::new(FrequencyPolicy   { min_accesses:         5 }));
        s.add(Box::new(SizeWeightedPolicy{ size_threshold_bytes: 1024 * 1024 })); // 1 MiB
        s.add(Box::new(MaxAgePolicy      { max_age:              Duration::from_secs(86400 * 7) })); // 1 week
        s
    }

    pub fn add(&mut self, policy: Box<dyn RetentionPolicy>) {
        self.policies.push(policy);
    }

    /// Composite score: product of all policy scores.
    /// If any policy returns MAX the entry is fully protected.
    pub fn score(&self, record: &AccessRecord) -> f64 {
        self.policies.iter().fold(1.0f64, |acc, p| {
            let s = p.score(record);
            if s == f64::MAX { f64::MAX } else { acc * s }
        })
    }
}

// ---------------------------------------------------------------------------
// AccessRegistry — per-store tracking table
// ---------------------------------------------------------------------------

pub struct AccessRegistry {
    records: HashMap<VectorId, AccessRecord>,
    scorer:  RetentionScorer,
}

impl AccessRegistry {
    pub fn new(scorer: RetentionScorer) -> Self {
        Self { records: HashMap::new(), scorer }
    }

    /// Register a newly inserted vector.
    pub fn on_insert(&mut self, id: VectorId, payload_bytes: usize, path: Option<String>) {
        let mut rec  = AccessRecord::new(id, payload_bytes);
        rec.path     = path;
        self.records.insert(id, rec);
    }

    /// Mark a vector as accessed.
    pub fn on_access(&mut self, id: VectorId) {
        if let Some(rec) = self.records.get_mut(&id) {
            rec.touch();
        }
    }

    pub fn on_remove(&mut self, id: VectorId) {
        self.records.remove(&id);
    }

    pub fn pin(&mut self, id: VectorId) {
        if let Some(rec) = self.records.get_mut(&id) { rec.pinned = true; }
    }

    pub fn unpin(&mut self, id: VectorId) {
        if let Some(rec) = self.records.get_mut(&id) { rec.pinned = false; }
    }

    /// Return up to `limit` eviction candidates, lowest score first.
    pub fn eviction_candidates(&self, limit: usize) -> Vec<VectorId> {
        let mut scored: Vec<(f64, VectorId)> = self
            .records
            .values()
            .filter(|r| !r.pinned)
            .map(|r| (self.scorer.score(r), r.vector_id))
            .filter(|(s, _)| *s < f64::MAX)
            .collect();
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    /// Return candidates older than `max_age` (for the MaxAge hard eviction path).
    pub fn aged_candidates(&self, max_age: Duration) -> Vec<VectorId> {
        self.records
            .values()
            .filter(|r| !r.pinned && r.age() >= max_age)
            .map(|r| r.vector_id)
            .collect()
    }

    pub fn record_count(&self) -> usize { self.records.len() }

    pub fn total_payload_bytes(&self) -> usize {
        self.records.values().map(|r| r.payload_bytes).sum()
    }
}
