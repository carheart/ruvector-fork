//! Main VectorDB API

use crate::distance::calculate_distance;
use crate::error::{Result, VectorDbError};
use crate::index::{HnswConfig, HnswIndex};
use crate::storage::Storage;
use crate::types::*;
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Main Vector Database
pub struct VectorDB {
    config: VectorDbConfig,
    storage: Arc<Storage>,
    index: Arc<HnswIndex>,
    stats: Arc<RwLock<VectorDbStats>>,
}

impl VectorDB {
    /// Create a new vector database with configuration
    pub fn new(config: VectorDbConfig) -> Result<Self> {
        let storage = Arc::new(Storage::new(&config.storage_path)?);

        let hnsw_config = HnswConfig {
            m: config.hnsw_m,
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
            metric: config.distance_metric,
            dimensions: config.dimensions,
        };

        let index = Arc::new(HnswIndex::new(hnsw_config));

        let stats = Arc::new(RwLock::new(VectorDbStats {
            total_vectors: 0,
            index_size_bytes: 0,
            storage_size_bytes: 0,
            avg_query_latency_us: 0.0,
            qps: 0.0,
        }));

        Ok(Self {
            config,
            storage,
            index,
            stats,
        })
    }

    /// Create a builder for configuring the database
    pub fn builder() -> VectorDbBuilder {
        VectorDbBuilder::default()
    }

    /// Insert a vector entry
    pub fn insert(&self, entry: VectorEntry) -> Result<String> {
        // Validate dimensions
        if entry.vector.len() != self.config.dimensions {
            return Err(VectorDbError::InvalidDimensions {
                expected: self.config.dimensions,
                actual: entry.vector.len(),
            });
        }

        // Store in storage layer
        self.storage.insert(&entry)?;

        // Insert into index
        self.index.insert(entry.id.clone(), entry.vector)?;

        // Update stats
        self.stats.write().total_vectors += 1;

        Ok(entry.id)
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(&self, entries: Vec<VectorEntry>) -> Result<Vec<String>> {
        // Validate all dimensions first
        for entry in &entries {
            if entry.vector.len() != self.config.dimensions {
                return Err(VectorDbError::InvalidDimensions {
                    expected: self.config.dimensions,
                    actual: entry.vector.len(),
                });
            }
        }

        // Store in storage layer
        self.storage.insert_batch(&entries)?;

        // Build index entries
        let index_entries: Vec<(String, Vec<f32>)> = entries
            .iter()
            .map(|e| (e.id.clone(), e.vector.clone()))
            .collect();

        // Insert into index
        self.index.insert_batch(index_entries)?;

        // Update stats
        self.stats.write().total_vectors += entries.len();

        Ok(entries.into_iter().map(|e| e.id).collect())
    }

    /// Search for similar vectors
    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let start = Instant::now();

        // Validate query vector dimensions
        if query.vector.len() != self.config.dimensions {
            return Err(VectorDbError::InvalidDimensions {
                expected: self.config.dimensions,
                actual: query.vector.len(),
            });
        }

        let user_k = query.k;
        let has_filters = query.filters.as_ref().map_or(false, |f| !f.is_empty());

        // Post-filter overfetch. When metadata filters are present, the
        // index returns top-K by raw distance and we trim to matching
        // metadata afterwards. With imbalanced bucket sizes (production
        // reality: a single dominant `node_type` swamps top-K, while small
        // buckets at 4-8 % of corpus get ~zero hits even for k=10), small
        // buckets return zero unless we expand the candidate pool. Match
        // the application-layer overfetch policy that downstream callers
        // (e.g. Atlas Data Fabric `filter_pushdown.go`) already use:
        // Factor=20, Min=500, Cap=2000. Without filters this is a no-op.
        const OVERFETCH_FACTOR: usize = 20;
        const OVERFETCH_MIN: usize = 500;
        const OVERFETCH_CAP: usize = 2000;
        let fetch_k = if has_filters {
            user_k
                .saturating_mul(OVERFETCH_FACTOR)
                .max(OVERFETCH_MIN)
                .min(OVERFETCH_CAP)
        } else {
            user_k
        };

        // Build a derived query that drives the index toward the larger
        // candidate pool. `ef_search` mirrors `k` so the inner search
        // visits enough nodes (the index respects `max(ef, k)` since the
        // companion fix in `index::HnswIndex::search`).
        let mut fetch_query = query.clone();
        fetch_query.k = fetch_k;
        if fetch_k > query.ef_search.unwrap_or(0) {
            fetch_query.ef_search = Some(fetch_k);
        }

        // Search index
        let mut results = self.index.search(&fetch_query)?;

        // Enrich results with metadata if needed
        for result in &mut results {
            if let Some(metadata) = self.storage.get_metadata(&result.id)? {
                result.metadata = metadata;
            }
        }

        // Apply metadata filters if specified
        if let Some(filters) = &query.filters {
            results.retain(|r| {
                filters
                    .iter()
                    .all(|(key, value)| r.metadata.get(key).map(|v| v == value).unwrap_or(false))
            });
        }

        // Brute-force fallback. Even with the overfetch above, post-filter
        // top-K cannot reliably surface vectors from very small buckets
        // (e.g. one-of-a-kind `node_type` values like a singleton schema
        // record). When the fast path under-delivers — i.e. we asked for
        // `user_k` matches but got fewer — fall back to a full scan over
        // the storage layer, filtering by metadata and ranking by distance
        // to the query. O(n) in total vectors but only fires for the
        // filtered path on small buckets, where it's both fast (matching
        // set is small) and necessary (HNSW post-filter can't find them).
        if has_filters && results.len() < user_k {
            let bf_results = self.brute_force_filtered_search(&query)?;
            // bf_results already ranked ascending by distance; replace.
            results = bf_results;
        }

        // Trim back to the caller's requested k after post-filter.
        results.truncate(user_k);

        // Update stats
        let latency_us = start.elapsed().as_micros() as f64;
        let mut stats = self.stats.write();
        stats.avg_query_latency_us = (stats.avg_query_latency_us * 0.9) + (latency_us * 0.1);

        Ok(results)
    }

    /// Exhaustive filtered search over the storage layer. Used as the
    /// guaranteed-correctness fallback when HNSW + post-filter under-
    /// delivers (typically: requested filter matches a very small bucket
    /// that the top-K HNSW pool cannot represent). Iterates every stored
    /// id, applies the metadata filter, computes distance to the query,
    /// returns the top-K by ascending distance.
    fn brute_force_filtered_search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let filters = match query.filters.as_ref() {
            Some(f) if !f.is_empty() => f,
            _ => return Ok(Vec::new()),
        };

        let mut hits: Vec<SearchResult> = Vec::new();
        for id in self.storage.get_all_ids()? {
            let metadata = match self.storage.get_metadata(&id)? {
                Some(m) => m,
                None => continue,
            };

            let matches = filters.iter().all(|(key, value)| {
                metadata.get(key).map(|v| v == value).unwrap_or(false)
            });
            if !matches {
                continue;
            }

            let vec = match self.storage.get(&id)? {
                Some(v) => v,
                None => continue,
            };

            let dist =
                calculate_distance(&query.vector, &vec, self.config.distance_metric)
                    .unwrap_or(f32::MAX);

            if let Some(threshold) = query.threshold {
                if dist > threshold {
                    continue;
                }
            }

            hits.push(SearchResult {
                id,
                score: dist,
                metadata,
                vector: None,
            });
        }

        hits.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(query.k);

        Ok(hits)
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let deleted = self.storage.delete(id)?;

        if deleted {
            self.index.remove(id)?;
            self.stats.write().total_vectors = self.stats.write().total_vectors.saturating_sub(1);
        }

        Ok(deleted)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        if let Some(vector) = self.storage.get(id)? {
            let metadata = self.storage.get_metadata(id)?.unwrap_or_default();

            Ok(Some(VectorEntry {
                id: id.to_string(),
                vector,
                metadata,
                timestamp: chrono::Utc::now().timestamp(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get database statistics
    pub fn stats(&self) -> VectorDbStats {
        self.stats.read().clone()
    }

    /// Get total number of vectors
    pub fn count(&self) -> Result<usize> {
        self.storage.count()
    }

    /// Get all vector IDs
    pub fn get_all_ids(&self) -> Result<Vec<String>> {
        self.storage.get_all_ids()
    }
}

/// Builder for VectorDB configuration
#[derive(Debug, Clone, Default)]
pub struct VectorDbBuilder {
    config: VectorDbConfig,
}

impl VectorDbBuilder {
    /// Set vector dimensions
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.config.dimensions = dimensions;
        self
    }

    /// Set maximum number of elements
    pub fn max_elements(mut self, max_elements: usize) -> Self {
        self.config.max_elements = max_elements;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Set HNSW M parameter
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.config.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction parameter
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.config.hnsw_ef_construction = ef;
        self
    }

    /// Set HNSW ef_search parameter
    pub fn hnsw_ef_search(mut self, ef: usize) -> Self {
        self.config.hnsw_ef_search = ef;
        self
    }

    /// Set quantization type
    pub fn quantization(mut self, qtype: QuantizationType) -> Self {
        self.config.quantization = qtype;
        self
    }

    /// Set storage path
    pub fn storage_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.storage_path = path.as_ref().to_string_lossy().to_string();
        self
    }

    /// Enable or disable memory mapping
    pub fn mmap_vectors(mut self, mmap: bool) -> Self {
        self.config.mmap_vectors = mmap;
        self
    }

    /// Build the VectorDB instance
    pub fn build(self) -> Result<VectorDB> {
        VectorDB::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_vector_db_basic_operations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let db = VectorDB::builder()
            .dimensions(3)
            .storage_path(&path)
            .build()
            .unwrap();

        // Insert
        let entry = VectorEntry {
            id: "test1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: std::collections::HashMap::new(),
            timestamp: 0,
        };

        let id = db.insert(entry).unwrap();
        assert_eq!(id, "test1");

        // Search
        let query = SearchQuery {
            vector: vec![0.9, 0.1, 0.0],
            k: 1,
            filters: None,
            threshold: None,
            ef_search: None,
        };

        let results = db.search(query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test1");

        // Delete
        assert!(db.delete("test1").unwrap());
        assert_eq!(db.count().unwrap(), 0);
    }
}
