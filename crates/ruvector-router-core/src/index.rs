//! HNSW index implementation

use crate::distance::calculate_distance;
use crate::error::{Result, VectorDbError};
use crate::types::{DistanceMetric, SearchQuery, SearchResult};
use parking_lot::RwLock;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

/// HNSW Index configuration
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// M parameter - number of connections per node
    pub m: usize,
    /// ef_construction - size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// ef_search - size of dynamic candidate list during search
    pub ef_search: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Number of dimensions
    pub dimensions: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimensions: 384,
        }
    }
}

#[derive(Clone)]
struct Neighbor {
    id: String,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        // Compare by distance AND id so two distinct nodes at the same
        // distance are not treated as equal (otherwise BinaryHeap can lose
        // distinct candidates when distances tie).
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural ordering by distance: greater distance is "greater".
        // BinaryHeap<Neighbor> is therefore a max-heap on distance — peek
        // returns the FURTHEST element, which is the correct invariant for
        // the `result` set during HNSW search (we evict the furthest when a
        // closer neighbour arrives).
        //
        // For the `candidates` set we want a min-heap (peek = closest), so
        // the search loop wraps Neighbor in `std::cmp::Reverse`.
        //
        // Tie-break by id keeps the comparator total — required because
        // `f32::partial_cmp` returns None for NaN (which can't reach here in
        // practice but the fallback keeps Ord consistent).
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Simplified HNSW index
pub struct HnswIndex {
    config: HnswConfig,
    vectors: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    graph: Arc<RwLock<HashMap<String, Vec<String>>>>,
    entry_point: Arc<RwLock<Option<String>>>,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            vectors: Arc::new(RwLock::new(HashMap::new())),
            graph: Arc::new(RwLock::new(HashMap::new())),
            entry_point: Arc::new(RwLock::new(None)),
        }
    }

    /// Insert a vector into the index
    pub fn insert(&self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::InvalidDimensions {
                expected: self.config.dimensions,
                actual: vector.len(),
            });
        }

        // Store vector
        self.vectors.write().insert(id.clone(), vector.clone());

        // Initialize graph connections and check if this is the first vector
        // IMPORTANT: Release all locks before calling search_knn_internal to avoid deadlock
        // (parking_lot::RwLock is NOT reentrant)
        let is_first = {
            let mut graph = self.graph.write();
            graph.insert(id.clone(), Vec::new());

            let mut entry_point = self.entry_point.write();
            if entry_point.is_none() {
                *entry_point = Some(id.clone());
                return Ok(());
            }
            false
        }; // All locks released here

        if is_first {
            return Ok(());
        }

        // Find nearest neighbors (safe now - no locks held).
        // Use the full ef_construction beam — a wider candidate pool gives
        // higher-quality outgoing edges and is essential for late-inserted
        // nodes to find each other rather than collapsing back to whatever
        // neighbours the (single, fixed) entry point can see. Previously this
        // was clamped to `min(ef_construction, m*2)`, which silently cut the
        // pool from 200 down to 32 with default config and was the principal
        // reason filter-search lost late-inserted node_types at scale.
        let beam = self.config.ef_construction.max(self.config.m);
        let neighbors = self.search_knn_internal(&vector, beam);

        // Re-acquire graph lock for modifications, plus a read lock on the
        // vector store so the pruning step can compute distances. The
        // newly-inserted vector is already in `self.vectors` (line above),
        // so distance computation against it is well-defined.
        //
        // Lock order matters: `search_knn_internal` acquires
        // `vectors.read() -> graph.read() -> entry_point.read()`. To avoid a
        // multi-thread deadlock under concurrent inserts (parking_lot is
        // writer-preferring, so a pending writer on `vectors` would block
        // new readers), we must take `vectors` BEFORE `graph` here too.
        // Acquiring `graph.write()` after `vectors.read()` is then safe:
        // graph and vectors are independent locks, and the order matches
        // the read-only path above.
        let vectors_for_prune = self.vectors.read();
        let mut graph = self.graph.write();

        // Connect to nearest neighbors (bidirectional)
        for neighbor in neighbors.iter().take(self.config.m) {
            if let Some(connections) = graph.get_mut(&id) {
                connections.push(neighbor.id.clone());
            }

            if let Some(neighbor_connections) = graph.get_mut(&neighbor.id) {
                neighbor_connections.push(id.clone());

                // Distance-based pruning: when a node's connection list grows
                // past 2M, keep the M neighbours that are CLOSEST to the
                // owning node. The previous implementation called
                // `truncate(m)` which is FIFO — it kept the OLDEST M edges.
                // At scale that sediments connectivity around the first-
                // inserted vectors and disconnects later clusters, so a
                // search starting from the (fixed) entry point can never
                // reach them. Distance-based pruning is the standard HNSW
                // heuristic-1 selection.
                if neighbor_connections.len() > self.config.m * 2 {
                    if let Some(neighbor_vec) = vectors_for_prune.get(&neighbor.id) {
                        let mut scored: Vec<(String, f32)> = neighbor_connections
                            .iter()
                            .filter_map(|cid| {
                                vectors_for_prune.get(cid).map(|cv| {
                                    let d = calculate_distance(neighbor_vec, cv, self.config.metric)
                                        .unwrap_or(f32::MAX);
                                    (cid.clone(), d)
                                })
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                        });
                        scored.truncate(self.config.m);
                        *neighbor_connections =
                            scored.into_iter().map(|(cid, _)| cid).collect();
                    } else {
                        // Owning vector unexpectedly absent — keep the M most
                        // recently added edges rather than the oldest, which
                        // is at worst no-worse than the previous FIFO
                        // behaviour.
                        let drop_count = neighbor_connections.len() - self.config.m;
                        neighbor_connections.drain(..drop_count);
                    }
                }
            }
        }

        Ok(())
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(&self, vectors: Vec<(String, Vec<f32>)>) -> Result<()> {
        for (id, vector) in vectors {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        // ef must be at least `k` — otherwise the inner search visits only
        // `ef` nodes and `take(query.k)` returns at most `ef` results,
        // capping recall regardless of how large the caller's `k` is. The
        // post-filter overfetch in `VectorDB::search` relies on driving the
        // candidate pool via large `k`, so this invariant is load-bearing.
        let ef_search = query
            .ef_search
            .unwrap_or(self.config.ef_search)
            .max(query.k);
        let candidates = self.search_knn_internal(&query.vector, ef_search);

        let mut results = Vec::new();
        for candidate in candidates.into_iter().take(query.k) {
            // Apply distance threshold if specified
            if let Some(threshold) = query.threshold {
                if candidate.distance > threshold {
                    continue;
                }
            }

            results.push(SearchResult {
                id: candidate.id,
                score: candidate.distance,
                metadata: HashMap::new(),
                vector: None,
            });
        }

        Ok(results)
    }

    /// Internal k-NN search implementation
    ///
    /// Maintains two heaps with deliberately opposite orientations:
    ///   * `candidates` — min-heap on distance (peek = closest unvisited).
    ///     Achieved with `BinaryHeap<Reverse<Neighbor>>` because Neighbor's
    ///     natural Ord is by-distance ascending (so unwrapping into a max-
    ///     heap and reversing yields a min-heap).
    ///   * `result`     — max-heap on distance (peek = furthest current
    ///     result). Plain `BinaryHeap<Neighbor>` gives this directly.
    ///
    /// The previous implementation used Neighbor's reversed Ord for both
    /// heaps, which made `result.peek()` return the CLOSEST result. The
    /// search-loop termination check then read "is the next candidate worse
    /// than my best result" (almost always yes once result was non-empty),
    /// terminating the search after visiting only a handful of nodes around
    /// the entry point. The replacement branch evicted the BEST result on
    /// every improvement instead of the worst, so the final set was a
    /// churned beam, not the top-ef closest. Both bugs are fixed by getting
    /// the heap orientations right.
    fn search_knn_internal(&self, query: &[f32], ef: usize) -> Vec<Neighbor> {
        let vectors = self.vectors.read();
        let graph = self.graph.read();
        let entry_point = self.entry_point.read();

        if entry_point.is_none() {
            return Vec::new();
        }

        let entry_id = entry_point.as_ref().unwrap();
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
        let mut result: BinaryHeap<Neighbor> = BinaryHeap::new();

        // Calculate distance to entry point
        if let Some(entry_vec) = vectors.get(entry_id) {
            let dist = calculate_distance(query, entry_vec, self.config.metric).unwrap_or(f32::MAX);

            let neighbor = Neighbor {
                id: entry_id.clone(),
                distance: dist,
            };

            candidates.push(Reverse(neighbor.clone()));
            result.push(neighbor);
            visited.insert(entry_id.clone());
        }

        // Search phase
        while let Some(Reverse(current)) = candidates.pop() {
            // Standard HNSW termination: stop when the closest unvisited
            // candidate is further than the FURTHEST current result AND we
            // have at least `ef` results already. `result.peek()` correctly
            // returns the furthest result because `result` is a max-heap.
            if let Some(furthest) = result.peek() {
                if current.distance > furthest.distance && result.len() >= ef {
                    break;
                }
            }

            // Explore neighbors
            if let Some(neighbors) = graph.get(&current.id) {
                for neighbor_id in neighbors {
                    if visited.contains(neighbor_id) {
                        continue;
                    }

                    visited.insert(neighbor_id.clone());

                    if let Some(neighbor_vec) = vectors.get(neighbor_id) {
                        let dist = calculate_distance(query, neighbor_vec, self.config.metric)
                            .unwrap_or(f32::MAX);

                        let neighbor = Neighbor {
                            id: neighbor_id.clone(),
                            distance: dist,
                        };

                        // Add to candidates queue regardless — the
                        // termination check above prunes its growth.
                        candidates.push(Reverse(neighbor.clone()));

                        // Maintain the top-ef closest results: if we have
                        // capacity, push; otherwise replace the current
                        // FURTHEST result if the new neighbour is closer.
                        if result.len() < ef {
                            result.push(neighbor);
                        } else if let Some(furthest) = result.peek() {
                            if dist < furthest.distance {
                                result.pop();
                                result.push(neighbor);
                            }
                        }
                    }
                }
            }
        }

        // Convert to ascending-by-distance vector (closest first).
        let mut sorted_results: Vec<Neighbor> = result.into_iter().collect();
        sorted_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        sorted_results
    }

    /// Remove a vector from the index
    pub fn remove(&self, id: &str) -> Result<bool> {
        let mut vectors = self.vectors.write();
        let mut graph = self.graph.write();

        if vectors.remove(id).is_none() {
            return Ok(false);
        }

        // Remove from graph
        graph.remove(id);

        // Remove references from other nodes
        for connections in graph.values_mut() {
            connections.retain(|conn_id| conn_id != id);
        }

        // Update entry point if needed
        let mut entry_point = self.entry_point.write();
        if entry_point.as_ref() == Some(&id.to_string()) {
            *entry_point = vectors.keys().next().cloned();
        }

        Ok(true)
    }

    /// Get total number of vectors in index
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_and_search() {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            dimensions: 3,
        };

        let index = HnswIndex::new(config);

        // Insert vectors
        index.insert("v1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert("v2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        index.insert("v3".to_string(), vec![0.0, 0.0, 1.0]).unwrap();

        // Search
        let query = SearchQuery {
            vector: vec![0.9, 0.1, 0.0],
            k: 2,
            filters: None,
            threshold: None,
            ef_search: None,
        };

        let results = index.search(&query).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1"); // Should be closest
    }

    #[test]
    fn test_hnsw_multiple_inserts_no_deadlock() {
        // Regression test for issue #133: VectorDb.insert() deadlocks on second call
        // The bug was caused by holding write locks while calling search_knn_internal,
        // which tries to acquire read locks on the same RwLocks (parking_lot is not reentrant)
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
            dimensions: 128,
        };

        let index = HnswIndex::new(config);

        // Insert many vectors to ensure we exercise the KNN search path
        for i in 0..20 {
            let mut vector = vec![0.0f32; 128];
            vector[i % 128] = 1.0;
            index.insert(format!("v{}", i), vector).unwrap();
        }

        assert_eq!(index.len(), 20);

        // Verify search still works
        let query = SearchQuery {
            vector: vec![1.0; 128],
            k: 5,
            filters: None,
            threshold: None,
            ef_search: None,
        };

        let results = index.search(&query).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_hnsw_concurrent_inserts() {
        use std::sync::Arc;
        use std::thread;

        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            dimensions: 3,
        };

        let index = Arc::new(HnswIndex::new(config));

        // Spawn multiple threads to insert concurrently
        let mut handles = vec![];
        for t in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let id = format!("t{}_v{}", t, i);
                    let vector = vec![t as f32, i as f32, 0.0];
                    index_clone.insert(id, vector).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(index.len(), 40);
    }

    /// Regression test for bug #97 (Atlas Data Fabric): filter-search returned
    /// zero rows for late-inserted node_type buckets at production scale even
    /// though the underlying vectors were persisted. Root cause was a triad
    /// of bugs in this file: the `result` BinaryHeap was inverted (peek
    /// returned the closest, not the furthest); `ef_construction` was clamped
    /// to `m*2` at insert time, starving the construction beam; and
    /// neighbour-list pruning was FIFO, so growing nodes lost their newer
    /// (closer-to-late-inserts) edges and the late-inserted clusters became
    /// unreachable from the fixed entry point.
    ///
    /// This test exercises recall@1 on a graph that's large enough to expose
    /// the original bugs (search would terminate inside a small region around
    /// the entry point). With the fix, every inserted vector must be its own
    /// nearest neighbour when queried.
    #[test]
    fn test_recall_at_1_with_biased_insertion_order() {
        // 64-dim is enough to give random vectors distinct distances; the
        // production failure is dimension-independent — the bug is in graph
        // construction and search termination, not in the metric.
        let dimensions = 64;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            metric: DistanceMetric::Cosine,
            dimensions,
        };

        let index = HnswIndex::new(config);

        // Deterministic pseudo-random vectors — no `rand` dep needed and the
        // test is reproducible. SplitMix64-style step gives well-spread bits.
        fn random_vec(seed: u64, dim: usize) -> Vec<f32> {
            let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            (0..dim)
                .map(|_| {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = s;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    // Map u64 → f32 in [-1, 1)
                    ((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0
                })
                .collect()
        }

        // Mirrors the production insertion ordering of bug #97:
        // a tiny "schema" bucket inserted first becomes the entry point, then
        // a moderate "table" bucket, then a large "column" bucket inserted
        // last — exactly the shape that triggered FIFO truncation to drop the
        // late-cluster edges.
        let buckets: &[(&str, usize)] = &[("schema", 1), ("table", 80), ("column", 1500)];
        let total: usize = buckets.iter().map(|(_, n)| *n).sum();

        let mut all_ids: Vec<String> = Vec::with_capacity(total);
        let mut all_vecs: Vec<Vec<f32>> = Vec::with_capacity(total);

        let mut idx: u64 = 0;
        for (label, count) in buckets {
            for _ in 0..*count {
                let id = format!("{}_{}", label, idx);
                let v = random_vec(idx, dimensions);
                index.insert(id.clone(), v.clone()).unwrap();
                all_ids.push(id);
                all_vecs.push(v);
                idx += 1;
            }
        }
        assert_eq!(index.len(), total);

        // Recall@1: querying with a vector EXACTLY equal to an inserted one
        // must return that vector as the top-1 result. On the broken impl,
        // late-inserted "column" vectors are unreachable from the entry
        // point so this assertion fires for any column index.
        // We sample 30 evenly spaced indices across the full range so any
        // bucket-specific failure surfaces.
        let probes = 30usize;
        let mut misses: Vec<String> = Vec::new();
        for i in 0..probes {
            let probe_idx = (i * (total - 1)) / (probes - 1);
            let q = SearchQuery {
                vector: all_vecs[probe_idx].clone(),
                k: 1,
                filters: None,
                threshold: None,
                ef_search: None,
            };
            let results = index.search(&q).unwrap();
            if results.is_empty() || results[0].id != all_ids[probe_idx] {
                misses.push(format!(
                    "probe={} expected={} got={}",
                    probe_idx,
                    all_ids[probe_idx],
                    results.first().map(|r| r.id.clone()).unwrap_or_default()
                ));
            }
        }
        assert!(
            misses.is_empty(),
            "recall@1 regression — {}/{} probes missed (bug #97 reproduces): {:?}",
            misses.len(),
            probes,
            misses
        );
    }

    /// Companion test: even when search is stricter (k=10, biased query), a
    /// random query must reach hits across ALL inserted clusters when ef is
    /// large enough. This is the unit-level analogue of the Go-side
    /// `TestUpsert_SearchableByMetadataFilter_LiveScale` — the Rust index is
    /// metadata-blind, so we approximate "hits in every bucket" by checking
    /// that the top-K result set spans at least two of the three label
    /// prefixes (the broken impl returns 10 hits all from the first two
    /// buckets).
    #[test]
    fn test_search_covers_all_clusters_at_scale() {
        let dimensions = 64;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            metric: DistanceMetric::Cosine,
            dimensions,
        };
        let index = HnswIndex::new(config);

        fn random_vec(seed: u64, dim: usize) -> Vec<f32> {
            let mut s = seed.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
            (0..dim)
                .map(|_| {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = s;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    ((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0
                })
                .collect()
        }

        let buckets: &[(&str, usize)] = &[("schema", 1), ("table", 80), ("column", 1500)];
        let mut idx: u64 = 0;
        for (label, count) in buckets {
            for _ in 0..*count {
                let id = format!("{}_{}", label, idx);
                index.insert(id, random_vec(idx, dimensions)).unwrap();
                idx += 1;
            }
        }

        // Random query at insertion-time-distant seed.
        let q = SearchQuery {
            vector: random_vec(99_999, dimensions),
            k: 50,
            filters: None,
            threshold: None,
            ef_search: None,
        };
        let results = index.search(&q).unwrap();
        assert_eq!(results.len(), 50, "search must return k results");

        let mut buckets_hit = HashSet::new();
        for r in &results {
            // Bucket name is the prefix before the first underscore.
            if let Some(prefix) = r.id.split('_').next() {
                buckets_hit.insert(prefix.to_string());
            }
        }
        assert!(
            buckets_hit.len() >= 2,
            "top-50 hits must span multiple buckets — broken HNSW only reaches the entry-point cluster. Hit: {:?}",
            buckets_hit
        );
    }
}
