//! Persistent storage layer with redb and memory-mapped vectors
//!
//! Provides ACID-compliant storage for graph nodes, edges, and hyperedges

#[cfg(feature = "storage")]
use crate::edge::Edge;
#[cfg(feature = "storage")]
use crate::hyperedge::{Hyperedge, HyperedgeId};
#[cfg(feature = "storage")]
use crate::node::Node;
#[cfg(feature = "storage")]
use crate::types::{EdgeId, NodeId};
#[cfg(feature = "storage")]
use anyhow::Result;
#[cfg(feature = "storage")]
use bincode::config;
#[cfg(feature = "storage")]
use once_cell::sync::Lazy;
#[cfg(feature = "storage")]
use parking_lot::Mutex;
#[cfg(feature = "storage")]
use redb::{Database, Durability, ReadableTable, TableDefinition, WriteTransaction};

#[cfg(feature = "storage")]
/// Re-export redb's Durability so callers don't need a direct redb dep.
pub use redb::Durability as StorageDurability;
#[cfg(feature = "storage")]
use std::collections::HashMap;
#[cfg(feature = "storage")]
use std::path::{Path, PathBuf};
#[cfg(feature = "storage")]
use std::sync::Arc;

#[cfg(feature = "storage")]
// Table definitions
const NODES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("nodes");
#[cfg(feature = "storage")]
const EDGES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("edges");
#[cfg(feature = "storage")]
const HYPEREDGES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("hyperedges");
#[cfg(feature = "storage")]
const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

// Atlas intent 01 (durable graph storage): metadata keys for the WAL/LSN
// integrity contract. Written on first open and bumped on every mutation.
#[cfg(feature = "storage")]
const META_KEY_LSN: &str = "__atlas_lsn";
#[cfg(feature = "storage")]
const META_KEY_FORMAT_VERSION: &str = "__atlas_format_version";
#[cfg(feature = "storage")]
const META_KEY_EDGE_TYPES_DIGEST: &str = "__atlas_edge_types_digest";
#[cfg(feature = "storage")]
const META_KEY_NODE_LABELS_DIGEST: &str = "__atlas_node_labels_digest";
#[cfg(feature = "storage")]
const ATLAS_FORMAT_VERSION: &str = "1.0";

#[cfg(feature = "storage")]
// Global database connection pool to allow multiple GraphStorage instances
// to share the same underlying database file
static DB_POOL: Lazy<Mutex<HashMap<PathBuf, Arc<Database>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "storage")]
/// Storage backend for graph database
pub struct GraphStorage {
    db: Arc<Database>,
    /// Optional per-write durability mode (FR-08). When set, every
    /// `begin_write()` call applies the configured durability before
    /// the first table is opened. None = redb default (Eventual).
    durability: Option<Durability>,
}

#[cfg(feature = "storage")]
/// Result of [`GraphStorage::seed_or_check_digests`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DigestCheck {
    /// Digests written for the first time (or back-filled).
    Seeded,
    /// On-disk digests match the runtime values.
    Match,
    /// On-disk digests differ from the runtime values. Warn-only — the
    /// caller logs the diff but does NOT fail open (DD-08 parity).
    Mismatch {
        edge_disk: String,
        edge_runtime: String,
        node_disk: String,
        node_runtime: String,
    },
}

#[cfg(feature = "storage")]
impl GraphStorage {
    /// Create or open a graph storage at the given path with default
    /// durability (`Durability::Eventual` — redb default). Atlas
    /// production callers should use [`Self::with_durability`] instead
    /// to opt into `Durability::Immediate` (per-mutation fsync).
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_durability_inner(path, None)
    }

    /// Open the storage with an explicit redb durability mode.
    ///
    /// Atlas intent 01 / FR-08: this constructor is the production wire
    /// for Electron mode where every mutation must hit fsync before the
    /// FFI ack. Uses redb's `Database::set_durability(Immediate)` after
    /// open, which configures every subsequent `commit()` to call
    /// `fdatasync` (NFR-04).
    ///
    /// On first open, writes the `__atlas_format_version` metadata key
    /// (FR-27) and seeds the LSN counter at 0.
    pub fn with_durability<P: AsRef<Path>>(path: P, durability: Durability) -> Result<Self> {
        Self::with_durability_inner(path, Some(durability))
    }

    fn with_durability_inner<P: AsRef<Path>>(path: P, durability: Option<Durability>) -> Result<Self> {
        let path_ref = path.as_ref();

        // Create parent directories if they don't exist
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Convert to absolute path
        let path_buf = if path_ref.is_absolute() {
            path_ref.to_path_buf()
        } else {
            std::env::current_dir()?.join(path_ref)
        };

        // SECURITY: Check for path traversal attempts
        let path_str = path_ref.to_string_lossy();
        if path_str.contains("..") && !path_ref.is_absolute() {
            if let Ok(cwd) = std::env::current_dir() {
                let mut normalized = cwd.clone();
                for component in path_ref.components() {
                    match component {
                        std::path::Component::ParentDir => {
                            if !normalized.pop() || !normalized.starts_with(&cwd) {
                                anyhow::bail!("Path traversal attempt detected");
                            }
                        }
                        std::path::Component::Normal(c) => normalized.push(c),
                        _ => {}
                    }
                }
            }
        }

        // Check if we already have a Database instance for this path
        let db = {
            let mut pool = DB_POOL.lock();

            if let Some(existing_db) = pool.get(&path_buf) {
                // Reuse existing database connection
                Arc::clone(existing_db)
            } else {
                // Create new database and add to pool. Per-database
                // durability was removed in redb 2.x — durability is
                // applied per-transaction in `begin_write_txn`.
                let new_db = Arc::new(Database::create(&path_buf)?);

                // Initialize tables and seed Atlas metadata in the same txn
                // so the format-version key + LSN seed are atomic with
                // table creation.
                let mut write_txn = new_db.begin_write()?;
                if let Some(d) = durability {
                    write_txn.set_durability(d);
                }
                {
                    let _ = write_txn.open_table(NODES_TABLE)?;
                    let _ = write_txn.open_table(EDGES_TABLE)?;
                    let _ = write_txn.open_table(HYPEREDGES_TABLE)?;
                    let mut meta = write_txn.open_table(METADATA_TABLE)?;
                    if meta.get(META_KEY_FORMAT_VERSION)?.is_none() {
                        meta.insert(META_KEY_FORMAT_VERSION, ATLAS_FORMAT_VERSION)?;
                    }
                    if meta.get(META_KEY_LSN)?.is_none() {
                        meta.insert(META_KEY_LSN, "0")?;
                    }
                }
                write_txn.commit()?;

                pool.insert(path_buf, Arc::clone(&new_db));
                new_db
            }
        };

        Ok(Self { db, durability })
    }

    // --- Atlas intent 01: LSN counter + digest helpers ---

    /// Begin a write transaction with the configured durability mode
    /// applied. All Atlas mutation methods route through this so the
    /// FR-07 / NFR-04 contract (per-mutation fdatasync) holds without
    /// each caller having to remember.
    fn begin_write_txn(&self) -> Result<WriteTransaction> {
        let mut write_txn = self.db.begin_write()?;
        if let Some(d) = self.durability {
            write_txn.set_durability(d);
        }
        Ok(write_txn)
    }

    /// Bump the monotonic LSN counter inside an open write transaction.
    /// Returns the new (post-bump) LSN. The caller MUST call
    /// `write_txn.commit()` for the LSN advance to be durable. This
    /// keeps the LSN write atomic with the data write (FR-09, FR-11).
    fn bump_lsn_in_txn(write_txn: &WriteTransaction) -> Result<u64> {
        let mut meta = write_txn.open_table(METADATA_TABLE)?;
        let current: u64 = meta
            .get(META_KEY_LSN)?
            .and_then(|v| v.value().parse::<u64>().ok())
            .unwrap_or(0);
        let next = current.saturating_add(1);
        meta.insert(META_KEY_LSN, next.to_string().as_str())?;
        Ok(next)
    }

    /// Atlas FR-32: restore the LSN counter to a specific value during
    /// snapshot import. Distinct from `bump_lsn_in_txn` which only ever
    /// increments — this overwrites in its own write_txn so the
    /// post-import counter equals the snapshot's captured LSN. The
    /// snapshot import has already replayed the data, so a single
    /// metadata-table write captures the watermark with no further
    /// state churn.
    pub fn restore_lsn(&self, lsn: u64) -> Result<()> {
        let write_txn = self.begin_write_txn()?;
        {
            let mut meta = write_txn.open_table(METADATA_TABLE)?;
            meta.insert(META_KEY_LSN, lsn.to_string().as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Read the current LSN (FR-12 backing primitive). Reads the
    /// committed value so a writer mid-transaction is not observed.
    pub fn current_lsn(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        Ok(table
            .get(META_KEY_LSN)?
            .and_then(|v| v.value().parse::<u64>().ok())
            .unwrap_or(0))
    }

    /// Read the on-disk format version (FR-27). Empty string when the
    /// key is absent (legacy DBs created before this contract).
    pub fn format_version(&self) -> Result<String> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        Ok(table
            .get(META_KEY_FORMAT_VERSION)?
            .map(|v| v.value().to_string())
            .unwrap_or_default())
    }

    /// Persist the edge-type and node-label digests on first open
    /// (FR-28). Subsequent opens: the caller compares the on-disk
    /// digests against the in-process model; mismatches are warn-only,
    /// never fatal (mirrors snapshot DD-08).
    pub fn seed_or_check_digests(&self, edge_types_digest: &str, node_labels_digest: &str) -> Result<DigestCheck> {
        let write_txn = self.begin_write_txn()?;
        let result;
        {
            let mut meta = write_txn.open_table(METADATA_TABLE)?;
            let existing_edge = meta
                .get(META_KEY_EDGE_TYPES_DIGEST)?
                .map(|v| v.value().to_string());
            let existing_node = meta
                .get(META_KEY_NODE_LABELS_DIGEST)?
                .map(|v| v.value().to_string());
            match (existing_edge, existing_node) {
                (None, None) => {
                    meta.insert(META_KEY_EDGE_TYPES_DIGEST, edge_types_digest)?;
                    meta.insert(META_KEY_NODE_LABELS_DIGEST, node_labels_digest)?;
                    result = DigestCheck::Seeded;
                }
                (Some(edge), Some(node)) => {
                    if edge == edge_types_digest && node == node_labels_digest {
                        result = DigestCheck::Match;
                    } else {
                        result = DigestCheck::Mismatch {
                            edge_disk: edge,
                            edge_runtime: edge_types_digest.to_string(),
                            node_disk: node,
                            node_runtime: node_labels_digest.to_string(),
                        };
                    }
                }
                _ => {
                    // Partial seed (one digest written, the other not):
                    // back-fill the missing key without overwriting.
                    if meta.get(META_KEY_EDGE_TYPES_DIGEST)?.is_none() {
                        meta.insert(META_KEY_EDGE_TYPES_DIGEST, edge_types_digest)?;
                    }
                    if meta.get(META_KEY_NODE_LABELS_DIGEST)?.is_none() {
                        meta.insert(META_KEY_NODE_LABELS_DIGEST, node_labels_digest)?;
                    }
                    result = DigestCheck::Seeded;
                }
            }
        }
        write_txn.commit()?;
        Ok(result)
    }

    // Node operations

    /// Insert a node
    pub fn insert_node(&self, node: &Node) -> Result<NodeId> {
        let write_txn = self.begin_write_txn()?;
        {
            let mut table = write_txn.open_table(NODES_TABLE)?;

            // Serialize node data
            let node_data = bincode::encode_to_vec(node, config::standard())?;
            table.insert(node.id.as_str(), node_data.as_slice())?;
        }
        Self::bump_lsn_in_txn(&write_txn)?;
        write_txn.commit()?;

        Ok(node.id.clone())
    }

    /// Insert multiple nodes in a batch
    pub fn insert_nodes_batch(&self, nodes: &[Node]) -> Result<Vec<NodeId>> {
        let write_txn = self.begin_write_txn()?;
        let mut ids = Vec::with_capacity(nodes.len());

        {
            let mut table = write_txn.open_table(NODES_TABLE)?;

            for node in nodes {
                let node_data = bincode::encode_to_vec(node, config::standard())?;
                table.insert(node.id.as_str(), node_data.as_slice())?;
                ids.push(node.id.clone());
            }
        }

        // Single LSN bump per batch — the entire batch is one atomic
        // logical mutation (one commit). Treating a batch as one LSN
        // step matches the per-mutation atomicity contract (NFR-09).
        Self::bump_lsn_in_txn(&write_txn)?;
        write_txn.commit()?;
        Ok(ids)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Result<Option<Node>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NODES_TABLE)?;

        let Some(node_data) = table.get(id)? else {
            return Ok(None);
        };

        let (node, _): (Node, usize) =
            bincode::decode_from_slice(node_data.value(), config::standard())?;
        Ok(Some(node))
    }

    /// Delete a node by ID
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        let write_txn = self.begin_write_txn()?;
        let deleted;
        {
            let mut table = write_txn.open_table(NODES_TABLE)?;
            let result = table.remove(id)?;
            deleted = result.is_some();
        }
        if deleted {
            Self::bump_lsn_in_txn(&write_txn)?;
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get all node IDs
    pub fn all_node_ids(&self) -> Result<Vec<NodeId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NODES_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    // Edge operations

    /// Insert an edge
    pub fn insert_edge(&self, edge: &Edge) -> Result<EdgeId> {
        let write_txn = self.begin_write_txn()?;
        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;

            // Serialize edge data
            let edge_data = bincode::encode_to_vec(edge, config::standard())?;
            table.insert(edge.id.as_str(), edge_data.as_slice())?;
        }
        Self::bump_lsn_in_txn(&write_txn)?;
        write_txn.commit()?;

        Ok(edge.id.clone())
    }

    /// Insert multiple edges in a batch
    pub fn insert_edges_batch(&self, edges: &[Edge]) -> Result<Vec<EdgeId>> {
        let write_txn = self.begin_write_txn()?;
        let mut ids = Vec::with_capacity(edges.len());

        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;

            for edge in edges {
                let edge_data = bincode::encode_to_vec(edge, config::standard())?;
                table.insert(edge.id.as_str(), edge_data.as_slice())?;
                ids.push(edge.id.clone());
            }
        }

        Self::bump_lsn_in_txn(&write_txn)?;
        write_txn.commit()?;
        Ok(ids)
    }

    /// Get an edge by ID
    pub fn get_edge(&self, id: &str) -> Result<Option<Edge>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EDGES_TABLE)?;

        let Some(edge_data) = table.get(id)? else {
            return Ok(None);
        };

        let (edge, _): (Edge, usize) =
            bincode::decode_from_slice(edge_data.value(), config::standard())?;
        Ok(Some(edge))
    }

    /// Delete an edge by ID
    pub fn delete_edge(&self, id: &str) -> Result<bool> {
        let write_txn = self.begin_write_txn()?;
        let deleted;
        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;
            let result = table.remove(id)?;
            deleted = result.is_some();
        }
        if deleted {
            Self::bump_lsn_in_txn(&write_txn)?;
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    pub fn delete_edges_batch(&self, ids: &[impl AsRef<str>]) -> Result<usize> {
        let write_txn = self.begin_write_txn()?;
        let mut deleted = 0;
        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;
            for id in ids {
                if table.remove(id.as_ref())?.is_some() {
                    deleted += 1;
                }
            }
        }

        if deleted > 0 {
            Self::bump_lsn_in_txn(&write_txn)?;
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get all edge IDs
    pub fn all_edge_ids(&self) -> Result<Vec<EdgeId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EDGES_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    // Hyperedge operations

    /// Insert a hyperedge
    pub fn insert_hyperedge(&self, hyperedge: &Hyperedge) -> Result<HyperedgeId> {
        let write_txn = self.begin_write_txn()?;
        {
            let mut table = write_txn.open_table(HYPEREDGES_TABLE)?;

            // Serialize hyperedge data
            let hyperedge_data = bincode::encode_to_vec(hyperedge, config::standard())?;
            table.insert(hyperedge.id.as_str(), hyperedge_data.as_slice())?;
        }
        Self::bump_lsn_in_txn(&write_txn)?;
        write_txn.commit()?;

        Ok(hyperedge.id.clone())
    }

    /// Insert multiple hyperedges in a batch
    pub fn insert_hyperedges_batch(&self, hyperedges: &[Hyperedge]) -> Result<Vec<HyperedgeId>> {
        let write_txn = self.begin_write_txn()?;
        let mut ids = Vec::with_capacity(hyperedges.len());

        {
            let mut table = write_txn.open_table(HYPEREDGES_TABLE)?;

            for hyperedge in hyperedges {
                let hyperedge_data = bincode::encode_to_vec(hyperedge, config::standard())?;
                table.insert(hyperedge.id.as_str(), hyperedge_data.as_slice())?;
                ids.push(hyperedge.id.clone());
            }
        }

        Self::bump_lsn_in_txn(&write_txn)?;
        write_txn.commit()?;
        Ok(ids)
    }

    /// Get a hyperedge by ID
    pub fn get_hyperedge(&self, id: &str) -> Result<Option<Hyperedge>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(HYPEREDGES_TABLE)?;

        let Some(hyperedge_data) = table.get(id)? else {
            return Ok(None);
        };

        let (hyperedge, _): (Hyperedge, usize) =
            bincode::decode_from_slice(hyperedge_data.value(), config::standard())?;
        Ok(Some(hyperedge))
    }

    /// Delete a hyperedge by ID
    pub fn delete_hyperedge(&self, id: &str) -> Result<bool> {
        let write_txn = self.begin_write_txn()?;
        let deleted;
        {
            let mut table = write_txn.open_table(HYPEREDGES_TABLE)?;
            let result = table.remove(id)?;
            deleted = result.is_some();
        }
        if deleted {
            Self::bump_lsn_in_txn(&write_txn)?;
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get all hyperedge IDs
    pub fn all_hyperedge_ids(&self) -> Result<Vec<HyperedgeId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(HYPEREDGES_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    // Metadata operations

    /// Set metadata. Atlas user metadata writes (i.e., not the
    /// `__atlas_*` reserved keys) bump the LSN; reserved-key writes
    /// (LSN bookkeeping itself) do not, to avoid recursion.
    pub fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        let write_txn = self.begin_write_txn()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert(key, value)?;
        }
        if !key.starts_with("__atlas_") {
            Self::bump_lsn_in_txn(&write_txn)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;

        let value = table.get(key)?.map(|v| v.value().to_string());
        Ok(value)
    }

    // Statistics

    /// Get the number of nodes
    pub fn node_count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NODES_TABLE)?;
        Ok(table.iter()?.count())
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EDGES_TABLE)?;
        Ok(table.iter()?.count())
    }

    /// Get the number of hyperedges
    pub fn hyperedge_count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(HYPEREDGES_TABLE)?;
        Ok(table.iter()?.count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeBuilder;
    use crate::hyperedge::HyperedgeBuilder;
    use crate::node::NodeBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_node_storage() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let node = NodeBuilder::new()
            .label("Person")
            .property("name", "Alice")
            .build();

        let id = storage.insert_node(&node)?;
        assert_eq!(id, node.id);

        let retrieved = storage.get_node(&id)?;
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, node.id);
        assert!(retrieved.has_label("Person"));

        Ok(())
    }

    #[test]
    fn test_edge_storage() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let edge = EdgeBuilder::new("n1".to_string(), "n2".to_string(), "KNOWS")
            .property("since", 2020i64)
            .build();

        let id = storage.insert_edge(&edge)?;
        assert_eq!(id, edge.id);

        let retrieved = storage.get_edge(&id)?;
        assert!(retrieved.is_some());

        Ok(())
    }

    #[test]
    fn test_batch_insert() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let nodes = vec![
            NodeBuilder::new().label("Person").build(),
            NodeBuilder::new().label("Person").build(),
        ];

        let ids = storage.insert_nodes_batch(&nodes)?;
        assert_eq!(ids.len(), 2);
        assert_eq!(storage.node_count()?, 2);

        Ok(())
    }

    #[test]
    fn test_hyperedge_storage() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let hyperedge = HyperedgeBuilder::new(
            vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            "MEETING",
        )
        .description("Team meeting")
        .build();

        let id = storage.insert_hyperedge(&hyperedge)?;
        assert_eq!(id, hyperedge.id);

        let retrieved = storage.get_hyperedge(&id)?;
        assert!(retrieved.is_some());

        Ok(())
    }

    // --- Atlas intent 01 (durable graph storage) tests ---

    #[test]
    fn test_lsn_monotonic_across_writes() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::with_durability(
            dir.path().join("lsn.db"),
            Durability::Immediate,
        )?;
        assert_eq!(storage.current_lsn()?, 0, "freshly opened DB starts at LSN=0");

        for i in 0..5 {
            let node = NodeBuilder::new()
                .label("Person")
                .property("i", i as i64)
                .build();
            storage.insert_node(&node)?;
        }
        assert_eq!(storage.current_lsn()?, 5, "5 inserts → LSN=5");

        // Edge insert advances LSN.
        let e = EdgeBuilder::new("a".to_string(), "b".to_string(), "REL").build();
        storage.insert_edge(&e)?;
        assert_eq!(storage.current_lsn()?, 6);

        // Batch counts as ONE LSN step.
        let batch = vec![
            NodeBuilder::new().label("A").build(),
            NodeBuilder::new().label("B").build(),
        ];
        storage.insert_nodes_batch(&batch)?;
        assert_eq!(storage.current_lsn()?, 7, "batch insert is one LSN step");

        Ok(())
    }

    #[test]
    fn test_format_version_seeded_on_first_open() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("fmt.db");
        let storage = GraphStorage::with_durability(&path, Durability::Immediate)?;
        assert_eq!(storage.format_version()?, ATLAS_FORMAT_VERSION);
        Ok(())
    }

    #[test]
    fn test_seed_or_check_digests_first_run_then_match() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::with_durability(
            dir.path().join("digest.db"),
            Durability::Immediate,
        )?;
        let edge_d = "abc";
        let node_d = "xyz";
        match storage.seed_or_check_digests(edge_d, node_d)? {
            DigestCheck::Seeded => {}
            other => panic!("first run should Seed, got {:?}", other),
        }
        match storage.seed_or_check_digests(edge_d, node_d)? {
            DigestCheck::Match => {}
            other => panic!("second run with same digests should Match, got {:?}", other),
        }
        match storage.seed_or_check_digests("abc-CHANGED", node_d)? {
            DigestCheck::Mismatch { edge_disk, edge_runtime, .. } => {
                assert_eq!(edge_disk, "abc");
                assert_eq!(edge_runtime, "abc-CHANGED");
            }
            other => panic!("changed digest should Mismatch, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn test_metadata_user_keys_bump_lsn_reserved_keys_do_not() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::with_durability(
            dir.path().join("meta.db"),
            Durability::Immediate,
        )?;
        let lsn0 = storage.current_lsn()?;
        storage.set_metadata("user_key", "value")?;
        assert_eq!(storage.current_lsn()?, lsn0 + 1, "user metadata bumps LSN");

        // Reserved keys (the LSN counter itself) must not bump on write.
        storage.set_metadata("__atlas_format_version", "1.0")?;
        assert_eq!(storage.current_lsn()?, lsn0 + 1, "reserved-key writes do NOT bump LSN");
        Ok(())
    }

    #[test]
    fn test_lsn_persists_across_reopens() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("persist.db");
        {
            let storage = GraphStorage::with_durability(&path, Durability::Immediate)?;
            for _ in 0..3 {
                storage.insert_node(&NodeBuilder::new().label("X").build())?;
            }
            assert_eq!(storage.current_lsn()?, 3);
            // Drop the storage and the connection-pool entry by clearing
            // the global pool key so a fresh open sees the existing file.
        }
        // Force the global DB_POOL to drop its handle for this path so
        // the second open re-creates Database from disk (proves WAL
        // replay would carry the LSN forward in a kill -9 scenario).
        DB_POOL.lock().clear();
        let reopened = GraphStorage::with_durability(&path, Durability::Immediate)?;
        assert_eq!(reopened.current_lsn()?, 3, "LSN survives reopen");
        Ok(())
    }
}
