/// Storage abstraction for graph snapshot files.
///
/// Separates the *what* (read/write graph artifacts) from the *where*
/// (local disk today, S3 sync tomorrow).
///
/// ## What is covered
///
/// "Cold" snapshot files produced after each merge:
///   nodes.json, edges.bin, edge_types.json, edge_contexts.json,
///   edge_vectors.bin, edge_endpoints.json
///
/// ## What is NOT covered
///
/// - `delta.wal` — crash-recovery log; must stay on local disk with
///   append-only semantics and fsync guarantees.
/// - `vectors.bin` / `edge_vectors.bin` mmap reads — the OS memory-map
///   requires a local file descriptor; `local_path()` is provided so
///   callers that need direct file access can still get it.
///
/// ## Adding S3 sync (future)
///
/// Implement `S3SyncStorage` that wraps `LocalStorage`:
///   - `write_bytes` → write locally, then async-push to S3
///   - `read_bytes`  → read locally (never from S3 on hot path)
/// Cold-start restore: download snapshot from S3 → local before boot.

use std::path::Path;
use anyhow::Result;

pub mod local;
pub use local::LocalStorage;

pub trait StorageBackend: Send + Sync {
    fn write_bytes(&self, relative_path: &str, data: &[u8]) -> Result<()>;
    fn read_bytes(&self, relative_path: &str) -> Result<Vec<u8>>;
    fn exists(&self, relative_path: &str) -> bool;

    fn read_string(&self, relative_path: &str) -> Result<String> {
        let bytes = self.read_bytes(relative_path)?;
        String::from_utf8(bytes).map_err(|e| anyhow::anyhow!("invalid UTF-8 in {relative_path}: {e}"))
    }

    fn write_string(&self, relative_path: &str, s: &str) -> Result<()> {
        self.write_bytes(relative_path, s.as_bytes())
    }

    /// Base local filesystem path.
    ///
    /// Callers that need a real `Path` (mmap, `VectorStore::write`, WAL)
    /// use this to construct absolute paths directly.
    fn local_path(&self) -> &Path;
}
