use std::path::{Path, PathBuf};
use anyhow::Result;
use super::StorageBackend;

/// Local-filesystem storage backend.
///
/// All paths are resolved relative to `base`.
/// This is the only storage backend in Phase 1.
/// S3 sync will be added as a wrapper around this type.
pub struct LocalStorage {
    base: PathBuf,
}

impl LocalStorage {
    pub fn new(base: PathBuf) -> Self {
        Self { base }
    }
}

impl StorageBackend for LocalStorage {
    fn write_bytes(&self, relative_path: &str, data: &[u8]) -> Result<()> {
        let path = self.base.join(relative_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, data)?;
        Ok(())
    }

    fn read_bytes(&self, relative_path: &str) -> Result<Vec<u8>> {
        Ok(std::fs::read(self.base.join(relative_path))?)
    }

    fn exists(&self, relative_path: &str) -> bool {
        self.base.join(relative_path).exists()
    }

    fn local_path(&self) -> &Path {
        &self.base
    }
}
