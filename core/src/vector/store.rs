use std::path::Path;
use memmap2::Mmap;
use anyhow::Result;
use bytemuck;

/// Memory-mapped vector store.
///
/// Binary layout of vectors.bin:
///   [header: dim(u32) + num_vecs(u32)] + [vec_0: f32*dim] + [vec_1: f32*dim] + ...
///
/// Total size = 8 + num_vecs * dim * 4 bytes
///
/// mmap lets the OS manage which pages stay in RAM.
/// Reading a vector = zero-copy pointer arithmetic, no syscall.
pub struct VectorStore {
    mmap: Mmap,
    pub dim: usize,
    pub num_vecs: usize,
}

impl VectorStore {
    const HEADER_BYTES: usize = 8; // dim(u32) + num_vecs(u32)

    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < Self::HEADER_BYTES {
            anyhow::bail!("vectors.bin too small — missing header");
        }

        let dim = u32::from_le_bytes(mmap[0..4].try_into()?) as usize;
        let num_vecs = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;

        let expected = Self::HEADER_BYTES + num_vecs * dim * 4;
        if mmap.len() < expected {
            anyhow::bail!(
                "vectors.bin size mismatch: expected {} bytes, got {}",
                expected, mmap.len()
            );
        }

        tracing::info!("opened vector store: {} vecs × {} dim", num_vecs, dim);
        Ok(Self { mmap, dim, num_vecs })
    }

    /// Zero-copy: returns a reference into the mmap'd region.
    /// No allocation. Lifetime tied to `self`.
    #[inline]
    pub fn get(&self, id: u32) -> &[f32] {
        let offset = Self::HEADER_BYTES + id as usize * self.dim * 4;
        let bytes = &self.mmap[offset..offset + self.dim * 4];
        bytemuck::cast_slice(bytes)
    }

    /// Build vectors.bin from a Vec of embeddings.
    pub fn write(path: &Path, dim: usize, vectors: &[Vec<f32>]) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;

        // header
        if vectors.len() > u32::MAX as usize {
            anyhow::bail!("too many vectors: {} exceeds u32::MAX", vectors.len());
        }
        f.write_all(&(dim as u32).to_le_bytes())?;
        f.write_all(&(vectors.len() as u32).to_le_bytes())?;

        // vectors
        for vec in vectors {
            assert_eq!(vec.len(), dim, "dimension mismatch in vector store write");
            let bytes = bytemuck::cast_slice::<f32, u8>(vec);
            f.write_all(bytes)?;
        }

        tracing::info!("wrote {} vectors to {}", vectors.len(), path.display());
        Ok(())
    }
}

/// Cosine similarity between two equal-length slices.
/// Assumes vectors are pre-normalised (unit length) — then dot product == cosine sim.
///
/// On aarch64 (Apple Silicon / ARM): explicit NEON with 4-way unrolled FMA.
/// Processes 16 f32 per iteration (4 × float32x4_t), then reduces horizontally.
/// Falls back to auto-vectorised iterator on all other targets.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        cosine_sim_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(target_arch = "aarch64")]
fn cosine_sim_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len().min(b.len());
    let chunks16 = n / 16;
    let remainder = n % 16;

    unsafe {
        // 4 independent accumulators — lets the CPU pipeline FMA units in parallel.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut ap = a.as_ptr();
        let mut bp = b.as_ptr();

        for _ in 0..chunks16 {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap),      vld1q_f32(bp));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(4)),  vld1q_f32(bp.add(4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(8)),  vld1q_f32(bp.add(8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(12)), vld1q_f32(bp.add(12)));
            ap = ap.add(16);
            bp = bp.add(16);
        }

        // tree-reduce the 4 accumulators, then horizontal-add to scalar
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(acc0);

        // scalar tail (< 16 elements)
        let base = chunks16 * 16;
        for i in 0..remainder {
            sum += *a.get_unchecked(base + i) * *b.get_unchecked(base + i);
        }

        sum
    }
}

/// Brute-force top-k search. Used in Phase 1 / testing.
/// Replace with HNSW for production.
pub fn brute_search(store: &VectorStore, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scores: Vec<(u32, f32)> = (0..store.num_vecs as u32)
        .map(|id| (id, cosine_sim(store.get(id), query)))
        .collect();

    // partial sort — only need top-k
    let n = k.min(scores.len());
    scores.select_nth_unstable_by(n - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}
