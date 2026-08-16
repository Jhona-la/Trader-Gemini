use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::path::Path;

/// FASE 13: Zero-Copy Memory-Mapped Tensor Dumper
/// Permite volcar tensores de la red neuronal al Data Lakehouse en O(1) sin syscalls.
pub struct LakehouseMmap {
    mmap: Arc<MmapMut>,
    offset: AtomicUsize,
    capacity: usize,
}

// We need unsafe impls because MmapMut pointers are raw inside our structure, but we only mutate atomically.
// Note: MmapMut does not implement Clone, but we wrapped it in Arc. We cannot mutate it safely through Arc 
// without unsafe code, but we guarantee disjoint writes via atomic offset.
unsafe impl Send for LakehouseMmap {}
unsafe impl Sync for LakehouseMmap {}

impl LakehouseMmap {
    pub fn new<P: AsRef<Path>>(path: P, size_mb: usize) -> Result<Self, String> {
        let capacity = size_mb * 1024 * 1024;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("Failed to open mmap file: {}", e))?;

        file.set_len(capacity as u64)
            .map_err(|e| format!("Failed to set file size: {}", e))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| format!("Failed to mmap: {}", e))?
        };

        Ok(Self {
            mmap: Arc::new(mmap),
            offset: AtomicUsize::new(0),
            capacity,
        })
    }

    /// Escribe un tensor float al lakehouse crudo
    #[inline(always)]
    pub fn append_tensor(&self, timestamp: u64, features: &[f64], probabilities: &[f64]) -> Result<(), &'static str> {
        // Calculate needed bytes
        // 8 bytes (timestamp) + 4 bytes (features len) + 4 bytes (probs len) + arrays
        let needed = 8 + 4 + 4 + (features.len() * 8) + (probabilities.len() * 8);

        let current_offset = self.offset.fetch_add(needed, Ordering::SeqCst);
        if current_offset + needed > self.capacity {
            return Err("Lakehouse Mmap is full. Need rotation.");
        }

        // Get mutable reference to the slice in memory without locking (since offset is unique to this caller)
        let ptr = self.mmap.as_ptr() as *mut u8;
        let mut cursor = current_offset;

        unsafe {
            // Write timestamp
            std::ptr::copy_nonoverlapping(&timestamp as *const u64 as *const u8, ptr.add(cursor), 8);
            cursor += 8;

            // Write features len
            let f_len = features.len() as u32;
            std::ptr::copy_nonoverlapping(&f_len as *const u32 as *const u8, ptr.add(cursor), 4);
            cursor += 4;

            // Write features
            let f_bytes = features.len() * 8;
            std::ptr::copy_nonoverlapping(features.as_ptr() as *const u8, ptr.add(cursor), f_bytes);
            cursor += f_bytes;

            // Write probs len
            let p_len = probabilities.len() as u32;
            std::ptr::copy_nonoverlapping(&p_len as *const u32 as *const u8, ptr.add(cursor), 4);
            cursor += 4;

            // Write probs
            let p_bytes = probabilities.len() * 8;
            std::ptr::copy_nonoverlapping(probabilities.as_ptr() as *const u8, ptr.add(cursor), p_bytes);
        }

        Ok(())
    }
}
