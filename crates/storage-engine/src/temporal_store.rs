use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_stream_si128, _mm_set_epi64x, __m128i};

/// 🚀 ALGORITMO #89: TEMPORAL OBJECT STORE (SSD Picosecond Writer)
/// Almacenamiento ultra-masivo para Genomas de IA y Market Data usando
/// bloques pre-alocados (1GB) en Mmap para evitar latencia de Syscalls.
pub struct TemporalObjectStore {
    mmap: MmapMut,
    write_cursor: AtomicUsize,
    capacity: usize,
}

// Ensure thread safety manually since MmapMut is not Sync by default
unsafe impl Send for TemporalObjectStore {}
unsafe impl Sync for TemporalObjectStore {}

impl TemporalObjectStore {
    /// Pre-aloca un bloque gigante en el SSD (ej. 1GB = 1,073,741,824 bytes)
    pub fn new<P: AsRef<Path>>(path: P, prealloc_bytes: usize) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        // Pre-allocate space to avoid fragmentation and file extension latencies at runtime
        file.set_len(prealloc_bytes as u64)?;
        
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            mmap,
            write_cursor: AtomicUsize::new(0),
            capacity: prealloc_bytes,
        })
    }

    /// Escribe un bloque binario de 64 bytes directamente a la caché de memoria
    /// usando instrucciones Non-Temporal SIMD, evitando contaminar el Caché L1/L2
    /// del procesador, forzando al hardware a volcarlo al Mmap (SSD).
    #[inline(always)]
    pub fn write_tensor_block_64(&self, data: &[f64; 8]) -> Result<(), &'static str> {
        let current_offset = self.write_cursor.fetch_add(64, Ordering::AcqRel);
        
        if current_offset + 64 > self.capacity {
            // Buffer lleno (En un entorno real implementaríamos Rotación Log-Structured)
            return Err("TemporalObjectStore capacity exceeded");
        }

        unsafe {
            let base_ptr = self.mmap.as_ptr().add(current_offset);
            
            #[cfg(target_arch = "x86_64")]
            {
                let dest_ptr = base_ptr as *mut __m128i;
                let src_ptr = data.as_ptr();
                
                // Procesar 64 bytes (4 bloques de 128-bits)
                for i in 0..4 {
                    let chunk = _mm_set_epi64x(
                        *src_ptr.add(i * 2 + 1) as i64, 
                        *src_ptr.add(i * 2) as i64
                    );
                    // _mm_stream_si128: Escribe directo a memoria RAM (Mmap) bypasseando CPU Cache
                    _mm_stream_si128(dest_ptr.add(i), chunk);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // Fallback genérico para arquitecturas no-x86_64 (como ARM/M1)
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8, 
                    base_ptr as *mut u8, 
                    64
                );
            }
        }
        
        Ok(())
    }
}
