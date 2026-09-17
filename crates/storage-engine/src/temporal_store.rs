use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128i, _mm_set_epi64x, _mm_stream_si128};

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
        // FIX #1436: Sanitización de finitud en el tensor antes de persistencia SIMD
        let mut safe_data = *data;
        for val in &mut safe_data {
            if !val.is_finite() {
                *val = 0.0;
            }
        }

        let usable_capacity = (self.capacity / 64) * 64;
        if usable_capacity == 0 {
            return Err("TemporalObjectStore capacity is zero");
        }
        // D-517: Rotación circular lock-free (Ring Wrap-Around) permanente
        // Evita agotar el buffer de 1GB y previene panics/errores en producción 24/7.
        let raw_offset = self.write_cursor.fetch_add(64, Ordering::AcqRel);
        let current_offset = raw_offset % usable_capacity;

        unsafe {
            // FIX #660: Puntero base a memoria mapeada y memory fence _mm_sfence()
            let base_ptr = (self.mmap.as_ptr() as *mut u8).add(current_offset);

            #[cfg(target_arch = "x86_64")]
            {
                let dest_ptr = base_ptr as *mut __m128i;
                let src_ptr = safe_data.as_ptr();

                // Procesar 64 bytes (4 bloques de 128-bits) preservando la representación IEEE-754 exacta
                for i in 0..4 {
                    let w1 = (*src_ptr.add(i * 2 + 1)).to_bits() as i64;
                    let w0 = (*src_ptr.add(i * 2)).to_bits() as i64;
                    let chunk = _mm_set_epi64x(w1, w0);
                    // _mm_stream_si128: Escribe directo a memoria RAM (Mmap) bypasseando CPU Cache
                    if (dest_ptr.add(i) as usize) % 16 == 0 {
                        _mm_stream_si128(dest_ptr.add(i), chunk);
                    } else {
                        // FIX #1533: Uso de safe_data sanitizado en rama no alineada
                        std::ptr::copy_nonoverlapping(
                            safe_data.as_ptr() as *const u8,
                            base_ptr,
                            64,
                        );
                        break;
                    }
                }
                std::arch::x86_64::_mm_sfence();
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // FIX #1533: Fallback genérico usando safe_data sanitizado
                std::ptr::copy_nonoverlapping(safe_data.as_ptr() as *const u8, base_ptr, 64);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_object_store_write_tensor() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!(
            "test_temporal_store_{}.dat",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let store = TemporalObjectStore::new(&path, 1024).expect("Failed to create temporal store");
        let tensor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(store.write_tensor_block_64(&tensor).is_ok());

        let nan_tensor = [f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(store.write_tensor_block_64(&nan_tensor).is_ok());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_temporal_object_store_capacity_overflow_boundary() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!(
            "test_temporal_overflow_{}.dat",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // 128 bytes allows exactly 2 blocks of 64 bytes
        let store = TemporalObjectStore::new(&path, 128).expect("Failed to create temporal store");
        let tensor = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        assert!(store.write_tensor_block_64(&tensor).is_ok());
        assert!(store.write_tensor_block_64(&tensor).is_ok());
        // 3rd block must fail with capacity exceeded
        assert!(store.write_tensor_block_64(&tensor).is_err());

        let _ = std::fs::remove_file(path);
    }
}
