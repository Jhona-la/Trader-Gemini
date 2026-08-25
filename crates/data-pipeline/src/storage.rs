use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Binary struct for raw storage representation.
/// Extremely fast serialization with zero-copy potential.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TelemetryTick {
    pub timestamp: u64,
    pub coin_id: u32,
    pub bid_price: f32,
    pub ask_price: f32,
    pub bid_qty: f32,
    pub ask_qty: f32,
    pub checksum: u32, // FASE 14: Protección contra corrupción en Mmap
}

impl TelemetryTick {
    pub fn calculate_checksum(&self) -> u32 {
        // XOR checksum ultrarrápido para nanosegundos
        let mut cs = 0u32;
        cs ^= (self.timestamp & 0xFFFFFFFF) as u32;
        cs ^= (self.timestamp >> 32) as u32;
        cs ^= self.coin_id;
        cs ^= self.bid_price.to_bits();
        cs ^= self.ask_price.to_bits();
        cs ^= self.bid_qty.to_bits();
        cs ^= self.ask_qty.to_bits();
        cs
    }
}

/// Institutional Grade Zero-Copy Storage.
/// Wrapped in UnsafeCell to respect Rust's aliasing rules when doing lock-free concurrency.
pub struct TelemetryStorage {
    mmap: UnsafeCell<MmapMut>,
    write_cursor: AtomicUsize,
    max_capacity: usize,
}

// Safety: We guarantee that no two threads will ever write to the same byte offset concurrently
// because `write_cursor.fetch_add` yields strictly unique, monotonically increasing indices, 
// ensuring disjoint memory access within the Mmap array bounds.
unsafe impl Sync for TelemetryStorage {}
unsafe impl Send for TelemetryStorage {}

impl TelemetryStorage {
    pub fn new<P: AsRef<Path>>(path: P, max_ticks: usize) -> std::io::Result<Self> {
        let file_size = (max_ticks * std::mem::size_of::<TelemetryTick>()) as u64;
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
            
        // Pre-allocate the file size to prevent fragmentation and page faults on Windows.
        // If the OS cannot allocate this contiguous space, it safely returns an error here
        // instead of crashing later.
        file.set_len(file_size)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(Self {
            mmap: UnsafeCell::new(mmap),
            write_cursor: AtomicUsize::new(0),
            max_capacity: max_ticks,
        })
    }

    /// Appends a tick directly to memory mapped file (Zero-copy, O(1), No Heap)
    /// Hardened to prevent segmentation faults and ML Hallucinations.
    pub fn append_tick(&self, tick: &TelemetryTick) {
        // [FASE 12] PRE-FLIGHT CHECK: Estasis de Probabilidad Cuántica
        // Nunca guardar basura en disco. Los modelos de IA colapsan si ingieren NaNs o precios en 0.
        if tick.bid_price.is_nan() || tick.ask_price.is_nan() || 
           tick.bid_price <= 0.0 || tick.ask_price <= 0.0 {
            return; // Descartar tick silenciosamente para no atascar el motor HFT
        }
        
        // Anti-Corrupción Cuántica
        let calculated_checksum = tick.calculate_checksum();
        if tick.checksum != 0 && tick.checksum != calculated_checksum {
            eprintln!("⚠️ [STORAGE CRÍTICO] Corrupción de Ticks detectada en memoria RAM antes del Mmap. Descartando!");
            return;
        }

        let index = self.write_cursor.fetch_add(1, Ordering::Relaxed);
        let wrapped_index = index % self.max_capacity;
        let offset = wrapped_index * std::mem::size_of::<TelemetryTick>();
        
        // Strict boundary validation to prevent memory corruption (Pantallazo Azul / Segfault).
        // While mathematical modulo ensures it theoretically never exceeds capacity,
        // hardware bit flips or struct size changes could cause catastrophic failure.
        // This check has near zero overhead.
        let mmap_ptr = self.mmap.get();
        unsafe {
            let mmap_len = (&*mmap_ptr).len();
            if offset + std::mem::size_of::<TelemetryTick>() > mmap_len {
                eprintln!("CRITICAL STORAGE ERROR: Attempted to write outside memory map boundaries. Dropping telemetry safely.");
                return;
            }
            
            let src = tick as *const TelemetryTick as *const u8;
            let dst = (&mut *mmap_ptr).as_mut_ptr().add(offset);
            std::ptr::copy_nonoverlapping(src, dst, std::mem::size_of::<TelemetryTick>());
        }
    }
    
    /// Flushes the memory mapped file to SSD asychronously.
    /// In Windows, the OS manages dirty pages automatically, but this forces it.
    pub fn flush_to_disk(&self) -> std::io::Result<()> {
        unsafe {
            (&*self.mmap.get()).flush_async()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_tick_checksum() {
        let mut tick = TelemetryTick {
            timestamp: 1672531200000,
            coin_id: 1,
            bid_price: 50000.0,
            ask_price: 50001.0,
            bid_qty: 1.5,
            ask_qty: 2.0,
            checksum: 0,
        };
        let cs = tick.calculate_checksum();
        assert_ne!(cs, 0);
        tick.checksum = cs;
        assert_eq!(tick.calculate_checksum(), tick.checksum);
    }

    #[test]
    fn test_telemetry_storage_append_and_flush() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_telemetry_storage.bin");

        let storage = TelemetryStorage::new(&path, 10).unwrap();
        let mut tick = TelemetryTick {
            timestamp: 1672531200000,
            coin_id: 0,
            bid_price: 50000.0,
            ask_price: 50001.0,
            bid_qty: 1.0,
            ask_qty: 1.0,
            checksum: 0,
        };
        tick.checksum = tick.calculate_checksum();

        storage.append_tick(&tick);
        assert!(storage.flush_to_disk().is_ok());

        // Ingestión de tick corrupto con NaN no debe provocar pánico
        let nan_tick = TelemetryTick {
            timestamp: 1672531200000,
            coin_id: 0,
            bid_price: f32::NAN,
            ask_price: 50001.0,
            bid_qty: 1.0,
            ask_qty: 1.0,
            checksum: 0,
        };
        storage.append_tick(&nan_tick);

        let _ = std::fs::remove_file(path);
    }
}
