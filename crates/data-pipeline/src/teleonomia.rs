use bytemuck::{Pod, Zeroable};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RingBufferRecord {
    pub timestamp: u64,
    pub coin_id: u64,
    pub dt: f64,
    pub price: f64,
    pub features: [f64; 54],
}

unsafe impl Zeroable for RingBufferRecord {}
unsafe impl Pod for RingBufferRecord {}

/// A memory-mapped/atomic shared state to allow the Evolution Daemon (NEAT)
/// to read real-time market data directly from the GodEngineCore
/// without using locks or blocking the HFT thread.
pub struct TeleonomiaState {
    // 30 coins, each with 54 features
    pub omni_features: Vec<Vec<AtomicU64>>,
    pub dt: Vec<AtomicU64>,
    pub current_price: Vec<AtomicU64>,

    // Quantum Zero-Alloc Ring Buffer mapped to SSD
    mmap_file: Option<Mutex<MmapMut>>,
    head: AtomicUsize,
    capacity: usize,
}

impl TeleonomiaState {
    pub fn new(num_coins: usize, num_features: usize) -> Self {
        let mut omni_features = Vec::with_capacity(num_coins);
        for _ in 0..num_coins {
            let mut coin_feats = Vec::with_capacity(num_features);
            for _ in 0..num_features {
                coin_feats.push(AtomicU64::new(0));
            }
            omni_features.push(coin_feats);
        }

        let mut dt = Vec::with_capacity(num_coins);
        let mut current_price = Vec::with_capacity(num_coins);
        for _ in 0..num_coins {
            dt.push(AtomicU64::new(0));
            current_price.push(AtomicU64::new(0));
        }

        // Setup Zero-Alloc Ring Buffer on disk
        let capacity = 100_000; // 100k records
        let record_size = std::mem::size_of::<RingBufferRecord>();
        let total_size = capacity * record_size;

        let path = "data/lakehouse";
        if std::fs::metadata(path).is_err() {
            let _ = std::fs::create_dir_all(path);
        }
        let ring_file_path = "data/lakehouse/quantum_ring_buffer.bin";
        let mmap_file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(ring_file_path)
        {
            Ok(file) => {
                if file.metadata().map(|m| m.len()).unwrap_or(0) < total_size as u64 {
                    if let Err(e) = file.set_len(total_size as u64) {
                        eprintln!(
                            "⚠️ [TELEONOMIA] Fallo al establecer tamaño del Ring Buffer SSD: {}",
                            e
                        );
                    }
                }
                match unsafe { MmapMut::map_mut(&file) } {
                    Ok(mmap) => Some(Mutex::new(mmap)),
                    Err(e) => {
                        eprintln!("⚠️ [TELEONOMIA] Fallo al mapear Ring Buffer SSD: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("⚠️ [TELEONOMIA] No se pudo crear {}: {}", ring_file_path, e);
                None
            }
        };

        Self {
            omni_features,
            dt,
            current_price,
            mmap_file,
            head: AtomicUsize::new(0),
            capacity,
        }
    }

    #[inline(always)]
    pub fn write_features(&self, coin_id: usize, features: &[f64; 54], dt: f64, price: f64) {
        if coin_id >= self.omni_features.len() {
            return;
        }

        // FIX #695: Sanitizar finitud de los 54 features, dt y precio
        let mut safe_features = *features;
        for f in &mut safe_features {
            if !f.is_finite() {
                *f = 0.0;
            }
        }
        let safe_dt = if dt.is_finite() && dt >= 0.0 { dt } else { 0.0 };
        let safe_price = if price.is_finite() && price > 0.0 {
            price
        } else {
            0.0
        };

        for (i, &val) in safe_features.iter().enumerate() {
            if i < self.omni_features[coin_id].len() {
                self.omni_features[coin_id][i].store(val.to_bits(), Ordering::Relaxed);
            }
        }
        self.dt[coin_id].store(safe_dt.to_bits(), Ordering::Relaxed);
        self.current_price[coin_id].store(safe_price.to_bits(), Ordering::Relaxed);

        // Persist to SSD via Zero-Alloc Ring Buffer
        if let Some(mmap_mutex) = &self.mmap_file {
            let index = self.head.fetch_add(1, Ordering::Relaxed) % self.capacity;
            let record = RingBufferRecord {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                coin_id: coin_id as u64,
                dt: safe_dt,
                price: safe_price,
                features: safe_features,
            };

            let bytes = bytemuck::bytes_of(&record);
            let offset = index * std::mem::size_of::<RingBufferRecord>();

            // Lock to ensure reliable persistence to disk without dropping telemetry frames
            if let Ok(mut mmap) = mmap_mutex.lock() {
                let end = offset + bytes.len();
                if end <= mmap.len() {
                    // FIX #911: Prevención de Torn Reads en IPC sin Mutex inter-procesos
                    // Escribimos primero el payload (offset + 8) y luego el timestamp (primeros 8 bytes)
                    // con una barrera de memoria (Release) para garantizar causalidad.
                    let (ts_bytes, data_bytes) = bytes.split_at(8);
                    mmap[offset + 8..end].copy_from_slice(data_bytes);
                    std::sync::atomic::fence(Ordering::Release);
                    mmap[offset..offset + 8].copy_from_slice(ts_bytes);
                }
            }
        }
    }

    #[inline(always)]
    pub fn read_features(&self, coin_id: usize) -> ([f64; 54], f64, f64) {
        let mut feats = [0.0; 54];
        if coin_id < self.omni_features.len() {
            for i in 0..54 {
                if i < self.omni_features[coin_id].len() {
                    feats[i] =
                        f64::from_bits(self.omni_features[coin_id][i].load(Ordering::Relaxed));
                }
            }
        }
        let dt = if coin_id < self.dt.len() {
            f64::from_bits(self.dt[coin_id].load(Ordering::Relaxed))
        } else {
            0.0
        };
        let price = if coin_id < self.current_price.len() {
            f64::from_bits(self.current_price[coin_id].load(Ordering::Relaxed))
        } else {
            0.0
        };

        (feats, dt, price)
    }
}

// Global Singleton access if needed, though passing Arc is better.
lazy_static::lazy_static! {
    pub static ref GLOBAL_TELEONOMIA: std::sync::Arc<TeleonomiaState> = std::sync::Arc::new(TeleonomiaState::new(30, 54));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teleonomia_write_and_read() {
        let state = TeleonomiaState::new(10, 54);
        let mut feats = [0.0; 54];
        feats[0] = 1.23;
        feats[53] = 4.56;

        state.write_features(0, &feats, 0.05, 50000.0);
        let (read_feats, dt, price) = state.read_features(0);

        assert_eq!(read_feats[0], 1.23);
        assert_eq!(read_feats[53], 4.56);
        assert_eq!(dt, 0.05);
        assert_eq!(price, 50000.0);
    }

    #[test]
    fn test_teleonomia_nan_sanitization() {
        let state = TeleonomiaState::new(10, 54);
        let mut feats = [f64::NAN; 54];
        feats[10] = 100.0;

        state.write_features(1, &feats, f64::NAN, f64::NAN);
        let (read_feats, dt, price) = state.read_features(1);

        assert_eq!(read_feats[0], 0.0);
        assert_eq!(read_feats[10], 100.0);
        assert_eq!(dt, 0.0);
        assert_eq!(price, 0.0);
    }
}
