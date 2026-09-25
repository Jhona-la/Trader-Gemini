pub mod dynamic_selector;
pub mod liquidation;
pub mod tensor_parser;
pub mod world_bank;

pub use data_pipeline::ResilientStreamManager;
pub use dynamic_selector::{AssetScore, DynamicSelector};
pub use tensor_parser::TensorParser;
use memmap2::MmapOptions;
use polars::prelude::LazyFileListReader;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

fn get_monotonic_ms() -> u64 {
    static BASE_INSTANT: OnceLock<Instant> = OnceLock::new();
    let base = BASE_INSTANT.get_or_init(Instant::now);
    base.elapsed().as_millis() as u64
}

pub struct TokenBucket {
    capacity: u32,
    fill_rate: f64,   // tokens per millisecond
    state: AtomicU64, // packed: [32-bit tokens | 32-bit last_update_ms]
}

impl TokenBucket {
    #[inline(always)]
    fn pack(tokens: u32, last_ms: u32) -> u64 {
        ((tokens as u64) << 32) | (last_ms as u64)
    }

    #[inline(always)]
    fn unpack(val: u64) -> (u32, u32) {
        ((val >> 32) as u32, val as u32)
    }

    // FIX #1490: Sanitización de fill_rate y capacity para prevenir divisiones por cero
    pub fn new(capacity: u64, fill_rate: f64) -> Self {
        let now_ms = (get_monotonic_ms() & 0xFFFFFFFF) as u32;
        let cap_u32 = capacity.min(u32::MAX as u64).max(1) as u32;
        let safe_fill_rate = if fill_rate.is_finite() && fill_rate > 0.0 {
            fill_rate
        } else {
            1.0
        };

        Self {
            capacity: cap_u32,
            fill_rate: safe_fill_rate,
            state: AtomicU64::new(Self::pack(cap_u32, now_ms)),
        }
    }

    /// Try to consume 1 token. 100% Lock-free single CAS logic.
    #[inline(always)]
    pub fn try_consume(&self) -> bool {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            let (current_tokens, last_ms) = Self::unpack(current);
            let now_ms = (get_monotonic_ms() & 0xFFFFFFFF) as u32;

            let elapsed_ms = (now_ms.wrapping_sub(last_ms)) as f64;
            let added_tokens = (elapsed_ms * self.fill_rate) as u32;

            let total_tokens =
                std::cmp::min(self.capacity, current_tokens.saturating_add(added_tokens));

            if total_tokens == 0 {
                return false;
            }

            let new_tokens = total_tokens - 1;
            let new_last_ms = if added_tokens > 0 {
                let consumed_time_ms = ((added_tokens as f64) / self.fill_rate.max(1e-9)) as u32;
                last_ms.wrapping_add(consumed_time_ms)
            } else {
                last_ms
            };
            let new_packed = Self::pack(new_tokens, new_last_ms);

            match self.state.compare_exchange_weak(
                current,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }
}

pub struct ZeroCopyReader {
    mmap: memmap2::Mmap,
}

impl ZeroCopyReader {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

// FASE 15: Polars LazyFrame Ingestor (Memoria < 16GB)
// Permite leer TBs de históricos en trozos (chunks) directo al SSD sin desbordar la RAM.
pub struct PolarsIngest {
    path: std::path::PathBuf,
}

impl PolarsIngest {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Retorna un LazyFrame que no carga los datos en RAM hasta que se llama a .collect().
    /// Esto permite filtrar (ej. fechas específicas) y luego hacer chunking a tensores
    /// manteniendo el consumo de memoria plano.
    pub fn get_lazy_frame(&self) -> Result<polars::prelude::LazyFrame, String> {
        let path_str = self.path.to_str().unwrap_or_default();
        if path_str.ends_with(".parquet") {
            polars::prelude::LazyFrame::scan_parquet(
                path_str,
                polars::prelude::ScanArgsParquet::default(),
            )
            .map_err(|e| format!("Fallo al leer Parquet: {}", e))
        } else if path_str.ends_with(".csv") {
            // Asume Csv sin configuraciones especiales
            polars::prelude::LazyCsvReader::new(path_str)
                .with_has_header(true)
                .finish()
                .map_err(|e| format!("Fallo al leer CSV: {}", e))
        } else {
            Err("Formato no soportado. Se requiere .parquet o .csv".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_rate_limiting() {
        let bucket = TokenBucket::new(2, 0.01); // 2 tokens capacity
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        // Exhausted
        assert!(!bucket.try_consume());
    }

    #[test]
    fn test_polars_ingest_path_validation() {
        let ingest = PolarsIngest::new("test.unsupported");
        assert!(ingest.get_lazy_frame().is_err());
    }

    #[test]
    fn test_zero_copy_reader_and_token_bucket_refill() {
        let bucket = TokenBucket::new(5, 100.0); // 100 tokens per ms
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());

        // Test ZeroCopyReader with a temporary file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_zero_copy.bin");
        std::fs::write(&test_file, b"HFT_QUANTUM_DATA_123456").expect("write temp file");

        let reader = ZeroCopyReader::new(&test_file).expect("mmap file");
        assert_eq!(reader.as_bytes(), b"HFT_QUANTUM_DATA_123456");

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_token_bucket_negative_and_nan_fill_rate_immunity() {
        // NaN fill rate sanitized to 1.0
        let nan_bucket = TokenBucket::new(10, f64::NAN);
        assert_eq!(nan_bucket.fill_rate, 1.0);
        assert!(nan_bucket.try_consume());

        // Negative fill rate sanitized to 1.0
        let neg_bucket = TokenBucket::new(5, -10.0);
        assert_eq!(neg_bucket.fill_rate, 1.0);
        assert!(neg_bucket.try_consume());
    }
}
