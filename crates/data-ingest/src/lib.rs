pub mod dynamic_selector;
pub mod world_bank;
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
    capacity: u64,
    fill_rate: f64, // tokens per millisecond
    tokens: AtomicU64,
    last_update: AtomicU64,
}

impl TokenBucket {
    pub fn new(capacity: u64, fill_rate: f64) -> Self {
        let now = get_monotonic_ms();

        Self {
            capacity,
            fill_rate,
            tokens: AtomicU64::new(capacity),
            last_update: AtomicU64::new(now),
        }
    }

    /// Try to consume 1 token. Lock-free logic.
    pub fn try_consume(&self) -> bool {
        loop {
            let now = get_monotonic_ms();

            let last = self.last_update.load(Ordering::SeqCst);
            let current_tokens = self.tokens.load(Ordering::SeqCst);

            let elapsed_ms = now.saturating_sub(last);
            let added_tokens = (elapsed_ms as f64 * self.fill_rate) as u64;

            let mut new_tokens = std::cmp::min(self.capacity, current_tokens + added_tokens);

            if new_tokens == 0 {
                return false;
            }

            new_tokens -= 1;

            // Try to update time and tokens atomically via CAS-like loop approach
            // In a strict high-concurrency setting, we might use a spin-loop with compare_exchange
            if self
                .tokens
                .compare_exchange(
                    current_tokens,
                    new_tokens,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                if added_tokens > 0 {
                    let _ = self.last_update.compare_exchange(
                        last,
                        now,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    );
                }
                return true;
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
