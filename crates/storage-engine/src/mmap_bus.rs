use memmap2::{MmapMut, MmapOptions};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128i, _mm_set_epi64x, _mm_stream_si128};
use std::cell::UnsafeCell;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Un frame de telemetría de 64 bytes (1 cache line exacto)
/// #[repr(C)] garantiza que los datos se guarden tal cual en la memoria (y por tanto, en el SSD).
#[repr(C, align(64))]
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct TelemetryFrame {
    pub timestamp_ns: u64, // 8 bytes
    pub subsystem_id: u8,  // 1 byte (Ej: 0 = Risk, 1 = ML, 2 = Execution, 3 = OS Memory)
    pub frame_type: u8,    // 1 byte (Ej: 0 = Base, 1 = Tensor State, 2 = Quant Stats, 3 = OS Meta)
    pub padding: [u8; 6],  // 6 bytes (total 16)
    pub payload: [f64; 6], // 48 bytes (total 64). Multiplexado según frame_type.
}

// Constantes de Sub-sistemas de Telemetría (FASE XLIII)
pub const SUBSYSTEM_RISK_KELLY: u8 = 0;
pub const SUBSYSTEM_TENSOR_ML: u8 = 1;
pub const SUBSYSTEM_QUANT_MATH: u8 = 2;
pub const SUBSYSTEM_OS_MEM: u8 = 3;
/// Feedback de autoevolución: predicción ML vs resultado real.
/// Consumido por online_daemon para el Shadow Forest.
pub const SUBSYSTEM_TENSOR_PREDICTOR: u8 = 12;
pub const FRAME_PREDICTION_VS_REALITY: u8 = 30;

/// Writer global lazy: el hot path del god-engine escribe frames de
/// predicción-vs-realidad sin construir el bus cada tick.
static GLOBAL_TELEMETRY_WRITER: std::sync::OnceLock<Option<MmapTelemetryBus>> =
    std::sync::OnceLock::new();

/// Inicializa el bus global (llamar UNA vez al arranque desde god_engine).
pub fn init_global_telemetry(path: &str) {
    match MmapTelemetryBus::new(path) {
        Ok(bus) => {
            let _ = GLOBAL_TELEMETRY_WRITER.set(Some(bus));
        }
        Err(_) => {
            let _ = GLOBAL_TELEMETRY_WRITER.set(None);
        }
    }
}

/// Escribe un frame de predicción-vs-realidad al bus global (si inicializado).
/// No-op si el bus no existe (compatible con tests y backtests sin telemetría).
pub fn write_prediction_vs_reality(ml_prob: f64, is_long: bool, net_pnl_pct: f64, atr_pct: f64) {
    if let Some(Some(bus)) = GLOBAL_TELEMETRY_WRITER.get() {
        bus.write_trace(
            SUBSYSTEM_TENSOR_PREDICTOR,
            FRAME_PREDICTION_VS_REALITY,
            [
                ml_prob,
                if is_long { 1.0 } else { 0.0 },
                0.0,
                net_pnl_pct,
                atr_pct,
                0.0,
            ],
        );
    }
}

// Tipos de frames Multiplexados
pub const FRAME_TYPE_TENSOR_ENTROPY: u8 = 10;
pub const FRAME_TYPE_BAYESIAN_PROB: u8 = 11;
pub const FRAME_TYPE_HURST_EXPONENT: u8 = 12;

const RING_CAPACITY: usize = 1_000_000; // ~64 MB
const HEADER_SIZE: usize = 64; // Guardamos metadatos atómicos al principio

/// Bus lock-free Mmap para latencia O(1) picosegundos
pub struct MmapTelemetryBus {
    mmap: UnsafeCell<MmapMut>,
}

unsafe impl Send for MmapTelemetryBus {}
unsafe impl Sync for MmapTelemetryBus {}

impl MmapTelemetryBus {
    /// Inicializa o abre el archivo mapeado en memoria (RAM transparente sobre SSD).
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file_size = HEADER_SIZE + (RING_CAPACITY * std::mem::size_of::<TelemetryFrame>());

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let metadata = file.metadata()?;
        if metadata.len() < file_size as u64 {
            file.set_len(file_size as u64)?;
            // Pre-fault the file with zeros to prevent allocation latencies during runtime
            file.write_all(&vec![0; file_size])?;
            file.flush()?;
        }

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            mmap: UnsafeCell::new(mmap),
        })
    }

    /// Obtiene una referencia a los punteros atómicos en el header
    #[inline(always)]
    fn get_head_ptr(&self) -> &AtomicUsize {
        unsafe {
            let mmap = &*self.mmap.get();
            &*(mmap.as_ptr() as *const AtomicUsize)
        }
    }

    /// Escribe una traza atómicamente en memoria, que Windows paginará al SSD
    #[inline(always)]
    pub fn write_trace(&self, subsystem: u8, frame_type: u8, payload: [f64; 6]) {
        // FIX #658: Sanitizar finitud de los flotantes en payload
        let mut safe_payload = payload;
        for p in &mut safe_payload {
            if !p.is_finite() {
                *p = 0.0;
            }
        }

        let head = self.get_head_ptr();
        // Atomic fetch_add reserves a unique slot in the ring buffer across all concurrent threads
        let slot_idx = head.fetch_add(1, Ordering::AcqRel);
        let current_idx = slot_idx % RING_CAPACITY;

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let frame = TelemetryFrame {
            timestamp_ns,
            subsystem_id: subsystem,
            frame_type,
            padding: [0; 6],
            payload: safe_payload,
        };

        unsafe {
            let mmap = &mut *self.mmap.get();
            let base_ptr = mmap.as_mut_ptr().add(HEADER_SIZE);
            let frame_ptr = (base_ptr as *mut TelemetryFrame).add(current_idx);

            // FASE XLII: Zero-Latency Telemetry (Non-Temporal Store)
            // Evitamos golpear el Caché L1/L2 del procesador usando intrínsecos SIMD
            #[cfg(target_arch = "x86_64")]
            {
                let ptr = frame_ptr as *mut __m128i;
                let payload_ptr = frame.payload.as_ptr();

                // Chunk 2, 3, 4: payload [f64; 6] (Preservar bits IEEE-754 exactos sin truncamiento a entero)
                // FIX #1002: Escribir el payload PRIMERO para evitar lecturas sucias (tearing) si el lector ve timestamp != 0
                let chunk2 = _mm_set_epi64x(
                    payload_ptr.add(1).read().to_bits() as i64,
                    payload_ptr.read().to_bits() as i64,
                );
                _mm_stream_si128(ptr.add(1), chunk2);

                let chunk3 = _mm_set_epi64x(
                    payload_ptr.add(3).read().to_bits() as i64,
                    payload_ptr.add(2).read().to_bits() as i64,
                );
                _mm_stream_si128(ptr.add(2), chunk3);

                let chunk4 = _mm_set_epi64x(
                    payload_ptr.add(5).read().to_bits() as i64,
                    payload_ptr.add(4).read().to_bits() as i64,
                );
                _mm_stream_si128(ptr.add(3), chunk4);

                // Chunk 1: timestamp (u64) + metadata (u64) - Escribir al final como commit del frame
                let meta_u64 = (frame.subsystem_id as u64) | ((frame.frame_type as u64) << 8);
                let chunk1 = _mm_set_epi64x(meta_u64 as i64, frame.timestamp_ns as i64);
                _mm_stream_si128(ptr, chunk1);

                // CRITICAL: Memory fence to ensure non-temporal stores reach memory BEFORE read
                std::arch::x86_64::_mm_sfence();
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                std::ptr::write_volatile(frame_ptr, frame);
            }
        }
    }
}

/// Lector lock-free del anillo de telemetría.
/// Permite extraer métricas en vivo (O(1)) de la memoria mapeada sin interferir con el motor.
pub struct MmapTelemetryReader {
    mmap: Option<memmap2::Mmap>,
    path: std::path::PathBuf,
    last_read_idx: usize,
}

impl MmapTelemetryReader {
    /// Abre el archivo mapeado en memoria en modo SÓLO LECTURA.
    // FIX #1499: Constructor no falible para latencia cero y drop explícito antes de borrar archivo
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let path_buf = path.as_ref().to_path_buf();
        let mmap = Self::open_mmap(&path_buf).ok();
        Self {
            mmap,
            path: path_buf,
            last_read_idx: 0,
        }
    }

    fn open_mmap(path: &std::path::Path) -> std::io::Result<memmap2::Mmap> {
        #[cfg(windows)]
        use std::os::windows::fs::OpenOptionsExt;

        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(windows)]
        opts.share_mode(3); // FILE_SHARE_READ | FILE_SHARE_WRITE

        let file = opts.open(path)?;
        unsafe { MmapOptions::new().map(&file) }
    }

    /// Lee todos los frames nuevos desde la última vez que fue invocado.
    /// Mantiene el mmap cacheado para latencia O(1) sin handle churn.
    pub fn read_latest_frames(&mut self) -> std::io::Result<Vec<TelemetryFrame>> {
        if self.mmap.is_none() {
            self.mmap = Self::open_mmap(&self.path).ok();
        }

        let mmap = match &self.mmap {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let head_ptr = unsafe { &*(mmap.as_ptr() as *const AtomicUsize) };
        let current_head = head_ptr.load(Ordering::Acquire);

        let mut frames = Vec::new();

        // FIX #602: Prevención de overflow y limitación de lote para cero picos de RAM en 16GB
        const MAX_BATCH_READ: usize = 10_000;
        if current_head > self.last_read_idx + MAX_BATCH_READ {
            self.last_read_idx = current_head - MAX_BATCH_READ;
        }

        let ring_start_offset = 64;
        let ring_bytes_len = std::mem::size_of::<[TelemetryFrame; RING_CAPACITY]>();

        if mmap.len() < ring_start_offset + ring_bytes_len {
            return Ok(frames);
        }

        let ring_ptr = unsafe { mmap.as_ptr().add(ring_start_offset) as *const TelemetryFrame };

        while self.last_read_idx < current_head {
            let slot = self.last_read_idx % RING_CAPACITY;
            let frame = unsafe { std::ptr::read_volatile(ring_ptr.add(slot)) };

            // FIX #602: Descartar frames no inicializados (timestamp_ns == 0)
            if frame.timestamp_ns != 0 {
                frames.push(frame);
            }
            self.last_read_idx += 1;
        }

        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_telemetry_bus_write_and_read() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(12345);
        let path = temp_dir.join(format!("test_mmap_bus_{}.dat", unique_id));

        {
            let bus = MmapTelemetryBus::new(&path).expect("Failed to create mmap bus");
            bus.write_trace(
                SUBSYSTEM_RISK_KELLY,
                FRAME_TYPE_BAYESIAN_PROB,
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            );
            bus.write_trace(
                SUBSYSTEM_TENSOR_ML,
                FRAME_TYPE_TENSOR_ENTROPY,
                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            );
        }

        {
            let mut reader = MmapTelemetryReader::new(&path);
            let frames = reader.read_latest_frames().expect("Failed to read frames");
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[0].subsystem_id, SUBSYSTEM_RISK_KELLY);
            assert_eq!(frames[0].payload[0], 1.0);
            assert_eq!(frames[1].subsystem_id, SUBSYSTEM_TENSOR_ML);
            assert_eq!(frames[1].payload[0], 0.5);
            drop(reader);
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_mmap_telemetry_bus_nan_payload_sanitization() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(54321);
        let path = temp_dir.join(format!("test_mmap_bus_nan_{}.dat", unique_id));

        {
            let bus = MmapTelemetryBus::new(&path).expect("Failed to create mmap bus");
            bus.write_trace(
                SUBSYSTEM_QUANT_MATH,
                FRAME_TYPE_HURST_EXPONENT,
                [f64::NAN, f64::INFINITY, -f64::INFINITY, 0.75, 10.0, 20.0],
            );
        }

        {
            let mut reader = MmapTelemetryReader::new(&path);
            let frames = reader.read_latest_frames().expect("Failed to read frames");
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].payload[0], 0.0); // Sanitized NaN -> 0.0
            assert_eq!(frames[0].payload[1], 0.0); // Sanitized Inf -> 0.0
            assert_eq!(frames[0].payload[2], 0.0); // Sanitized -Inf -> 0.0
            assert_eq!(frames[0].payload[3], 0.75);
            drop(reader);
        }

        let _ = std::fs::remove_file(path);
    }
}
