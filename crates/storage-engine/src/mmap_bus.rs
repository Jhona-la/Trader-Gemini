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
    pub padding: [u8; 2],  // 2 bytes
    /// CERT-M6-H01 — SEQLOCK por frame: escritor marca seq impar antes de
    /// modificar el slot y seq par al commit; lector descarta si seq cambió
    /// o quedó impar (frame rasgado en reuso de slot). Vive en bytes 12-15
    /// (bits 32-63 de meta_u64), antes padding muerto.
    pub seq: u32,         // 4 bytes (total 16)
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
    write_prediction_vs_reality_ext(ml_prob, is_long, net_pnl_pct, atr_pct, 0.0, 0.50);
}

/// #23: Escribe predicción-vs-realidad con el vector completo de 6 variables microestructurales
pub fn write_prediction_vs_reality_ext(
    ml_prob: f64,
    is_long: bool,
    net_pnl_pct: f64,
    atr_pct: f64,
    obi: f64,
    hurst: f64,
) {
    if let Some(Some(bus)) = GLOBAL_TELEMETRY_WRITER.get() {
        bus.write_trace(
            SUBSYSTEM_TENSOR_PREDICTOR,
            FRAME_PREDICTION_VS_REALITY,
            [
                ml_prob,
                if is_long { 1.0 } else { 0.0 },
                obi,
                net_pnl_pct,
                atr_pct,
                hurst,
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

        unsafe {
            let mmap = &mut *self.mmap.get();
            let base_ptr = mmap.as_mut_ptr().add(HEADER_SIZE);
            let frame_ptr = (base_ptr as *mut TelemetryFrame).add(current_idx);

            // CERT-M6-H01 — El truco "commit al final" (FIX #1002) NO ordena
            // non-temporal stores: sin fence intermedio, el lector podía ver
            // chunk1 (commit) ANTES que el payload — frame rasgado invisible
            // al filtro timestamp != 0. Seqlock clásico por frame:
            //   (1) invalidate  → seq IMPAR visible,
            //   (2) payload,
            //   (3) commit      → seq PAR + timestamp/meta.
            // Lector: seq par e igual antes/después ⇒ frame íntegro.
            let seq_ptr = (frame_ptr as *mut u8).add(12) as *const u32;
            let seq_begin = seq_ptr.read_volatile() | 1; // impar SIEMPRE
            let seq_end = seq_begin.wrapping_add(1); // par
            let meta_base = (subsystem as u64) | ((frame_type as u64) << 8);

            // FASE XLII: Zero-Latency Telemetry (Non-Temporal Store)
            // Evitamos golpear el Caché L1/L2 del procesador usando intrínsecos SIMD
            #[cfg(target_arch = "x86_64")]
            {
                let ptr = frame_ptr as *mut __m128i;

                // (1) INVALIDATE: chunk1 con seq impar — el slot queda "en obra".
                let chunk1_odd = _mm_set_epi64x(
                    (meta_base | ((seq_begin as u64) << 32)) as i64,
                    timestamp_ns as i64,
                );
                _mm_stream_si128(ptr, chunk1_odd);
                // CRITICAL: seq impar debe ser visible ANTES del payload —
                // los NT stores entre sí NO están ordenados (núcleo de M6-H01).
                std::arch::x86_64::_mm_sfence();

                // (2) PAYLOAD: chunks 2-4 [f64; 6] (bits IEEE-754 exactos).
                let payload_ptr = safe_payload.as_ptr();
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

                // (3) COMMIT: timestamp + metadata + seq par.
                let chunk1_commit = _mm_set_epi64x(
                    (meta_base | ((seq_end as u64) << 32)) as i64,
                    timestamp_ns as i64,
                );
                _mm_stream_si128(ptr, chunk1_commit);
                std::arch::x86_64::_mm_sfence();
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let seq_mut = (frame_ptr as *mut u8).add(12) as *mut u32;
                seq_mut.write_volatile(seq_begin);
                std::sync::atomic::fence(Ordering::Release);
                let frame = TelemetryFrame {
                    timestamp_ns,
                    subsystem_id: subsystem,
                    frame_type,
                    padding: [0; 2],
                    seq: seq_end,
                    payload: safe_payload,
                };
                std::ptr::write_volatile(frame_ptr, frame);
                std::sync::atomic::fence(Ordering::Release);
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

            // CERT-M6-H01 — validación seqlock: el frame sólo es íntegro si
            // seq estaba PAR (escritor no en medio) y NO cambió durante la
            // copia (reuso de slot por wraparound del anillo). El filtro
            // timestamp != 0 (FIX #602) SOLO cubría slots jamás inicializados,
            // no tearing en reuso.
            unsafe {
                let frame_ptr = ring_ptr.add(slot);
                let seq_ptr = (frame_ptr as *const u8).add(12) as *const u32;
                let seq_before = seq_ptr.read_volatile();
                let frame = std::ptr::read_volatile(frame_ptr);
                let seq_after = seq_ptr.read_volatile();

                if (seq_before & 1) == 0 && seq_before == seq_after && frame.timestamp_ns != 0 {
                    frames.push(frame);
                }
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

    /// CERT-M6-H01 — regresión del seqlock: un slot con seq IMPAR (escritor
    /// a mitad de update, estado que el filtro timestamp != 0 dejaba pasar
    /// como frame "válido" con payload viejo) DEBE ser descartado por el
    /// lector. Y un frame comprometido correctamente lleva seq PAR.
    #[test]
    fn test_mmap_seqlock_discards_torn_frames() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(99999);
        let path = temp_dir.join(format!("test_mmap_seqlock_{}.dat", unique_id));

        {
            let bus = MmapTelemetryBus::new(&path).expect("Failed to create mmap bus");
            bus.write_trace(SUBSYSTEM_RISK_KELLY, FRAME_TYPE_BAYESIAN_PROB, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
            bus.write_trace(SUBSYSTEM_TENSOR_ML, FRAME_TYPE_TENSOR_ENTROPY, [9.9; 6]);

            // Simular escritor interrumpido en el slot 1: seq impar in-place
            // (segundo mapping del mismo archivo, mismo mecanismo que un
            // segundo proceso escritor).
            let corruptor = MmapTelemetryBus::new(&path).expect("Failed to reopen mmap bus");
            unsafe {
                let mmap = &*corruptor.mmap.get();
                let base_ptr = mmap.as_ptr().add(HEADER_SIZE) as *const u8;
                let seq_ptr = base_ptr.add(1 * std::mem::size_of::<TelemetryFrame>() + 12) as *mut u32;
                let cur = seq_ptr.read_volatile();
                assert_eq!(cur % 2, 0, "frame comprometido debe tener seq PAR");
                seq_ptr.write_volatile(cur | 1); // escritor "congelado" a mitad
            }
        }

        {
            let mut reader = MmapTelemetryReader::new(&path);
            let frames = reader.read_latest_frames().expect("Failed to read frames");
            // Slot 0 íntegro; slot 1 rasgado (seq impar) → descartado.
            assert_eq!(frames.len(), 1, "frame rasgado NO debe pasar el lector");
            assert_eq!(frames[0].subsystem_id, SUBSYSTEM_RISK_KELLY);
            assert_eq!(frames[0].seq % 2, 0);
            assert_eq!(frames[0].payload[0], 1.5);
            drop(reader);
        }

        let _ = std::fs::remove_file(path);
    }
}
