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
        let head = self.get_head_ptr();
        // FASE XLIII RCU Fix: Determine index, write data, SFENCE, then publish head.
        let current_head = head.load(Ordering::Acquire);
        let current_idx = current_head % RING_CAPACITY;

        let frame = TelemetryFrame {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            subsystem_id: subsystem,
            frame_type,
            padding: [0; 6],
            payload,
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

                // Chunk 1: timestamp (u64) + metadata (u64)
                let meta_u64 = (frame.subsystem_id as u64) | ((frame.frame_type as u64) << 8);
                let chunk1 = _mm_set_epi64x(meta_u64 as i64, frame.timestamp_ns as i64);
                _mm_stream_si128(ptr, chunk1);

                // Chunk 2, 3, 4: payload [f64; 6]
                let chunk2 = _mm_set_epi64x(*payload_ptr.add(1) as i64, *payload_ptr as i64);
                _mm_stream_si128(ptr.add(1), chunk2);

                let chunk3 = _mm_set_epi64x(*payload_ptr.add(3) as i64, *payload_ptr.add(2) as i64);
                _mm_stream_si128(ptr.add(2), chunk3);

                let chunk4 = _mm_set_epi64x(*payload_ptr.add(5) as i64, *payload_ptr.add(4) as i64);
                _mm_stream_si128(ptr.add(3), chunk4);

                // CRITICAL: Memory fence to ensure non-temporal stores reach memory BEFORE index publishes
                std::arch::x86_64::_mm_sfence();
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                std::ptr::write_volatile(frame_ptr, frame);
            }
        }

        // Publish the write (RCU commit)
        head.store(current_head + 1, Ordering::Release);
    }
}

/// Lector lock-free del anillo de telemetría.
/// Permite extraer métricas en vivo (O(1)) de la memoria mapeada sin interferir con el motor.
pub struct MmapTelemetryReader {
    mmap: MmapOptions,
    path: std::path::PathBuf,
    last_read_idx: usize,
}

impl MmapTelemetryReader {
    /// Abre el archivo mapeado en memoria en modo SÓLO LECTURA.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Ok(Self {
            mmap: MmapOptions::new(),
            path: path.as_ref().to_path_buf(),
            last_read_idx: 0,
        })
    }

    /// Lee todos los frames nuevos desde la última vez que fue invocado.
    /// Si hay saturación (el escritor dio más de 1 vuelta completa), saltamos al head más reciente
    /// para siempre mantenernos en tiempo real (frontera de latencia).
    pub fn read_latest_frames(&mut self) -> std::io::Result<Vec<TelemetryFrame>> {
        #[cfg(windows)]
        use std::os::windows::fs::OpenOptionsExt;

        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(windows)]
        opts.share_mode(3); // FILE_SHARE_READ | FILE_SHARE_WRITE

        let file = opts.open(&self.path)?;
        let mmap = unsafe { self.mmap.map(&file)? };

        let head_ptr = unsafe { &*(mmap.as_ptr() as *const AtomicUsize) };
        let current_head = head_ptr.load(Ordering::Acquire);

        let mut frames = Vec::new();

        // Prevención de overflow / saturación de buffer
        if current_head > self.last_read_idx + RING_CAPACITY {
            // Saltamos al punto más reciente disponible, descartando lo muy viejo.
            self.last_read_idx = current_head.saturating_sub(RING_CAPACITY - 1);
        }

        let base_ptr = unsafe { mmap.as_ptr().add(HEADER_SIZE) };

        while self.last_read_idx < current_head {
            let ring_idx = self.last_read_idx % RING_CAPACITY;

            unsafe {
                let frame_ptr = (base_ptr as *const TelemetryFrame).add(ring_idx);
                let frame = std::ptr::read_volatile(frame_ptr);
                frames.push(frame);
            }

            self.last_read_idx += 1;
        }

        Ok(frames)
    }
}
