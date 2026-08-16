use std::fs::OpenOptions;
use std::sync::atomic::{AtomicUsize, Ordering};
use memmap2::MmapMut;

const EVENT_SIZE: usize = 64;
const MAX_EVENTS: usize = 1_000_000; // ~64MB file
const HEADER_SIZE: usize = 64;

/// 🧠 OMNI-TELEMETRY ZERO-COPY BUS
/// Proporciona telemetría distribuida "de pies a cabeza" sin afectar 
/// la latencia L1 (Lock-Free, Zero-Alloc). Usa Memmap sobre SSD.
pub struct ZeroCopyTelemetryBus {
    _mmap: MmapMut, // Se mantiene el ciclo de vida del memory map
    write_head: AtomicUsize,
    base_ptr: *mut u8,
    file_size: usize,
}

unsafe impl Send for ZeroCopyTelemetryBus {}
unsafe impl Sync for ZeroCopyTelemetryBus {}

impl ZeroCopyTelemetryBus {
    pub fn new(file_path: &str) -> Self {
        let file_size = HEADER_SIZE + (EVENT_SIZE * MAX_EVENTS);
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_path)
            .unwrap_or_else(|_| panic!("🛡️ [ZeroCopyTelemetryBus] Fallo al crear archivo {}", file_path));

        file.set_len(file_size as u64).unwrap();

        let mut mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file).unwrap() };
        let base_ptr = mmap.as_mut_ptr();

        Self {
            _mmap: mmap,
            write_head: AtomicUsize::new(HEADER_SIZE),
            base_ptr,
            file_size,
        }
    }

    /// Escribe un evento cuántico/tensorial directo al SSD en O(1) puro (~10 nanosegundos)
    /// - event_type: 1=ScalpIntent, 2=SwingIntent, 3=MOE_Fitness, 4=Hardware_Cycles, 5=IA_Tensor
    /// - source_id: ID de la moneda o subsistema
    /// - payload: Arreglo puro de 48 bytes (f64, etc.)
    #[inline(always)]
    pub fn record_event(&self, event_type: u8, source_id: u8, payload: &[u8; 48]) {
        let mut head = self.write_head.load(Ordering::Relaxed);
        let mut offset;
        loop {
            if head >= self.file_size {
                head = HEADER_SIZE;
            }
            offset = head;
            let next_head = head + EVENT_SIZE;
            
            match self.write_head.compare_exchange_weak(head, next_head, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => head = actual,
            }
        }

        // ⚡ HFT: Inyección directa al puntero virtual reservado atómicamente
        unsafe {
            let ptr = self.base_ptr.add(offset);
            
            // TS: 8 bytes
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
            std::ptr::copy_nonoverlapping(&ts as *const u64 as *const u8, ptr, 8);
            
            // Tipo y origen: 2 bytes
            *ptr.add(8) = event_type;
            *ptr.add(9) = source_id;
            
            // Payload: 48 bytes (offset 16)
            std::ptr::copy_nonoverlapping(payload.as_ptr(), ptr.add(16), 48);
        }
    }

    /// Extracción Matemática de Telemetría Avanzada (Tensores, Gradientes, Entropía)
    /// Convierte métricas f64 en bits para transferirlas en O(1) al Ring Buffer
    #[inline(always)]
    pub fn record_tensor_telemetry(
        &self, 
        source_id: u8, 
        gradient_loss: f64, 
        entropy: f64, 
        confidence: f64, 
        volatility_factor: f64,
        kelly_fraction: f64,
        dynamic_leverage: f64
    ) {
        let mut payload = [0u8; 48];
        unsafe {
            let p_ptr = payload.as_mut_ptr() as *mut f64;
            *p_ptr.add(0) = gradient_loss;
            *p_ptr.add(1) = entropy;
            *p_ptr.add(2) = confidence;
            *p_ptr.add(3) = volatility_factor;
            *p_ptr.add(4) = kelly_fraction;
            *p_ptr.add(5) = dynamic_leverage;
        }
        self.record_event(5, source_id, &payload); // event_type 5 = IA_Tensor
    }

    /// Escribe un evento estructurado de ROI (Retorno de Inversión) sin bloqueos.
    /// Payload de 48 bytes: [f64: roi_pre_fee, f64: roi_post_fee, f64: win_rate, f64: total_trades, u8 x 16: padding]
    #[inline(always)]
    pub fn record_roi_event(&self, source_id: u8, roi_pre_fee: f64, roi_post_fee: f64, win_rate: f64, total_trades: f64) {
        let mut payload = [0u8; 48];
        unsafe {
            let p_ptr = payload.as_mut_ptr() as *mut f64;
            *p_ptr.add(0) = roi_pre_fee;
            *p_ptr.add(1) = roi_post_fee;
            *p_ptr.add(2) = win_rate;
            *p_ptr.add(3) = total_trades;
        }
        self.record_event(6, source_id, &payload); // event_type 6 = ROI_Report
    }
}
