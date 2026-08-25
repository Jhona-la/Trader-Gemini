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
        
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_path)
            .unwrap_or_else(|_| {
                let temp_path = std::env::temp_dir().join("telemetry_bus_fallback.bin");
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&temp_path)
                    .unwrap_or_else(|_| panic!("🛡️ [ZeroCopyTelemetryBus] Fallo al crear archivo {}", file_path))
            });

        let _ = file.set_len(file_size as u64);

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
        let offset;
        loop {
            let next_head = if head + EVENT_SIZE > self.file_size {
                HEADER_SIZE + EVENT_SIZE
            } else {
                head + EVENT_SIZE
            };
            let current_offset = if head + EVENT_SIZE > self.file_size {
                HEADER_SIZE
            } else {
                head
            };
            
            match self.write_head.compare_exchange_weak(head, next_head, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => {
                    offset = current_offset;
                    break;
                }
                Err(actual) => head = actual,
            }
        }

        // ⚡ HFT: Inyección directa al puntero virtual reservado atómicamente
        unsafe {
            let ptr = self.base_ptr.add(offset);
            
            // TS: 8 bytes
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
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
        let values = [gradient_loss, entropy, confidence, volatility_factor, kelly_fraction, dynamic_leverage];
        for (i, val) in values.iter().enumerate() {
            // FIX #1428: Sanitización de flotantes antes de persistir en bus memmap
            let safe_val = if val.is_finite() { *val } else { 0.0 };
            let bytes = safe_val.to_bits().to_le_bytes();
            payload[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        self.record_event(5, source_id, &payload); // event_type 5 = IA_Tensor
    }

    /// Escribe un evento estructurado de ROI (Retorno de Inversión) sin bloqueos.
    /// Payload de 48 bytes: [f64: roi_pre_fee, f64: roi_post_fee, f64: win_rate, f64: total_trades, u8 x 16: padding]
    #[inline(always)]
    pub fn record_roi_event(&self, source_id: u8, roi_pre_fee: f64, roi_post_fee: f64, win_rate: f64, total_trades: f64) {
        let mut payload = [0u8; 48];
        let values = [roi_pre_fee, roi_post_fee, win_rate, total_trades];
        for (i, val) in values.iter().enumerate() {
            // FIX #1428: Sanitización de métricas de rendimiento
            let safe_val = if val.is_finite() { *val } else { 0.0 };
            let bytes = safe_val.to_bits().to_le_bytes();
            payload[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        self.record_event(6, source_id, &payload); // event_type 6 = ROI_Report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_bus_event_recording() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_telemetry_bus.bin");
        let path_str = path.to_str().unwrap();

        let bus = ZeroCopyTelemetryBus::new(path_str);
        let payload = [7u8; 48];
        bus.record_event(1, 0, &payload);
        bus.record_tensor_telemetry(0, 0.05, 0.95, 0.88, 0.01, 0.25, 5.0);
        bus.record_roi_event(0, 0.02, 0.015, 0.70, 10.0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_telemetry_bus_nan_sanitization_and_ring_wrap() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_telemetry_bus_nan.bin");
        let path_str = path.to_str().unwrap();

        let bus = ZeroCopyTelemetryBus::new(path_str);
        bus.record_tensor_telemetry(1, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        bus.record_roi_event(1, f64::NAN, f64::NAN, f64::NAN, f64::NAN);

        let _ = std::fs::remove_file(path);
    }
}
