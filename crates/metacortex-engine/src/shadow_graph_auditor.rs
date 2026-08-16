use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::cell::UnsafeCell;

/// FASE 2: MOTOR DE AUDITORÍA SHADOW GRAPH (FORENSE)
/// Compara en tiempo real la expectativa teórica del modelo vs la realidad.
/// Funciona como un Ring Buffer limitado a 10,000 eventos para proteger la RAM (16GB).

const SHADOW_RING_SIZE: usize = 16384; // 16K eventos (Potencia de 2 para bitmask)
const SHADOW_RING_MASK: usize = SHADOW_RING_SIZE - 1;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ShadowEvent {
    pub tick_id: u64,
    pub expected_prob: f64,
    pub actual_slippage: f64,
    pub latency_ms: u64,
    pub pnl_drift: f64,
}

#[repr(C, align(64))]
pub struct ShadowGraphAuditor {
    buffer: UnsafeCell<[ShadowEvent; SHADOW_RING_SIZE]>,
    head: AtomicUsize,
    // Métricas de salud general
    pub aggregate_drift: AtomicU64,
    pub critical_drift_alarms: AtomicUsize,
}

unsafe impl Sync for ShadowGraphAuditor {}
unsafe impl Send for ShadowGraphAuditor {}

impl ShadowGraphAuditor {
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([ShadowEvent::default(); SHADOW_RING_SIZE]),
            head: AtomicUsize::new(0),
            aggregate_drift: AtomicU64::new(0.0_f64.to_bits()),
            critical_drift_alarms: AtomicUsize::new(0),
        }
    }

    /// O(1) Lock-free push event
    #[inline(always)]
    pub fn record_event(&self, event: ShadowEvent) {
        let current_head = self.head.load(Ordering::Relaxed);
        let idx = current_head & SHADOW_RING_MASK;
        
        unsafe {
            (*self.buffer.get())[idx] = event;
        }
        
        self.head.store(current_head.wrapping_add(1), Ordering::Release);
        
        // Acumulación cruda del Drift. 
        // Si el drift se vuelve muy negativo, el bot está perdiendo su borde matemático.
        let mut current_bits = self.aggregate_drift.load(Ordering::Relaxed);
        loop {
            let current_drift = f64::from_bits(current_bits);
            let new_drift = current_drift + event.pnl_drift;
            match self.aggregate_drift.compare_exchange_weak(current_bits, new_drift.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(b) => current_bits = b,
            }
        }

        if event.latency_ms > 100 || event.pnl_drift < -0.5 {
            self.critical_drift_alarms.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Evaluador de Divergencias de Producción (Metacórtex Drift Analyzer)
    /// Este método se llamaría asíncronamente desde el hilo del Consejo de Seniors.
    pub fn evaluate_system_drift(&self) -> bool {
        let alarms = self.critical_drift_alarms.load(Ordering::Relaxed);
        let agg_drift_bits = self.aggregate_drift.load(Ordering::Relaxed);
        let agg_drift = f64::from_bits(agg_drift_bits);
        
        // Si hay más de 50 alarmas críticas o el drift acumulado destruye > 5% del margen
        if alarms > 50 || agg_drift < -5.0 {
            // Se requiere Hot-Swapping genético urgente
            true
        } else {
            false
        }
    }
}

impl Default for ShadowGraphAuditor {
    fn default() -> Self {
        Self::new()
    }
}
