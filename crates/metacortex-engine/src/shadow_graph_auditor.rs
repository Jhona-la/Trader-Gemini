use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
        if !event.pnl_drift.is_finite() {
            return;
        }

        let current_head = self.head.fetch_add(1, Ordering::AcqRel);
        let idx = current_head & SHADOW_RING_MASK;

        unsafe {
            (*self.buffer.get())[idx] = event;
        }

        // FIX #618: Acumulación lock-free segura con validación de finitud y memoria AcqRel
        let mut current_bits = self.aggregate_drift.load(Ordering::Acquire);
        loop {
            let current_drift = f64::from_bits(current_bits);
            let safe_current = if current_drift.is_finite() {
                current_drift
            } else {
                0.0
            };
            let new_drift = safe_current + event.pnl_drift;
            if !new_drift.is_finite() {
                break;
            }
            match self.aggregate_drift.compare_exchange_weak(
                current_bits,
                new_drift.to_bits(),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_graph_auditor_record_and_drift() {
        let auditor = ShadowGraphAuditor::new();

        let event1 = ShadowEvent {
            tick_id: 1,
            expected_prob: 0.80,
            actual_slippage: 0.0001,
            latency_ms: 10,
            pnl_drift: 0.05,
        };
        auditor.record_event(event1);

        let agg = f64::from_bits(auditor.aggregate_drift.load(Ordering::Relaxed));
        assert!((agg - 0.05).abs() < 1e-6);
        assert!(!auditor.evaluate_system_drift());

        let event_bad = ShadowEvent {
            tick_id: 2,
            expected_prob: 0.80,
            actual_slippage: 0.005,
            latency_ms: 150, // Triggers alarm
            pnl_drift: -6.0, // Triggers drift threshold
        };
        auditor.record_event(event_bad);

        assert!(auditor.critical_drift_alarms.load(Ordering::Relaxed) >= 1);
        assert!(auditor.evaluate_system_drift());
    }

    #[test]
    fn test_shadow_graph_auditor_nan_immunity() {
        let auditor = ShadowGraphAuditor::new();
        let nan_event = ShadowEvent {
            tick_id: 3,
            expected_prob: f64::NAN,
            actual_slippage: 0.0,
            latency_ms: 5,
            pnl_drift: f64::NAN,
        };
        auditor.record_event(nan_event);

        let agg = f64::from_bits(auditor.aggregate_drift.load(Ordering::Relaxed));
        assert_eq!(agg, 0.0);
    }
}
