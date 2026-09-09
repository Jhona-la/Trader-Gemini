use crate::ebpf_core::KernelEvents;
use crate::pmu_sensor::PmuVector;
use std::sync::atomic::{AtomicU64, Ordering};

/// Motor Estadístico de Anomalías Base.
/// Usa EWMA (Exponentially Weighted Moving Average) y MAD (Median Absolute Deviation)
/// para rastrear latencias de cola y anomalías de hardware (context switches excesivos,
/// caída de IPC). No hace inferencia neuronal todavía; opera a la velocidad del hardware
/// para encontrar fallas determinísticas.
pub struct StatisticalAnomalyDetector {
    // EMA states (almacenado en u64 multiplicando x 10000 para lock-free concurrency)
    ipc_ema: AtomicU64,
    sched_delay_ema: AtomicU64,
    alpha: f64,
}

#[derive(Debug)]
pub struct AnomalyScore {
    pub is_anomaly: bool,
    pub score: f64,
    pub causal_hint: &'static str,
}

impl StatisticalAnomalyDetector {
    pub fn new(alpha: f64) -> Self {
        // FIX #1448: Sanitización de alpha en detector estadístico EWMA
        let safe_alpha = if alpha.is_finite() && alpha > 0.0 && alpha <= 1.0 {
            alpha
        } else {
            0.05
        };
        Self {
            ipc_ema: AtomicU64::new(20000), // 2.0 * 10000
            sched_delay_ema: AtomicU64::new(500),
            alpha: safe_alpha,
        }
    }

    /// Observa el vector de estado sin tocar el código fuente original del HFT.
    /// Retorna un score de anomalía y una "hipótesis causal".
    pub fn observe(&self, pmu: &PmuVector, ebpf: &KernelEvents) -> AnomalyScore {
        let current_ipc = if pmu.ipc.is_finite() && pmu.ipc >= 0.0 {
            pmu.ipc
        } else {
            2.0
        };
        let current_delay = ebpf.scheduler_delay_ns as f64;

        // Leer EMA actual
        let prev_ipc = self.ipc_ema.load(Ordering::Relaxed) as f64 / 10000.0;
        let prev_delay = self.sched_delay_ema.load(Ordering::Relaxed) as f64;

        // FIX #586: Detectar desviaciones críticas respecto a la línea base previa antes de absorber la anomalía
        // IPC cae más del 50% respecto al baseline histórico
        let is_ipc_anomaly = current_ipc < (prev_ipc * 0.5) && prev_ipc > 0.5;
        let is_delay_anomaly = current_delay > (prev_delay * 5.0) && current_delay > 1500.0;

        // Actualizar EWMA sólo con observaciones válidas o con tasa atenuada si hay anomalía
        let effective_alpha = if is_ipc_anomaly || is_delay_anomaly {
            self.alpha * 0.1
        } else {
            self.alpha
        };
        let new_ipc = prev_ipc + effective_alpha * (current_ipc - prev_ipc);
        let new_delay = prev_delay + effective_alpha * (current_delay - prev_delay);

        let clean_ipc = if new_ipc.is_finite() {
            new_ipc.max(0.0)
        } else {
            2.0
        };
        let clean_delay = if new_delay.is_finite() {
            new_delay.max(0.0)
        } else {
            500.0
        };

        self.ipc_ema
            .store((clean_ipc * 10000.0) as u64, Ordering::Relaxed);
        self.sched_delay_ema
            .store(clean_delay as u64, Ordering::Relaxed);

        if is_ipc_anomaly {
            return AnomalyScore {
                is_anomaly: true,
                score: 0.98,
                causal_hint: "IPC colapsó abruptamente. Posible trashing de L1 o contención en microarquitectura.",
            };
        }

        // Delay de scheduler se dispara más de 500% respecto al baseline (ej. de 500ns a 2500ns)
        if is_delay_anomaly {
            return AnomalyScore {
                is_anomaly: true,
                score: 0.95,
                causal_hint: "Latencia de scheduler disparada. Interference por IRQ de red o vecino ruidoso en CPU.",
            };
        }

        // Context switches inexplicables durante operaciones pinned
        if ebpf.context_switches > 10 {
            return AnomalyScore {
                is_anomaly: true,
                score: 0.90,
                causal_hint:
                    "Context switches detectados en CPU supuestamente aislado (Isolcpus roto).",
            };
        }

        AnomalyScore {
            is_anomaly: false,
            score: 0.0,
            causal_hint: "OK",
        }
    }
}
