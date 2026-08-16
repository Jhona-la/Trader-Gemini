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
        Self {
            ipc_ema: AtomicU64::new(20000), // 2.0 * 10000
            sched_delay_ema: AtomicU64::new(500),
            alpha,
        }
    }

    /// Observa el vector de estado sin tocar el código fuente original del HFT.
    /// Retorna un score de anomalía y una "hipótesis causal".
    pub fn observe(&self, pmu: &PmuVector, ebpf: &KernelEvents) -> AnomalyScore {
        let current_ipc = pmu.ipc;
        let current_delay = ebpf.scheduler_delay_ns as f64;

        // Leer EMA actual
        let prev_ipc = self.ipc_ema.load(Ordering::Relaxed) as f64 / 10000.0;
        let prev_delay = self.sched_delay_ema.load(Ordering::Relaxed) as f64;

        // Actualizar EWMA
        let new_ipc = prev_ipc + self.alpha * (current_ipc - prev_ipc);
        let new_delay = prev_delay + self.alpha * (current_delay - prev_delay);

        self.ipc_ema
            .store((new_ipc * 10000.0) as u64, Ordering::Relaxed);
        self.sched_delay_ema
            .store(new_delay as u64, Ordering::Relaxed);

        // Detectar desviaciones críticas
        // IPC cae más del 50% respecto al baseline
        if current_ipc < (new_ipc * 0.5) {
            return AnomalyScore {
                is_anomaly: true,
                score: 0.98,
                causal_hint: "IPC colapsó abruptamente. Posible trashing de L1 o contención en microarquitectura.",
            };
        }

        // Delay de scheduler se dispara más de 500% respecto al baseline (ej. de 500ns a 2500ns)
        if current_delay > (new_delay * 5.0) && current_delay > 1500.0 {
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
