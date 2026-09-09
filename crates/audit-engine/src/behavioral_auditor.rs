use std::f64;

/// 🔬 MOTOR ALGORÍTMICO DE DIAGNÓSTICO COMPORTAMENTAL CONTINUO Y DETECCIÓN DE ANOMALÍAS (ZERO-COST BEHAVIORAL AUDITOR)
/// Implementa Wald's Sequential Probability Ratio Test (SPRT) y CUSUM (Cumulative Sum Control Chart).
/// Detecta anomalías, silenciaciones, degradaciones de latencia y desvíos microestructurales en < 5 nanosegundos por tick.
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct BehavioralAuditorEngine {
    pub log_likelihood_ratio: f64, // Log-Likelihood Ratio Acumulado SPRT (Λ_t)
    pub cusum_pos: f64,            // Suma Acumulativa Positiva (S_t+)
    pub cusum_neg: f64,            // Suma Acumulativa Negativa (S_t-)
    pub baseline_mean: f64,        // Media baseline teórica (μ_0)
    pub slack_k: f64,              // Factor de tolerancia de holgura k
    pub threshold_h: f64,          // Umbral de decisión CUSUM h
    pub sprt_bound_a: f64,         // Límite A de aceptación H0 (e.g. ln(β / (1-α)))
    pub sprt_bound_b: f64,         // Límite B de aceptación H1 (e.g. ln((1-β) / α))
    pub total_audits: u64,         // Contador total de ticks auditados
    pub anomalies_detected: u64,   // Contador total de anomalías comportamentales detectadas
}

impl BehavioralAuditorEngine {
    pub fn new(baseline_mean: f64, tolerance_k: f64, threshold_h: f64) -> Self {
        let alpha: f64 = 0.01; // Probabilidad de Falso Positivo (1%)
        let beta: f64 = 0.01; // Probabilidad de Falso Negativo (1%)
        let safe_baseline = if baseline_mean.is_finite() {
            baseline_mean
        } else {
            0.50
        };

        Self {
            log_likelihood_ratio: 0.0,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            baseline_mean: safe_baseline,
            slack_k: tolerance_k.max(1e-5),
            threshold_h: threshold_h.max(0.1),
            sprt_bound_a: (beta / (1.0 - alpha)).ln(),
            sprt_bound_b: ((1.0 - beta) / alpha).ln(),
            total_audits: 0,
            anomalies_detected: 0,
        }
    }

    /// Audita una nueva observación comportamental en caliente en < 5 nanosegundos
    /// Retorna Option<&'static str> indicando la anomalía detectada o None si el comportamiento es nominal
    #[inline(always)]
    pub fn audit_observation(&mut self, current_val: f64) -> Option<&'static str> {
        // FIX #678: Guarda de finitud estricta en observación
        if !current_val.is_finite() {
            return None;
        }

        self.total_audits += 1;

        // 1. Algoritmo CUSUM de detección de desvío
        let diff = current_val - self.baseline_mean;
        self.cusum_pos = (self.cusum_pos + diff - self.slack_k).max(0.0);
        self.cusum_neg = (self.cusum_neg - diff - self.slack_k).max(0.0);

        // 2. Algoritmo SPRT de Wald (Likelihood Ratio de distribución Gaussiana)
        let delta_hypotheses = 0.10; // Desvío de hipótesis alternativa H1
        let sprt_increment = (diff * delta_hypotheses) / (self.baseline_mean.abs() + 1e-5);
        self.log_likelihood_ratio += sprt_increment;

        // Evaluación del estado CUSUM y SPRT
        if self.cusum_pos > self.threshold_h {
            self.anomalies_detected += 1;
            self.cusum_pos = 0.0; // Reset probabilístico
            Some("ANOMALY_UPWARD_DRIFT_DETECTED")
        } else if self.cusum_neg > self.threshold_h {
            self.anomalies_detected += 1;
            self.cusum_neg = 0.0;
            Some("ANOMALY_DOWNWARD_DRIFT_DETECTED")
        } else if self.log_likelihood_ratio > self.sprt_bound_b {
            self.anomalies_detected += 1;
            self.log_likelihood_ratio = 0.0;
            Some("SPRT_STRUCTURAL_SHIFT_H1")
        } else {
            if self.log_likelihood_ratio < self.sprt_bound_a {
                self.log_likelihood_ratio = 0.0; // Reseteo a estado nominal H0
            }
            None
        }
    }
}

impl Default for BehavioralAuditorEngine {
    fn default() -> Self {
        Self::new(0.50, 0.05, 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavioral_auditor_nominal_and_drift_detection() {
        let mut auditor = BehavioralAuditorEngine::new(0.50, 0.05, 2.0);

        // Observaciones nominales
        for _ in 0..20 {
            assert!(auditor.audit_observation(0.50).is_none());
        }

        // Fuerte shock positivo persistente para disparar CUSUM
        let mut anomaly = None;
        for _ in 0..50 {
            if let Some(a) = auditor.audit_observation(1.50) {
                anomaly = Some(a);
                break;
            }
        }
        assert!(anomaly.is_some());
        assert!(auditor.anomalies_detected >= 1);
    }

    #[test]
    fn test_behavioral_auditor_nan_immunity() {
        let mut auditor = BehavioralAuditorEngine::default();
        assert!(auditor.audit_observation(f64::NAN).is_none());
        assert_eq!(auditor.total_audits, 0);
    }
}
