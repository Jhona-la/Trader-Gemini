use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub symbol_id: usize,
    pub is_long: bool,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl_pct: f64,
    pub timestamp_ms: u64,
}

pub struct DriftAuditor {
    pub total_drift_pct: AtomicU64, // float representation via u64
    pub mismatch_count: AtomicU64,
    pub max_allowed_drift: f64,
}

impl Default for DriftAuditor {
    fn default() -> Self {
        Self {
            total_drift_pct: AtomicU64::new(0),
            mismatch_count: AtomicU64::new(0),
            max_allowed_drift: 0.05, // 5% de diferencia permitida entre Shadow y Live
        }
    }
}

impl DriftAuditor {
    pub fn new(max_drift: f64) -> Self {
        Self {
            total_drift_pct: AtomicU64::new(0),
            mismatch_count: AtomicU64::new(0),
            max_allowed_drift: max_drift,
        }
    }

    /// Compara un trade real con un trade simulado (Shadow)
    /// Devuelve Ok(drift) si está dentro del margen.
    /// Devuelve Err(drift) si ha saltado el Circuit Breaker.
    pub fn audit_execution(&self, real: &TradeResult, shadow: &TradeResult) -> Result<f64, f64> {
        // En un mundo ideal, real.pnl_pct == shadow.pnl_pct.
        // Si el real pierde dinero y el shadow gana, hay un "Negative Drift".
        let drift = shadow.pnl_pct - real.pnl_pct;

        // Acumulación atómica segura mediante bucle CAS
        // FIX #639: Ordenamiento atómico AcqRel y verificación de finitud estricta
        if drift.is_finite() {
            let mut current_bits = self.total_drift_pct.load(Ordering::Acquire);
            loop {
                let current_drift = f64::from_bits(current_bits);
                let new_drift = current_drift + drift;
                if !new_drift.is_finite() {
                    break;
                }
                match self.total_drift_pct.compare_exchange_weak(
                    current_bits,
                    new_drift.to_bits(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => current_bits = actual,
                }
            }
        }

        if drift.abs() > self.max_allowed_drift {
            self.mismatch_count.fetch_add(1, Ordering::Relaxed);
            return Err(drift); // CIRCUIT BREAKER TRIGGERED
        }

        Ok(drift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_auditor_execution() {
        let auditor = DriftAuditor::new(0.05);
        let real = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 60000.0,
            exit_price: 60600.0,
            pnl_pct: 0.01,
            timestamp_ms: 1000,
        };
        let shadow = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 60000.0,
            exit_price: 60660.0,
            pnl_pct: 0.011,
            timestamp_ms: 1000,
        };
        let res = auditor.audit_execution(&real, &shadow);
        assert!(res.is_ok());
        assert!((res.unwrap() - 0.001).abs() < 1e-6);

        let diverged_shadow = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 60000.0,
            exit_price: 66000.0,
            pnl_pct: 0.10,
            timestamp_ms: 1000,
        };
        let res_div = auditor.audit_execution(&real, &diverged_shadow);
        assert!(res_div.is_err());
    }

    #[test]
    fn test_drift_auditor_nan_immunity() {
        let auditor = DriftAuditor::default();
        let real = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 60000.0,
            exit_price: 60600.0,
            pnl_pct: f64::NAN,
            timestamp_ms: 1000,
        };
        let shadow = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 60000.0,
            exit_price: 60660.0,
            pnl_pct: 0.011,
            timestamp_ms: 1000,
        };
        let res = auditor.audit_execution(&real, &shadow);
        assert!(res.is_ok());
    }
}
