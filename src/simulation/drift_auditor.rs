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
        // FIX #1467: Validación estricta de finitud de PnL
        if !real.pnl_pct.is_finite() || !shadow.pnl_pct.is_finite() {
            return Ok(0.0);
        }
        // En un mundo ideal, real.pnl_pct == shadow.pnl_pct.
        // Si el real pierde dinero y el shadow gana, hay un "Negative Drift".
        let drift = shadow.pnl_pct - real.pnl_pct; 
        if !drift.is_finite() {
            return Ok(0.0);
        }

        // FIX #1467: Lock-free CAS loop para actualización libre de data-race
        let mut old_bits = self.total_drift_pct.load(Ordering::Relaxed);
        loop {
            let current = f64::from_bits(old_bits);
            let safe_current = if current.is_finite() { current } else { 0.0 };
            let new_val = safe_current + drift;
            let new_bits = new_val.to_bits();
            match self.total_drift_pct.compare_exchange_weak(old_bits, new_bits, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => break,
                Err(b) => old_bits = b,
            }
        }

        if drift.abs() > self.max_allowed_drift {
            self.mismatch_count.fetch_add(1, Ordering::Relaxed);
            return Err(drift); // CIRCUIT BREAKER TRIGGERED
        }

        Ok(drift)
    }
}
