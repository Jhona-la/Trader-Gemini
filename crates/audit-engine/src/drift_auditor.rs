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

/// A comparison failure is not evidence of a healthy execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftAuditError {
    InvalidLimit,
    IncompatibleTrade,
    NonfinitePnl,
    NonfiniteDifference,
    InvalidAccumulator,
    AccumulatorOverflow,
    Exceeded { drift: f64, limit: f64 },
}

/// Owns ONLY the drift entry veto; cannot clear a shared kill switch.
/// The count is operational policy, not confidence. Callers must supply distinct
/// matched outcomes: symbol/side agreement alone does not establish identity.
pub struct DriftRecovery {
    required_clean: std::num::NonZeroU32,
    clean: u32,
    blocked: bool,
}

impl DriftRecovery {
    pub fn new(required_clean: std::num::NonZeroU32) -> Self {
        Self { required_clean, clean: 0, blocked: false }
    }

    pub fn observe(&mut self, result: &Result<f64, DriftAuditError>) {
        if !matches!(result, Ok(drift) if drift.is_finite()) {
            self.blocked = true;
            self.clean = 0;
        } else if self.blocked {
            self.clean = self.clean.saturating_add(1);
            if self.clean >= self.required_clean.get() {
                self.blocked = false;
                self.clean = 0;
            }
        }
    }

    pub fn is_blocked(&self) -> bool { self.blocked }
    pub fn clean_count(&self) -> u32 { self.clean }
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

    /// Legacy adapter. Invalid evidence becomes Err(NaN), never Ok(NaN).
    /// New consumers use the typed API to distinguish invalidity from excess drift.
    pub fn audit_execution(&self, real: &TradeResult, shadow: &TradeResult) -> Result<f64, f64> {
        self.audit_execution_checked(real, shadow).map_err(|error| match error {
            DriftAuditError::Exceeded { drift, .. } => drift,
            _ => f64::NAN,
        })
    }

    /// Compares supplied fractional PnLs for compatible symbols/sides, not prices.
    /// Prices/timestamps are descriptive, not validated. No trade ID, generation,
    /// deduplication or independent shadow is established by this API.
    /// The signed total is telemetry; the limit applies to EACH observation.
    /// This method never activates or releases a trading interlock.
    pub fn audit_execution_checked(&self, real: &TradeResult, shadow: &TradeResult) -> Result<f64, DriftAuditError> {
        if !self.max_allowed_drift.is_finite() || self.max_allowed_drift < 0.0 {
            return Err(DriftAuditError::InvalidLimit);
        }
        if real.symbol_id != shadow.symbol_id || real.is_long != shadow.is_long {
            return Err(DriftAuditError::IncompatibleTrade);
        }
        if !real.pnl_pct.is_finite() || !shadow.pnl_pct.is_finite() {
            return Err(DriftAuditError::NonfinitePnl);
        }
        let drift = shadow.pnl_pct - real.pnl_pct;
        if !drift.is_finite() {
            return Err(DriftAuditError::NonfiniteDifference);
        }

        let mut current_bits = self.total_drift_pct.load(Ordering::Acquire);
        loop {
            let current_drift = f64::from_bits(current_bits);
            if !current_drift.is_finite() {
                return Err(DriftAuditError::InvalidAccumulator);
            }
            let new_drift = current_drift + drift;
            if !new_drift.is_finite() {
                return Err(DriftAuditError::AccumulatorOverflow);
            }
            match self.total_drift_pct.compare_exchange_weak(
                current_bits, new_drift.to_bits(), Ordering::AcqRel, Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current_bits = actual,
            }
        }

        if drift.abs() > self.max_allowed_drift {
            self.mismatch_count.fetch_add(1, Ordering::Relaxed);
            return Err(DriftAuditError::Exceeded { drift, limit: self.max_allowed_drift });
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
        let res = auditor.audit_execution_checked(&real, &shadow);
        assert_eq!(res, Err(DriftAuditError::NonfinitePnl));
    }
}
