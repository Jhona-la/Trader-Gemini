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
    blocked_since_ms: u64,
}

impl DriftRecovery {
    pub fn new(required_clean: std::num::NonZeroU32) -> Self {
        Self { required_clean, clean: 0, blocked: false, blocked_since_ms: 0 }
    }

    pub fn observe(&mut self, result: &Result<f64, DriftAuditError>) {
        self.observe_at(result, 0);
    }

    pub fn observe_at(&mut self, result: &Result<f64, DriftAuditError>, timestamp_ms: u64) {
        if !matches!(result, Ok(drift) if drift.is_finite()) {
            self.blocked = true;
            self.clean = 0;
            self.blocked_since_ms = timestamp_ms;
        } else if self.blocked {
            self.clean = self.clean.saturating_add(1);
            if self.clean >= self.required_clean.get() {
                self.blocked = false;
                self.clean = 0;
                self.blocked_since_ms = 0;
            }
        }
    }

    pub fn is_blocked(&self) -> bool { self.blocked }
    pub fn clean_count(&self) -> u32 { self.clean }
    pub fn blocked_since_ms(&self) -> u64 { self.blocked_since_ms }

    /// CL-xxx (F1-062, Ola Ω171): Prevención de estado absorbente cuando la cartera está inactiva.
    /// Si el veto de deriva está activo pero NO hay posiciones abiertas (`open_positions == 0`),
    /// el sistema no puede acumular cierres limpios porque las entradas están vetadas.
    /// Si han transcurrido al menos `cooldown_ms` desde el bloqueo sin nuevos incidentes,
    /// libera el veto y devuelve `true`.
    pub fn try_idle_recovery(&mut self, now_ms: u64, cooldown_ms: u64, open_positions: usize) -> bool {
        if self.blocked && open_positions == 0 {
            let inicio = if self.blocked_since_ms == 0 {
                self.blocked_since_ms = now_ms;
                now_ms
            } else {
                self.blocked_since_ms
            };
            if now_ms.saturating_sub(inicio) >= cooldown_ms {
                self.blocked = false;
                self.clean = 0;
                self.blocked_since_ms = 0;
                return true;
            }
        }
        false
    }
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

    /// CL-xxx (F1-062, Ola Ω171): Centinela contable directo en O(1) con cero asignaciones.
    /// Sustituye la sombra sintética `shadow = real * 0.95`.
    /// Verifica que el retorno sea finito y que no exceda el 100% no apalancado (|pnl_pct| <= 1.0).
    /// Devuelve Ok(-0.05 * real_pnl_pct) o Err(DriftAuditError).
    #[inline]
    pub fn centinela_contable(&self, real_pnl_pct: f64) -> Result<f64, DriftAuditError> {
        if !self.max_allowed_drift.is_finite() || self.max_allowed_drift < 0.0 {
            return Err(DriftAuditError::InvalidLimit);
        }
        if !real_pnl_pct.is_finite() {
            return Err(DriftAuditError::NonfinitePnl);
        }
        let drift = real_pnl_pct * 0.95 - real_pnl_pct;
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

    /// CL-xxx (F1-062, Ola Ω171): Audita la divergencia real de cartera entre el universo de control (ShadowForest) y la ejecución viva.
    /// Retorna Ok(divergencia) si |divergencia| <= max_divergence, o Err si no es finita o excede el límite.
    pub fn audit_control_divergence(&self, control_pct: f64, real_pct: f64, max_divergence: f64) -> Result<f64, DriftAuditError> {
        if !max_divergence.is_finite() || max_divergence < 0.0 {
            return Err(DriftAuditError::InvalidLimit);
        }
        if !control_pct.is_finite() || !real_pct.is_finite() {
            return Err(DriftAuditError::NonfinitePnl);
        }
        let divergencia = control_pct - real_pct;
        if !divergencia.is_finite() {
            return Err(DriftAuditError::NonfiniteDifference);
        }
        if divergencia.abs() > max_divergence {
            return Err(DriftAuditError::Exceeded { drift: divergencia, limit: max_divergence });
        }
        Ok(divergencia)
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

    #[test]
    fn test_centinela_contable_matches_synthetic_shadow_behavior() {
        let auditor = DriftAuditor::new(0.05);
        // Retorno normal dentro de bounds: 2%
        let res = auditor.centinela_contable(0.02);
        assert!(res.is_ok());
        assert!((res.unwrap() - (-0.001)).abs() < 1e-7);

        // Retorno no finito
        assert_eq!(auditor.centinela_contable(f64::NAN), Err(DriftAuditError::NonfinitePnl));
        assert_eq!(auditor.centinela_contable(f64::INFINITY), Err(DriftAuditError::NonfinitePnl));

        // Retorno corrupto > 100% no apalancado (|pnl_pct| > 1.0 => |drift| > 0.05)
        let res_corrupt = auditor.centinela_contable(1.20);
        assert!(matches!(res_corrupt, Err(DriftAuditError::Exceeded { .. })));
    }

    #[test]
    fn test_try_idle_recovery_releases_absorbing_state() {
        let mut r = DriftRecovery::new(std::num::NonZeroU32::new(10).unwrap());
        // Simular fallo contable en t = 1000ms
        r.observe_at(&Err(DriftAuditError::NonfinitePnl), 1_000);
        assert!(r.is_blocked());
        assert_eq!(r.blocked_since_ms(), 1_000);

        // Si hay posiciones abiertas, NO se libera por inactividad
        assert!(!r.try_idle_recovery(100_000, 60_000, 1));
        assert!(r.is_blocked());

        // Con 0 posiciones abiertas pero antes del timeout de 60s, NO se libera
        assert!(!r.try_idle_recovery(30_000, 60_000, 0));
        assert!(r.is_blocked());

        // Con 0 posiciones abiertas y 60s transcurridos (t = 61_000), SE LIBERA
        assert!(r.try_idle_recovery(61_000, 60_000, 0));
        assert!(!r.is_blocked());
        assert_eq!(r.blocked_since_ms(), 0);
    }

    #[test]
    fn test_audit_control_divergence() {
        let auditor = DriftAuditor::new(0.05);
        // Divergencia aceptable (2% control vs real)
        assert!(auditor.audit_control_divergence(0.10, 0.08, 0.05).is_ok());

        // Divergencia excesiva (8% control vs real, límite 5%)
        let err = auditor.audit_control_divergence(0.12, 0.04, 0.05);
        assert!(matches!(err, Err(DriftAuditError::Exceeded { .. })));

        // PnL no finito
        assert_eq!(auditor.audit_control_divergence(f64::NAN, 0.05, 0.05), Err(DriftAuditError::NonfinitePnl));
    }
}
