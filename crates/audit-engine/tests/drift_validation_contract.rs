use audit_engine::drift_auditor::{DriftAuditor, TradeResult};
use std::sync::atomic::Ordering;

fn trade(pnl_pct: f64) -> TradeResult {
    TradeResult {
        symbol_id: 0,
        is_long: true,
        entry_price: 100.0,
        exit_price: 101.0,
        pnl_pct,
        timestamp_ms: 1000,
    }
}

#[test]
fn nonfinite_pnl_never_counts_as_clean_evidence() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = DriftAuditor::default();
        assert!(a.audit_execution(&trade(value), &trade(0.0)).is_err());
        assert!(a.audit_execution(&trade(0.0), &trade(value)).is_err());
        assert_eq!(a.total_drift_pct.load(Ordering::Relaxed), 0.0_f64.to_bits());
    }
}

#[test]
fn nonfinite_or_negative_limit_never_authorizes_comparison() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        assert!(DriftAuditor::new(value)
            .audit_execution(&trade(0.0), &trade(0.0))
            .is_err());
    }
}

#[test]
fn incompatible_symbol_or_side_is_not_clean_evidence() {
    let a = DriftAuditor::default();
    let mut other = trade(0.0);
    other.symbol_id = 1;
    assert!(a.audit_execution(&trade(0.0), &other).is_err());
    other.symbol_id = 0;
    other.is_long = false;
    assert!(a.audit_execution(&trade(0.0), &other).is_err());
}

#[test]
fn overflowed_accumulator_is_explicit_failure_without_mutation() {
    let a = DriftAuditor::new(f64::MAX);
    a.total_drift_pct
        .store(f64::MAX.to_bits(), Ordering::Relaxed);
    assert!(a.audit_execution(&trade(0.0), &trade(f64::MAX)).is_err());
    assert_eq!(
        a.total_drift_pct.load(Ordering::Relaxed),
        f64::MAX.to_bits()
    );
}

#[test]
fn zero_limit_accepts_exact_match_and_boundary_is_inclusive() {
    assert_eq!(
        DriftAuditor::new(0.0).audit_execution(&trade(1.0), &trade(1.0)),
        Ok(0.0)
    );
    assert_eq!(
        DriftAuditor::new(0.05).audit_execution(&trade(0.0), &trade(0.05)),
        Ok(0.05)
    );
}

#[test]
fn typed_errors_distinguish_invalid_difference_and_accumulator() {
    use audit_engine::drift_auditor::DriftAuditError as E;
    let a = DriftAuditor::default();
    assert_eq!(
        a.audit_execution_checked(&trade(-f64::MAX), &trade(f64::MAX)),
        Err(E::NonfiniteDifference)
    );
    a.total_drift_pct
        .store(f64::NAN.to_bits(), Ordering::Relaxed);
    assert_eq!(
        a.audit_execution_checked(&trade(0.0), &trade(0.0)),
        Err(E::InvalidAccumulator)
    );
    assert!(f64::from_bits(a.total_drift_pct.load(Ordering::Relaxed)).is_nan());
}

#[test]
fn exceeded_limit_retains_finite_drift_and_counts_only_real_mismatches() {
    use audit_engine::drift_auditor::DriftAuditError as E;
    let a = DriftAuditor::new(0.05);
    assert_eq!(
        a.audit_execution_checked(&trade(0.0), &trade(0.1)),
        Err(E::Exceeded {
            drift: 0.1,
            limit: 0.05
        })
    );
    assert_eq!(a.mismatch_count.load(Ordering::Relaxed), 1);
    assert_eq!(
        a.audit_execution_checked(&trade(f64::NAN), &trade(0.0)),
        Err(E::NonfinitePnl)
    );
    assert_eq!(a.mismatch_count.load(Ordering::Relaxed), 1);
}
