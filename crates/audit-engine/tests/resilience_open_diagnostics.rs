//! Passing OPEN tests reproduce remaining limitations, not safety guarantees.
use audit_engine::{
    cybernetic_resilience::{CyberneticResilienceShield as Shield, SystemicHealthMetrics},
    drift_auditor::{DriftAuditError, DriftAuditor, DriftRecovery, TradeResult},
    resilience::ChaosMonkey,
};
use std::num::NonZeroU32;

#[test]
fn open_comprehensive_health_accepts_latency_rejected_by_simple_health() {
    let m = SystemicHealthMetrics {
        active_engines_count: 1,
        websocket_latency_ms: 1500,
        queue_dropped_events: 0,
        error_rate_pct: 0.0,
        consecutive_execution_fails: 0,
    };
    assert!(!Shield::audit_systemic_integrity(1, 1500));
    assert!(Shield::audit_comprehensive_health(&m).0);
}

#[test]
fn open_health_cannot_distinguish_old_total_drops_from_current_incident() {
    let m = SystemicHealthMetrics {
        active_engines_count: 1,
        websocket_latency_ms: 10,
        queue_dropped_events: 101,
        error_rate_pct: 0.0,
        consecutive_execution_fails: 0,
    };
    // No event time, measurement window or traffic denominator exists in the API.
    assert!(!Shield::audit_comprehensive_health(&m).0);
}

#[test]
fn open_invalid_chaos_probability_silently_becomes_no_drop() {
    let mut m = ChaosMonkey::new();
    for invalid in [f64::NAN, -0.5, f64::NEG_INFINITY] {
        m.drop_rate = invalid;
        assert!(m.inject_chaos().is_ok());
    }
}

#[test]
fn open_chaos_probability_above_one_is_not_reported_as_configuration_error() {
    let mut m = ChaosMonkey::new();
    m.drop_rate = 2.0;
    assert_eq!(m.inject_chaos().unwrap_err(), "Network Drop (Simulado)");
}

#[test]
fn open_same_outcome_repeated_ten_times_can_release_recovery() {
    let a = DriftAuditor::default();
    let t = TradeResult {
        symbol_id: 0,
        is_long: true,
        entry_price: 100.0,
        exit_price: 100.0,
        pnl_pct: 0.0,
        timestamp_ms: 1000,
    };
    let mut r = DriftRecovery::new(NonZeroU32::new(10).unwrap());
    r.observe(&Err(DriftAuditError::NonfinitePnl));
    for _ in 0..10 {
        r.observe(&a.audit_execution_checked(&t, &t));
    }
    assert!(!r.is_blocked()); // no outcome ID or deduplication contract yet
}

#[test]
fn open_drift_signed_total_cancels_opposite_deviations() {
    let a = DriftAuditor::new(0.05);
    let mut t = TradeResult {
        symbol_id: 0,
        is_long: true,
        entry_price: 100.0,
        exit_price: 100.0,
        pnl_pct: 0.0,
        timestamp_ms: 1000,
    };
    let base = t.clone();
    t.pnl_pct = 0.04;
    assert!(a.audit_execution_checked(&base, &t).is_ok());
    t.pnl_pct = -0.04;
    assert!(a.audit_execution_checked(&base, &t).is_ok());
    assert_eq!(
        f64::from_bits(a.total_drift_pct.load(std::sync::atomic::Ordering::Relaxed)),
        0.0
    );
}
