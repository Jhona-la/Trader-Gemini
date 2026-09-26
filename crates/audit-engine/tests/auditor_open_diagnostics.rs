use audit_engine::behavioral_auditor::BehavioralAuditorEngine;
use audit_engine::drift_auditor::{DriftAuditor, TradeResult};
use audit_engine::trajectory_auditor::{TrajectoryAuditor, TrajectoryStatus};

fn trade(pnl: f64) -> TradeResult {
    TradeResult {
        symbol_id: 0,
        is_long: true,
        entry_price: 100.0,
        exit_price: 101.0,
        pnl_pct: pnl,
        timestamp_ms: 1_000,
    }
}

#[test]
fn repaired_nan_drift_is_not_reported_as_success() {
    let result = DriftAuditor::default().audit_execution(&trade(f64::NAN), &trade(0.01));
    assert!(result.unwrap_err().is_nan());
}

#[test]
fn repaired_drift_rejects_unrelated_symbols_and_sides() {
    let mut unrelated = trade(0.01);
    unrelated.symbol_id = 8;
    unrelated.is_long = false;
    assert!(DriftAuditor::default().audit_execution(&trade(0.01), &unrelated).is_err());
}

#[test]
fn repaired_nan_drift_limit_cannot_authorize_comparison() {
    assert!(
        DriftAuditor::new(f64::NAN)
            .audit_execution(&trade(-1.0), &trade(1.0))
            .is_err()
    );
}

#[test]
fn open_missing_or_invalid_trajectory_is_perfectly_aligned() {
    let mut a = TrajectoryAuditor::new(1);
    for (symbol, price) in [(0, 100.0), (0, f64::NAN), (1, 100.0)] {
        assert_eq!(
            a.evaluate_tick(symbol, price, 1.0, 1_000),
            TrajectoryStatus::Aligned {
                coherence_score: 1.0
            }
        );
    }
}

#[test]
fn open_old_trajectory_tick_rewinds_time_and_adds_volume() {
    let mut a = TrajectoryAuditor::new(1);
    a.record_entry(0, true, 100.0, 1_000, 0.01, 1_000.0, 10_000);
    a.evaluate_tick(0, 100.0, 5.0, 2_000);
    a.evaluate_tick(0, 100.0, 5.0, 500);
    let t = a.tracks[0].as_ref().unwrap();
    assert_eq!(t.last_update_ms, 500);
    assert_eq!(t.accumulated_volume_usd, 10.0);
}

#[test]
fn open_perfect_trajectory_returns_zero_error_not_one_fidelity() {
    let mut a = TrajectoryAuditor::new(1);
    a.record_entry(0, true, 100.0, 1_000, 0.01, 0.0, 1_000);
    assert_eq!(a.record_exit(0, 101.0, 2_000), Some(0.0));
}

#[test]
fn open_gaussian_likelihood_does_not_accumulate_null_evidence_at_its_mean() {
    let mut a = BehavioralAuditorEngine::default();
    for _ in 0..20 {
        a.audit_observation(0.5);
    }
    // For two distinct normal means with shared finite variance, the exact
    // log-likelihood increment at mu0 is negative, not zero. No variance is supplied here.
    assert_eq!(a.log_likelihood_ratio, 0.0);
}

#[test]
fn open_invalid_behavioral_observation_has_same_return_as_nominal() {
    let mut a = BehavioralAuditorEngine::default();
    assert_eq!(a.audit_observation(f64::NAN), None);
    assert_eq!(a.total_audits, 0);
    assert_eq!(a.audit_observation(0.5), None);
    assert_eq!(a.total_audits, 1);
}

#[test]
fn open_infinite_cusum_parameters_are_accepted() {
    let a = BehavioralAuditorEngine::new(0.5, f64::INFINITY, f64::INFINITY);
    assert!(a.slack_k.is_infinite());
    assert!(a.threshold_h.is_infinite());
}
