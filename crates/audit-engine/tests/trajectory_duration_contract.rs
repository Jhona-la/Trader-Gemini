use audit_engine::trajectory_auditor::{
    TrajectoryAuditor, TrajectoryDivergenceReason, TrajectoryStatus,
};

#[test]
fn time_exhaustion_is_reachable_after_four_expected_durations() {
    let mut a = TrajectoryAuditor::new(1);
    a.record_entry(0, true, true, 100.0, 1_000, 0.01, 0.0, 1_000);
    let status = a.evaluate_tick(0, true, 99.6, 0.0, 6_000);
    assert!(
        matches!(status, TrajectoryStatus::Divergent {
        reason: TrajectoryDivergenceReason::TimeExhaustionWithoutProgress, score
    } if score == 5.0),
        "{status:?}"
    );
}

#[test]
fn time_exhaustion_does_not_trigger_at_exactly_four_durations() {
    let mut a = TrajectoryAuditor::new(1);
    a.record_entry(0, true, false, 100.0, 1_000, 0.01, 0.0, 1_000);
    assert!(matches!(
        a.evaluate_tick(0, false, 99.6, 0.0, 5_000),
        TrajectoryStatus::Aligned { .. }
    ));
}
