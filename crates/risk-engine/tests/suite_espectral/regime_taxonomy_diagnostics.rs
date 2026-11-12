//! These tests characterize OPEN FMT-091, not a continuous-state migration.
use risk_engine::regime::{MarketRegime, RegimeDetector};

#[test]
fn legacy_unknown_code_is_indistinguishable_from_range() {
    assert_eq!(MarketRegime::from(255), MarketRegime::Range);
}

#[test]
fn invalid_observation_retains_a_regime_without_an_age_or_quality_flag() {
    let mut detector = RegimeDetector::new(0.6, 0.02);
    assert_eq!(detector.update(0.9, -0.03), MarketRegime::Crash);
    assert_eq!(detector.update(f64::NAN, 0.03), MarketRegime::Crash);
}

#[test]
fn threshold_and_invalid_correlation_are_not_probabilistic_inference() {
    let mut detector = RegimeDetector::new(0.6, 0.02);
    assert_eq!(detector.update(0.6, 0.03), MarketRegime::Range);
    assert_eq!(detector.update(0.6 + 1e-12, 0.03), MarketRegime::BullRun);
    assert_eq!(detector.update(2.0, 0.03), MarketRegime::BullRun);
}
