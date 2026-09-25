//! XXXII: pure/synthetic diagnostics; no network, account, promotion or live feed.
use god_engine_core::{liquidation_feed, order_flow_aggregator::OrderFlowAggregatorEngine};

#[test]
fn flow_ratio_does_not_overflow_for_finite_balanced_volumes() {
    let (ratio, delta) = OrderFlowAggregatorEngine::aggregate_order_flow(f64::MAX, f64::MAX);
    assert_eq!(ratio, 0.5);
    assert_eq!(delta, 0.0);
}

#[test]
fn flow_ratio_does_not_erase_small_nonzero_volume_as_neutral() {
    let (ratio, delta) = OrderFlowAggregatorEngine::aggregate_order_flow(3e-20, 1e-20);
    assert!((ratio - 0.75).abs() < 1e-12);
    assert!(delta > 0.0);
}

#[test]
fn flow_ratio_is_invariant_to_finite_changes_of_volume_units() {
    for scale in [1e-200, 1e-20, 1.0, 1e200, 1e307] {
        let (ratio, _) = OrderFlowAggregatorEngine::aggregate_order_flow(3.0 * scale, scale);
        assert!((ratio - 0.75).abs() < 1e-12, "scale={scale} ratio={ratio}");
    }
}

/// OPEN legacy compatibility buffer; XXXIII operational delivery no longer uses it.
#[test]
fn open_liquidation_breaker_peek_loses_an_event_after_macro_consumption() {
    liquidation_feed::take_pending();
    liquidation_feed::bump(0.95);
    assert_eq!(liquidation_feed::take_pending(), 0.95);
    assert_eq!(liquidation_feed::peek_pending(), 0.0);
    assert_eq!(liquidation_feed::take_pending(), 0.0);
}

#[test]
fn liquidation_normalization_matches_its_documented_formula() {
    assert!((liquidation_feed::severity_from_notional(10_000.0) - 2.0 / 3.0).abs() < 1e-12);
    assert!((liquidation_feed::severity_from_notional(100_000.0) - 5.0 / 6.0).abs() < 1e-12);
    assert_eq!(liquidation_feed::severity_from_notional(1_000_000.0), 1.0);
}

/// OPEN: unsigned-to-signed conversion precedes subtraction and the clamp.
#[test]
fn open_clock_helper_wraps_a_representable_unsigned_timestamp() {
    use god_engine_core::latency_accelerator::LatencyAcceleratorEngine;
    assert_eq!(LatencyAcceleratorEngine::compute_ntp_clock_drift_correction(0, u64::MAX), -1);
    // The mathematical signed difference is positive and would clamp to +5000.
    // This is a domain-limit diagnostic, not a claim about current epoch values.
}
