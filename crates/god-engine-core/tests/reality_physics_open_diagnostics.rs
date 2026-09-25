use god_engine_core::reality_physics::RealityPhysics;

#[test]
fn open_negative_latency_hides_impact_behind_finite_floor() {
    let p = RealityPhysics::default();
    let valid = p
        .calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, 0.0)
        .0;
    let invalid = p
        .calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, -1.0)
        .0;
    assert!(invalid < valid); // sqrt(-1) -> NaN; max(floor) hides missing/invalid latency.
}

#[test]
fn open_finite_extreme_base_price_can_produce_infinite_execution_price() {
    let p = RealityPhysics::default();
    assert!(
        p.calculate_market_entry(f64::MAX, true, 1e6, 0.001, 0.0001, 15.0)
            .0
            .is_infinite()
    );
}

#[test]
fn open_maker_estimate_has_no_queue_fill_or_latency_condition() {
    let p = RealityPhysics::default();
    assert_eq!(p.calculate_maker_entry(100.0, true, 1000.0), (100.0, 0.2));
    assert_eq!(p.calculate_maker_entry(100.0, false, 1000.0), (100.0, 0.2));
}

#[test]
fn open_latency_struct_field_is_not_used_by_the_explicit_latency_argument() {
    let a = RealityPhysics::default();
    let mut b = a.clone();
    b.latency_penalty_ms = 10000;
    assert_eq!(
        a.calculate_market_entry(100.0, true, 1000.0, 0.001, 0.0001, 15.0),
        b.calculate_market_entry(100.0, true, 1000.0, 0.001, 0.0001, 15.0)
    );
}
