use strategy_core::{stat_arb::StatArbEngine, SignalType};

#[test]
fn open_debt_two_sample_studentization_cannot_reach_default_threshold() {
    let mut engine = StatArbEngine::new(2, 1.5);
    for price in [100.0, 10000.0, 0.01, 1e100, 1e-100, 100.0] {
        assert_eq!(engine.update(price, 100.0).signal, SignalType::Flat);
    }
}
