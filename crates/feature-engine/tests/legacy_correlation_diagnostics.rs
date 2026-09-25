//! OPEN FMT-140: root library has a different estimator, not a crate alias.
#[allow(dead_code)]
#[path = "../../../src/features/ewma.rs"]
pub mod legacy_ewma;
mod features {
    pub use crate::legacy_ewma as ewma;
}
#[allow(dead_code)]
#[path = "../../../src/features/correlation.rs"]
mod legacy_correlation;

fn seeded() -> legacy_correlation::MarketCorrelationHeatmap {
    let mut model = legacy_correlation::MarketCorrelationHeatmap::new(2, 10.0);
    for prices in [[100.0, 100.0], [110.0, 90.0], [99.0, 105.0], [113.0, 100.0]] {
        model.update(&prices);
    }
    model
}

#[test]
fn legacy_invalid_snapshot_still_changes_future_estimates() {
    let mut actual = seeded();
    let mut control = seeded();
    actual.update(&[f64::NAN, 150.0]);
    let difference = (actual.update(&[120.0, 110.0]) - control.update(&[120.0, 110.0])).abs();
    assert!(difference > 1e-6);
}
