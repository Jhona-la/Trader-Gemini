//! OPEN diagnostics plus FMT-175 regression checks repaired in round XVIII.
//! A passing OPEN diagnostic is not remediation.
use quantum_arena::{
    symbol_registry::{self, SymbolSpec},
    symbols,
};

fn spec(symbol: &str) -> SymbolSpec {
    SymbolSpec {
        symbol: symbol.into(),
        step_size: 0.05,
        tick_size: 0.01,
        min_qty: 0.05,
        min_notional: 0.01,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }
}

#[test]
fn open_debt_universe_rotation_rebinds_positions_while_cached_host_ids_do_not() {
    symbol_registry::update_registry(vec![
        spec("XVIIAAAUSDT"),
        spec("XVIIBBBUSDT"),
        spec("XVIICCCUSDT"),
    ]);
    symbols::update_dynamic_universe(vec!["XVIIAAAUSDT".into(), "XVIIBBBUSDT".into()]);
    let startup_ids =
        std::collections::HashMap::from([("XVIIAAAUSDT", 0usize), ("XVIIBBBUSDT", 1usize)]);
    // Production host gives this cache precedence over dynamic try_index.
    symbols::update_dynamic_universe(vec!["XVIICCCUSDT".into(), "XVIIAAAUSDT".into()]);
    assert_eq!(startup_ids["XVIIAAAUSDT"], 0);
    assert_eq!(symbol_registry::try_index("XVIIAAAUSDT"), Some(1));
    assert_eq!(
        symbol_registry::try_symbol(startup_ids["XVIIAAAUSDT"]).as_deref(),
        Some("XVIICCCUSDT")
    );
}

#[test]
fn regression_order_validation_rejects_nonfinite_numbers() {
    let s = spec("XVIIAAAUSDT");
    assert!(s.validate_order(1.0, f64::NAN).is_err());
    assert!(s.validate_order(f64::NAN, 100.0).is_err());
}

#[test]
fn regression_rounding_preserves_lot_count_and_never_increases_quantity() {
    let s = spec("XVIIAAAUSDT");
    let q = s.validate_order(0.06, 100.0).unwrap();
    // XVIII removes decimal-count rounding, which previously gave 0.1.
    assert_eq!(q, 0.05);
    let mut s = s;
    s.step_size = 0.025;
    s.min_qty = 0.025;
    let q = s.validate_order(0.09, 100.0).unwrap();
    assert_eq!(q, 3.0 * 0.025);
}

#[test]
fn open_debt_ranker_leverage_penalty_decreases_after_the_return_band_peak() {
    // Algebraic diagnostic of SymbolRankerEngine: max_x x*exp(-x/15)=15/e<5.52.
    // Dividing by5 can exceed1 near x15, but it decreases again for larger x.
    let penalty = |x: f64| ((x * (-x / 15.0).exp()) / 5.0).clamp(1.0, 5.0);
    assert!(penalty(15.0) > 1.0);
    assert_eq!(penalty(100.0), 1.0);
    assert!(penalty(100.0) < penalty(15.0));
}
