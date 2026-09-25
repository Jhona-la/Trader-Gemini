use quantum_arena::symbol_registry::SymbolSpec;

fn spec(step: f64) -> SymbolSpec {
    SymbolSpec {
        symbol: "AUDITUSDT".into(),
        step_size: step,
        tick_size: 0.01,
        min_qty: step,
        min_notional: 0.0,
        max_leverage: 1,
        maker_fee: 0.0,
        taker_fee: 0.0,
        is_shadow: false,
    }
}

#[test]
fn nonfinite_order_inputs_are_errors() {
    let s = spec(0.05);
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(s.validate_order(x, 100.0).is_err(), "quantity={x}");
        assert!(s.validate_order(1.0, x).is_err(), "price={x}");
    }
}

#[test]
fn invalid_step_is_not_replaced_with_one() {
    for x in [f64::NAN, f64::INFINITY, 0.0, -0.01] {
        assert!(spec(x).validate_order(10.0, 100.0).is_err(), "step={x}");
    }
}

#[test]
fn invalid_minima_are_errors() {
    for x in [f64::NAN, f64::INFINITY, -1.0] {
        let mut s = spec(0.05);
        s.min_qty = x;
        assert!(s.validate_order(10.0, 100.0).is_err());
        s.min_qty = 0.05;
        s.min_notional = x;
        assert!(s.validate_order(10.0, 100.0).is_err());
    }
}

#[test]
fn non_power_of_ten_step_does_not_increase_quantity() {
    assert_eq!(spec(0.05).validate_order(0.06, 100.0), Ok(0.05));
}

#[test]
fn non_power_of_ten_step_keeps_integer_lot_count() {
    let q = spec(0.025).validate_order(0.09, 100.0).unwrap();
    assert_eq!(q, 3.0 * 0.025);
    assert!(q <= 0.09);
}

#[test]
fn below_one_lot_and_overflow_are_errors() {
    let mut s = spec(0.05);
    s.min_qty = 0.0;
    assert!(s.validate_order(0.01, 100.0).is_err());
    assert!(spec(1.0).validate_order(2.0, f64::MAX).is_err());
    assert!(spec(f64::MIN_POSITIVE)
        .validate_order(f64::MAX, 1.0)
        .is_err());
}

#[test]
fn lot_count_must_have_unit_resolution() {
    assert!(spec(1.0).validate_order(2_f64.powi(53), 1.0).is_err());
}

#[test]
fn conservative_projection_and_minimum_notional_hold() {
    for step in [0.025, 0.05, 0.1, 0.125, 1.0, 2.5] {
        for n in 1..1000 {
            let raw = (f64::from(n) + 0.375) * step;
            let q = spec(step).validate_order(raw, 100.0).unwrap();
            assert!(q.is_finite() && q > 0.0 && q <= raw);
            assert_eq!(q, step + f64::from(n - 1) * step);
        }
    }
    let mut s = spec(0.05);
    s.min_notional = 10.0;
    assert!(s.validate_order(0.06, 100.0).is_err());
    assert_eq!(s.validate_order(0.125, 100.0), Ok(0.1));
}

#[test]
fn minimum_quantity_is_the_lattice_origin() {
    let mut s = spec(0.025);
    s.min_qty = 0.01;
    assert_eq!(s.validate_order(0.062, 100.0), Ok(0.01 + 2.0 * 0.025));
    assert!(s.validate_order(0.009, 100.0).is_err());
}

#[test]
fn binary_boundary_may_underfill_but_never_upsizes() {
    let q = spec(0.1).validate_order(0.3, 100.0).unwrap();
    assert!(q <= 0.3);
    // Explicit known limitation until quantities/filters retain decimal strings.
    assert_eq!(q, 0.2);
}
