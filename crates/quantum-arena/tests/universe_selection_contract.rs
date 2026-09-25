use quantum_arena::{
    active_universe::*,
    symbol_registry::{update_registry, SymbolSpec},
    symbols,
};

static TEST_STATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn checked_api_explains_shape_capital_and_forced_evidence_failures() {
    let _guard = setup(vec![spec(0, 0.01, 0.01)]);
    assert_eq!(
        try_calculate_dynamic_universe(13.0, &[1.0], &[], &[], &[]).unwrap_err(),
        UniverseSelectionError::LengthMismatch
    );
    assert_eq!(
        try_calculate_active_universe(0.0, &[1.0], &[0]).unwrap_err(),
        UniverseSelectionError::InvalidCapital
    );
    assert_eq!(
        try_calculate_active_universe(13.0, &[1.0], &[1]).unwrap_err(),
        UniverseSelectionError::InvalidForcedCoin { coin_id: 1 }
    );
    assert_eq!(
        try_calculate_active_universe(13.0, &[f64::NAN], &[0]).unwrap_err(),
        UniverseSelectionError::InvalidForcedEvidence { coin_id: 0 }
    );
}

#[test]
fn selected_vector_and_bitmap_cannot_disagree_on_capacity() {
    let _guard = setup((0..65).map(|i| spec(i, 0.01, 0.01)).collect());
    assert_eq!(
        try_calculate_active_universe(1e6, &[1.0; 65], &[]).unwrap_err(),
        UniverseSelectionError::BitmapCapacityExceeded { coin_id: 64 }
    );
}

#[test]
fn forced_membership_bypasses_admission_but_not_evidence_checks() {
    let mut s = spec(0, 1.0, 1.0);
    s.min_notional = 1e6;
    let _guard = setup(vec![s]);
    let (selected, bitmap) = try_calculate_active_universe(1.0, &[1.0], &[0, 0]).unwrap();
    assert_eq!(selected.len(), 1);
    assert!(is_coin_active(bitmap, 0));
    assert_eq!(
        try_calculate_dynamic_universe(1.0, &[1.0], &[f64::NAN], &[0.0], &[0]).unwrap_err(),
        UniverseSelectionError::InvalidForcedEvidence { coin_id: 0 }
    );
}

#[test]
fn malformed_spec_is_not_a_cheap_asset() {
    let mut s = spec(0, 0.01, 0.01);
    s.tick_size = 0.0;
    let _guard = setup(vec![s]);
    assert!(try_calculate_active_universe(13.0, &[1.0], &[])
        .unwrap()
        .0
        .is_empty());
}

fn setup(specs: Vec<SymbolSpec>) -> std::sync::MutexGuard<'static, ()> {
    let guard = TEST_STATE.lock().unwrap_or_else(|e| e.into_inner());
    symbols::update_dynamic_universe(specs.iter().map(|s| s.symbol.clone()).collect());
    update_registry(specs);
    guard
}
fn spec(i: usize, min_qty: f64, tick_size: f64) -> SymbolSpec {
    SymbolSpec {
        symbol: format!("QXVII{i}USDT"),
        step_size: min_qty,
        tick_size,
        min_qty,
        min_notional: 0.01,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }
}

#[test]
fn invalid_capital_never_invents_an_admission_budget() {
    let _guard = setup(vec![spec(0, 0.01, 0.01)]);
    for capital in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(max_active_coins_for_capital(capital), 0);
        assert!(calculate_active_universe(capital, &[1.0], &[]).0.is_empty());
    }
}

#[test]
fn forced_membership_is_not_a_finite_score_bonus() {
    let _guard = setup(vec![
        spec(0, 1.0, 1.0),
        spec(1, 1e-8, 1e-8),
        spec(2, 1e-8, 1e-8),
    ]);
    let selected = calculate_active_universe(13.0, &[1.0; 3], &[0]).0;
    assert!(selected.iter().any(|x| x.coin_id == 0));
    let selected = calculate_dynamic_universe(13.0, &[1.0; 3], &[1e9; 3], &[0.0, 1e8, 1e8], &[0]).0;
    assert!(selected.iter().any(|x| x.coin_id == 0));
}

#[test]
fn duplicate_forced_ids_do_not_expand_the_budget() {
    let _guard = setup((0..4).map(|i| spec(i, 0.01, 0.01)).collect());
    assert_eq!(
        calculate_active_universe(13.0, &[1.0; 4], &[0, 0, 0, 0])
            .0
            .len(),
        2
    );
}

#[test]
fn minimum_notional_is_not_minimum_quantity_times_price() {
    let mut s = spec(0, 0.001, 0.01);
    s.min_notional = 1000.0;
    let _guard = setup(vec![s]);
    assert!(calculate_active_universe(13.0, &[1.0], &[]).0.is_empty());
}

#[test]
fn per_instrument_leverage_bound_is_respected() {
    let mut s = spec(0, 10.0, 0.01);
    s.max_leverage = 1;
    let _guard = setup(vec![s]);
    assert!(calculate_active_universe(13.0, &[10.0], &[]).0.is_empty());
}

#[test]
fn nonfinite_price_cannot_become_a_candidate() {
    let _guard = setup(vec![spec(0, 0.01, 0.01)]);
    assert!(calculate_active_universe(13.0, &[f64::NAN], &[])
        .0
        .is_empty());
}

#[test]
fn mismatched_dynamic_vectors_abstain_instead_of_panicking() {
    let _guard = setup(vec![spec(0, 0.01, 0.01)]);
    let result =
        std::panic::catch_unwind(|| calculate_dynamic_universe(13.0, &[1.0], &[], &[], &[]));
    assert!(result.is_ok());
    assert!(result.unwrap().0.is_empty());
}

#[test]
fn nonfinite_dynamic_evidence_does_not_reach_sorting() {
    let _guard = setup(vec![spec(0, 0.01, 0.01)]);
    assert!(
        calculate_dynamic_universe(13.0, &[1.0], &[f64::INFINITY], &[1.0], &[])
            .0
            .is_empty()
    );
}
