//! LG5 descriptive-return contracts: known arithmetic, causal availability and
//! bounded identity. These do not certify prediction or executable profits.
use god_engine_core::calibration::WINDOW;
use god_engine_core::net_payoff::{EntryPayoffEvidence, NetPayoffModel, WATERMARK_COUNT};
use quantum_arena::position::MAX_SPECTRAL_SLOTS;
use strategy_core::{NetPayoffSource, SignalType};

fn entry(i: u64) -> EntryPayoffEvidence {
    EntryPayoffEvidence {
        slot: 0,
        generation: i + 1,
        is_long: true,
        entry_time_ms: 100 + i * 10,
        score: 0.6,
        tau_ms: 60_000,
        entry_price: 100.0,
        initial_tp_price: 102.25,
        initial_sl_price: 99.0,
        source: NetPayoffSource::LocalSimulation,
    }
}

fn record(model: &mut NetPayoffModel, e: EntryPayoffEvidence, r: f64) {
    assert!(model.observe(
        e,
        r * 100.0,
        100.0,
        1.0,
        e.entry_time_ms + 1,
        e.entry_time_ms + 1
    ));
}

fn estimate(model: &NetPayoffModel) -> strategy_core::NetPayoffEstimate {
    model
        .estimate(
            0,
            SignalType::Long,
            NetPayoffSource::LocalSimulation,
            1_000_000,
        )
        .unwrap()
}

#[test]
fn equal_win_rates_with_different_magnitudes_have_different_net_expectations() {
    let mut small_wins = NetPayoffModel::new(0);
    let mut large_wins = NetPayoffModel::new(0);
    for (i, (small, large)) in [(0.0002, 0.02), (-0.01, -0.01)].into_iter().enumerate() {
        record(&mut small_wins, entry(i as u64), small);
        record(&mut large_wins, entry(i as u64), large);
    }
    assert!((estimate(&small_wins).mean_net_return_pct + 0.0049).abs() < 1e-14);
    assert!((estimate(&large_wins).mean_net_return_pct - 0.005).abs() < 1e-14);
    assert!((estimate(&small_wins).standard_deviation - 0.0051).abs() < 1e-14);
}

#[test]
fn fees_are_already_net_and_margin_cannot_change_the_entry_notional_unit() {
    let mut model = NetPayoffModel::new(0);
    let gross_pnl = 2.0;
    let (entry_fee, exit_fee) = (0.2, 0.3);
    assert!(model.observe(
        entry(0),
        gross_pnl - entry_fee - exit_fee,
        100.0,
        1.0,
        105,
        105
    ));
    // Same economic return at tenfold fill size. Equal trade weights, not
    // weights proportional to dollars, exposure, collateral or leverage.
    assert!(model.observe(
        entry(1),
        10.0 * (gross_pnl - entry_fee - exit_fee),
        100.0,
        10.0,
        115,
        115
    ));
    let e = estimate(&model);
    assert!((e.mean_net_return_pct - 0.015).abs() < 1e-14);
    for margin in [1.0, 10.0, 50.0] {
        let collateral_roi = (gross_pnl - entry_fee - exit_fee) / margin;
        assert!((e.mean_net_return_pct - collateral_roi).abs() > 0.001);
    }
    let mut unequal = NetPayoffModel::new(0);
    assert!(unequal.observe(entry(0), 1.0, 100.0, 1.0, 105, 105));
    assert!(unequal.observe(entry(1), -20.0, 100.0, 10.0, 115, 115));
    assert!((estimate(&unequal).mean_net_return_pct + 0.005).abs() < 1e-14);
}

#[test]
fn partial_fill_uses_closed_quantity_and_does_not_use_initial_stop_as_unit() {
    let mut model = NetPayoffModel::new(0);
    let mut e = entry(0);
    e.initial_sl_price = 95.0;
    assert!(model.observe(e, 0.5, 101.0, 0.5, 105, 105));
    assert!((estimate(&model).mean_net_return_pct - 0.5 / 50.5).abs() < 1e-14);
}

#[test]
fn known_zero_is_not_unknown_and_one_observation_is_not_a_confidence_bound() {
    let mut model = NetPayoffModel::new(0);
    assert!(
        model
            .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 200)
            .is_none()
    );
    record(&mut model, entry(0), 0.0);
    let e = estimate(&model);
    assert_eq!(
        (e.mean_net_return_pct, e.standard_deviation, e.sample_count),
        (0.0, 0.0, 1)
    );
}

#[test]
fn coin_side_and_provenance_are_disjoint_but_score_tau_geometry_are_pooled() {
    let mut model = NetPayoffModel::new(0);
    record(&mut model, entry(0), 0.01);
    let mut long_other = entry(1);
    long_other.score = 0.9;
    long_other.tau_ms = 3_600_001;
    long_other.initial_tp_price = 120.0;
    long_other.initial_sl_price = 80.0;
    record(&mut model, long_other, 0.03);
    let mut short = entry(2);
    short.is_long = false;
    short.initial_tp_price = 99.0;
    short.initial_sl_price = 102.0;
    record(&mut model, short, -0.02);
    let mut exchange = entry(0);
    exchange.source = NetPayoffSource::ExchangeLocalEstimate;
    record(&mut model, exchange, 0.04);
    assert!((estimate(&model).mean_net_return_pct - 0.02).abs() < 1e-14);
    assert_eq!(estimate(&model).sample_count, 2);
    assert_eq!(
        model
            .estimate(0, SignalType::Short, NetPayoffSource::LocalSimulation, 1000)
            .unwrap()
            .mean_net_return_pct,
        -0.02
    );
    assert_eq!(
        model
            .estimate(
                0,
                SignalType::Long,
                NetPayoffSource::ExchangeLocalEstimate,
                1000
            )
            .unwrap()
            .mean_net_return_pct,
        0.04
    );
    assert!(
        model
            .estimate(1, SignalType::Long, NetPayoffSource::LocalSimulation, 1000)
            .is_none()
    );
    assert!(
        model
            .estimate(0, SignalType::Flat, NetPayoffSource::LocalSimulation, 1000)
            .is_none()
    );
}

#[test]
fn availability_clock_prevents_retroactive_queries_and_late_other_slot_is_valid() {
    let mut model = NetPayoffModel::new(0);
    assert!(model.observe(entry(0), 1.0, 100.0, 1.0, 105, 200));
    assert!(
        model
            .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 199)
            .is_none()
    );
    assert!(
        model
            .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 200)
            .is_some()
    );
    let mut late = entry(0);
    late.slot = 1;
    late.entry_time_ms = 50;
    assert!(model.observe(late, -1.0, 100.0, 1.0, 80, 250));
    let e = model
        .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 250)
        .unwrap();
    assert_eq!(
        (e.sample_start_ms, e.sample_end_ms, e.sample_count),
        (80, 105, 2)
    );
    assert_eq!(e.mean_net_return_pct, 0.0);
    assert!(
        model
            .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 249)
            .is_none()
    );
}

#[test]
fn duplicate_or_retired_slot_identity_cannot_change_data_clock_or_watermarks() {
    let mut model = NetPayoffModel::new(0);
    record(&mut model, entry(0), 0.01);
    let before = model.snapshot();
    assert!(!model.observe(entry(0), 50.0, 100.0, 1.0, 110, 9_000));
    assert_eq!(model.snapshot(), before);
    let mut simultaneous = entry(0);
    simultaneous.slot = 1;
    record(&mut model, simultaneous, 0.02);
    assert_eq!(estimate(&model).sample_count, 2);
    for i in 1..=WINDOW as u64 {
        record(&mut model, entry(i), 0.001);
    }
    let before = model.snapshot();
    assert!(!before.rows.iter().any(|row| row.entry == simultaneous));
    assert!(!model.observe(simultaneous, 9.0, 100.0, 1.0, 120, 9_000));
    assert!(!model.observe(entry(0), 9.0, 100.0, 1.0, 120, 9_000));
    assert_eq!(model.snapshot(), before);
}

#[test]
fn invalid_geometry_numbers_and_time_reject_without_partial_mutation() {
    let mut model = NetPayoffModel::new(0);
    record(&mut model, entry(0), 0.01);
    let before = model.snapshot();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!model.observe(entry(1), bad, 100.0, 1.0, 115, 115));
        let mut e = entry(1);
        e.score = bad;
        assert!(!model.observe(e, 1.0, 100.0, 1.0, 115, 115));
    }
    for (price, qty) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (100.0, 0.0),
        (100.0, -1.0),
        (f64::MAX, 2.0),
    ] {
        assert!(!model.observe(entry(1), 1.0, price, qty, 115, 115));
    }
    assert!(!model.observe(entry(1), 1.0, 100.0, 1.0, 109, 120));
    assert!(!model.observe(entry(1), 1.0, 100.0, 1.0, 115, 114));
    for mutate in [
        |e: &mut EntryPayoffEvidence| e.slot = MAX_SPECTRAL_SLOTS,
        |e: &mut EntryPayoffEvidence| e.tau_ms = 0,
        |e: &mut EntryPayoffEvidence| e.initial_sl_price = 101.0,
        |e: &mut EntryPayoffEvidence| e.initial_tp_price = 99.0,
        |e: &mut EntryPayoffEvidence| e.score = 1.1,
    ] {
        let mut e = entry(1);
        mutate(&mut e);
        assert!(!model.observe(e, 1.0, 100.0, 1.0, 115, 115));
    }
    assert_eq!(model.snapshot(), before);
}

#[test]
fn retained_uniform_window_has_an_independent_arithmetic_oracle() {
    let mut model = NetPayoffModel::new(0);
    for i in 0..600u64 {
        record(&mut model, entry(i), if i % 3 == 0 { 0.03 } else { -0.01 });
    }
    let retained: Vec<_> = (600 - WINDOW..600)
        .map(|i| if i % 3 == 0 { 0.03 } else { -0.01 })
        .collect();
    let mean = retained.iter().sum::<f64>() / WINDOW as f64;
    let sd = (retained.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / WINDOW as f64).sqrt();
    let e = estimate(&model);
    assert_eq!(e.sample_count, WINDOW);
    assert!((e.mean_net_return_pct - mean).abs() < 1e-14);
    assert!((e.standard_deviation - sd).abs() < 1e-14);
}

#[test]
fn large_finite_returns_keep_their_magnitude_without_clipping_or_sum_overflow() {
    let mut model = NetPayoffModel::new(0);
    assert!(model.observe(entry(0), 1e300, 1.0, 1.0, 105, 105));
    assert!(model.observe(entry(1), -1e300, 1.0, 1.0, 115, 115));
    let e = estimate(&model);
    assert_eq!(e.mean_net_return_pct, 0.0);
    assert_eq!(e.standard_deviation, 1e300);
}

#[test]
fn restore_is_atomic_for_bad_context_clock_length_and_duplicate_rows() {
    let mut model = NetPayoffModel::new(0);
    record(&mut model, entry(0), 0.01);
    let before = model.snapshot();
    let mut bad_states = Vec::new();
    let mut bad = before.clone();
    bad.rows.push(bad.rows[0]);
    bad_states.push(bad);
    let mut bad = before.clone();
    bad.clock_ms = Some(100);
    bad_states.push(bad);
    let mut bad = before.clone();
    bad.rows[0].entry.slot = MAX_SPECTRAL_SLOTS;
    bad_states.push(bad);
    let mut bad = before.clone();
    bad.watermarks.resize(WATERMARK_COUNT + 1, None);
    bad_states.push(bad);
    let mut bad = before.clone();
    bad.rows.resize(WINDOW + 1, bad.rows[0]);
    bad_states.push(bad);
    let mut bad = before.clone();
    bad.rows[0].net_pnl_usd = f64::NAN;
    bad_states.push(bad);
    let mut bad = before.clone();
    bad.watermarks[0] = None;
    bad_states.push(bad);
    for bad in bad_states {
        assert!(model.restore(&bad).is_err());
        assert_eq!(model.snapshot(), before);
    }
}
