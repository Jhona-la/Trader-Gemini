use super::*;

// External ledger of the forecasts actually emitted. No re-estimation at maturity.
fn issued_forecast_scores() -> (ForecastScore, f64, f64) {
    let mut tape = SpectralTape::new(&[100.0]);
    let mut f = HorizonForecaster::new(&tape, ForecastTarget::Volume, 100.0);
    f.warmup = 0;
    let mut issued: VecDeque<(u64, [f64; N_OBS], f64, f64)> = VecDeque::new();
    let (mut expected_model, mut expected_clim, mut scored) = (0.0, 0.0, 0);
    let (mut clim_sum, mut clim_n) = (0.0, 0.0);
    for i in 0..40u64 {
        let ts = i * 25;
        while issued.front().map(|s| s.0 + 100 < ts).unwrap_or(false) {
            let (_, totals0, pred, clim) = issued.pop_front().unwrap();
            let y =
                (tape.totals()[OBS_NOTIONAL] - totals0[OBS_NOTIONAL] + tape.notional_floor()).ln();
            expected_model = expected_model * f.clim_lambda + (y - pred).powi(2);
            expected_clim = expected_clim * f.clim_lambda + (y - clim).powi(2);
            scored += 1;
            clim_sum = clim_sum * f.clim_lambda + y;
            clim_n = clim_n * f.clim_lambda + 1.0;
        }
        f.before_trade(&tape, ts);
        tape.on_trade(ts, 100.0, if i % 7 < 3 { 1.0 } else { 9.0 }, true);
        f.after_trade(&tape, ts);
        if let Some((_, persist)) = f.features(&tape, ts) {
            let pred = f.last_prediction().unwrap();
            let clim = if clim_n > 0.0 {
                clim_sum / clim_n
            } else {
                persist
            };
            issued.push_back((ts, tape.totals(), pred, clim));
        }
    }
    assert!(scored > 20);
    assert_eq!(f.score.n, scored);
    (f.score, expected_model, expected_clim)
}

#[test]
fn si_scores_prediction_frozen_at_issue() {
    let (score, expected, _) = issued_forecast_scores();
    assert!(
        (score.sse_model - expected).abs() < 1e-10,
        "maturity score={} issued score={expected}",
        score.sse_model
    );
}

#[test]
fn si_scores_climatology_frozen_at_issue() {
    let (score, _, expected) = issued_forecast_scores();
    assert!(
        (score.sse_clim - expected).abs() < 1e-10,
        "maturity score={} issued score={expected}",
        score.sse_clim
    );
}

#[test]
fn si_warmup_readiness_is_decided_at_issue() {
    let mut tape = SpectralTape::new(&[100.0]);
    let mut f = HorizonForecaster::new(&tape, ForecastTarget::Volume, 100.0);
    f.warmup = 1;
    for ts in [0, 25, 50, 75, 100] {
        f.before_trade(&tape, ts);
        tape.on_trade(ts, 100.0, 1.0, true);
        f.after_trade(&tape, ts);
    }
    f.before_trade(&tape, 1_000);
    assert!(f.warmed_up());
    assert_eq!(
        f.score.n, 0,
        "all pending forecasts were issued before warmup"
    );
}

#[test]
fn si_invalid_trade_cannot_change_bank_or_consume_clock() {
    let mut bank = SpectralForecastBank::new();
    for i in 0..20 {
        bank.on_trade(i * 5_000, 100.0 + i as f64, 1.0, true);
    }
    let control = bank.clone();
    let before = format!("{bank:?}");
    for (ts, price, qty) in [
        (1_000_000, f64::NAN, 1.0),
        (1_000_000, 100.0, 0.0),
        (1, 100.0, 1.0),
        (1_000_000, f64::MAX, 1.0),
    ] {
        bank.on_trade(ts, price, qty, true);
        assert!(
            format!("{bank:?}") == before,
            "invalid trade changed bank: {ts}, {price}, {qty}"
        );
    }
    let mut control = control;
    bank.on_trade(100_000, 110.0, 2.0, false);
    control.on_trade(100_000, 110.0, 2.0, false);
    assert_eq!(format!("{bank:?}"), format!("{control:?}"));
}

#[test]
fn si_clock_near_u64_max_does_not_overflow() {
    let mut tape = SpectralTape::new(&[100.0]);
    let mut f = HorizonForecaster::new(&tape, ForecastTarget::Volume, 100.0);
    for ts in [u64::MAX - 50, u64::MAX - 25, u64::MAX] {
        f.before_trade(&tape, ts);
        tape.on_trade(ts, 100.0, 1.0, true);
        f.after_trade(&tape, ts);
    }
    f.before_trade(&tape, u64::MAX);
    assert_eq!(f.matured, 0);
    let pending = f.pending.len();
    f.after_trade(&tape, u64::MAX);
    assert_eq!(
        f.pending.len(),
        pending,
        "exhausted clock duplicated a sample"
    );
}

#[test]
fn si_flat_start_cannot_poison_variance_learning() {
    let mut tape = SpectralTape::new(&[100.0, 400.0]);
    let mut f = HorizonForecaster::new(&tape, ForecastTarget::Variance, 100.0);
    f.warmup = 1;
    for i in 0..100u64 {
        let ts = i * 25;
        f.before_trade(&tape, ts);
        tape.on_trade(
            ts,
            if i < 20 {
                100.0
            } else {
                100.0 + (i % 3) as f64
            },
            1.0,
            true,
        );
        f.after_trade(&tape, ts);
    }
    assert!(
        f.score.n > 20,
        "must resume learning once variance is measurable"
    );
    assert!(f.clim_sum.is_finite());
    assert!(f.score.sse_model.is_finite() && f.score.sse_clim.is_finite());
    assert!(f.coefficients().iter().all(|x| x.is_finite()));
}

#[test]
fn si_perfect_or_invalid_null_cannot_authorize_skill() {
    for (persist, clim) in [
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
        (-1.0, 1.0),
    ] {
        let score = ForecastScore {
            n: 30,
            sse_model: 0.1,
            sse_persist: persist,
            sse_clim: clim,
        };
        assert!(score.skill_vs_best_null() <= 0.0, "{score:?}");
    }
    let valid = ForecastScore {
        n: 30,
        sse_model: 0.1,
        sse_persist: 1.0,
        sse_clim: 2.0,
    };
    assert!((valid.skill_vs_best_null() - 0.9).abs() < 1e-12);
}

#[test]
fn si_rejected_first_trade_keeps_tape_unstarted() {
    let mut tape = SpectralTape::new(&[100.0]);
    let before = format!("{tape:?}");
    for (price, qty) in [
        (f64::NAN, 1.0),
        (100.0, f64::INFINITY),
        (f64::MAX, 2.0),
        (1e200, 1.0),
        (1e-200, 1e-200),
    ] {
        tape.on_trade(100, price, qty, true);
        assert!(format!("{tape:?}") == before);
    }
    tape.on_trade(1, 100.0, 1.0, true);
    assert!(tape.is_started());
    assert_eq!(tape.last_ts(), 1);
    assert_eq!(tape.totals(), [0.0, 100.0, 100.0, 1.0, 10_000.0]);
}

#[test]
fn si_window_includes_boundary_trade_but_not_next_trade() {
    let mut tape = SpectralTape::new(&[100.0]);
    let mut f = HorizonForecaster::new(&tape, ForecastTarget::Volume, 100.0);
    f.warmup = 0;
    tape.on_trade(0, 100.0, 1.0, true);
    tape.on_trade(25, 100.0, 1.0, true);
    f.after_trade(&tape, 25);
    let prediction = f.last_prediction().unwrap();
    f.before_trade(&tape, 125);
    assert_eq!(f.score.n, 0);
    tape.on_trade(125, 100.0, 3.0, true);
    f.before_trade(&tape, 126);
    // Only the boundary trade (300) plus the existing smoothing floor (100).
    assert_eq!(f.score.n, 1);
    assert!((f.score.sse_model - (400.0_f64.ln() - prediction).powi(2)).abs() < 1e-12);
    tape.on_trade(126, 100.0, 1_000.0, true);
    assert_eq!(f.score.n, 1);
    assert!(
        f.predict(&tape, 25).is_none(),
        "past query sees future tape state"
    );
}
