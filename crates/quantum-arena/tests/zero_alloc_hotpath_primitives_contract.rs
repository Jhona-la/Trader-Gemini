use quantum_arena::spectral_tape::{HorizonForecaster, ForecastTarget, SpectralTape, Rls, ANCLAS_PRONOSTICO};
use quantum_arena::state::CoinArena;
use quantum_arena::{symbol_registry, symbols};
use std::sync::atomic::Ordering;

static M: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_sigma_forecast_at_zero_alloc_log_interpolation() {
    let arena = CoinArena::new(1.0);
    
    // Sin valores positivos, debe retornar None
    assert_eq!(arena.sigma_forecast_at(1_000.0), None);
    assert_eq!(arena.sigma_forecast_at(-100.0), None);
    assert_eq!(arena.sigma_forecast_at(f64::NAN), None);

    // Asignar anclas 0 y 1
    // Anclas son: 250ms, 1s, 5s, 30s, 2m, 15m, 1h
    let s0 = 0.0010; // 10 bps
    let s1 = 0.0020; // 20 bps
    arena.sigma_forecast[0].store(s0, Ordering::Relaxed);
    arena.sigma_forecast[1].store(s1, Ordering::Relaxed);

    // Antes de la primera ancla debe dar el extremo s0
    let tau_sub = ANCLAS_PRONOSTICO[0] * 0.5;
    assert_eq!(arena.sigma_forecast_at(tau_sub), Some(s0));

    // En la primera ancla debe dar s0
    assert_eq!(arena.sigma_forecast_at(ANCLAS_PRONOSTICO[0]), Some(s0));

    // En el punto medio geométrico entre ancla 0 y ancla 1: sqrt(t0 * t1)
    let tau_mid = (ANCLAS_PRONOSTICO[0] * ANCLAS_PRONOSTICO[1]).sqrt();
    let expected_mid = (s0 * s1).sqrt();
    let actual_mid = arena.sigma_forecast_at(tau_mid).expect("mid interpolation must exist");
    assert!((actual_mid - expected_mid).abs() < 1e-9, "Actual: {}, Expected: {}", actual_mid, expected_mid);
}

#[test]
fn test_rls_update_and_predict_zero_alloc_equivalence() {
    let mut rls = Rls::new(4, 0.99, 1.0);
    let x = [1.0, 0.5, -0.2, 0.1];
    let y = 1.25;

    let pred_before = rls.predict(&x);
    assert!(pred_before.is_finite());

    // Actualizar RLS
    rls.update(&x, y);

    let pred_after = rls.predict(&x);
    assert!(pred_after.is_finite());
    // El error debe haberse reducido en la dirección de y
    assert!((pred_after - y).abs() < (pred_before - y).abs());
}

#[test]
fn test_horizon_forecaster_predict_zero_alloc() {
    let taus = [250.0, 1000.0, 5000.0, 30000.0, 120000.0, 900000.0, 3600000.0];
    let mut tape = SpectralTape::new(&taus);
    tape.on_trade(1_000, 50_000.0, 1.0, true);
    tape.on_trade(2_000, 50_100.0, 1.5, false);

    let forecaster = HorizonForecaster::new(&tape, ForecastTarget::Variance, 5_000.0);
    let pred = forecaster.predict(&tape, 2_000);
    // Debe devolver Option<f64> de forma determinista sin pánico
    if let Some(val) = pred {
        assert!(val.is_finite());
    }
}

#[test]
fn test_symbol_registry_slot_es_simbolo_and_with_symbol() {
    let _g = M.lock().unwrap_or_else(|e| e.into_inner());
    let universe = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
    symbols::update_dynamic_universe(universe);

    assert!(symbol_registry::slot_es_simbolo(0, "BTCUSDT"));
    assert!(symbol_registry::slot_es_simbolo(0, "btcusdt"));
    assert!(!symbol_registry::slot_es_simbolo(0, "ETHUSDT"));
    assert!(symbol_registry::slot_es_simbolo(1, "ETHUSDT"));
    assert!(!symbol_registry::slot_es_simbolo(2, "SOLUSDT"));

    let res = symbol_registry::with_symbol(0, |s| s.len());
    assert_eq!(res, Some(7));

    let res_none = symbol_registry::with_symbol(99, |s| s.len());
    assert_eq!(res_none, None);
}
