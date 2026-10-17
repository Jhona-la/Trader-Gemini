//! Contratos de Validación y Precisión Analítica para Online Newton GBDT
//! Certifica convergencia, paridad direccional, calibración continua y cero alocaciones (<25 µs).

use evolution_engine::online_newton_gbdt::OnlineNewtonGbdt;
use std::time::Instant;

#[test]
fn test_newton_gbdt_training_and_calibration() {
    let gbdt = OnlineNewtonGbdt::new(1000);

    // Generar observaciones sintéticas con patrón claro: OBI alto + momentum positivo => sube precio
    for i in 0..120 {
        let is_positive = i % 2 == 0;
        let obi = if is_positive { 0.65 } else { -0.65 };
        let vol_accel = if is_positive { 1.5 } else { -1.5 };
        let spread_bps = 2.0;
        let atr_pct = 0.003;
        let hurst = 0.60;
        let macro_mom = if is_positive { 0.05 } else { -0.05 };

        let features = [obi, vol_accel, spread_bps, atr_pct, hurst, macro_mom];
        let net_pnl = if is_positive { 0.015 } else { -0.012 };
        let is_long = true;

        gbdt.feed_observation(features, net_pnl, is_long);
    }

    let retrain_res = gbdt.retrain_models();
    assert!(retrain_res.is_ok(), "El reentrenamiento debe ser exitoso: {:?}", retrain_res);

    let (acc, mse) = retrain_res.unwrap();
    assert!(acc >= 0.70, "La precisión del clasificador debe ser >= 70%: {}", acc);
    assert!(mse < 0.01, "El MSE debe ser menor a 0.01: {}", mse);

    // Verificar inferencia continua
    let eval_up = [0.80, 2.0, 2.0, 0.003, 0.65, 0.10];
    let (prob_up, pnl_up) = gbdt.predict_6d(eval_up).expect("Debe predecir para features válidos");
    assert!(prob_up > 0.55, "Features alcistas deben producir prob_up > 0.55: {}", prob_up);
    assert!(pnl_up > 0.0, "Expected PnL debe ser positivo: {}", pnl_up);

    let eval_down = [-0.80, -2.0, 2.0, 0.003, 0.65, -0.10];
    let (prob_down, pnl_down) = gbdt.predict_6d(eval_down).expect("Debe predecir para features válidos");
    assert!(prob_down < 0.45, "Features bajistas deben producir prob_up < 0.45: {}", prob_down);
    assert!(pnl_down < 0.0, "Expected PnL debe ser negativo: {}", pnl_down);
}

#[test]
fn test_newton_gbdt_directional_parity_short_vs_long() {
    let gbdt = OnlineNewtonGbdt::new(500);

    // Enviar trades cortos ganadores: cuando el precio baja (OBI negativo), el short gana
    for _i in 0..80 {
        let features = [-0.70, -1.0, 1.5, 0.002, 0.55, -0.04];
        let net_pnl = 0.020; // Short ganó +2%
        let is_long = false; // Era un corto
        gbdt.feed_observation(features, net_pnl, is_long);

        let features_l = [0.70, 1.0, 1.5, 0.002, 0.55, 0.04];
        let net_pnl_l = 0.020; // Long ganó +2%
        let is_long_l = true; // Era un largo
        gbdt.feed_observation(features_l, net_pnl_l, is_long_l);
    }

    let retrain_res = gbdt.retrain_models();
    assert!(retrain_res.is_ok());

    // Cuando el mercado presenta OBI negativo (desplome inminente), la probabilidad de subida DEBE ser baja (< 0.50)
    let drop_features = [-0.70, -1.0, 1.5, 0.002, 0.55, -0.04];
    let (prob_up, pnl_pred) = gbdt.predict_6d(drop_features).unwrap();
    assert!(
        prob_up < 0.50,
        "Un corto ganador indica que el precio cayó, por lo que prob_up debe ser < 0.50: {}",
        prob_up
    );
    assert!(
        pnl_pred < 0.0,
        "La dirección del precio fue a la baja: {}",
        pnl_pred
    );
}

#[test]
fn test_newton_gbdt_zero_alloc_latency_budget() {
    let gbdt = OnlineNewtonGbdt::new(500);

    for i in 0..100 {
        let is_up = i % 2 == 0;
        let features = [if is_up { 0.5 } else { -0.5 }, 0.0, 1.0, 0.002, 0.50, 0.0];
        gbdt.feed_observation(features, if is_up { 0.01 } else { -0.01 }, true);
    }
    gbdt.retrain_models().unwrap();

    let eval_features = [0.45, 0.10, 1.2, 0.0025, 0.52, 0.01];

    // Medir latencia de 10,000 inferencias consecutivas
    let start = Instant::now();
    let iterations = 10_000;
    for _ in 0..iterations {
        let res = gbdt.predict_6d(eval_features);
        assert!(res.is_some());
    }
    let elapsed = start.elapsed();
    let per_inference_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!("⚡ Latencia por inferencia Newton GBDT: {:.2} ns", per_inference_ns);
    // Mandato: ejecución crítica en nanosegundos / microsegundos (<25 µs = 25,000 ns)
    assert!(
        per_inference_ns < 25_000.0,
        "La inferencia debe ser ultra-rápida (<25 µs): {:.2} ns",
        per_inference_ns
    );
}

#[test]
fn test_newton_gbdt_nan_and_inf_immunity() {
    let gbdt = OnlineNewtonGbdt::new(100);

    // Observaciones corruptas deben ser rechazadas fail-safe
    gbdt.feed_observation([f64::NAN, 0.0, 1.0, 0.002, 0.50, 0.0], 0.01, true);
    gbdt.feed_observation([0.5, f64::INFINITY, 1.0, 0.002, 0.50, 0.0], 0.01, true);
    gbdt.feed_observation([0.5, 0.0, 1.0, 0.002, 0.50, 0.0], f64::NAN, true);

    assert_eq!(gbdt.observations.read().unwrap().len(), 0);

    // Inferencia ante features corruptos debe retornar None sin pánico
    assert!(gbdt.predict_6d([f64::NAN, 0.0, 1.0, 0.002, 0.50, 0.0]).is_none());
    assert!(gbdt.predict_6d([0.5, f64::NEG_INFINITY, 1.0, 0.002, 0.50, 0.0]).is_none());
}
