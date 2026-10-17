//! Contrato de Certificación Ola Ω135:
//! Paridad Direccional de Lado (LC-F1-18 y LC-F1-67) y Paridad de Esquema en Shadow Forest
//!
//! ## QUÉ
//! Valida que el motor de inferencia online `TrueOnlineRandomForest` entrene sobre la dirección real
//! del mercado Δy = sign(side) · R_net, proyectando prob_up ∈ (0, 1) como P(precio sube).
//!
//! ## POR QUÉ
//! Antes de Ω135 (hallazgo LC-F1-18 / LC-F1-67), el clasificador aprendía P(net_pnl > 0) sin paridad
//! de lado. Por ende, un trade corto altamente ganador producía prob > 0.50 ("ganador"), que
//! el GodEngineCore interpretaba como "predijo subida (Long)". Esto causaba que el core penalizara
//! erróneamente la confianza de órdenes cortas y penalizara la autoridad Wilson del bosque en cierres ganadores.
//!
//! ## PARA QUÉ
//! Garantizar simetría matemática y física en el espacio Hilbert L^2 para Long y Short sobre
//! micro-cuentas ($13 USD), preservando el crecimiento compuesto sin vetos autodestructivos.

use evolution_engine::online_random_forest::TrueOnlineRandomForest;

#[test]
fn test_directional_parity_winning_short_predicts_downwards() {
    let forest = TrueOnlineRandomForest::new(1_000);

    // Entrenar con 60 observaciones de cortos ganadores (precio cayó cuando OBI era marcadamente negativo)
    // y 60 observaciones de cortos perdedores (precio subió cuando OBI rebotó)
    for i in 0..60 {
        let x = (i as f64) * 0.1;
        // Cortos ganadores: fuerte presión de venta en el libro, velocidad negativa, retorno positivo para el short
        let features_win = [
            -0.65 - 0.1 * (x.sin().abs()), // OBI bajista
            -1.2,                           // aceleración negativa
            8.0,                            // spread en bps
            0.003,                          // ATR%
            0.55,                           // Hurst
            -1.5,                           // velocidad de precio fuertemente negativa
        ];
        // Short ganó +1.2% (el mercado bajó)
        forest.shadow_evaluate_with_features_dir(features_win, 0.012, false);

        // Cortos perdedores: OBI positivo que rebotó contra el short
        let features_loss = [
            0.65 + 0.1 * (x.cos().abs()),  // OBI alcista
            1.2,                            // aceleración positiva
            8.0,                            // spread en bps
            0.003,                          // ATR%
            0.55,                           // Hurst
            1.5,                            // velocidad de precio positiva
        ];
        // Short perdió -0.8% (el mercado subió)
        forest.shadow_evaluate_with_features_dir(features_loss, -0.008, false);
    }

    let retrain_res = forest.retrain_models();
    assert!(retrain_res.is_ok(), "El reentrenamiento con dos clases direccionales debe ser exitoso");
    assert!(forest.is_trained());

    // Al evaluar una condición fuertemente bajista, el bosque debe predecir que el precio VA A BAJAR:
    // por ende prob_up < 0.50 (indicando señal short / bajada)
    let bearish_eval = [-0.70, -1.5, 8.0, 0.003, 0.55, -2.0];
    let (prob_up, pnl_dir) = forest.predict_6d(bearish_eval)
        .expect("Inferencia 6D válida");

    assert!(
        prob_up < 0.50,
        "LC-F1-18 FIX: Un mercado con venta masiva y cortos ganadores debe predecir prob_up < 0.50 (actual: {})",
        prob_up
    );
    assert!(
        pnl_dir < 0.0,
        "El PnL direccional del precio debe ser negativo (bajada esperada, actual: {})",
        pnl_dir
    );
}

#[test]
fn test_directional_parity_winning_long_predicts_upwards() {
    let forest = TrueOnlineRandomForest::new(1_000);

    // Entrenar con largos ganadores (mercado subió con OBI alcista) y largos perdedores
    for i in 0..60 {
        let x = (i as f64) * 0.1;
        let features_win = [
            0.70 + 0.1 * (x.sin().abs()),
            1.5,
            6.0,
            0.0025,
            0.60,
            2.0,
        ];
        // Long ganó +1.5% (el mercado subió)
        forest.shadow_evaluate_with_features_dir(features_win, 0.015, true);

        let features_loss = [
            -0.70 - 0.1 * (x.cos().abs()),
            -1.5,
            6.0,
            0.0025,
            0.60,
            -2.0,
        ];
        // Long perdió -0.7% (el mercado cayó)
        forest.shadow_evaluate_with_features_dir(features_loss, -0.007, true);
    }

    forest.retrain_models().expect("Entrenamiento exitoso");

    let bullish_eval = [0.75, 1.8, 6.0, 0.0025, 0.60, 2.2];
    let (prob_up, pnl_dir) = forest.predict_6d(bullish_eval)
        .expect("Inferencia 6D válida");

    assert!(
        prob_up > 0.50,
        "Un mercado con compra agresiva y longs ganadores debe predecir prob_up > 0.50 (actual: {})",
        prob_up
    );
    assert!(
        pnl_dir > 0.0,
        "El PnL direccional del precio debe ser positivo (subida esperada, actual: {})",
        pnl_dir
    );
}

#[test]
fn test_nan_and_infinity_rejection_in_shadow_evaluate() {
    let forest = TrueOnlineRandomForest::new(100);

    let corrupt_features = [f64::NAN, 0.0, 1.0, 0.002, 0.50, 0.0];
    forest.shadow_evaluate_with_features_dir(corrupt_features, 0.01, true);

    let obs_count = forest.observations.read().unwrap().len();
    assert_eq!(obs_count, 0, "Observación con NaN en features debe ser descartada inmediatamente");

    let valid_features = [0.5, 0.0, 1.0, 0.002, 0.50, 0.0];
    forest.shadow_evaluate_with_features_dir(valid_features, f64::INFINITY, true);

    let obs_count2 = forest.observations.read().unwrap().len();
    assert_eq!(obs_count2, 0, "Observación con Inf en retorno debe ser descartada inmediatamente");
}

#[test]
fn test_predict_6d_rejects_nan_inputs() {
    let forest = TrueOnlineRandomForest::new(100);
    // Sin entrenar o con features inválidas, predict_6d retorna None con seguridad
    let corrupt_features = [0.5, f64::NAN, 1.0, 0.002, 0.50, 0.0];
    assert_eq!(forest.predict_6d(corrupt_features), None);
}
