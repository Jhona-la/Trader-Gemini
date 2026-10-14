//! CONTRATO FORMAL: NORMALIZACIÓN DINÁMICA DE VOLATILIDAD Y SLIPPAGE EN METACORTEX (Ola Ω102)
//!
//! # Propósito y Fundamento Físico-Matemático
//! Este contrato formal certifica la erradicación de los umbrales estáticos congelados en `SeniorVolatilidad`
//! y `SeniorEjecucion` en `crates/metacortex-engine/src/consejo_seniors.rs`.
//!
//! 1. `evaluate_volatility_regime`:
//!    Preserva paridad bit a bit exacta con las cotas canónicas de microestructura (10 bps calma, 40 bps peligro).
//! 2. `evaluate_volatility_regime_with_bounds`:
//!    Permite parametrizar las bandas de calma y peligro para activos de alta beta (altcoins) o regímenes
//!    específicos, evitando que altcoins con ATR natural de 50-100 bps sean aplastadas al piso de 0.40.
//! 3. `evaluate_volatility_regime_spectral`:
//!    Modula suavemente las bandas de volatilidad en función de la escala temporal espectral s ∈ [0, 1]
//!    (desde s = 0 en τ = 30 s hasta s = 1 en τ = 12 h) bajo difusión continua Brownian/Lévy.
//! 4. Inmunidad matemática estricta a NaN, Infinitos y valores negativos.
//! 5. `evaluate_execution_slippage` y `evaluate_execution_slippage_with_bounds`:
//!    Desacoplamiento funcional del cálculo de slippage efectivo y presupuesto máximo con modulación
//!    hidrodinámica Navier-Stokes y factor ATR relativo.

use metacortex_engine::consejo_seniors::{
    evaluate_execution_slippage, evaluate_execution_slippage_with_bounds,
    evaluate_volatility_regime, evaluate_volatility_regime_spectral,
    evaluate_volatility_regime_with_bounds, CALM_ATR_PCT, DANGER_ATR_PCT,
};

#[test]
fn ola102_volatilidad_regime_paridad_canonica() {
    // 1. Calma absoluta (10 bps): convicción plena 1.0
    let conf_calm = evaluate_volatility_regime(CALM_ATR_PCT);
    assert!(
        (conf_calm - 1.0).abs() < 1e-12,
        "En calma absoluta (10 bps), convicción debe ser 1.0 exacta, obtenido: {conf_calm}"
    );

    // 2. Cero volatilidad: convicción plena 1.0
    let conf_zero = evaluate_volatility_regime(0.0);
    assert!(
        (conf_zero - 1.0).abs() < 1e-12,
        "Con volatilidad cero, convicción debe ser 1.0 exacta"
    );

    // 3. Peligro máximo (40 bps): piso defensivo 0.40
    let conf_danger = evaluate_volatility_regime(DANGER_ATR_PCT);
    assert!(
        (conf_danger - 0.40).abs() < 1e-12,
        "En peligro máximo (40 bps), convicción debe ser exactamente 0.40, obtenido: {conf_danger}"
    );

    // 4. Volatilidad extrema (50 bps > 40 bps): clamp al piso defensivo 0.40
    let conf_extreme = evaluate_volatility_regime(0.0050);
    assert!(
        (conf_extreme - 0.40).abs() < 1e-12,
        "Volatilidad superior a peligro debe permanecer en el piso de 0.40"
    );

    // 5. Punto medio exacto (25 bps = 10 + 15 bps):
    // factor = 1.0 - 0.6 * (15 / 30) = 1.0 - 0.3 = 0.70
    let conf_mid = evaluate_volatility_regime(0.0025);
    assert!(
        (conf_mid - 0.70).abs() < 1e-12,
        "Punto medio (25 bps) debe dar 0.70 exacto, obtenido: {conf_mid}"
    );
}

#[test]
fn ola102_volatilidad_regime_dinamica_con_cotas_desacopladas() {
    // Configuración para Altcoin de alta beta (ej. SOL / MEME):
    // calma natural en 30 bps (0.0030), peligro en 120 bps (0.0120).
    let alt_calm = 0.0030;
    let alt_danger = 0.0120;

    // En su calma natural (30 bps), debe otorgar convicción plena 1.0
    // (anteriormente con umbral fijo de 10 bps, 30 bps hubiera recibido 0.60 de convicción).
    let conf_alt_calm = evaluate_volatility_regime_with_bounds(alt_calm, alt_calm, alt_danger);
    assert!(
        (conf_alt_calm - 1.0).abs() < 1e-12,
        "Altcoin en su calma intrínseca debe tener convicción 1.0"
    );

    // En 75 bps (punto medio del rango de altcoin: 30 + 45 = 75 bps):
    // factor = 1.0 - 0.6 * (45 / 90) = 0.70
    let conf_alt_mid = evaluate_volatility_regime_with_bounds(0.0075, alt_calm, alt_danger);
    assert!(
        (conf_alt_mid - 0.70).abs() < 1e-12,
        "Punto medio en altcoin (75 bps) debe dar 0.70 exacto, obtenido: {conf_alt_mid}"
    );

    // En peligro de altcoin (120 bps): piso defensivo 0.40
    let conf_alt_danger = evaluate_volatility_regime_with_bounds(alt_danger, alt_calm, alt_danger);
    assert!(
        (conf_alt_danger - 0.40).abs() < 1e-12,
        "En peligro de altcoin debe dar 0.40 exacto"
    );

    // Robustez ante cotas anómalas (danger <= calm):
    let conf_inverted = evaluate_volatility_regime_with_bounds(0.0020, 0.0050, 0.0020);
    assert!(
        conf_inverted.is_finite() && (0.40..=1.0).contains(&conf_inverted),
        "Cotas invertidas deben autocorregirse sin pánico ni NaN"
    );
}

#[test]
fn ola102_volatilidad_regime_spectral_expansion_continua() {
    // En s = 0.0 (banda ultra-rápida, 30 s), paridad canónica
    let conf_s0_calm = evaluate_volatility_regime_spectral(0.0010, 0.0, 0.0010, 0.0040, 2.0);
    let conf_s0_danger = evaluate_volatility_regime_spectral(0.0040, 0.0, 0.0010, 0.0040, 2.0);
    assert!((conf_s0_calm - 1.0).abs() < 1e-12);
    assert!((conf_s0_danger - 0.40).abs() < 1e-12);

    // Para un trade con ATR de 60 bps (0.0060):
    // - En s = 0.0 (scalp 30 s): 60 bps > 40 bps peligro → convicción 0.40 (peligroso para micro-scalp)
    // - En s = 0.5 (intermedio ~19 min): bandas se expanden a calma 20 bps, peligro 80 bps → convicción 0.60
    // - En s = 1.0 (swing 12 h): bandas se expanden a calma 30 bps, peligro 120 bps → convicción 0.80
    let conf_s0 = evaluate_volatility_regime_spectral(0.0060, 0.0, 0.0010, 0.0040, 2.0);
    let conf_s05 = evaluate_volatility_regime_spectral(0.0060, 0.5, 0.0010, 0.0040, 2.0);
    let conf_s1 = evaluate_volatility_regime_spectral(0.0060, 1.0, 0.0010, 0.0040, 2.0);

    assert_eq!(conf_s0, 0.40, "Para scalp rápido 60 bps debe saturar a peligro");
    assert!(
        (conf_s05 - 0.60).abs() < 1e-12,
        "Para escala intermedia 60 bps debe dar 0.60 exacto, obtenido: {conf_s05}"
    );
    assert!(
        (conf_s1 - 0.80).abs() < 1e-12,
        "Para swing lento 60 bps debe dar 0.80 exacto, obtenido: {conf_s1}"
    );

    // Monotonía estricta de la adaptación espectral:
    assert!(
        conf_s1 > conf_s05 && conf_s05 > conf_s0,
        "La convicción debe ser estrictamente monótona con el horizonte espectral para absorber volatilidad"
    );
}

#[test]
fn ola102_volatilidad_regime_inmunidad_a_nan_y_negativos() {
    let conf_nan = evaluate_volatility_regime(f64::NAN);
    assert_eq!(conf_nan, 1.0, "NaN en atr_pct debe retornar seguro 1.0");

    let conf_neg = evaluate_volatility_regime(-0.05);
    assert_eq!(conf_neg, 1.0, "Valor negativo en atr_pct debe clampear a calma 1.0");

    let conf_inf = evaluate_volatility_regime(f64::INFINITY);
    assert_eq!(conf_inf, 0.40, "Infinito en atr_pct debe activar piso defensivo 0.40");

    let conf_spectral_nan = evaluate_volatility_regime_spectral(0.0025, f64::NAN, 0.0010, 0.0040, 2.0);
    assert!(
        conf_spectral_nan.is_finite() && (conf_spectral_nan - 0.70).abs() < 1e-12,
        "NaN en spectral_s debe degradar limpiamente a s = 0.0"
    );
}

#[test]
fn ola102_execution_slippage_paridad_y_dinamismo() {
    // 1. Régimen laminar (Re laminar = 0.96, slippage 30 bps, extremo rápido s = 0, ATR 10 bps):
    // effective_slippage = 30.0 * (1.0 + (1.0 - 0.96) * 0.50) = 30.60 bps
    // max_slippage = 35.0 * 1.0 = 35.0 bps
    // is_veto = 30.60 > 35.0 = false
    let (eff_lam, max_lam, veto_lam) = evaluate_execution_slippage(30.0, 0.96, 0.0, 0.0010);
    assert!((eff_lam - 30.60).abs() < 1e-10);
    assert!((max_lam - 35.00).abs() < 1e-10);
    assert!(!veto_lam, "30 bps en régimen laminar no debe vetar");

    // 2. Régimen turbulento (Re laminar = 0.013, slippage 30 bps):
    // effective_slippage = 30.0 * (1.0 + (1.0 - 0.013) * 0.50) = 44.805 bps
    // is_veto = 44.805 > 35.0 = true
    let (eff_turb, _max_turb, veto_turb) = evaluate_execution_slippage(30.0, 0.013, 0.0, 0.0010);
    assert!((eff_turb - 44.805).abs() < 1e-10);
    assert!(veto_turb, "30 bps amplificado por turbulencia DEBE vetar");

    // 3. Parámetros desacoplados con cotas personalizadas:
    // fast_floor = 50.0 bps, slow_span = 100.0 bps
    let (eff_custom, max_custom, veto_custom) =
        evaluate_execution_slippage_with_bounds(30.0, 0.013, 0.5, 0.0010, 50.0, 100.0);
    // base_slippage = 50.0 + 100.0 * 0.5 = 100.0 bps
    assert!((eff_custom - 44.805).abs() < 1e-10);
    assert!((max_custom - 100.00).abs() < 1e-10);
    assert!(!veto_custom, "Con tolerancia desacoplada de 100 bps no debe vetar");
}
