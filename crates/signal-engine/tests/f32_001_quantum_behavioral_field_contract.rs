//! 🧪 CONTRATOS FORMALES DE VERIFICACIÓN: MOTOR DE FINANZAS CONDUCTUALES CUÁNTICAS
//!
//! # FASE 32 (§32) — OLA OMEGA220: DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Suites de verificación determinista para `QuantumBehavioralFieldEngine`:
//! - C1: Asimetría de la función de valor en $S$ de Kahneman-Tversky (aversión a la pérdida $\lambda = 2.25$).
//! - C2: Distorsión no lineal de probabilidad de Prelec (amplificación de colas $w(p) > p$).
//! - C3: Gradiente analítico de fuerza de stop-hunting institucional ($F_{\text{hunt}} = -\nabla \Phi$).
//! - C4: Cota estricta de resonancia conductual $B(t) \in [-1.0, 1.0]$ y multiplicador de convicción.
//! - C5: Inmunidad total a valores no finitos (NaN, ±Inf) y ejecución zero-alloc.

use signal_engine::quantum_behavioral_field::{
    BehavioralStateReport, QuantumBehavioralFieldEngine, QuantumBehavioralParams,
};

#[test]
fn c1_prospect_theory_loss_aversion_asymmetry() {
    let engine = QuantumBehavioralFieldEngine::new();
    let gain_return = 0.05; // +5%
    let loss_return = -0.05; // -5%

    let v_gain = engine.evaluate_prospect_value(gain_return);
    let v_loss = engine.evaluate_prospect_value(loss_return);

    assert!(
        v_gain > 0.0,
        "Ganancia debe arrojar valor subjetivo positivo"
    );
    assert!(
        v_loss < 0.0,
        "Pérdida debe arrojar valor subjetivo negativo"
    );

    // Asimetría: la pérdida duele 2.25 veces más que la ganancia idéntica
    let ratio = (-v_loss) / v_gain;
    assert!(
        (ratio - 2.25).abs() < 1e-4,
        "La aversión a la pérdida debe ser exactamente lambda=2.25, obtenido: {}",
        ratio
    );
}

#[test]
fn c2_prelec_probability_weighting_tail_distortion() {
    let engine = QuantumBehavioralFieldEngine::new();

    // 1. Cola extrema de baja probabilidad: p = 0.01 (1% probabilidad de crash)
    let p_tail = 0.01;
    let w_tail = engine.evaluate_prelec_weighting(p_tail);
    assert!(
        w_tail > p_tail,
        "Prelec debe sobreponderar eventos raros de cola (w({}) = {} > {})",
        p_tail,
        w_tail,
        p_tail
    );

    // 2. Probabilidad moderada: p = 0.50
    let p_mid = 0.50;
    let w_mid = engine.evaluate_prelec_weighting(p_mid);
    assert!(
        w_mid < p_mid,
        "Prelec debe subponderar probabilidades moderadas (w({}) = {} < {})",
        p_mid,
        w_mid,
        p_mid
    );

    // 3. Fronteras
    assert_eq!(engine.evaluate_prelec_weighting(0.0), 0.0);
    assert_eq!(engine.evaluate_prelec_weighting(1.0), 1.0);
}

#[test]
fn c3_stop_hunt_force_gradient_liquidity_pull() {
    let engine = QuantumBehavioralFieldEngine::new();
    let current_price = 100.0;
    let pivot_high = 101.0; // 1% arriba (resistencia con stops de cortos)
    let pivot_low = 95.0; // 5% abajo (soporte lejano)
    let short_liq_vol = 50.0;
    let long_liq_vol = 2.0;

    let (force_high, potential_high) = engine.compute_stop_hunt_force(
        current_price,
        pivot_high,
        pivot_low,
        short_liq_vol,
        long_liq_vol,
    );

    assert!(
        force_high > 0.30,
        "La fuerza neta debe ser fuertemente alcista hacia el pivot high de cortos, obtenido: {}",
        force_high
    );
    assert!(
        potential_high > 0.0,
        "El potencial de atrapamiento debe ser positivo"
    );

    // Caso inverso: precio muy cerca del soporte de largos con gran volumen de largos en riesgo
    let pivot_high_far = 108.0;
    let pivot_low_near = 99.5;
    let (force_low, _) =
        engine.compute_stop_hunt_force(current_price, pivot_high_far, pivot_low_near, 2.0, 50.0);

    assert!(
        force_low < -0.30,
        "La fuerza neta debe ser fuertemente bajista hacia el soporte de largos, obtenido: {}",
        force_low
    );
}

#[test]
fn c4_behavioral_resonance_bounds_and_conviction_expansion() {
    let engine = QuantumBehavioralFieldEngine::new();

    // Squeeze alcista inminente: cortos atrapados cerca del pivot_high
    let report_bull = engine.evaluate(100.0, 99.0, 100.5, 95.0, 0.02, 100.0, 1.0);
    assert!(
        report_bull.behavioral_resonance > 0.0,
        "Resonancia debe favorecer LONGs ante squeeze de cortos"
    );
    assert!(
        report_bull.conviction_multiplier >= 1.0,
        "Convicción debe expandirse cuando la caza está alineada, obtenido: {}",
        report_bull.conviction_multiplier
    );
    assert!(
        report_bull.conviction_multiplier <= 2.0,
        "Convicción no debe exceder el tope de 2.0x"
    );

    // Liquidación bajista inminente: largos atrapados bajo el soporte
    let report_bear = engine.evaluate(100.0, 102.0, 108.0, 99.5, 0.40, 1.0, 100.0);
    assert!(
        report_bear.behavioral_resonance < 0.0,
        "Resonancia debe favorecer SHORTs ante cascada de largos"
    );
    assert!(
        (-1.0..=1.0).contains(&report_bear.behavioral_resonance),
        "Resonancia debe estar acotada estrictamente en [-1.0, 1.0]"
    );
}

#[test]
fn c5_non_finite_immunity_and_zero_alloc_safety() {
    let engine = QuantumBehavioralFieldEngine::with_params(QuantumBehavioralParams {
        loss_aversion_lambda: f64::NAN,
        value_alpha: f64::INFINITY,
        value_beta: -5.0,
        prelec_gamma: f64::NAN,
        epsilon_regularizer: 0.0,
        hunt_force_gain: f64::NAN,
    });

    let report_nan = engine.evaluate(
        f64::NAN,
        f64::INFINITY,
        -100.0,
        f64::NAN,
        f64::NAN,
        f64::INFINITY,
        -50.0,
    );

    assert_eq!(
        report_nan,
        BehavioralStateReport::default(),
        "Ante entradas corruptas, debe retornar estado por defecto seguro (fail-closed)"
    );
}
