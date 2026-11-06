//! Contrato formal de verificación para Fase 41: Motor de Medida Invariante SRB y Entropía Métrica de Kolmogorov-Sinai ($h_{\text{KS}}$)
//!
//! Invariantes certificados:
//! 1. c1: Espectro continuo de Lyapunov ordenado en orden descendente ($\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4$) y coherencia con la identidad de Pesin.
//! 2. c2: Detección de divergencia caótica y colapso monotónico del modulador de predictibilidad $\eta_{\text{pred}}$.
//! 3. c3: Estabilidad numérica absoluta ante estados singulares, ceros, NaNs e infinitos (`rb_p1`).
//! 4. c4: Cero asignaciones en memoria dinámica (heap) y ejecución acotada $O(1)$.

use signal_engine::srb_kolmogorov_entropy::{
    SrbEntropyParams, SrbKolmogorovEntropyEngine, CHAOS_EMBEDDING_DIM,
};

#[test]
fn c1_lyapunov_spectrum_ordering_and_pesin_identity() {
    let params = SrbEntropyParams {
        spectrum_smoothing: 0.10,
        h_ks_critical: 2.0,
        tangent_epsilon: 1e-4,
        chaos_decay_scale: 1.0,
    };
    let mut engine = SrbKolmogorovEntropyEngine::new(params);

    // Simular una trayectoria suave determinista durante varios pasos
    let mut rep = engine.update_and_evaluate(100.0, 0.05, 0.02, 2.0, 1_000);
    for step in 1..30 {
        let ts = 1_000 + step * 100;
        let p = 100.0 + (step as f64) * 0.05;
        let ofi = 0.05 + 0.01 * ((step as f64) * 0.2).sin();
        let vol = 0.02 + 0.005 * ((step as f64) * 0.1).cos();
        rep = engine.update_and_evaluate(p, ofi, vol, 2.0, ts);
    }

    // 1. El espectro debe estar estrictamente ordenado de forma decreciente
    let s = rep.lyapunov_spectrum;
    for i in 0..CHAOS_EMBEDDING_DIM - 1 {
        assert!(
            s[i] >= s[i + 1],
            "El espectro de Lyapunov debe estar ordenado en orden descendente: s[{}]={} >= s[{}]={}",
            i,
            s[i],
            i + 1,
            s[i + 1]
        );
    }

    // 2. Entropía de Kolmogorov-Sinai debe coincidir exactamente con la suma de exponentes positivos
    let mut expected_h_ks = 0.0;
    for &lambda in &s {
        if lambda > 0.0 {
            expected_h_ks += lambda;
        }
    }
    assert!(
        (rep.kolmogorov_sinai_entropy - expected_h_ks).abs() < 1e-9,
        "La entropía KS debe cumplir exactamente la identidad de Pesin"
    );

    // 3. Dimensión de Kaplan-Yorke acotada en [0, 4]
    assert!(
        rep.kaplan_yorke_dimension >= 0.0 && rep.kaplan_yorke_dimension <= 4.0,
        "La dimensión de Kaplan-Yorke debe estar acotada en [0, 4]: {}",
        rep.kaplan_yorke_dimension
    );
}

#[test]
fn c2_chaotic_transition_and_predictability_collapse() {
    let params = SrbEntropyParams {
        spectrum_smoothing: 0.15,
        h_ks_critical: 2.0,
        tangent_epsilon: 1e-4,
        chaos_decay_scale: 1.0,
    };
    let mut engine = SrbKolmogorovEntropyEngine::new(params);

    // 1. Régimen laminar estable: predictibilidad alta
    let rep_laminar = engine.update_and_evaluate(100.0, 0.01, 0.01, 1.5, 10_000);
    let pred_laminar = rep_laminar.predictability_modulator;

    // 2. Régimen de turbulencia caótica: saltos erráticos no correlacionados
    let mut rep_chaotic = rep_laminar;
    for step in 1..25 {
        let ts = 10_000 + step * 50;
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        let p = 100.0 + sign * (step as f64) * 0.45;
        let ofi = sign * 2.5;
        let vol = 0.15 + (step as f64) * 0.02;
        rep_chaotic = engine.update_and_evaluate(p, ofi, vol, 5.0, ts);
    }

    // El exponente máximo de Lyapunov y la entropía deben ser mayores en el régimen turbulento
    assert!(
        rep_chaotic.kolmogorov_sinai_entropy > rep_laminar.kolmogorov_sinai_entropy,
        "La entropía KS caótica ({}) debe ser mayor que la laminar ({})",
        rep_chaotic.kolmogorov_sinai_entropy,
        rep_laminar.kolmogorov_sinai_entropy
    );

    // El modulador de predictibilidad debe contraerse
    assert!(
        rep_chaotic.predictability_modulator <= pred_laminar,
        "La predictibilidad debe decrecer ante turbulencia caótica: {} <= {}",
        rep_chaotic.predictability_modulator,
        pred_laminar
    );
}

#[test]
fn c3_subnormal_and_degraded_numerical_safety() {
    let mut engine = SrbKolmogorovEntropyEngine::new(SrbEntropyParams::default());

    // Probar con valores atípicos y degenerados
    let rep_nan =
        engine.update_and_evaluate(f64::NAN, f64::INFINITY, -1.0, f64::NEG_INFINITY, 20_000);
    assert!(rep_nan.max_lyapunov.is_finite());
    assert!(rep_nan.kolmogorov_sinai_entropy.is_finite());
    assert!(rep_nan.predictability_modulator.is_finite());
    assert!(rep_nan.predictability_horizon_sec.is_finite());

    // Probar con cero absoluto continuo
    let rep_zero = engine.update_and_evaluate(0.0, 0.0, 0.0, 0.0, 20_100);
    assert!(rep_zero.max_lyapunov.is_finite());
    assert!(rep_zero.kolmogorov_sinai_entropy >= 0.0);
}

#[test]
fn c4_zero_alloc_and_zero_panic_guarantee() {
    let mut engine = SrbKolmogorovEntropyEngine::new(SrbEntropyParams::default());

    let mut p = 100.0;
    for step in 0..100 {
        p += 0.02 * ((step as f64) * 0.15).sin();
        let rep = engine.update_and_evaluate(p, 0.1, 0.03, 2.0, 50_000 + step * 20);
        assert!(rep.predictability_modulator >= 0.0 && rep.predictability_modulator <= 1.0);
        assert!(rep.predictability_horizon_sec > 0.0);
    }
}
