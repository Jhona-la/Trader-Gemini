//! Contrato formal de verificación para F1-082 (LC-F1-16 / D-641).
//!
//! Erradicación del salto discontinuo en `capital_scale` al cruzar `capital_ratio == kelly_survival_cap_ratio`
//! mediante la variedad continua C^1 en log(ratio) con interpolación cúbica de Hermite (smoothstep).

use risk_engine::kelly::{calculate_kelly_fraction, continuous_capital_scale};

const PHI: f64 = 1.618033988749895;

#[test]
fn contrato_1_continuidad_infinitesimal_en_frontera_rho() {
    let rhos = [1.0, 1.344958, PHI, 2.0, 2.5, 3.0];
    let base = 13.0;

    for &rho in &rhos {
        let cap_threshold = base * rho;
        let delta = 1e-5;

        // Evaluación a ambos lados del umbral
        let f_below = calculate_kelly_fraction(
            0.60,
            2.0,
            cap_threshold - delta,
            base,
            rho,
            1.5,
            0.01,
            1.0,
            0.35,
            0.5,
        );
        let f_above = calculate_kelly_fraction(
            0.60,
            2.0,
            cap_threshold + delta,
            base,
            rho,
            1.5,
            0.01,
            1.0,
            0.35,
            0.5,
        );

        let diff = (f_above - f_below).abs();
        assert!(
            diff < 1e-4,
            "Fallo Contrato 1: discontinuidad C^0 en rho={rho}: f_below={f_below}, f_above={f_above}, diff={diff}"
        );

        // Comprobación directa sobre la función pura continuous_capital_scale
        let scale_below = continuous_capital_scale(rho - 1e-6, rho, 1.5, 0.35, 0.60, 0.5);
        let scale_above = continuous_capital_scale(rho + 1e-6, rho, 1.5, 0.35, 0.60, 0.5);
        let scale_diff = (scale_above - scale_below).abs();
        assert!(
            scale_diff < 1e-5,
            "Fallo Contrato 1: continuous_capital_scale discontinuo en rho={rho}: diff={scale_diff}"
        );
    }
}

#[test]
fn contrato_2_barrido_denso_sin_discontinuidades_de_escalon() {
    let base = 13.0;
    let rho = PHI;
    let steps = 2000;
    let cap_min = 5.0;
    let cap_max = 100.0;
    let step_size = (cap_max - cap_min) / steps as f64;

    let mut prev_f = calculate_kelly_fraction(
        0.55, 1.8, cap_min, base, rho, 1.2, 0.01, 1.0, 0.35, 0.5,
    );

    let mut max_jump = 0.0_f64;

    for i in 1..=steps {
        let cap = cap_min + i as f64 * step_size;
        let f = calculate_kelly_fraction(
            0.55, 1.8, cap, base, rho, 1.2, 0.01, 1.0, 0.35, 0.5,
        );

        let diff = (f - prev_f).abs();
        if diff > max_jump {
            max_jump = diff;
        }

        // En un paso de 0.0475 USD, el cambio máximo no debe exceder 0.005
        // (el antiguo salto discreto era de ~0.0738 de golpe en 0.000002 USD).
        assert!(
            diff < 0.005,
            "Fallo Contrato 2: salto discontinuo detectado en cap={cap}: diff={diff}"
        );

        prev_f = f;
    }

    assert!(
        max_jump > 0.0 && max_jump < 0.002,
        "Fallo Contrato 2: gradiente anómalo o plano: max_jump={max_jump}"
    );
}

#[test]
fn contrato_3_monotonia_y_coherencia_asintotica() {
    let base = 13.0;
    let rho = PHI;

    // Región de supervivencia extrema: capital muy pequeño
    let f_micro = calculate_kelly_fraction(
        0.55, 1.5, 6.5, base, rho, 1.0, 0.01, 1.0, 0.35, 0.5,
    );
    // Región de transición: capital alrededor del umbral
    let f_trans = calculate_kelly_fraction(
        0.55, 1.5, 21.0, base, rho, 1.0, 0.01, 1.0, 0.35, 0.5,
    );
    // Región de expansión extrema: capital acumulado grande
    let f_macro = calculate_kelly_fraction(
        0.55, 1.5, 65.0, base, rho, 1.0, 0.01, 1.0, 0.35, 0.5,
    );

    assert!(
        f_trans >= f_micro,
        "Fallo Contrato 3: escala en transición ({f_trans}) debe ser >= supervivencia ({f_micro})"
    );
    assert!(
        f_macro >= f_trans,
        "Fallo Contrato 3: escala en expansión ({f_macro}) debe ser >= transición ({f_trans})"
    );
}

#[test]
fn contrato_4_modulacion_espectral_global() {
    let base = 13.0;
    let rho = PHI;
    let test_capitals = [10.0, 15.0, 21.03, 30.0, 60.0];

    for &cap in &test_capitals {
        let f_anti = calculate_kelly_fraction(
            0.55, 1.3, cap, base, rho, 1.0, 0.01, 1.0, 0.35, 0.0,
        );
        let f_neut = calculate_kelly_fraction(
            0.55, 1.3, cap, base, rho, 1.0, 0.01, 1.0, 0.35, 0.5,
        );
        let f_pers = calculate_kelly_fraction(
            0.55, 1.3, cap, base, rho, 1.0, 0.01, 1.0, 0.35, 1.0,
        );

        assert!(
            f_pers >= f_neut,
            "Fallo Contrato 4 en cap={cap}: persistente ({f_pers}) < neutro ({f_neut})"
        );
        assert!(
            f_neut >= f_anti,
            "Fallo Contrato 4 en cap={cap}: neutro ({f_neut}) < anti-persistente ({f_anti})"
        );
    }
}

#[test]
fn contrato_5_resiliencia_a_entradas_degeneradas_o_no_finitas() {
    // Casos no finitos deben devolver 0.0 sin entrar en pánico
    assert_eq!(
        continuous_capital_scale(f64::NAN, PHI, 1.0, 0.35, 0.55, 0.5),
        0.0
    );
    assert_eq!(
        continuous_capital_scale(1.0, f64::INFINITY, 1.0, 0.35, 0.55, 0.5),
        0.0
    );
    assert_eq!(
        continuous_capital_scale(1.0, PHI, f64::NAN, 0.35, 0.55, 0.5),
        0.0
    );
    assert_eq!(
        continuous_capital_scale(1.0, PHI, 1.0, f64::NEG_INFINITY, 0.55, 0.5),
        0.0
    );

    // Capital ratio muy bajo o negativo se acota limpiamente a 0.01 sin NaN
    let scale_neg = continuous_capital_scale(-5.0, PHI, 1.0, 0.35, 0.55, 0.5);
    assert!(scale_neg.is_finite() && scale_neg > 0.0);

    // calculate_kelly_fraction con entradas anómalas
    let f_nan = calculate_kelly_fraction(
        0.55, 1.5, 13.0, 13.0, f64::NAN, 1.0, 0.01, 1.0, 0.35, 0.5,
    );
    assert_eq!(f_nan, 0.0);
}
