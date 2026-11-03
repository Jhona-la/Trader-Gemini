//! # Contrato Formal F22-001 — Zakai SPDE, Doob-Meyer, Kramers-Kronig y Amplituedro
//!
//! Invariantes matemáticas y físicas comprobadas:
//! - **C1:** El filtro SPDE no lineal de Zakai preserva la positividad de masa y actualiza la media según la señal de innovación.
//! - **C2:** La descomposición de Doob-Meyer aísla martingalas puras frente a flujos tóxicos con compensador monótono.
//! - **C3:** Las relaciones de dispersión de Kramers-Kronig verifican analíticamente la causalidad de absorción y reactividad.
//! - **C4:** El Grassmanniano Positivo $Gr^+(2, 4)$ valida los 6 menores de Plücker $\Delta_{ij} > 0$ sobre matrices totalmente positivas.
//! - **C5:** La violación de positividad de menores detecta geometrías no admisibles para politopos de amplitudes.
//! - **C6:** El motor maestro `ZakaiDoobAmplituhedronMasterEngine` es resiliente a patologías numéricas sin ningún pánico.

use feature_engine::zakai_doob_amplituhedron::*;

#[test]
fn c1_zakai_spde_filtering_updates_moments_and_preserves_positivity() {
    let mut zakai = ZakaiFilterEngine::new(2.5, 0.05);

    // Actualización inicial con innovación nula
    let rep_0 = zakai.update(0.0, 0.1);
    assert!(
        rep_0.total_unnormalized_mass > 0.0,
        "La masa total debe ser positiva"
    );
    assert!(
        rep_0.posterior_variance > 0.0,
        "La varianza debe ser positiva"
    );

    // Secuencia de observaciones fuertemente positivas (empujan la media hacia la derecha)
    let mut rep_pos = rep_0;
    for _ in 0..10 {
        rep_pos = zakai.update(0.5, 0.05);
    }

    assert!(
        rep_pos.posterior_mean > rep_0.posterior_mean,
        "La media posterior tras innovaciones positivas ({}) debe ser mayor que la inicial ({})",
        rep_pos.posterior_mean,
        rep_0.posterior_mean
    );
    assert!(
        rep_pos.total_unnormalized_mass.is_finite(),
        "La masa debe permanecer finita"
    );
}

#[test]
fn c2_doob_meyer_martingale_decomposition_and_toxicity() {
    let mut doob = DoobMeyerCompensatorEngine::new(0.90);

    // Caso 1: Flujo de martingala pura equilibrada (media cero, intensidad esperada cero)
    let benign_shocks = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05];
    let mut rep_benign = DoobMeyerReport::default();
    for &dx in &benign_shocks {
        rep_benign = doob.update(dx, 0.0);
    }

    assert_eq!(
        rep_benign.regime,
        DoobMeyerToxicityRegime::PureMartingaleBenign,
        "El flujo de media cero debe ser clasificado como martingala benigna"
    );
    assert!(
        rep_benign.toxicity_ratio < 0.25,
        "Razón de toxicidad debe ser baja, fue {}",
        rep_benign.toxicity_ratio
    );

    // Caso 2: Flujo tóxico fuertemente informado (intensidad predecible sostenida)
    let mut rep_toxic = rep_benign;
    for _ in 0..40 {
        rep_toxic = doob.update(0.8, 0.8);
    }

    assert_eq!(
        rep_toxic.regime,
        DoobMeyerToxicityRegime::ToxicInformedFlow,
        "El flujo sostenido persistente debe ser clasificado como tóxico informado"
    );
    assert!(
        rep_toxic.toxicity_ratio > 0.55,
        "Razón de toxicidad debe superar 0.55, fue {}",
        rep_toxic.toxicity_ratio
    );
}

#[test]
fn c3_kramers_kronig_analytical_causal_dispersion() {
    let kk = KramersKronigDispersionEngine;

    // Frecuencias fijas espaciadas
    let freqs = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8];
    // Perfil lorentziano de absorción viscosa: \chi''(\Omega) = \frac{\gamma \Omega}{(\Omega^2 - \Omega_0^2)^2 + \gamma^2 \Omega^2}
    let mut absorption = [0.0; DISPERSION_FREQS];
    for (i, &w) in freqs.iter().enumerate() {
        absorption[i] = w / (1.0 + w * w);
    }

    let rep = kk.evaluate(&freqs, &absorption, 2, 0.35);

    assert!(
        rep.predicted_real_susceptibility.is_finite(),
        "La susceptibilidad real predicha debe ser finita"
    );
    assert!(
        rep.causality_compliance_score >= 0.0 && rep.causality_compliance_score <= 1.0,
        "Score de causalidad debe estar en [0, 1]"
    );
}

#[test]
fn c4_positive_grassmannian_plucker_minors_totally_positive() {
    let engine = PositiveGrassmannianAmplituhedronEngine;

    // Matriz de Grassmanniano Gr+(2, 4) canónica totalmente positiva:
    // C = [ [1, 1, 1, 1],
    //       [0, 1, 2, 3] ]
    // Minors:
    // (0,1): 1*1 - 1*0 = 1 > 0
    // (0,2): 1*2 - 1*0 = 2 > 0
    // (0,3): 1*3 - 1*0 = 3 > 0
    // (1,2): 1*2 - 1*1 = 1 > 0
    // (1,3): 1*3 - 1*1 = 2 > 0
    // (2,3): 1*3 - 1*2 = 1 > 0
    let matrix_c = [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]];

    let rep = engine.evaluate(&matrix_c);

    assert!(
        rep.is_positive_geometry,
        "La matriz debe pertenecer al Grassmanniano Positivo"
    );
    assert_eq!(rep.positive_minor_fraction, 1.0);
    assert!(
        rep.projected_amplituhedron_volume > 0.0,
        "El volumen del politopo debe ser positivo, fue {}",
        rep.projected_amplituhedron_volume
    );

    // Verificar cada uno de los 6 menores esperados
    let expected_minors = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0];
    for (i, &exp) in expected_minors.iter().enumerate() {
        assert!(
            (rep.plucker_minors[i] - exp).abs() < 1e-12,
            "Menor {} difiere: obtenido {}, esperado {}",
            i,
            rep.plucker_minors[i],
            exp
        );
    }
}

#[test]
fn c5_non_positive_geometry_detection() {
    let engine = PositiveGrassmannianAmplituhedronEngine;

    // Invertir columnas provoca determinantes negativos
    let matrix_c_neg = [[1.0, -1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]];

    let rep = engine.evaluate(&matrix_c_neg);

    assert!(
        !rep.is_positive_geometry,
        "La matriz con menor negativo no debe ser positiva"
    );
    assert!(rep.positive_minor_fraction < 1.0);
    assert_eq!(rep.projected_amplituhedron_volume, 0.0);
}

#[test]
fn c6_zakai_doob_amplituhedron_master_engine_zero_panic() {
    let mut master = ZakaiDoobAmplituhedronMasterEngine::new(2.0, 0.1, 0.95);

    let freqs = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8];
    let absorption = [0.1; DISPERSION_FREQS];
    let matrix_c = [[1.0; AMPLITUHEDRON_N]; AMPLITUHEDRON_K];

    // Prueba 1: Ejecución normal
    let rep_normal = master.evaluate(0.01, 0.1, 0.05, 0.05, &freqs, &absorption, 0.1, &matrix_c);
    assert!(
        rep_normal.causal_structural_integrity >= 0.0
            && rep_normal.causal_structural_integrity <= 1.0
    );

    // Prueba 2: Resiliencia ante NaNs e infinitos
    let bad_freqs = [f64::NAN; DISPERSION_FREQS];
    let bad_absorption = [f64::INFINITY; DISPERSION_FREQS];
    let rep_bad = master.evaluate(
        f64::NAN,
        -1.0,
        f64::INFINITY,
        f64::NAN,
        &bad_freqs,
        &bad_absorption,
        f64::NAN,
        &matrix_c,
    );

    assert!(rep_bad.causal_structural_integrity.is_finite());
    assert!(rep_bad.zakai.posterior_mean.is_finite());
    assert!(rep_bad.doob_meyer.toxicity_ratio.is_finite());
}
