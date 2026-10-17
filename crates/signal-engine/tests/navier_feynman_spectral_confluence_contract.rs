//! Formal Contract Tests for Ola Ω146: Navier-Stokes Manifold & Feynman Propagator
//! 15-Motor Spectral Consensus Integration & Prequential Skill Calibration.

use signal_engine::skill_motores::{
    SkillMotores, MOTOR_FEYNMAN, MOTOR_NAVIER_STOKES, MOTORES, NOMBRES_MOTORES,
    NUM_PARES_MOTOR_ESCALA, PISO_EXPLORACION,
};
use signal_engine::voto_espectral::{ESCALAS_VOTO, VotoEspectral};

#[test]
fn contract_15_motors_constants_and_family_multiplicity() {
    assert_eq!(
        MOTORES, 15,
        "Spectral consensus must encompass exactly 15 physics/quantum engines"
    );
    assert_eq!(
        NUM_PARES_MOTOR_ESCALA,
        15 * 32,
        "Ville Bonferroni family multiplicity must equal 15 engines * 32 scales = 480"
    );
    assert_eq!(NOMBRES_MOTORES.len(), 15);
    assert_eq!(NOMBRES_MOTORES[MOTOR_NAVIER_STOKES], "navier_stokes");
    assert_eq!(NOMBRES_MOTORES[MOTOR_FEYNMAN], "feynman");
    assert_eq!(MOTOR_NAVIER_STOKES, 13);
    assert_eq!(MOTOR_FEYNMAN, 14);
}

#[test]
fn contract_cold_start_15_motor_weights_identical_to_equal_weights() {
    let sm = SkillMotores::new();
    let pesos = sm.pesos();

    // Verify all 15 motors start at exploration floor across all 32 scales
    for m in 0..MOTORES {
        for k in 0..ESCALAS_VOTO {
            assert_eq!(
                pesos[m][k], PISO_EXPLORACION,
                "Motor {m} scale {k} must start at PISO_EXPLORACION"
            );
        }
    }

    // Build 15 synthetic votes with different patterns
    let mut votos = Vec::with_capacity(MOTORES);
    for m in 0..MOTORES {
        let val = (m as f64 - 7.0) / 10.0; // from -0.7 to +0.7
        votos.push(VotoEspectral::desde_escalar(val));
    }

    let c_pesos_iguales = VotoEspectral::consenso(&votos, &[1.0; MOTORES]);
    let c_consenso_escala = VotoEspectral::consenso_por_escala(&votos, &pesos);

    for k in 0..ESCALAS_VOTO {
        let diff = (c_pesos_iguales.en_escala(k) - c_consenso_escala.en_escala(k)).abs();
        assert!(
            diff < 1e-12,
            "Scale {k}: equal weight consensus ({}) must match cold-start scale consensus ({}) within FP epsilon",
            c_pesos_iguales.en_escala(k),
            c_consenso_escala.en_escala(k)
        );
    }
}

#[test]
fn contract_navier_stokes_and_feynman_influence_in_spectral_consensus() {
    let sm = SkillMotores::new();
    let pesos = sm.pesos();

    // 13 engines neutral (0.0), Navier-Stokes bullish (+0.8), Feynman bullish (+0.6)
    let mut votos = Vec::with_capacity(MOTORES);
    for m in 0..MOTORES {
        if m == MOTOR_NAVIER_STOKES {
            votos.push(VotoEspectral::desde_escalar(0.80));
        } else if m == MOTOR_FEYNMAN {
            votos.push(VotoEspectral::desde_escalar(0.60));
        } else {
            votos.push(VotoEspectral::desde_escalar(0.00));
        }
    }

    let c = VotoEspectral::consenso_por_escala(&votos, &pesos);

    // Expected weighted average across 15 motors in cold start (all weights = 0.15):
    // sum_wv = 0.15 * 0.80 + 0.15 * 0.60 = 0.15 * 1.40 = 0.21
    // sum_w = 15 * 0.15 = 2.25
    // expected = 0.21 / 2.25 = 1.40 / 15 ≈ 0.09333333333333334
    for k in 0..ESCALAS_VOTO {
        let v_k = c.en_escala(k);
        let expected = 1.40 / 15.0;
        assert!(
            (v_k - expected).abs() < 1e-12,
            "Scale {k}: Navier-Stokes and Feynman must actively drive the consensus vote: {v_k} vs {expected}"
        );
    }

    // Contrast with the previous dead-edge bug where only 13 motors were included:
    // With 13 motors omitting Navier-Stokes and Feynman, the consensus was 0.00!
    assert!(c.en_escala(0) > 0.05, "Consensus must not be zeroed out");
}

#[test]
fn contract_prequential_skill_learning_for_navier_and_feynman() {
    let mut sm = SkillMotores::new();
    let scale_k = 19; // ~137 s scale

    // Initial setup: arm motors at scale_k
    let mut votos_armado = vec![VotoEspectral::desde_escalar(0.0); MOTORES];
    votos_armado[MOTOR_NAVIER_STOKES] = VotoEspectral::desde_escalar(1.0);
    votos_armado[MOTOR_FEYNMAN] = VotoEspectral::desde_escalar(0.9);

    sm.observar_maduracion(scale_k, 1_000, 0.0, &votos_armado);

    // Observe aligned block returns (r = +0.04) across consecutive block maturations
    // Ville e-process family threshold for M=480: 480 / 0.05 = 9600
    // Factor per aligned step = 1 + 0.10 * 1 = 1.10
    // 1.10^97 ≈ 10334 > 9600.
    for i in 2..=105u64 {
        sm.observar_maduracion(scale_k, i * 1_000, 0.04, &votos_armado);
    }

    let peso_navier = sm.peso(MOTOR_NAVIER_STOKES, scale_k);
    let peso_feynman = sm.peso(MOTOR_FEYNMAN, scale_k);
    let peso_neutral = sm.peso(0, scale_k);

    assert_eq!(
        peso_neutral, PISO_EXPLORACION,
        "Neutral motor without active votes must remain at exploration floor"
    );
    assert!(
        peso_navier > PISO_EXPLORACION + 0.10,
        "Navier-Stokes skill weight must increase with aligned prequential evidence: {peso_navier}"
    );
    assert!(
        peso_feynman > PISO_EXPLORACION + 0.10,
        "Feynman Propagator skill weight must increase with aligned prequential evidence: {peso_feynman}"
    );

    let (maduros, pmax) = sm.diagnostico_banda();
    assert!(
        maduros >= 2,
        "Band diagnostics must register at least 2 mature motor-scale pairs: {maduros}"
    );
    assert!(
        pmax > PISO_EXPLORACION,
        "Max weight in band must exceed exploration floor: {pmax}"
    );
}

#[test]
fn contract_anti_aligned_navier_feynman_retains_exploration_floor_without_inversion() {
    let mut sm = SkillMotores::new();
    let scale_k = 12;

    // Both engines vote long (+1.0) but market drops continuously (r = -0.05)
    let mut votos_contra = vec![VotoEspectral::desde_escalar(0.0); MOTORES];
    votos_contra[MOTOR_NAVIER_STOKES] = VotoEspectral::desde_escalar(1.0);
    votos_contra[MOTOR_FEYNMAN] = VotoEspectral::desde_escalar(1.0);

    sm.observar_maduracion(scale_k, 1_000, 0.0, &votos_contra);

    for i in 2..=80u64 {
        sm.observar_maduracion(scale_k, i * 1_000, -0.05, &votos_contra);
    }

    let peso_navier = sm.peso(MOTOR_NAVIER_STOKES, scale_k);
    let peso_feynman = sm.peso(MOTOR_FEYNMAN, scale_k);

    assert_eq!(
        peso_navier, PISO_EXPLORACION,
        "Anti-aligned Navier-Stokes must remain at exploration floor, never inverted"
    );
    assert_eq!(
        peso_feynman, PISO_EXPLORACION,
        "Anti-aligned Feynman must remain at exploration floor, never inverted"
    );
}
