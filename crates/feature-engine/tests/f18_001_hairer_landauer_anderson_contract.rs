//! Suite de Contratos Formales F18-001 — Estructuras de Regularidad de Martin Hairer (Paraproducto de Bony & BPHZ),
//! Conductancia Cuántica de Landauer-Büttiker y Localización de Anderson por Matrices de Transferencia Simplécticas.
//!
//! Verifica las 6 propiedades físico-matemáticas de la Fase 18:
//! - C1: Propiedad de grupo de estructura de Hairer $\Gamma_{h_1} \circ \Gamma_{h_2} \equiv \Gamma_{h_1 + h_2}$ con error $\le 10^{-15}$.
//! - C2: Descomposición exacta de Paraproducto de Bony ($f \cdot g = \Pi_<(f,g) + \Pi_>(f,g) + \Pi_\circ(f,g)$) y sustracción del contratérmino ultravioleta BPHZ $C_\epsilon$.
//! - C3: Transparencia balística en resonancia de Landauer-Büttiker ($T_n = 1.0 \implies G/G_0 = M$, Ruido de Disparo $S_{\text{shot}} = 0$, Factor de Fano $F = 0$).
//! - C4: Máximo de Ruido de Disparo de Büttiker en semi-transparencia ($T_n = 0.5 \implies F = 0.5$) y límite Poissoniano de túnel débil ($T_n \to 0 \implies F \to 1$).
//! - C5: Invariante simpléctica $\det M_k \equiv 1.0$ y transición de banda conductora ordenada ($\gamma_{\text{loc}} \approx 0$) a localización exponencial de Anderson bajo desorden fuerte ($\xi_{\text{loc}} < 1.0$).
//! - C6: Inmunidad total frente a entradas `NaN` / `±Inf` en `HairerLandauerAndersonMasterEngine`.

use feature_engine::{
    AndersonLocalizationEngine, HairerLandauerAndersonMasterEngine, HairerRegularityEngine,
    LandauerButtikerEngine, MesoscopicTransportRegime, MESOSCOPIC_CHANNELS,
};

#[test]
fn c1_hairer_structure_group_cocycle_identity() {
    let base_coords = [1.5, -0.8, 2.2, 0.4, -1.1];
    let h1 = 0.65;
    let h2 = -0.25;

    let step1 = HairerRegularityEngine::apply_structure_group_gamma(&base_coords, h1);
    let composed = HairerRegularityEngine::apply_structure_group_gamma(&step1, h2);
    let direct = HairerRegularityEngine::apply_structure_group_gamma(&base_coords, h1 + h2);

    for i in 0..5 {
        assert!(
            (composed[i] - direct[i]).abs() < 1e-14,
            "Hairer structure group cocycle failed at index {i}"
        );
    }
}

#[test]
fn c2_bony_paraproduct_and_bphz_renormalization_subtraction() {
    let engine = HairerRegularityEngine::default();
    let f = [2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0]; // media = 3.0, fluctuación = ±1.0
    let g = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]; // media = 2.0, fluctuación = ±1.0

    // Sin varianza ultravioleta (sigma^2 = 0 => C_eps = 0), la suma de parte suave (3*2=6) + resonante (1*1=1) reproduce la media exacta de f*g = (2+12)/2 = 7.0
    let rep_classical = engine.evaluate_renormalized_product(&f, &g, 0.0);
    assert!((rep_classical.paraproduct_smooth_part - 6.0).abs() < 1e-14);
    assert!((rep_classical.resonant_raw_term - 1.0).abs() < 1e-14);
    assert!((rep_classical.bphz_counterterm_c_eps - 0.0).abs() < 1e-14);
    assert!((rep_classical.renormalized_wick_product - 7.0).abs() < 1e-14);
    assert!(rep_classical.structure_group_invariance_error < 1e-14);

    // Con varianza ultravioleta (sigma^2 = 1.0), BPHZ sustrae C_eps > 0 del término resonante
    let rep_renorm = engine.evaluate_renormalized_product(&f, &g, 1.0);
    assert!(rep_renorm.bphz_counterterm_c_eps > 0.0);
    assert!(
        (rep_renorm.renormalized_wick_product - (7.0 - rep_renorm.bphz_counterterm_c_eps)).abs()
            < 1e-13
    );
}

#[test]
fn c3_landauer_buttiker_ballistic_transparency_zero_shot_noise() {
    let engine = LandauerButtikerEngine {
        coupling_width_gamma: 1.0,
    };
    // Cuando E == V_n en todos los canales, T_n(E) == 1.0 (transmisión balística perfecta)
    let barriers = [2.5; MESOSCOPIC_CHANNELS];
    let rep = engine.evaluate_conductance(&barriers, 2.5);

    assert!((rep.dimensionless_conductance - (MESOSCOPIC_CHANNELS as f64)).abs() < 1e-14);
    assert!(rep.buttiker_shot_noise.abs() < 1e-14);
    assert!(rep.quantum_fano_factor.abs() < 1e-14);
}

#[test]
fn c4_landauer_buttiker_half_transmission_and_poissonian_tunnel_limit() {
    let engine = LandauerButtikerEngine {
        coupling_width_gamma: 1.0,
    };
    // Cuando |E - V_n| == Gamma, T_n = Gamma^2 / (Gamma^2 + Gamma^2) = 0.5 => Fano Factor F == 0.5
    let barriers_half = [1.0; MESOSCOPIC_CHANNELS];
    let rep_half = engine.evaluate_conductance(&barriers_half, 2.0);
    assert!((rep_half.dimensionless_conductance - 4.0).abs() < 1e-14);
    assert!((rep_half.quantum_fano_factor - 0.5).abs() < 1e-14);

    // En barrera muy alta (|E - V_n| >> Gamma), T_n -> 0 y el Factor de Fano tiende al límite Poissoniano F -> 1.0
    let barriers_high = [100.0; MESOSCOPIC_CHANNELS];
    let rep_tunnel = engine.evaluate_conductance(&barriers_high, 0.0);
    assert!(rep_tunnel.dimensionless_conductance < 1e-2);
    assert!(rep_tunnel.quantum_fano_factor > 0.99);
}

#[test]
fn c5_anderson_localization_unimodular_det_and_disorder_transition() {
    let engine = AndersonLocalizationEngine;

    // 1. Retículo periódico perfectamente ordenado (eps_k = 0) dentro de la banda conductora |E| < 2 (E = 0):
    // M_k = [[0, -1], [1, 0]] es una rotación ortogonal pura => ||psi|| constante, gamma_loc == 0
    let ordered = [0.0; MESOSCOPIC_CHANNELS];
    let rep_ordered = engine.evaluate_localization(&ordered, 0.0);
    assert!((rep_ordered.transfer_matrix_det - 1.0).abs() < 1e-15);
    assert!(rep_ordered.furstenberg_lyapunov_exponent < 1e-12);
    assert!(rep_ordered.localization_length_xi >= 1.0e5);

    // 2. Retículo con desorden fuerte (spoofing/muros erráticos |eps_k| >> 2):
    // Induce crecimiento exponencial de Furstenberg y localización fuerte (xi_loc < 1.0)
    let disordered = [15.0, -22.0, 18.0, -30.0, 25.0, -19.0, 28.0, -24.0];
    let rep_disordered = engine.evaluate_localization(&disordered, 0.0);
    assert!((rep_disordered.transfer_matrix_det - 1.0).abs() < 1e-15);
    assert!(rep_disordered.furstenberg_lyapunov_exponent > 2.0);
    assert!(rep_disordered.localization_length_xi < 0.5);
}

#[test]
fn c6_nan_inf_adversarial_immunity_in_master_engine() {
    let master = HairerLandauerAndersonMasterEngine::default();
    let bad_f = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0,
        -2.0,
        0.0,
        3.0,
        f64::NAN,
    ];
    let bad_g = [
        1.0,
        f64::NAN,
        2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.0,
        0.5,
    ];

    let rep = master.evaluate_cycle(&bad_f, &bad_g, f64::NAN, &bad_f, f64::INFINITY);
    assert!(rep.hairer.renormalized_wick_product.is_finite());
    assert!(rep.landauer.dimensionless_conductance.is_finite());
    assert!(rep.anderson.localization_length_xi.is_finite());
    assert!(rep.mesoscopic_conviction.is_finite());
    assert!(matches!(
        rep.regime,
        MesoscopicTransportRegime::BallisticConductingChannel
            | MesoscopicTransportRegime::DiffusiveMetallicFlow
            | MesoscopicTransportRegime::AndersonLocalizedInsulatorVeto
    ));
}
