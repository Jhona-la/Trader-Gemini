//! # F13-001: Contratos Formales de Geometría No Conmutativa de Connes, Hudson-Parthasarathy & Redes KAN Clifford
//!
//! Certifica los 6 invariantes físico-matemáticos del módulo `connes_clifford_kan`:
//! - **C1:** Axiomas métricos de la distancia espectral de Connes $d_{\text{Connes}}(\rho_1, \rho_2)$ y conmutador $[\hat{X}, \hat{P}]$.
//! - **C2:** Conservación exacta de traza $\text{Tr}(\rho_t) \equiv 1$, positividad en la bola de Bloch y conteo de saltos de Poisson en la QSDE de Hudson-Parthasarathy.
//! - **C3:** Axiomas del Álgebra Geométrica de Clifford $\mathcal{C}\ell(3, 0)$ ($e_i^2 = 1$, $e_i e_j = -e_j e_i$, $e_{123}^2 = -1$ y asociatividad $(AB)C = A(BC)$).
//! - **C4:** Isometría exacta de rotores espinoriales $M' = R M \tilde{R}$ ($\|M'\|_2 = \|M\|_2$) y rotación exacta de $\pi/2$ en el plano $e_{12}$.
//! - **C5:** Partición exacta de la unidad ($\sum_{m=0}^3 B_m(t) \equiv 1$) y continuidad $C^2$ de las aristas KAN B-Spline cúbicas.
//! - **C6:** Integración E2E del motor `ConnesCliffordKanEngine` y seguridad total (cero pánicos) ante entradas corruptas (`NaN`, `±Inf`, vacíos).

use feature_engine::{
    CliffordMultivector3D, ConnesCliffordKanEngine, ConnesSpectralTripleEngine,
    CubicBSplineKanEdge, HudsonParthasarathyEvolutor, HudsonParthasarathyState,
    NonCommutativeRegime,
};

#[test]
fn test_c1_connes_spectral_distance_metric_axioms() {
    let engine = ConnesSpectralTripleEngine::new(1.0, 2.5);
    let prices = [100.0, 101.0, 102.0, 103.0];
    let dist_a = [10.0, 0.0, 0.0, 0.0];
    let dist_b = [0.0, 10.0, 0.0, 0.0];
    let dist_c = [0.0, 0.0, 0.0, 10.0];
    let uniform_flow = [5.0, 5.0, 5.0, 5.0];
    let gradient_flow = [12.0, -4.0, 8.0, -6.0];

    // 1. Identidad de indiscernibles: d(a, a) == 0 y flujo uniforme => [\hat{X}, \hat{P}] == 0
    let rep_same = engine.evaluate_spectral_triple(&prices, &dist_a, &dist_a, &uniform_flow);
    assert!(rep_same.connes_spectral_distance.abs() < 1e-12);
    assert!(rep_same.planck_market_constant.abs() < 1e-12);

    // 2. Simetría: d(a, b) == d(b, a)
    let rep_ab = engine.evaluate_spectral_triple(&prices, &dist_a, &dist_b, &gradient_flow);
    let rep_ba = engine.evaluate_spectral_triple(&prices, &dist_b, &dist_a, &gradient_flow);
    assert!((rep_ab.connes_spectral_distance - rep_ba.connes_spectral_distance).abs() < 1e-12);
    assert!(rep_ab.planck_market_constant > 0.0);

    // 3. Monotonía con transporte espacial: mover masa del nivel 0 al 3 cuesta 3x más que del 0 al 1
    let rep_ac = engine.evaluate_spectral_triple(&prices, &dist_a, &dist_c, &gradient_flow);
    assert!(rep_ac.connes_spectral_distance > rep_ab.connes_spectral_distance);
    assert!((rep_ac.connes_spectral_distance - 3.0 * rep_ab.connes_spectral_distance).abs() < 1e-9);
}

#[test]
fn test_c2_hudson_parthasarathy_qsde_trace_and_bloch_positivity() {
    let evolutor = HudsonParthasarathyEvolutor::default();
    let mut state = HudsonParthasarathyState::default();
    let mut expected_jumps = 0u64;

    for step in 0..200u32 {
        let drive = ((step as f64) * 0.1).sin() * 1.5;
        let da = ((step as f64) * 0.37).cos() * 0.8;
        let jump = step.is_multiple_of(17);
        if jump {
            expected_jumps += 1;
        }
        state = evolutor.step_qsde(&state, drive, da, jump, 0.01);

        assert!((state.trace - 1.0).abs() < 1e-12);
        let bloch_norm_sq = state.bloch_x * state.bloch_x
            + state.bloch_y * state.bloch_y
            + state.bloch_z * state.bloch_z;
        assert!(bloch_norm_sq <= 1.0 + 1e-12);
        assert!(state.purity >= 0.5 - 1e-12 && state.purity <= 1.0 + 1e-12);
    }

    assert_eq!(state.gauge_jumps_observed, expected_jumps);
}

#[test]
fn test_c3_clifford_geometric_algebra_associativity_and_anticommutation() {
    let e1 = CliffordMultivector3D::from_vector(1.0, 0.0, 0.0);
    let e2 = CliffordMultivector3D::from_vector(0.0, 1.0, 0.0);
    let e3 = CliffordMultivector3D::from_vector(0.0, 0.0, 1.0);

    // 1. e_i^2 = +1
    assert_eq!(
        e1.geometric_product(&e1),
        CliffordMultivector3D::from_scalar(1.0)
    );
    assert_eq!(
        e2.geometric_product(&e2),
        CliffordMultivector3D::from_scalar(1.0)
    );
    assert_eq!(
        e3.geometric_product(&e3),
        CliffordMultivector3D::from_scalar(1.0)
    );

    // 2. Anticonmutación: e1 * e2 = - e2 * e1 = e12
    let e12 = e1.geometric_product(&e2);
    let e21 = e2.geometric_product(&e1);
    assert_eq!(e12.coeffs[4], 1.0);
    assert_eq!(e21.coeffs[4], -1.0);

    // 3. Pseudoescalar e123^2 = -1
    let e123 = e12.geometric_product(&e3);
    assert_eq!(e123.coeffs[7], 1.0);
    let e123_sq = e123.geometric_product(&e123);
    assert_eq!(e123_sq, CliffordMultivector3D::from_scalar(-1.0));

    // 4. Asociatividad exacta: (A * B) * C == A * (B * C)
    let a = CliffordMultivector3D {
        coeffs: [0.5, -1.2, 0.8, 2.1, -0.4, 1.3, -0.9, 0.7],
    };
    let b = CliffordMultivector3D {
        coeffs: [-0.3, 0.9, -1.5, 0.4, 1.1, -0.6, 0.2, -1.0],
    };
    let c = CliffordMultivector3D {
        coeffs: [1.4, 0.2, -0.7, -1.1, 0.8, 0.5, -1.3, 0.3],
    };

    let lhs = a.geometric_product(&b).geometric_product(&c);
    let rhs = a.geometric_product(&b.geometric_product(&c));
    for (&l, &r) in lhs.coeffs.iter().zip(rhs.coeffs.iter()) {
        assert!((l - r).abs() < 1e-11);
    }
}

#[test]
fn test_c4_clifford_spinor_rotor_isometry_and_90deg_rotation() {
    // Rotación de +90 grados (pi/2 rad) en el plano e1 ^ e2 lleva e1 -> e2
    let rotor_90 = CliffordMultivector3D::rotor_from_bivector_angle(
        1.0,
        0.0,
        0.0,
        std::f64::consts::FRAC_PI_2,
    );
    let e1 = CliffordMultivector3D::from_vector(1.0, 0.0, 0.0);
    let rotated = e1.apply_rotor(&rotor_90);

    assert!(rotated.coeffs[0].abs() < 1e-12);
    assert!(rotated.coeffs[1].abs() < 1e-12);
    assert!((rotated.coeffs[2] - 1.0).abs() < 1e-12);
    assert!(rotated.coeffs[3].abs() < 1e-12);

    // Conservación exacta de la norma euclidiana en un multivector general
    let general = CliffordMultivector3D {
        coeffs: [1.0, -2.0, 3.5, -0.5, 1.2, -0.8, 2.4, -1.5],
    };
    let rotor_arbitrary = CliffordMultivector3D::rotor_from_bivector_angle(0.6, -1.1, 0.9, 1.2345);
    let transformed = general.apply_rotor(&rotor_arbitrary);
    assert!((general.norm_sq() - transformed.norm_sq()).abs() < 1e-10);
}

#[test]
fn test_c5_cubic_bspline_partition_of_unity_and_kan_continuity() {
    for i in 0..=100 {
        let t = (i as f64) / 100.0;
        let weights = CubicBSplineKanEdge::cubic_bspline_basis_weights(t);
        let mut sum = 0.0_f64;
        for &w in &weights {
            assert!(w >= -1e-15);
            sum += w;
        }
        assert!((sum - 1.0).abs() < 1e-14);
    }

    // Continuidad suave de la arista KAN a lo largo de [-1, 1]
    let edge = CubicBSplineKanEdge::default();
    let mut prev = edge.evaluate(-1.0);
    for i in 1..=200 {
        let x = -1.0 + 2.0 * (i as f64) / 200.0;
        let curr = edge.evaluate(x);
        assert!(curr.is_finite());
        assert!((curr - prev).abs() < 0.05);
        prev = curr;
    }
}

#[test]
fn test_c6_full_connes_clifford_kan_cycle_and_adversarial_safety() {
    let engine = ConnesCliffordKanEngine::default();
    let prices = [100.0, 100.5, 101.0, 101.5];
    let bids = [25.0, 15.0, 10.0, 5.0];
    let asks = [5.0, 10.0, 15.0, 25.0];
    let flows = [4.5, 2.0, -0.5, -1.0];

    let mut qsde = HudsonParthasarathyState::default();
    for _ in 0..10 {
        let rep = engine.evaluate_cycle(&prices, &bids, &asks, &flows, &qsde, 0.05);
        assert!(rep.directional_conviction.is_finite());
        assert!(rep.directional_conviction >= -1.0 && rep.directional_conviction <= 1.0);
        qsde = rep.qsde_state;
    }

    // Caso adversarial con NaN / Inf / vacíos
    let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -99.0];
    let bad_rep =
        engine.evaluate_cycle(&bad_vals, &bad_vals, &bad_vals, &bad_vals, &qsde, f64::NAN);
    assert!(bad_rep.directional_conviction.is_finite());
    assert!(bad_rep.connes.connes_spectral_distance.is_finite());
    assert!(bad_rep.qsde_state.purity.is_finite());

    let empty_rep = engine.evaluate_cycle(
        &[],
        &[],
        &[],
        &[],
        &HudsonParthasarathyState::default(),
        0.01,
    );
    assert_eq!(
        empty_rep.regime,
        NonCommutativeRegime::HeisenbergUncertaintyVeto
    );
}
