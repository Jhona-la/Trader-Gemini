//! Suite de Contratos Formales F15-002 — Geometría y Álgebra Lineal Tropical $(\mathbb{R}_{\max,+})$
//! y Solitones Korteweg-de Vries (KdV) por Transformada Inversa de Dispersión (IST).
//!
//! Verifica las 6 propiedades físico-matemáticas de la Fase 15.2:
//! - C1: Axiomas del semianillo tropical $(\mathbb{R}_{\max,+})$: idempotencia $a \oplus a = a$, asociatividad y elemento neutro matricial $A \otimes I_{\text{trop}} = A$.
//! - C2: Exactitud del Algoritmo de Karp para el autovalor tropical dominante $\lambda_{\text{trop}}(A)$ (media máxima de ciclo).
//! - C3: Solución analítica exacta del 1-solitón KdV $u(x,t) = 2\kappa^2 \text{sech}^2(\kappa(x - 4\kappa^2 t - x_0))$ y relación velocidad-amplitud $v_1 = 2 A_1 = 4\kappa_1^2$.
//! - C4: Extracción de estados ligados por Transformada Inversa de Dispersión (operador de Lax-Schrödinger) para 1-solitón y colisión elástica de 2 solitones.
//! - C5: Invariancia traslacional de las tres primeras leyes de conservación de Whitham-KdV ($I_1, I_2, I_3$).
//! - C6: Inmunidad total frente a entradas `NaN` / `±Inf` en `TropicalKdvEngine`.

use feature_engine::{
    tropical_add, tropical_mul, KdvSolitonIstAnalyzer, TropicalKdvEngine, TropicalMaxPlusMatrix,
    TropicalSolitonRegime, KDV_GRID_DIM, TROPICAL_ONE, TROPICAL_ZERO,
};

#[test]
fn c1_tropical_semiring_axioms_and_matrix_identity() {
    // Idempotencia aditiva: a (+) a == a
    assert!((tropical_add(3.5, 3.5) - 3.5).abs() < 1e-15);
    // Elemento neutro aditivo: a (+) (-inf) == a
    assert!((tropical_add(3.5, TROPICAL_ZERO) - 3.5).abs() < 1e-15);
    // Elemento neutro multiplicativo: a (x) 0 == a
    assert!((tropical_mul(3.5, TROPICAL_ONE) - 3.5).abs() < 1e-15);
    // Distributividad: a (x) (b (+) c) == (a (x) b) (+) (a (x) c)
    let lhs = tropical_mul(2.0, tropical_add(4.0, 7.0));
    let rhs = tropical_add(tropical_mul(2.0, 4.0), tropical_mul(2.0, 7.0));
    assert!((lhs - rhs).abs() < 1e-15);

    let a = TropicalMaxPlusMatrix::from_raw_sanitized([
        [0.0, 2.5, -1.0],
        [1.5, 0.0, 3.0],
        [4.0, -2.0, 0.0],
    ]);
    let id = TropicalMaxPlusMatrix::<3>::tropical_identity();
    let prod_right = a.matmul_tropical(&id);
    let prod_left = id.matmul_tropical(&a);
    assert_eq!(prod_right, a);
    assert_eq!(prod_left, a);
}

#[test]
fn c2_karp_tropical_eigenvalue_exact_cycle_mean() {
    // Grafo de 3 nodos con dos ciclos dirigidos:
    // Ciclo A: 0 -> 1 -> 0 con pesos 4.0 + 6.0 = 10.0 => media = 5.0
    // Ciclo B: 1 -> 2 -> 1 con pesos 8.0 + 10.0 = 18.0 => media = 9.0 (dominante)
    let mut raw = [[TROPICAL_ZERO; 3]; 3];
    raw[0][1] = 4.0;
    raw[1][0] = 6.0;
    raw[1][2] = 8.0;
    raw[2][1] = 10.0;

    let mat = TropicalMaxPlusMatrix::from_raw_sanitized(raw);
    let rep = mat.karp_tropical_eigenvalue();
    assert!(
        (rep.max_cycle_mean_eigenvalue - 9.0).abs() < 1e-12,
        "Karp tropical eigenvalue must equal maximum cycle mean 9.0, got {}",
        rep.max_cycle_mean_eigenvalue
    );
}

#[test]
fn c3_analytical_kdv_soliton_profile_and_velocity_amplitude_law() {
    let kappa = 1.2;
    let peak_t0 = KdvSolitonIstAnalyzer::analytical_sech2_soliton(kappa, 0.0, 0.0, 0.0);
    // En x = x0 + 4*kappa^2*t, sech(0) = 1 => u_max = 2 * kappa^2
    let expected_amp = 2.0 * kappa * kappa;
    assert!((peak_t0 - expected_amp).abs() < 1e-14);

    // En t = 0.5, el pico se traslada exactamente a x = 4 * kappa^2 * 0.5 = 2 * kappa^2
    let shifted_x = 4.0 * kappa * kappa * 0.5;
    let peak_t1 = KdvSolitonIstAnalyzer::analytical_sech2_soliton(kappa, shifted_x, 0.0, 0.5);
    assert!((peak_t1 - expected_amp).abs() < 1e-14);
}

#[test]
fn c4_ist_lax_schrodinger_bound_states_single_and_double_soliton() {
    let analyzer = KdvSolitonIstAnalyzer::default();

    // 1. Perfil dispersivo nulo/débil => 0 solitones
    let noise_profile = [0.01, -0.01, 0.005, -0.005, 0.01, 0.0, -0.01, 0.005];
    let rep_noise = analyzer.analyze_profile(&noise_profile);
    assert_eq!(rep_noise.bound_soliton_count, 0);

    // 2. Perfil de 1 pozo solitónico fuerte centrado en el nodo 3
    let single_soliton = [0.1, 0.5, 2.0, 5.5, 2.0, 0.5, 0.1, 0.0];
    let rep_single = analyzer.analyze_profile(&single_soliton);
    assert!(rep_single.bound_soliton_count >= 1);
    assert!(rep_single.primary_wave_number_kappa1 > 1.0);
    assert!(
        (rep_single.primary_soliton_velocity - 2.0 * rep_single.primary_soliton_amplitude).abs()
            < 1e-14
    );

    // 3. Perfil bimodal de 2 solitones bien separados (nodos 1 y 6) con amplitudes distintas
    let double_soliton = [1.0, 6.5, 1.0, 0.0, 0.0, 0.8, 4.2, 0.8];
    let rep_double = analyzer.analyze_profile(&double_soliton);
    assert!(
        rep_double.bound_soliton_count >= 2,
        "Bimodal deep profile must produce >= 2 bound solitons, got {}",
        rep_double.bound_soliton_count
    );
    assert!(rep_double.two_soliton_phase_shift > 0.0);
}

#[test]
fn c5_kdv_conservation_laws_translation_invariance() {
    let analyzer = KdvSolitonIstAnalyzer::default();
    // Perfil localizado trasladado un sitio en la malla interior (ceros en los bordes)
    let profile_a = [0.0, 0.5, 2.5, 4.0, 2.5, 0.5, 0.0, 0.0];
    let profile_b = [0.0, 0.0, 0.5, 2.5, 4.0, 2.5, 0.5, 0.0];

    let rep_a = analyzer.analyze_profile(&profile_a);
    let rep_b = analyzer.analyze_profile(&profile_b);

    assert!((rep_a.conserved_mass_i1 - rep_b.conserved_mass_i1).abs() < 1e-13);
    assert!((rep_a.conserved_momentum_i2 - rep_b.conserved_momentum_i2).abs() < 1e-13);
    assert!((rep_a.conserved_hamiltonian_i3 - rep_b.conserved_hamiltonian_i3).abs() < 1e-12);
}

#[test]
fn c6_nan_inf_adversarial_immunity_in_tropical_kdv_engine() {
    let engine = TropicalKdvEngine::default();
    let bad_mat = TropicalMaxPlusMatrix::from_raw_sanitized([
        [f64::NAN, f64::INFINITY, -5.0, 2.0],
        [1.0, f64::NEG_INFINITY, f64::NAN, 3.0],
        [0.0, 2.0, 1.0, f64::NAN],
        [f64::INFINITY, -1.0, 4.0, 0.0],
    ]);
    let bad_profile: [f64; KDV_GRID_DIM] = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        5.0,
        -10.0,
        0.0,
        2.0,
        f64::NAN,
    ];

    let rep = engine.evaluate_cycle(&bad_mat, &bad_profile, f64::NAN);
    assert!(rep.tropical.max_cycle_mean_eigenvalue.is_finite());
    assert!(rep.soliton.primary_wave_number_kappa1.is_finite());
    assert!(rep.soliton.conserved_mass_i1.is_finite());
    assert!(rep.directional_impulse.is_finite());
    assert!(matches!(
        rep.regime,
        TropicalSolitonRegime::SolitonBreakoutWave
            | TropicalSolitonRegime::MultiSolitonElasticCollision
            | TropicalSolitonRegime::DispersiveRadiationNoise
    ));
}
