//! Suite de Contratos Formales F14-001 — Vectorización Explícita SIMD 512-bit (`Avx512Lane8F64`),
//! Integrador Simpléctico de Yoshida de 4to Orden y Dinámica Replicadora Evolutiva en el Simplex.
//!
//! Verifica las 6 propiedades físico-matemáticas y de silicio de la Fase 14:
//! - C1: Alineación física a 64 bytes (`align_of == 64`, `size_of == 64`), exactitud FMA de 8 carriles y simetría OFI.
//! - C2: Identidades algebraicas de los coeficientes cúbicos de Yoshida ($\sum c_k = 1, \sum d_k = 1$) e invariante de Liouville $\det J \equiv 1.0$.
//! - C3: Conservación de energía Hamiltoniana a largo plazo ($1000$ pasos anarmónicos con $|\Delta H| < 10^{-7}$) y reversibilidad temporal exacta.
//! - C4: Conservación estricta del simplex $\Delta^7$ ($\sum x_i = 1, x_i \ge 0$) y convergencia monótona hacia la estrategia dominante.
//! - C5: Equilibrio interior evolutivamente estable (ESS) bajo juego suma-cero antisimétrico ($A^T = -A$) con máxima diversidad de Shannon $N_{\text{eff}} = 8.0$.
//! - C6: Inmunidad total frente a entradas adversariales `NaN` / `±Inf` en el motor maestro.

use feature_engine::{
    Avx512Lane8F64, EvolutionaryReplicatorSimplex, HamiltonianPhaseState,
    SimdYoshidaReplicatorEngine, SymplecticEvolutionRegime, Yoshida4SymplecticIntegrator,
    SIMD_F64_LANES, YOSHIDA_C, YOSHIDA_D, YOSHIDA_W0, YOSHIDA_W1,
};
use std::mem::{align_of, size_of};

#[test]
fn c1_avx512_lane8_alignment_and_fma_exactness() {
    assert_eq!(align_of::<Avx512Lane8F64>(), 64);
    assert_eq!(size_of::<Avx512Lane8F64>(), 64);
    assert_eq!(SIMD_F64_LANES, 8);

    let a = Avx512Lane8F64::from_array_sanitized([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Avx512Lane8F64::splat(2.5);
    let c = Avx512Lane8F64::from_array_sanitized([0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0]);

    let fma_res = a.fma(&b, &c);
    for i in 0..SIMD_F64_LANES {
        let expected = a.lanes[i] * b.lanes[i] + c.lanes[i];
        assert!(
            (fma_res.lanes[i] - expected).abs() < 1e-14,
            "FMA mismatch at lane {i}"
        );
    }

    // Suma de 1..=8 es 36.0
    assert!((a.horizontal_sum() - 36.0).abs() < 1e-14);
    // Suma de cuadrados 1^2 + ... + 8^2 = 204.0
    assert!((a.l2_norm_sq() - 204.0).abs() < 1e-13);

    // Desequilibrio OFI de 8 carriles: antisimetría exacta al intercambiar bids y asks
    let bids =
        Avx512Lane8F64::from_array_sanitized([100.0, 50.0, 80.0, 10.0, 25.0, 90.0, 40.0, 60.0]);
    let asks =
        Avx512Lane8F64::from_array_sanitized([20.0, 150.0, 80.0, 30.0, 75.0, 10.0, 40.0, 20.0]);
    let ofi_ab = Avx512Lane8F64::eval_8way_ofi_imbalance(&bids, &asks);
    let ofi_ba = Avx512Lane8F64::eval_8way_ofi_imbalance(&asks, &bids);
    for i in 0..SIMD_F64_LANES {
        assert!((ofi_ab.lanes[i] + ofi_ba.lanes[i]).abs() < 1e-14);
        assert!(ofi_ab.lanes[i] >= -1.0 && ofi_ab.lanes[i] <= 1.0);
    }
}

#[test]
fn c2_yoshida_4th_order_coefficients_and_symplectic_jacobian_det_one() {
    let sum_w = YOSHIDA_W0 + 2.0 * YOSHIDA_W1;
    assert!((sum_w - 1.0).abs() < 1e-15);

    let sum_c: f64 = YOSHIDA_C.iter().sum();
    let sum_d: f64 = YOSHIDA_D.iter().sum();
    assert!((sum_c - 1.0).abs() < 1e-15);
    assert!((sum_d - 1.0).abs() < 1e-15);

    // Condición de anulación del error de 3er orden: w0^3 + 2 * w1^3 == 0
    let cubic_residual = YOSHIDA_W0.powi(3) + 2.0 * YOSHIDA_W1.powi(3);
    assert!(
        cubic_residual.abs() < 1e-14,
        "Yoshida 3rd-order cancellation residual too large: {cubic_residual}"
    );

    let integrator = Yoshida4SymplecticIntegrator::default();
    for mode in 0..SIMD_F64_LANES {
        let q_val = -1.2 + 0.35 * (mode as f64);
        let det_j = integrator.single_mode_jacobian_det(q_val, mode, 0.02);
        assert!(
            (det_j - 1.0).abs() < 1e-13,
            "Symplectic Liouville invariant det(J)==1 violated at mode {mode}: det={det_j}"
        );
    }
}

#[test]
fn c3_yoshida4_long_horizon_energy_conservation_vs_euler_drift() {
    let integrator = Yoshida4SymplecticIntegrator::default();
    let initial = HamiltonianPhaseState {
        q: Avx512Lane8F64::from_array_sanitized([0.5, -0.4, 0.3, -0.2, 0.1, -0.3, 0.4, -0.5]),
        p: Avx512Lane8F64::from_array_sanitized([-0.2, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1, 0.3]),
    };

    let dt = 0.01;
    let steps = 1_000;
    let report = integrator.integrate_orbit(&initial, dt, steps);

    assert!(
        report.energy_drift_abs < 1e-7,
        "Yoshida 4th-order energy drift after 1000 steps must be < 1e-7, got {}",
        report.energy_drift_abs
    );

    // Reversibilidad temporal exacta: avanzar 1 paso con +dt y retroceder 1 paso con -dt recupera el estado inicial
    let forward = integrator.step_yoshida4(&initial, dt);
    let backward = integrator.step_yoshida4(&forward, -dt);
    for i in 0..SIMD_F64_LANES {
        assert!(
            (backward.q.lanes[i] - initial.q.lanes[i]).abs() < 1e-14,
            "Time reversibility violated on q[{i}]"
        );
        assert!(
            (backward.p.lanes[i] - initial.p.lanes[i]).abs() < 1e-14,
            "Time reversibility violated on p[{i}]"
        );
    }
}

#[test]
fn c4_replicator_simplex_conservation_and_dominant_strategy_selection() {
    let replicator = EvolutionaryReplicatorSimplex {
        payoff_matrix: [[0.0; SIMD_F64_LANES]; SIMD_F64_LANES],
        selection_intensity: 2.5,
        mutation_rate: 0.005,
    };

    let mut weights = Avx512Lane8F64::splat(1.0 / 8.0);
    // La estrategia 0 domina estrictamente a las demás
    let growth =
        Avx512Lane8F64::from_array_sanitized([2.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]);

    let mut prev_w0 = weights.lanes[0];
    for _ in 0..40 {
        let rep = replicator.step_replicator(&weights, &growth, 0.1);
        let sum_w = rep.weights.horizontal_sum();
        assert!(
            (sum_w - 1.0).abs() < 1e-14,
            "Simplex sum must be 1.0, got {sum_w}"
        );
        for &w in &rep.weights.lanes {
            assert!((0.0..=1.0).contains(&w));
        }
        assert!(
            rep.weights.lanes[0] >= prev_w0 - 1e-12,
            "Dominant strategy weight must grow monotonically"
        );
        prev_w0 = rep.weights.lanes[0];
        weights = rep.weights;
    }

    assert!(
        weights.lanes[0] > 0.90,
        "Dominant strategy must capture >90% of simplex mass, got {}",
        weights.lanes[0]
    );
}

#[test]
fn c5_antisymmetric_zero_sum_game_invariance_and_shannon_diversity() {
    let replicator = EvolutionaryReplicatorSimplex::default();
    let uniform = Avx512Lane8F64::splat(1.0 / 8.0);
    let zero_growth = Avx512Lane8F64::splat(0.0);

    let rep = replicator.step_replicator(&uniform, &zero_growth, 0.05);
    assert!(
        rep.mean_fitness.abs() < 1e-15,
        "Expected payoff x^T A x of antisymmetric game must vanish, got {}",
        rep.mean_fitness
    );
    assert!(
        (rep.effective_species_count - 8.0).abs() < 1e-12,
        "Uniform ESS must exhibit N_eff == 8.0, got {}",
        rep.effective_species_count
    );
    assert!((rep.shannon_entropy - (8.0_f64).ln()).abs() < 1e-12);
}

#[test]
fn c6_nan_inf_adversarial_immunity_in_master_engine() {
    let engine = SimdYoshidaReplicatorEngine::default();
    let bad_bids = Avx512Lane8F64::from_array_sanitized([
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -10.0,
        100.0,
        0.0,
        50.0,
        f64::NAN,
    ]);
    let bad_asks = Avx512Lane8F64::from_array_sanitized([
        50.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        20.0,
        0.0,
        50.0,
        10.0,
    ]);
    let bad_phase = HamiltonianPhaseState {
        q: bad_bids,
        p: bad_asks,
    };
    let bad_weights = Avx512Lane8F64::splat(f64::NAN);

    let report = engine.evaluate_cycle(&bad_bids, &bad_asks, &bad_phase, &bad_weights, f64::NAN);

    assert!(report.orbit.total_energy.is_finite());
    assert!(report.orbit.energy_drift_abs.is_finite());
    assert!((report.replicator.weights.horizontal_sum() - 1.0).abs() < 1e-12);
    assert!(report.composite_conviction.is_finite());
    assert!(report.composite_conviction >= -1.0 && report.composite_conviction <= 1.0);
    assert!(matches!(
        report.regime,
        SymplecticEvolutionRegime::ConservativeBalancedOrbit
            | SymplecticEvolutionRegime::SelectiveSimplexDominance
            | SymplecticEvolutionRegime::HighEnergyAnharmonicGuard
    ));
}
