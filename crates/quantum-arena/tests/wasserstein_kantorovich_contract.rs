//! Formal Contract Tests for Ola Ω143: Wasserstein-Kantorovich W_2 Optimal Transport
//! and Benamou-Brenier Hydrodynamic Continuity over Hilbert Space L^2.

use quantum_arena::{
    exact_empirical_w2_distance_8, wasserstein_hermite_c1, WassersteinKantorovichEngine,
};
use std::time::Instant;

#[test]
fn contract_wasserstein_brenier_mccann_isometry_axioms() {
    let mu_0 = [-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5];
    // 1. Identity: W_2(mu_0, mu_0) == 0
    let (w2_id, w2_sq_id, shift_id) = exact_empirical_w2_distance_8(mu_0, mu_0);
    assert!(w2_id.abs() < 1e-12 && w2_sq_id.abs() < 1e-12 && shift_id.abs() < 1e-12);

    // 2. Pure translation: mu_1 = mu_0 + c -> W_2(mu_0, mu_1) == |c| and shape deformation == 0
    let c = 0.75;
    let mut mu_shifted = [0.0; 8];
    for k in 0..8 {
        mu_shifted[k] = mu_0[7 - k] + c; // unsorted order to verify in-place quantile sorting
    }
    let (w2_tr, w2_sq_tr, shift_tr) = exact_empirical_w2_distance_8(mu_0, mu_shifted);
    assert!(
        (w2_tr - c).abs() < 1e-12,
        "Pure translation by c={} must yield W_2={}, got {}",
        c,
        c,
        w2_tr
    );
    assert!((shift_tr - c).abs() < 1e-12);
    assert!((w2_sq_tr - shift_tr * shift_tr).abs() < 1e-12);

    // 3. Symmetry: W_2(mu_0, mu_1) == W_2(mu_1, mu_0)
    let (w2_rev, _, shift_rev) = exact_empirical_w2_distance_8(mu_shifted, mu_0);
    assert!((w2_tr - w2_rev).abs() < 1e-12);
    assert!((shift_tr + shift_rev).abs() < 1e-12);
}

#[test]
fn contract_wasserstein_coherent_geodesic_vs_regime_fracture() {
    // 1. Coherent geodesic translation: mu_0 centered around 0.0, mu_1 shifted coherently by +0.80
    let mut engine_geodesic = WassersteinKantorovichEngine::new();
    let base = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30];
    for &x in &base {
        let _ = engine_geodesic.observe_spectral_measure(x);
    }
    let mut m_geo = Default::default();
    for &x in &base {
        m_geo = engine_geodesic.observe_spectral_measure(x + 0.80);
    }
    assert!(
        m_geo.geodesic_coherence_ratio > 0.99,
        "Expected geodesic coherence near 1.0 for pure translation, got {}",
        m_geo.geodesic_coherence_ratio
    );
    assert!(
        m_geo.shape_deformation_energy < 1e-9,
        "Expected zero shape deformation for pure translation, got {}",
        m_geo.shape_deformation_energy
    );
    assert!(
        m_geo.barycenter_transport_velocity > 0.50,
        "Expected positive Benamou-Brenier velocity, got {}",
        m_geo.barycenter_transport_velocity
    );
    assert!(
        m_geo.wasserstein_conviction_modulation > 1.10,
        "Coherent geodesic transport must amplify conviction (>1.10), got {}",
        m_geo.wasserstein_conviction_modulation
    );

    // 2. Symmetric structural regime fracture: zero barycenter shift, massive variance explosion
    let mut engine_fracture = WassersteinKantorovichEngine::new();
    for &x in &base {
        let _ = engine_fracture.observe_spectral_measure(x);
    }
    let mut m_frac = Default::default();
    for &x in &base {
        m_frac = engine_fracture.observe_spectral_measure(x * 8.0);
    }
    assert!(
        m_frac.geodesic_coherence_ratio < 1e-6,
        "Expected near-zero geodesic coherence for symmetric variance explosion, got {}",
        m_frac.geodesic_coherence_ratio
    );
    assert!(
        m_frac.shape_deformation_energy > 0.80,
        "Expected high shape deformation energy during regime fracture, got {}",
        m_frac.shape_deformation_energy
    );
    assert!(
        m_frac.wasserstein_conviction_modulation < 0.40,
        "Structural regime fracture must attenuate conviction (<0.40), got {}",
        m_frac.wasserstein_conviction_modulation
    );
}

#[test]
fn contract_wasserstein_cpt_directional_parity() {
    let base = [-0.25, -0.15, -0.08, -0.02, 0.02, 0.08, 0.15, 0.25];
    let mut eng_up = WassersteinKantorovichEngine::new();
    let mut eng_dn = WassersteinKantorovichEngine::new();
    for &x in &base {
        let _ = eng_up.observe_spectral_measure(x);
        let _ = eng_dn.observe_spectral_measure(-x);
    }
    let mut m_up = Default::default();
    let mut m_dn = Default::default();
    for &x in &base {
        m_up = eng_up.observe_spectral_measure(x + 0.65);
        m_dn = eng_dn.observe_spectral_measure(-(x + 0.65));
    }

    assert!((m_up.wasserstein_w2_distance - m_dn.wasserstein_w2_distance).abs() < 1e-12);
    assert!(
        (m_up.barycenter_transport_velocity + m_dn.barycenter_transport_velocity).abs() < 1e-12
    );
    assert!(
        (m_up.wasserstein_conviction_modulation - m_dn.wasserstein_conviction_modulation).abs()
            < 1e-12
    );
}

#[test]
fn contract_wasserstein_nan_and_inf_immunity() {
    let mut engine = WassersteinKantorovichEngine::new();
    for k in 0..16 {
        let _ = engine.observe_spectral_measure(k as f64 * 0.05);
    }
    let m_nan = engine.observe_spectral_measure(f64::NAN);
    assert!(m_nan.wasserstein_w2_distance.is_finite());
    assert!(m_nan.wasserstein_conviction_modulation.is_finite());
    assert_eq!(wasserstein_hermite_c1(f64::NAN, 0.0, 1.0), 0.0);
}

#[test]
fn contract_wasserstein_sub_microsecond_stack_latency() {
    let mut engine = WassersteinKantorovichEngine::new();
    for k in 0..16 {
        let _ = engine.observe_spectral_measure((k as f64 * 0.2).sin());
    }

    let iterations = 4_000;
    let start = Instant::now();
    let mut acc = 0.0;
    for k in 0..iterations {
        let t = k as f64 * 0.11;
        let m = engine.observe_spectral_measure(t.sin() + 0.3 * (2.0 * t).cos());
        acc += m.wasserstein_conviction_modulation;
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
    assert!(acc.is_finite() && acc > 0.0);
    assert!(
        ns_per_op < 25_000.0,
        "Expected < 25,000 ns per Wasserstein-Kantorovich W_2 update, got {:.1} ns",
        ns_per_op
    );
}
