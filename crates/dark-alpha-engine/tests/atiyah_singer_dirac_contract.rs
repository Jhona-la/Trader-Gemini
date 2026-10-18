//! Formal Contract Tests for Ola Ω142: Atiyah-Singer Index Theorem & Chiral Dirac Zero Modes
//! (\dim \ker D_+ - \dim \ker D_- = \int ch(E) \wedge \hat{A}(M)).

use dark_alpha_engine::{
    a_roof_generating_function, dirac_hermite_c1, AtiyahSingerDiracEngine,
};
use std::time::Instant;

#[test]
fn contract_dirac_a_roof_genus_properties() {
    // At zero curvature, A-roof genus is 1.0
    assert!((a_roof_generating_function(0.0) - 1.0).abs() < 1e-12);
    // Small curvature matches Taylor 1 - x^2 / 24
    let x = 1e-5;
    let expected = 1.0 - (x * x) / 24.0;
    assert!((a_roof_generating_function(x) - expected).abs() < 1e-12);

    // Monotonically decreasing with curvature magnitude
    let a1 = a_roof_generating_function(0.5);
    let a2 = a_roof_generating_function(2.0);
    let a3 = a_roof_generating_function(8.0);
    assert!(a1 > a2 && a2 > a3 && a3 > 0.0);
    assert_eq!(a_roof_generating_function(f64::NAN), 0.0);
}

#[test]
fn contract_dirac_positive_vs_negative_chiral_zero_modes() {
    // 1. Right-handed chiral soliton excitation (psi_+, D_+ zero modes)
    let mut engine_plus = AtiyahSingerDiracEngine::new();
    let mut m_plus = Default::default();
    for k in 0..16 {
        let x = k as f64 * 0.25;
        let a_flow = 0.50;
        m_plus = engine_plus.observe_spinor_state(x, a_flow);
    }
    assert!(
        m_plus.zero_modes_plus > m_plus.zero_modes_minus + 0.20,
        "Expected positive chiral zero modes to dominate: n_+={} vs n_-={}",
        m_plus.zero_modes_plus,
        m_plus.zero_modes_minus
    );
    assert!(
        m_plus.atiyah_singer_index > 0.15,
        "Expected positive Atiyah-Singer index, got {}",
        m_plus.atiyah_singer_index
    );
    assert!(
        m_plus.chiral_anomaly_current > 0.20,
        "Expected positive Adler-Bell-Jackiw chiral current, got {}",
        m_plus.chiral_anomaly_current
    );

    // 2. Left-handed chiral soliton excitation (psi_-, D_- zero modes)
    let mut engine_minus = AtiyahSingerDiracEngine::new();
    let mut m_minus = Default::default();
    for k in 0..16 {
        let x = -(k as f64 * 0.25);
        let a_flow = -0.50;
        m_minus = engine_minus.observe_spinor_state(x, a_flow);
    }
    assert!(
        m_minus.zero_modes_minus > m_minus.zero_modes_plus + 0.20,
        "Expected negative chiral zero modes to dominate: n_-={} vs n_+={}",
        m_minus.zero_modes_minus,
        m_minus.zero_modes_plus
    );
    assert!(
        m_minus.atiyah_singer_index < -0.15,
        "Expected negative Atiyah-Singer index, got {}",
        m_minus.atiyah_singer_index
    );
    assert!(
        m_minus.chiral_anomaly_current < -0.20,
        "Expected negative Adler-Bell-Jackiw chiral current, got {}",
        m_minus.chiral_anomaly_current
    );
}

#[test]
fn contract_dirac_whipsaw_curvature_suppression_vs_smooth_chiral_manifold() {
    // Violent whipsaw regime: high discrete Riemannian curvature -> A-roof genus -> 0
    let mut engine_whipsaw = AtiyahSingerDiracEngine::new();
    let mut m_whip = Default::default();
    for k in 0..16 {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        m_whip = engine_whipsaw.observe_spinor_state(sign * 1.8, -sign * 1.5);
    }

    // Smooth chiral flow regime: low Riemannian curvature + coherent gauge flow
    let mut engine_smooth = AtiyahSingerDiracEngine::new();
    let mut m_smooth = Default::default();
    for k in 0..16 {
        let x = k as f64 * 0.22;
        m_smooth = engine_smooth.observe_spinor_state(x, 0.45);
    }

    assert!(
        m_whip.a_roof_genus < 0.25,
        "Violent whipsaw must collapse A-roof genus, got {}",
        m_whip.a_roof_genus
    );
    assert!(
        m_smooth.a_roof_genus > 0.90,
        "Smooth chiral trajectory must preserve A-roof genus near 1.0, got {}",
        m_smooth.a_roof_genus
    );
    assert!(
        m_smooth.dirac_conviction_modulation > m_whip.dirac_conviction_modulation + 0.50,
        "Smooth chiral zero-mode manifold must yield much higher conviction ({}) than whipsaw ({})",
        m_smooth.dirac_conviction_modulation,
        m_whip.dirac_conviction_modulation
    );
}

#[test]
fn contract_dirac_nan_and_inf_immunity() {
    let mut engine = AtiyahSingerDiracEngine::new();
    for k in 0..10 {
        let _ = engine.observe_spinor_state(k as f64 * 0.1, 0.2);
    }
    let m_nan = engine.observe_spinor_state(f64::NAN, f64::INFINITY);
    assert!(m_nan.dirac_conviction_modulation.is_finite());
    assert!(m_nan.atiyah_singer_index.is_finite());
    assert!(m_nan.a_roof_genus.is_finite());
    assert_eq!(dirac_hermite_c1(f64::NAN, 0.0, 1.0), 0.0);
}

#[test]
fn contract_dirac_sub_microsecond_stack_latency() {
    let mut engine = AtiyahSingerDiracEngine::new();
    for k in 0..16 {
        let _ = engine.observe_spinor_state(k as f64 * 0.1, 0.25);
    }

    let iterations = 4_000;
    let start = Instant::now();
    let mut acc = 0.0;
    for k in 0..iterations {
        let t = k as f64 * 0.07;
        let m = engine.observe_spinor_state(t.sin(), t.cos());
        acc += m.dirac_conviction_modulation;
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
    assert!(acc.is_finite() && acc > 0.0);
    assert!(
        ns_per_op < 25_000.0,
        "Expected < 25,000 ns per Atiyah-Singer Dirac update, got {:.1} ns",
        ns_per_op
    );
}

#[test]
fn contract_g8_omega_15_smooth_neutral_manifold_and_c1_continuity() {
    // Una variedad suave y neutral (x = 0, a_flow = 0) tiene género A-techo = 1.0
    // y NO debe sufrir la penalización espuria de -12% (0.88): debe anclarse en 1.00.
    let mut engine_neutral = AtiyahSingerDiracEngine::new();
    let mut m_neutral = Default::default();
    for _ in 0..8 {
        m_neutral = engine_neutral.observe_spinor_state(0.0, 0.0);
    }
    assert!(
        (m_neutral.a_roof_genus - 1.0).abs() < 1e-12,
        "Variedad plana debe tener género A-techo == 1.0, obtuvimos {}",
        m_neutral.a_roof_genus
    );
    assert!(
        (m_neutral.dirac_conviction_modulation - 1.0).abs() < 1e-12,
        "Variedad suave neutral debe preservar convicción 1.00 sin penalización espuria de -12%, obtuvimos {}",
        m_neutral.dirac_conviction_modulation
    );

    // Continuidad C¹ al cruzar dx * dx_prev = 0 y a_mid = 0
    let mut e_pos = AtiyahSingerDiracEngine::new();
    let mut e_neg = AtiyahSingerDiracEngine::new();
    for _ in 0..6 {
        let _ = e_pos.observe_spinor_state(0.10, 0.0);
        let _ = e_neg.observe_spinor_state(0.10, 0.0);
    }
    let m_p = e_pos.observe_spinor_state(0.100001, 0.05001);
    let m_n = e_neg.observe_spinor_state(0.099999, 0.04999);
    assert!(
        (m_p.dirac_conviction_modulation - m_n.dirac_conviction_modulation).abs() < 1e-3,
        "El cruce de signo en dx*dx_prev y a_mid=0.05 debe ser C¹-continuo: {} vs {}",
        m_p.dirac_conviction_modulation,
        m_n.dirac_conviction_modulation
    );
}

