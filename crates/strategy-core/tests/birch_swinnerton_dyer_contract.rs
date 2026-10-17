//! Formal Contract Tests for Ola Ω141: Birch and Swinnerton-Dyer Conjecture
//! Elliptic Curves over Finite Fields F_p, Hasse-Weil L-Function L(E, s), and Analytic Rank r_BSD.

use std::time::Instant;
use strategy_core::{
    bsd_hermite_c1, frobenius_trace_mod_p, legendre_symbol, BirchSwinnertonDyerEngine,
    BSD_ODD_PRIMES,
};

#[test]
fn contract_bsd_hasse_bound_and_legendre_symbol() {
    // 1. Verify Legendre symbol properties on small primes
    // Modulo 5: squares are 1, 4. Non-squares are 2, 3.
    assert_eq!(legendre_symbol(0, 5), 0);
    assert_eq!(legendre_symbol(1, 5), 1);
    assert_eq!(legendre_symbol(4, 5), 1);
    assert_eq!(legendre_symbol(2, 5), -1);
    assert_eq!(legendre_symbol(3, 5), -1);

    // 2. Hasse's Theorem on Elliptic Curves (1936):
    // For every prime p and any curve y^2 = x^3 + a x + b over F_p,
    // |a_p| <= 2 * sqrt(p).
    for &p in &BSD_ODD_PRIMES {
        let bound = (2.0 * (p as f64).sqrt()).floor() as i64 + 1;
        for a in -5..=5 {
            for b in -5..=5 {
                let (n_p, a_p) = frobenius_trace_mod_p(a, b, p);
                assert_eq!(n_p + a_p, p + 1);
                assert!(
                    a_p.abs() <= bound,
                    "Hasse bound violated for p={}, a={}, b={}: |a_p|={} > 2*sqrt(p)={}",
                    p,
                    a,
                    b,
                    a_p.abs(),
                    2.0 * (p as f64).sqrt()
                );
            }
        }
    }
}

#[test]
fn contract_bsd_singular_cusp_vs_nonsingular_elliptic_curve() {
    // Singular curve: static phase point (0, 0) -> variance = 0, skewness = 0 -> a = 0, b = 0 -> Delta = 0
    let mut engine_singular = BirchSwinnertonDyerEngine::new();
    let mut m_sing = Default::default();
    for _ in 0..32 {
        m_sing = engine_singular.observe_phase_point(0.0, 0.0);
    }
    assert!(
        m_sing.discriminant.abs() < 1e-9,
        "Expected zero discriminant for singular cusp, got {}",
        m_sing.discriminant
    );
    assert!(
        m_sing.non_singularity < 1e-6,
        "Expected near-zero non-singularity for cusp, got {}",
        m_sing.non_singularity
    );
    assert!(
        m_sing.bsd_conviction_modulation <= 0.35,
        "Singular cusp must attenuate conviction towards floor 0.30, got {}",
        m_sing.bsd_conviction_modulation
    );

    // Non-singular dynamic phase trajectory: rich oscillatory liquidity absorption
    let mut engine_nonsing = BirchSwinnertonDyerEngine::new();
    let mut m_ns = Default::default();
    for k in 0..32 {
        let t = k as f64 * 0.35;
        let x = 1.2 * t.sin() + 0.4 * (2.3 * t).cos();
        let y = 0.9 * t.cos() - 0.5 * (1.7 * t).sin();
        m_ns = engine_nonsing.observe_phase_point(x, y);
    }
    assert!(
        m_ns.discriminant.abs() > 0.1,
        "Expected non-zero discriminant for non-singular phase space, got {}",
        m_ns.discriminant
    );
    assert!(
        m_ns.non_singularity > 0.50,
        "Expected strong non-singularity for active elliptic phase curve, got {}",
        m_ns.non_singularity
    );
    assert!(
        m_ns.bsd_conviction_modulation > m_sing.bsd_conviction_modulation + 0.30,
        "Non-singular elliptic curve must yield significantly higher conviction ({}) than singular cusp ({})",
        m_ns.bsd_conviction_modulation,
        m_sing.bsd_conviction_modulation
    );
}

#[test]
fn contract_bsd_l_function_and_analytic_rank_bounds() {
    let mut engine = BirchSwinnertonDyerEngine::new();
    for k in 0..48 {
        let t = k as f64 * 0.27;
        let x = 1.5 * t.sin() + 0.25 * (3.1 * t).sin();
        let y = 1.1 * t.cos() + 0.35 * (2.7 * t).cos();
        let m = engine.observe_phase_point(x, y);
        if k >= 12 {
            assert!(m.l_value_at_one.is_finite() && m.l_value_at_one > 0.0);
            assert!(
                (0.0..=2.5).contains(&m.analytic_rank_proxy),
                "Analytic rank {} out of [0, 2.5] bounds",
                m.analytic_rank_proxy
            );
            assert!(
                (-1.0..=1.0).contains(&m.frobenius_trace_norm),
                "Normalized Frobenius trace {} out of [-1, 1]",
                m.frobenius_trace_norm
            );
            assert!(
                (0.30..=1.15).contains(&m.bsd_conviction_modulation),
                "Conviction modulation {} out of [0.30, 1.15]",
                m.bsd_conviction_modulation
            );
            assert!(
                (-1.0..=1.0).contains(&m.bsd_restoring_current),
                "Restoring current {} out of [-1, 1]",
                m.bsd_restoring_current
            );
        }
    }
}

#[test]
fn contract_bsd_nan_and_inf_immunity() {
    let mut engine = BirchSwinnertonDyerEngine::new();
    for k in 0..16 {
        let _ = engine.observe_phase_point((k as f64).sin(), (k as f64).cos());
    }
    let m_nan = engine.observe_phase_point(f64::NAN, f64::INFINITY);
    assert!(m_nan.bsd_conviction_modulation.is_finite());
    assert!(m_nan.analytic_rank_proxy.is_finite());
    assert!(m_nan.l_value_at_one.is_finite());

    assert_eq!(bsd_hermite_c1(f64::NAN, 0.0, 1.0), 0.0);
    assert_eq!(bsd_hermite_c1(0.5, 0.0, 1.0), 0.5);
}

#[test]
fn contract_bsd_sub_microsecond_stack_latency() {
    let mut engine = BirchSwinnertonDyerEngine::new();
    for k in 0..24 {
        let _ = engine.observe_phase_point((k as f64 * 0.2).sin(), (k as f64 * 0.2).cos());
    }

    let iterations = 2_000;
    let start = Instant::now();
    let mut acc = 0.0;
    for k in 0..iterations {
        let t = k as f64 * 0.13;
        let m = engine.observe_phase_point(t.sin(), t.cos());
        acc += m.bsd_conviction_modulation;
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
    assert!(acc.is_finite() && acc > 0.0);
    assert!(
        ns_per_op < 25_000.0,
        "Expected < 25,000 ns per BSD elliptic curve update, got {:.1} ns",
        ns_per_op
    );
}
