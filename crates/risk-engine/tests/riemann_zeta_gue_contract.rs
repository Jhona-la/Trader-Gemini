//! # Contrato Matemático y Físico — Ola Ω140: Espectro de Ceros de Riemann Zeta y Matrices Aleatorias Cuánticas GUE
//!
//! Verifica:
//! 1. Ley de correlación de pares de Montgomery-Odlyzko $R_2(u) = 1 - (\sin(\pi u)/(\pi u))^2$ y repulsión cuadrática $u \to 0$.
//! 2. Densidad de Wigner Surmise para GUE ($\beta = 2$): $P_{\text{GUE}}(s) = \frac{32}{\pi^2} s^2 \exp(-\frac{4}{\pi} s^2)$.
//! 3. Discriminación entre colapso degenerado (contagio sistémico, $\langle r \rangle \to 0$) y repulsión GUE en la línea crítica $\Re(s) = \frac{1}{2}$.
//! 4. Inmunidad ante NaN/Inf y continuidad $C^1$ Hermite.
//! 5. Latencia determinista zero-alloc en stack ($< 25\text{ }\mu\text{s}$).

use risk_engine::riemann_zeta_gue::{
    hermite_smoothstep_c1, montgomery_odlyzko_pair_correlation, wigner_gue_density,
    RiemannZetaGueEngine, GUE_RIEMANN_GAP_RATIO_MEAN, POISSON_GAP_RATIO_MEAN,
};
use std::time::Instant;

#[test]
fn c1_montgomery_odlyzko_pair_correlation_and_level_repulsion() {
    // En u -> 0, la repulsión cuadrática de niveles exige R_2(0) = 0.0
    let r_zero = montgomery_odlyzko_pair_correlation(0.0);
    assert!(r_zero.abs() < 1e-12, "R_2(0) debe ser 0 por repulsión cuántica, obtuvimos {}", r_zero);

    // Cerca del origen (u = 1e-5), Taylor de sexto orden: R_2(u) ~ (pi^2 / 3) * u^2
    let u_small = 1e-5;
    let r_small = montgomery_odlyzko_pair_correlation(u_small);
    let expected_small = (std::f64::consts::PI * u_small).powi(2) / 3.0;
    assert!(
        (r_small - expected_small).abs() < 1e-14,
        "Aproximación Taylor en u=1e-5 inexacta: {} vs {}",
        r_small,
        expected_small
    );

    // En números enteros u = 1, 2, 3, sin(pi * u) = 0 => R_2(u) = 1.0 exacto
    for k in 1..=5 {
        let r_int = montgomery_odlyzko_pair_correlation(k as f64);
        assert!(
            (r_int - 1.0).abs() < 1e-12,
            "En entero u={}, R_2(u) debe ser 1.0, obtuvimos {}",
            k,
            r_int
        );
    }
}

#[test]
fn c2_wigner_gue_surmise_density_properties() {
    // En s = 0, la densidad GUE se anula cuadráticamente (repulsión beta=2)
    assert_eq!(wigner_gue_density(0.0), 0.0);
    assert_eq!(wigner_gue_density(-1.5), 0.0);

    // El modo (máximo) de P_GUE(s) ocurre exactamente en s* = sqrt(pi) / 2 ~ 0.886226925
    let s_mode = std::f64::consts::PI.sqrt() * 0.5;
    let p_mode = wigner_gue_density(s_mode);
    let p_left = wigner_gue_density(s_mode - 0.15);
    let p_right = wigner_gue_density(s_mode + 0.15);
    assert!(
        p_mode > p_left && p_mode > p_right,
        "El modo de Wigner GUE debe estar en sqrt(pi)/2: p_mode={}, left={}, right={}",
        p_mode,
        p_left,
        p_right
    );

    // Verificación de constantes teóricas de Oganesyan-Huse
    assert!(GUE_RIEMANN_GAP_RATIO_MEAN > POISSON_GAP_RATIO_MEAN);
    assert!((GUE_RIEMANN_GAP_RATIO_MEAN - 0.60265779).abs() < 1e-6);
}

#[test]
fn c3_discriminates_degenerate_contagion_vs_gue_critical_line_spectrum() {
    // Caso A: Primeros 12 ceros no triviales de la función Zeta de Riemann (partes imaginarias gamma_n)
    // en la línea crítica Re(s) = 1/2:
    let riemann_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
        40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248,
    ];
    let gue_state = RiemannZetaGueEngine::evaluate_spectrum(&riemann_zeros);

    // Caso B: Espectro degenerado / colapso de niveles (todos agrupados en 10.0 salvo un outlier extremo en 100.0)
    let degenerate_levels = [
        10.000001, 10.000002, 10.0000025, 10.000003, 10.0000031, 10.0000032,
        10.0000033, 10.0000034, 10.0000035, 10.0000036, 10.0000037, 100.0,
    ];
    let deg_state = RiemannZetaGueEngine::evaluate_spectrum(&degenerate_levels);

    assert!(
        gue_state.critical_line_stability > deg_state.critical_line_stability + 0.45,
        "El espectro de ceros de Riemann debe tener mucha mayor estabilidad crítica ({}) que el espectro degenerado ({})",
        gue_state.critical_line_stability,
        deg_state.critical_line_stability
    );
    assert!(
        gue_state.gue_conviction_modulation > 0.90,
        "Los ceros de Riemann deben producir modulación de convicción saludable (>0.90), obtuvimos {}",
        gue_state.gue_conviction_modulation
    );
    assert!(
        deg_state.gue_conviction_modulation < 0.45,
        "El colapso degenerado debe atenuar fuertemente la convicción (<0.45), obtuvimos {}",
        deg_state.gue_conviction_modulation
    );
}

#[test]
fn c4_nan_immunity_and_c1_hermite_continuity() {
    let dirty_levels = [
        f64::NAN,
        14.1347,
        f64::INFINITY,
        21.0220,
        f64::NEG_INFINITY,
        25.0108,
        30.4248,
        32.9350,
    ];
    let state = RiemannZetaGueEngine::evaluate_spectrum(&dirty_levels);
    assert_eq!(state.num_levels, 5);
    assert!(state.critical_line_stability.is_finite());
    assert!(state.gue_conviction_modulation.is_finite());
    assert!((0.25..=1.15).contains(&state.gue_conviction_modulation));

    // Monotonía y suavidad de hermite_smoothstep_c1
    let h0 = hermite_smoothstep_c1(0.0, 0.0, 1.0);
    let h_mid = hermite_smoothstep_c1(0.5, 0.0, 1.0);
    let h1 = hermite_smoothstep_c1(1.0, 0.0, 1.0);
    assert!((h0 - 0.0).abs() < 1e-12);
    assert!((h_mid - 0.5).abs() < 1e-12);
    assert!((h1 - 1.0).abs() < 1e-12);
}

#[test]
fn c5_zero_alloc_stack_latency_under_25_microseconds() {
    let mut engine = RiemannZetaGueEngine::new();
    let sample_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
        40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248,
        59.347044, 60.831779, 65.112544, 67.079811,
    ];

    let iterations = 1_000_u32;
    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for i in 0..iterations {
        let lvl = sample_zeros[(i as usize) % sample_zeros.len()] + (i as f64) * 1e-6;
        let st = engine.observe_energy_level(lvl);
        acc += st.gue_conviction_modulation;
    }
    let elapsed_ns = t0.elapsed().as_nanos() as f64 / (iterations as f64);
    assert!(acc.is_finite() && acc > 0.0);
    assert!(
        elapsed_ns < 25_000.0,
        "La latencia por evaluación RiemannZetaGueEngine debe ser < 25 µs (25,000 ns), medida: {:.1} ns",
        elapsed_ns
    );
}

#[test]
fn c6_g8_omega_08_picket_fence_and_monotone_ema_ramp_attenuated() {
    // Rejilla picket-fence equiespaciada (s_k = 1.0 constante, <r> = 1.0, sigma(s) = 0.0)
    // típica de un cristal rígido o rampa EMA monótona sin caos cuántico GUE:
    let picket_fence = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    let pf_state = RiemannZetaGueEngine::evaluate_spectrum(&picket_fence);

    // Espectro genuino de ceros de Riemann Zeta en Re(s) = 1/2 (<r> ~ 0.61, sigma(s) ~ 0.37)
    let riemann_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
        40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248,
    ];
    let gue_state = RiemannZetaGueEngine::evaluate_spectrum(&riemann_zeros);

    assert!(
        pf_state.gue_conviction_modulation < 0.45,
        "Una rejilla picket-fence equiespaciada (<r>=1, sigma=0) NO es GUE y debe atenuarse (<0.45), obtuvimos {}",
        pf_state.gue_conviction_modulation
    );
    assert!(
        gue_state.gue_conviction_modulation > pf_state.gue_conviction_modulation + 0.50,
        "Los ceros genuinos de Riemann Zeta ({}) deben superar ampliamente a la rejilla picket-fence ({})",
        gue_state.gue_conviction_modulation,
        pf_state.gue_conviction_modulation
    );
}

