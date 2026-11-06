//! Contrato Formal de Propiedades Matemáticas e Invariantes: Fase 35 (§35)
//! Dinámica Hidrodinámica de Shockwaves de Riemann-Burgers y Cole-Hopf en L2/L3
//!
//! - C1: Condición de salto de Rankine-Hugoniot para la velocidad de propagación $s = (u_L + u_R)/2$.
//! - C2: Escalamiento analítico del ancho de capa de choque de Cole-Hopf $\delta = 4\nu / \Delta u$.
//! - C3: Detección precisa de bolsa de aire de liquidez (air pocket) ante colapso de densidad.
//! - C4: Convicción hidrodinámica acotada en $[-1.0, 1.0]$ y alineada con la advección.
//! - C5: Invarianza ante valores patológicos (cero pánicos `rb_p1`).

use signal_engine::{RiemannBurgersParams, RiemannBurgersShockwaveEngine};

#[test]
fn c1_rankine_hugoniot_shock_velocity() {
    let mut engine = RiemannBurgersShockwaveEngine::new(RiemannBurgersParams {
        market_viscosity_nu: 0.10,
        ..Default::default()
    });

    // Inyectar flujo agresor alcista constante OFI = 0.60, OBI = 0.20
    let mut report = signal_engine::BurgersShockwaveReport::default();
    for _ in 0..10 {
        report = engine.evaluate_shockwave(0.60, 0.20, 0.0010, 1.0);
    }

    // u_L converge a 0.60, u_R converge a -0.20
    // s = (0.60 + (-0.20)) / 2 = 0.20
    assert!(report.shock_velocity_s > 0.10);
    assert!(report.shock_velocity_s < 0.30);
    assert!(report.shock_jump_amplitude > 0.50);
}

#[test]
fn c2_cole_hopf_layer_width_scaling() {
    let mut engine_small_jump = RiemannBurgersShockwaveEngine::default();
    let mut engine_large_jump = RiemannBurgersShockwaveEngine::default();

    let mut report_small = signal_engine::BurgersShockwaveReport::default();
    let mut report_large = signal_engine::BurgersShockwaveReport::default();

    for _ in 0..10 {
        report_small = engine_small_jump.evaluate_shockwave(0.10, 0.0, 0.0010, 1.0);
        report_large = engine_large_jump.evaluate_shockwave(0.90, 0.90, 0.0010, 1.0);
    }

    // El ancho de choque de Cole-Hopf $\delta = 4\nu / \Delta u$ debe ser estrictamente menor
    // para un salto violento (large jump) que para un salto pequeño
    assert!(report_large.shock_layer_width < report_small.shock_layer_width);
    assert!(report_large.shock_layer_width > 0.0);
    assert!(report_small.shock_layer_width > 0.0);
}

#[test]
fn c3_air_pocket_detection_on_liquidity_collapse() {
    let mut engine = RiemannBurgersShockwaveEngine::new(RiemannBurgersParams {
        shock_jump_threshold: 0.40,
        market_viscosity_nu: 0.03, // Viscosidad muy baja (mercado delgado)
        ..Default::default()
    });

    // Salto violento con libro muy delgado (depth = 0.08)
    let mut report = signal_engine::BurgersShockwaveReport::default();
    for _ in 0..15 {
        report = engine.evaluate_shockwave(0.85, 0.85, 0.0050, 0.08);
    }

    assert!(report.air_pocket_detected);
    assert!(report.liquidity_density_at_front <= 0.25);
    assert!(report.shock_jump_amplitude >= 0.40);
}

#[test]
fn c4_hydrodynamic_conviction_boundedness() {
    let mut engine_long = RiemannBurgersShockwaveEngine::default();
    let mut engine_short = RiemannBurgersShockwaveEngine::default();

    let mut report_long = signal_engine::BurgersShockwaveReport::default();
    let mut report_short = signal_engine::BurgersShockwaveReport::default();

    for _ in 0..10 {
        report_long = engine_long.evaluate_shockwave(0.80, 0.50, 0.0010, 1.0);
        report_short = engine_short.evaluate_shockwave(-0.80, -0.50, 0.0010, 1.0);
    }

    assert!(report_long.hydrodynamic_conviction > 0.0);
    assert!(report_long.hydrodynamic_conviction <= 1.0);
    assert!(report_short.hydrodynamic_conviction < 0.0);
    assert!(report_short.hydrodynamic_conviction >= -1.0);
}

#[test]
fn c5_zero_panic_boundary_invariance() {
    let mut engine = RiemannBurgersShockwaveEngine::default();

    let cases = [
        (f64::NAN, 0.0, 0.0010, 1.0),
        (0.0, f64::INFINITY, 0.0010, 1.0),
        (0.5, -0.5, f64::NEG_INFINITY, 1.0),
        (0.5, -0.5, 0.0010, f64::NAN),
        (-100.0, 100.0, -0.05, -5.0),
        (0.0, 0.0, 0.0, 0.0),
    ];

    for &(ofi, obi, spread, depth) in cases.iter() {
        let report = engine.evaluate_shockwave(ofi, obi, spread, depth);
        assert!(report.shock_velocity_s.is_finite());
        assert!(report.shock_jump_amplitude.is_finite());
        assert!(report.shock_layer_width.is_finite());
        assert!(report.liquidity_density_at_front.is_finite());
        assert!(report.hydrodynamic_conviction.is_finite());
        assert!(report.time_to_impact_ms.is_finite());
    }
}
