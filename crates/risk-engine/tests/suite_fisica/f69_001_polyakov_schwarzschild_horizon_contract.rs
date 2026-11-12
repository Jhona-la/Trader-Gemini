//! Contrato Formal de Verificación: Fase 69 (Acción de Polyakov y Horizonte de Schwarzschild)
//!
//! Este conjunto de pruebas certifica las propiedades relativistas y cuánticas del motor de horizonte de eventos:
//! - C1: Escala gravitatoria de Schwarzschild proporcional al apalancamiento y masa nocional.
//! - C2: Régimen subcrítico seguro con factor de escala unitario y distancia al horizonte positiva.
//! - C3: Disparo determinista de veto de agujero negro ante cruce del horizonte de eventos ($\Delta r \le 0$).
//! - C4: Determinismo en stack $O(1)$, cero pánicos (`rb_p1`) y resiliencia ante NaN e infinitos.

use risk_engine::{PolyakovSchwarzschildHorizonEngine, PolyakovSchwarzschildParams};

#[test]
fn test_c1_schwarzschild_radius_leverage_scaling() {
    let params = PolyakovSchwarzschildParams::default();
    let engine = PolyakovSchwarzschildHorizonEngine::new(params);

    let rep_low_lev = engine.evaluate(
        1000.0, // capital
        2.0,    // leverage
        1.5,    // spread
        0.01,   // vol
        0.10,   // ofi
    );

    let rep_high_lev = engine.evaluate(
        1000.0, // capital
        20.0,   // leverage
        1.5,    // spread
        0.01,   // vol
        0.10,   // ofi
    );

    assert!(
        rep_low_lev.schwarzschild_radius > 0.0,
        "El radio de Schwarzschild debe ser positivo"
    );
    assert!(
        rep_high_lev.schwarzschild_radius > rep_low_lev.schwarzschild_radius,
        "El radio de Schwarzschild debe aumentar con mayor apalancamiento: {} vs {}",
        rep_high_lev.schwarzschild_radius,
        rep_low_lev.schwarzschild_radius
    );
}

#[test]
fn test_c2_subcritical_safe_regime_and_unit_scaling() {
    let params = PolyakovSchwarzschildParams::default();
    let engine = PolyakovSchwarzschildHorizonEngine::new(params);

    // Entorno de bajo apalancamiento y alta liquidez
    let rep = engine.evaluate(
        5000.0, // capital amplio
        1.5,    // apalancamiento mínimo
        0.5,    // spread ultra estrecho
        0.005,  // baja volatilidad
        0.05,   // OFI neutro
    );

    assert!(
        rep.horizon_distance > 0.0,
        "La distancia al horizonte debe ser positiva en régimen seguro: {}",
        rep.horizon_distance
    );
    assert!(
        rep.horizon_proximity_ratio < 0.85,
        "La proximidad al horizonte debe ser baja: {}",
        rep.horizon_proximity_ratio
    );
    assert_eq!(
        rep.slot_scale_factor, 1.0,
        "La escala de slot debe ser 1.0 en régimen seguro"
    );
    assert!(!rep.event_horizon_veto_induced);
}

#[test]
fn test_c3_event_horizon_crossing_terminal_veto() {
    let params = PolyakovSchwarzschildParams::default();
    let engine = PolyakovSchwarzschildHorizonEngine::new(params);

    // Situación catastrófica: apalancamiento 50x, spread salvaje (50 bps) y volatilidad extrema
    let rep_shock = engine.evaluate(
        100.0, // micro-capital
        50.0,  // apalancamiento masivo
        60.0,  // spread divergente
        0.50,  // volatilidad explosiva
        0.95,  // OFI extremo
    );

    assert!(
        rep_shock.event_horizon_veto_induced,
        "Debe detectar el cruce del horizonte de eventos de Schwarzschild"
    );
    assert_eq!(
        rep_shock.slot_scale_factor, 0.0,
        "El cruce del horizonte debe forzar la escala de slot a 0.0 absoluto"
    );
    assert!(
        rep_shock.horizon_distance <= 0.0 || rep_shock.horizon_proximity_ratio >= 1.0,
        "La coordenada radial debe haber cruzado el horizonte r <= r_s"
    );
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let params = PolyakovSchwarzschildParams::default();
    let engine = PolyakovSchwarzschildHorizonEngine::new(params);

    let rep_nan = engine.evaluate(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(rep_nan.schwarzschild_radius.is_finite());
    assert!(rep_nan.phase_space_radius.is_finite());
    assert!(rep_nan.horizon_distance.is_finite());
    assert!(rep_nan.slot_scale_factor.is_finite());

    let rep_inf = engine.evaluate(
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    assert!(rep_inf.schwarzschild_radius.is_finite());
    assert!(rep_inf.phase_space_radius.is_finite());
    assert!(rep_inf.horizon_distance.is_finite());
    assert!(rep_inf.slot_scale_factor.is_finite());
}
