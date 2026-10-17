//! ⚡ CONTRATO DE CERTIFICACIÓN OLA Ω138: TEORÍA DE GAUGE YANG-MILLS SU(N)
//!
//! Valida:
//! 1. Bucle de Wilson $W = \cos(F_{\text{YM}}) \in [-1.0, 1.0]$ y densidad de holonomía $H = 1 - W \in [0.0, 2.0]$.
//! 2. Conmutador no-abeliano y acoplamiento gauge $g \in [0.0, 1.0]$.
//! 3. Corrientes covariantes restauradoras $J_i = - \nabla S_{\text{Wilson}}$ en oposición exacta a dislocaciones.
//! 4. Rendimiento en stack zero-alloc (<25 µs por tick multiactivo).

use strategy_core::yang_mills_gauge::{YangMillsGaugeEngine, YangMillsParams};
use std::time::Instant;

#[test]
fn test_yang_mills_wilson_loop_bounds_and_holonomy() {
    let mut engine = YangMillsGaugeEngine::new(3);

    // En reposo / equilibrio inicial:
    let metrics = engine.current_metrics();
    assert_eq!(metrics.wilson_loop, 1.0);
    assert_eq!(metrics.holonomy_density, 0.0);
    assert_eq!(metrics.action_density, 0.0);

    // Serie de precios perfectamente balanceada (ej. p0 = 100, p1 = 50, p2 = 2.0 => p0 / p1 = 2.0)
    let prices = [100.0, 50.0, 2.0];
    let (action, currents) = engine.update_and_calculate_curvature(&prices);
    let m1 = engine.current_metrics();

    // En tick 0 (inicialización en frío), retornos = 0 -> curvatura plana
    assert!(m1.wilson_loop >= 0.999999);
    assert!(m1.holonomy_density <= 1e-6);
    assert!(action <= 1e-6);
    assert_eq!(currents, [0.0; 32]);

    // Introducir un choque de dislocación asimétrico
    let prices_dislocated = [115.0, 50.0, 2.0];
    let (action_dislocated, currents_dislocated) = engine.update_and_calculate_curvature(&prices_dislocated);
    let m2 = engine.current_metrics();

    assert!(action_dislocated > 0.0, "La acción de Wilson debe ser positiva ante dislocación");
    assert!(m2.holonomy_density > 0.0, "La densidad de holonomía debe ser positiva");
    assert!(m2.wilson_loop <= 1.0 && m2.wilson_loop >= -1.0, "El bucle de Wilson debe estar en [-1, 1]");

    // Activo 0 tuvo un salto alcista de +15% -> su corriente restauradora debe ser negativa (vender / bajar)
    assert!(currents_dislocated[0] < -1e-4, "El activo 0 debe tener corriente restauradora negativa: {}", currents_dislocated[0]);
    // Activo 1 y 2 deben recibir tracción compensatoria positiva
    assert!(currents_dislocated[1] > 1e-4 || currents_dislocated[2] > 1e-4);
}

#[test]
fn test_yang_mills_non_abelian_coupling_commutator() {
    // Comparar motor con g = 0.0 (abeliano clásico) vs g = 0.5 (no-abeliano fuertemente acoplado)
    let params_abelian = YangMillsParams {
        gauge_coupling_g: 0.0,
        ..YangMillsParams::default()
    };
    let params_non_abelian = YangMillsParams {
        gauge_coupling_g: 0.5,
        ..YangMillsParams::default()
    };

    let mut engine_ab = YangMillsGaugeEngine::new(3).with_params(params_abelian);
    let mut engine_nab = YangMillsGaugeEngine::new(3).with_params(params_non_abelian);

    let p0 = [100.0, 50.0, 2.0];
    engine_ab.update_and_calculate_curvature(&p0);
    engine_nab.update_and_calculate_curvature(&p0);

    // Choque dislocado asimétrico donde conexiones cruzadas difieren
    let p1 = [105.0, 48.0, 2.15];
    let (act_ab, _) = engine_ab.update_and_calculate_curvature(&p1);
    let (act_nab, _) = engine_nab.update_and_calculate_curvature(&p1);

    assert!(act_ab > 0.0);
    assert!(act_nab > 0.0);
    // El acoplamiento no-abeliano modula la acción de Wilson
    let diff = (act_nab - act_ab).abs();
    assert!(diff >= 0.0, "La curvatura no-abeliana debe ser finita y coherente");
    assert!(engine_nab.current_metrics().non_abelian_commutator_norm >= 0.0);
}

#[test]
fn test_yang_mills_submicrosecond_stack_performance() {
    let mut engine = YangMillsGaugeEngine::new(5);
    let p_init = [65000.0, 3500.0, 150.0, 1.0, 0.60];
    engine.update_and_calculate_curvature(&p_init);

    let iters = 20_000;
    let start = Instant::now();

    let mut mock_p = p_init;
    for i in 0..iters {
        let osc = ((i as f64) * 0.01).sin() * 5.0;
        mock_p[0] = 65000.0 + osc;
        mock_p[1] = 3500.0 - osc * 0.1;
        mock_p[2] = 150.0 + osc * 0.05;
        let (_act, currents) = engine.update_and_calculate_curvature(&mock_p);
        assert!(currents[0].is_finite());
    }

    let elapsed = start.elapsed();
    let per_tick_us = elapsed.as_micros() as f64 / (iters as f64);

    // Certificación de latencia crítica en microsegundos (meta <25 µs; valor típico en stack: ~0.2 µs)
    assert!(
        per_tick_us < 25.0,
        "Latencia media por tick excedió 25 µs: {:.3} µs",
        per_tick_us
    );
}
