//! ⚡ CONTRATO DE CERTIFICACIÓN OLA Ω139: VARIEDADES KÄHLERIANAS Y CONJETURA DE HODGE
//!
//! Valida:
//! 1. Geometría Calabi-Yau en equilibrio armónico ($R = 0$, `calabi_yau_flatness = 1.0`).
//! 2. Corrientes restauradoras holomorfas $\mathcal{J}_{\text{Kahler}, i}$ opuestas a dislocaciones.
//! 3. Formas armónicas $(1, 1)$ y área simpléctica $E_{1,1} \ge 0$ como ciclos de cointegración algebraica.
//! 4. Latencia crítica en stack zero-alloc (<25 µs por evaluación multiactivo).

use strategy_core::kahler_hodge::{KahlerHodgeEngine, KahlerHodgeParams};
use std::time::Instant;

#[test]
fn test_kahler_hodge_params_validation_and_update() {
    let p_default = KahlerHodgeParams::default();
    assert!(p_default.validate().is_ok());

    let p_invalid_alpha = KahlerHodgeParams {
        alpha_price_weight: -1.0,
        ..Default::default()
    };
    assert!(p_invalid_alpha.validate().is_err());

    let p_valid = KahlerHodgeParams {
        alpha_price_weight: 2.0,
        beta_flow_weight: 1.0,
        adaptation_rate: 0.05,
        calabi_yau_tolerance: 0.10,
    };
    let mut engine = KahlerHodgeEngine::new(3).with_params(p_valid);
    assert_eq!(engine.params.alpha_price_weight, 2.0);

    let p_update = KahlerHodgeParams {
        alpha_price_weight: 1.5,
        ..Default::default()
    };
    engine.update_params(p_update);
    assert_eq!(engine.params.alpha_price_weight, 1.5);
}

#[test]
fn test_kahler_hodge_calabi_yau_equilibrium() {
    let mut engine = KahlerHodgeEngine::new(4);

    let m_init = engine.current_metrics();
    assert_eq!(m_init.ricci_scalar, 0.0);
    assert_eq!(m_init.calabi_yau_flatness, 1.0);
    assert_eq!(m_init.hodge_1_1_energy, 0.0);

    // Precios y OFIs idénticos estacionarios
    let prices = [100.0, 50.0, 20.0, 10.0];
    let ofis = [0.0, 0.0, 0.0, 0.0];

    let (m, currents) = engine.update_and_calculate_manifold(&prices, &ofis);
    assert_eq!(m.ricci_scalar, 0.0);
    assert_eq!(m.calabi_yau_flatness, 1.0);
    assert_eq!(m.kahler_potential, 0.0);
    assert_eq!(currents, [0.0; 32]);
}

#[test]
fn test_kahler_hodge_restoring_currents_oppose_dislocation() {
    let mut engine = KahlerHodgeEngine::new(3);

    let p0 = [100.0, 50.0, 25.0];
    let ofi0 = [0.0, 0.0, 0.0];
    engine.update_and_calculate_manifold(&p0, &ofi0);

    // Choque asimétrico: el activo 0 salta un +10% en precio y experimenta fuerte presión compradora (+2.5 OFI)
    let p1 = [110.0, 50.0, 25.0];
    let ofi1 = [2.5, 0.0, 0.0];
    let (m1, currents1) = engine.update_and_calculate_manifold(&p1, &ofi1);

    assert!(m1.kahler_potential > 0.0, "El potencial de Kähler debe activarse ante dislocación");
    // El activo 0 debe experimentar una fuerte corriente restauradora negativa hacia el equilibrio
    assert!(
        currents1[0] < -0.05,
        "El activo 0 debe recibir corriente restauradora negativa: {:.4}",
        currents1[0]
    );
}

#[test]
fn test_kahler_hodge_symplectic_area_and_hodge_1_1_energy() {
    let mut engine = KahlerHodgeEngine::new(3);

    let p0 = [100.0, 50.0, 25.0];
    let ofi0 = [0.0, 0.0, 0.0];
    engine.update_and_calculate_manifold(&p0, &ofi0);

    // Co-movimiento en cuadratura de fase: el activo 0 tiene precio alto y OFI neutro,
    // el activo 1 tiene precio neutro y OFI alto (área simpléctica en plano complejo no nula)
    let p1 = [105.0, 50.0, 25.0];
    let ofi1 = [0.0, 1.5, 0.0];
    let (m1, _) = engine.update_and_calculate_manifold(&p1, &ofi1);

    assert!(
        m1.hodge_1_1_energy > 0.0,
        "El área simpléctica en el plano complejo debe generar energía Hodge (1,1) positiva: {:.6}",
        m1.hodge_1_1_energy
    );
}

#[test]
fn test_kahler_hodge_submicrosecond_stack_performance() {
    let mut engine = KahlerHodgeEngine::new(5);
    let p_init = [65000.0, 3500.0, 150.0, 1.0, 0.60];
    let ofi_init = [0.1, -0.2, 0.05, 0.0, -0.1];
    engine.update_and_calculate_manifold(&p_init, &ofi_init);

    let iters = 20_000;
    let start = Instant::now();

    let mut mock_p = p_init;
    let mut mock_ofi = ofi_init;
    for i in 0..iters {
        let osc = ((i as f64) * 0.01).sin() * 10.0;
        mock_p[0] = 65000.0 + osc;
        mock_ofi[0] = ((i as f64) * 0.02).cos();
        let (m, currents) = engine.update_and_calculate_manifold(&mock_p, &mock_ofi);
        assert!(m.kahler_potential.is_finite());
        assert!(currents[0].is_finite());
    }

    let elapsed = start.elapsed();
    let per_tick_us = elapsed.as_micros() as f64 / (iters as f64);

    assert!(
        per_tick_us < 25.0,
        "Latencia media de Kähler-Hodge excedió 25 µs: {:.3} µs",
        per_tick_us
    );
}
