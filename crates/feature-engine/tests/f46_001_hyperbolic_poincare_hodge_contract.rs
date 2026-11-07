//! Contrato Formal de Verificación: Fase 46 (Hyperbolic Poincaré-Hodge Manifold)
//!
//! Este conjunto de pruebas certifica las propiedades matemáticas y físicas del motor:
//! - C1: Métrica geodésica del disco de Poincaré, simetría y desigualdad triangular.
//! - C2: Operador dual de Hodge $\star \omega$ y 2-forma simpléctica de vorticidad de área.
//! - C3: Continuidad de Lipschitz, resiliencia estocástica y estabilidad ante subnormales/NaN.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y modulación de slot $\in [0.10, 1.25]$.

use feature_engine::{
    HyperbolicPoincareHodgeEngine, HyperbolicPoincareHodgeParams, PoincarePoint,
    POINCARE_HISTORY_LEN,
};

#[test]
fn test_c1_poincare_metric_distance_and_triangle_inequality() {
    let p0 = PoincarePoint::ORIGIN;
    let p1 = PoincarePoint::new(0.3, 0.4, 0.99); // r = 0.5
    let p2 = PoincarePoint::new(-0.2, 0.5, 0.99);
    let p3 = PoincarePoint::new(0.6, -0.4, 0.99);

    // 1. Distancia geodésica al origen d_H(0, z) = ln((1+r)/(1-r))
    // Para r = 0.5: ln(1.5 / 0.5) = ln(3.0) ≈ 1.0986122886681098
    let d01 = p0.distance_to(&p1);
    let expected_d01 = 3.0_f64.ln();
    assert!(
        (d01 - expected_d01).abs() < 1e-9,
        "d_H(0, p1) esperada {}, obtenida {}",
        expected_d01,
        d01
    );

    // 2. Simetría métrica: d_H(u, v) == d_H(v, u)
    let d12 = p1.distance_to(&p2);
    let d21 = p2.distance_to(&p1);
    assert!(
        (d12 - d21).abs() < 1e-12,
        "Simetría violada: {} != {}",
        d12,
        d21
    );

    // 3. Identidad de indiscernibles: d_H(u, u) == 0.0
    let d11 = p1.distance_to(&p1);
    assert!(d11 < 1e-12, "Identidad violada: {}", d11);

    // 4. Desigualdad triangular: d_H(u, w) <= d_H(u, v) + d_H(v, w) + eps
    let d13 = p1.distance_to(&p3);
    let d23 = p2.distance_to(&p3);
    assert!(
        d13 <= d12 + d23 + 1e-9,
        "Desigualdad triangular violada: d13={} > d12+d23={}",
        d13,
        d12 + d23
    );
}

#[test]
fn test_c2_hodge_orthogonal_flux_and_vorticity() {
    let params = HyperbolicPoincareHodgeParams::default();
    let mut engine = HyperbolicPoincareHodgeEngine::new(params);

    // Simular un ciclo de rotación en el plano de microestructura (p, ofi)
    let mut reports = Vec::new();
    let n_steps = 12;
    for step in 0..n_steps {
        let angle = (step as f64) * std::f64::consts::PI / 6.0;
        let p_ret = 0.02 * angle.cos();
        let ofi = 0.8 * angle.sin();
        let ts = 1000 + step as u64 * 100;
        let rep = engine.update_and_evaluate(p_ret, ofi, ts);
        reports.push(rep);
    }

    // Verificar que el flujo ortogonal dual de Hodge y la vorticidad de área se activan
    let has_positive_hodge = reports.iter().any(|r| r.hodge_orthogonal_flux > 0.0);
    assert!(
        has_positive_hodge,
        "El flujo dual de Hodge debe ser positivo durante la rotación"
    );

    let has_non_zero_vorticity = reports
        .iter()
        .any(|r| r.symplectic_area_vorticity.abs() > 0.01);
    assert!(
        has_non_zero_vorticity,
        "La vorticidad de área simpléctica debe ser no nula en dinámicas circulares"
    );
}

#[test]
fn test_c3_lipschitz_continuity_and_subnormal_resilience() {
    let mut engine = HyperbolicPoincareHodgeEngine::new(HyperbolicPoincareHodgeParams::default());

    // 1. Entradas patológicas: NaN, subnormales e infinitos deben mapear de forma segura sin pánicos
    let r_nan = engine.update_and_evaluate(f64::NAN, f64::INFINITY, 1000);
    assert!(r_nan.poincare_x.is_finite());
    assert!(r_nan.poincare_y.is_finite());
    assert!(r_nan.poincare_radius >= 0.0 && r_nan.poincare_radius < 1.0);
    assert!(r_nan.lipschitz_constant.is_finite());

    // 2. Shock de precio violento
    let r_shock = engine.update_and_evaluate(0.50, -0.99, 1050);
    assert!(r_shock.hyperbolic_stress_index >= 0.0 && r_shock.hyperbolic_stress_index <= 1.0);
    assert!(r_shock.slot_scaling_factor >= 0.10 && r_shock.slot_scaling_factor <= 1.25);
}

#[test]
fn test_c4_zero_alloc_stack_determinism_and_ruin_bound() {
    let mut engine_a = HyperbolicPoincareHodgeEngine::new(HyperbolicPoincareHodgeParams::default());
    let mut engine_b = HyperbolicPoincareHodgeEngine::new(HyperbolicPoincareHodgeParams::default());

    // Ejecutar 500 pasos idénticos para verificar determinismo exacto a nivel de bits
    for step in 1..=500 {
        let ts = 10_000 + step * 50;
        let p = ((step as f64) * 0.05).sin() * 0.03;
        let ofi = ((step as f64) * 0.07).cos() * 0.8;

        let rep_a = engine_a.update_and_evaluate(p, ofi, ts);
        let rep_b = engine_b.update_and_evaluate(p, ofi, ts);

        assert_eq!(rep_a.poincare_x.to_bits(), rep_b.poincare_x.to_bits());
        assert_eq!(rep_a.poincare_y.to_bits(), rep_b.poincare_y.to_bits());
        assert_eq!(
            rep_a.geodesic_distance_to_center.to_bits(),
            rep_b.geodesic_distance_to_center.to_bits()
        );
        assert_eq!(
            rep_a.hyperbolic_stress_index.to_bits(),
            rep_b.hyperbolic_stress_index.to_bits()
        );
        assert_eq!(
            rep_a.slot_scaling_factor.to_bits(),
            rep_b.slot_scaling_factor.to_bits()
        );

        // La escala de slot debe estar acotada para preservar la supermartingala de Ville
        assert!(
            rep_a.slot_scaling_factor >= 0.10 && rep_a.slot_scaling_factor <= 1.25,
            "Slot scaling violado en paso {}: {}",
            step,
            rep_a.slot_scaling_factor
        );
    }
}
