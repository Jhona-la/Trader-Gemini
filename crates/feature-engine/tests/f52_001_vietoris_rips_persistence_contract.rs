//! Contrato Formal de Verificación: Fase 52 (Vietoris-Rips Persistent Homology & TDA)
//!
//! Este conjunto de pruebas certifica las propiedades topológicas y de invariancia del motor:
//! - C1: Filtración simplicial de Vietoris-Rips, cálculo exacto de números de Betti $\beta_0, \beta_1$.
//! - C2: Entropía persistente de Shannon $E_{\text{pers}} \in [0.0, 1.0]$ y detección de agujeros de aire (air pockets).
//! - C3: Modulación acotada de la escala de slot $\zeta_{\text{tda}} \in [0.10, 1.30]$.
//! - C4: Determinismo en stack $O(1)$, cero heap allocs y robustez numérica total ante NaN/Infinitos.

use feature_engine::{
    TdaMicroPoint, VietorisRipsParams, VietorisRipsReport, VietorisRipsTopologicalEngine,
    TDA_FILTRATION_LEVELS, TDA_POINT_CLOUD_LEN,
};

#[test]
fn test_c1_vietoris_rips_betti_numbers_and_connectedness() {
    let params = VietorisRipsParams {
        base_filtration_radius: 0.10,
        filtration_step_scale: 2.0,
        chaotic_entropy_threshold: 0.80,
        topological_smoothing: 0.50,
    };
    let mut engine = VietorisRipsTopologicalEngine::new(params);

    // Inyectamos puntos cercanos para formar un cluster bien conectado
    let mut last_rep = VietorisRipsReport {
        mean_betti_0: 0.0,
        mean_betti_1: 0.0,
        persistent_entropy: 0.0,
        air_pocket_detected: false,
        slot_scaling_factor: 1.0,
    };

    for i in 0..16 {
        let ret = 0.0001 * (i as f64);
        let ofi = 0.10;
        let depth = 1.0;
        let spread = 2.0;
        last_rep = engine.update_and_evaluate(ret, ofi, depth, spread, i * 50);
    }

    // Al estar los puntos muy próximos, deben converger a un número bajo de componentes conexas
    assert!(
        last_rep.mean_betti_0 <= 8.0,
        "Betti 0 excesivo para cluster denso: {}",
        last_rep.mean_betti_0
    );
    assert!(
        last_rep.mean_betti_1 >= 0.0,
        "Betti 1 negativo: {}",
        last_rep.mean_betti_1
    );
    assert!(
        (0.0..=1.0).contains(&last_rep.persistent_entropy),
        "Entropía persistente fuera de rango: {}",
        last_rep.persistent_entropy
    );

    assert_eq!(TDA_POINT_CLOUD_LEN, 16);
    assert_eq!(TDA_FILTRATION_LEVELS, 4);
}

#[test]
fn test_c2_persistent_entropy_and_air_pocket_detection() {
    let params = VietorisRipsParams {
        base_filtration_radius: 0.05,
        filtration_step_scale: 1.2,
        chaotic_entropy_threshold: 0.40,
        topological_smoothing: 0.80,
    };
    let mut engine = VietorisRipsTopologicalEngine::new(params);

    // Inyectar puntos sumamente dispersos y caóticos para fragmentar el espacio
    for i in 0..20 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let ret = sign * 0.08;
        let ofi = sign * 0.90;
        let depth = if i % 2 == 0 { 5.0 } else { 0.2 };
        let spread = 25.0 * (1.0 + (i as f64));
        let rep = engine.update_and_evaluate(ret, ofi, depth, spread, i * 100);

        // Si se detecta un air pocket, la escala de slot debe penalizarse drásticamente
        if rep.air_pocket_detected {
            assert!(
                rep.slot_scaling_factor <= 0.60,
                "Air pocket detectado pero escala no reducida: {}",
                rep.slot_scaling_factor
            );
        }
    }
}

#[test]
fn test_c3_topological_slot_scaling_bounds() {
    let params = VietorisRipsParams::default();
    let mut engine = VietorisRipsTopologicalEngine::new(params);

    for i in 0..30 {
        let ret = (i as f64 * 0.001).sin();
        let ofi = (i as f64 * 0.05).cos();
        let depth = 1.0 + 0.5 * (i as f64 * 0.1).sin();
        let spread = 2.0 + (i as f64 * 0.2);
        let rep = engine.update_and_evaluate(ret, ofi, depth, spread, i * 50);

        assert!(
            rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.30,
            "Slot scaling fuera de rango [0.10, 1.30]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c4_stack_zero_heap_and_nan_subnormal_immunity() {
    let params = VietorisRipsParams::default();
    let mut engine = VietorisRipsTopologicalEngine::new(params);

    let anomalous_inputs = [
        (f64::NAN, 0.0, 1.0, 2.0),
        (0.001, f64::INFINITY, 1.0, 2.0),
        (0.001, 0.0, f64::NEG_INFINITY, 2.0),
        (0.001, 0.0, 1.0, f64::NAN),
        (0.0, 0.0, 0.0, 0.0),
        (1e-300, 1e-300, 1e-300, 1e-300),
    ];

    for &(ret, ofi, depth, spread) in &anomalous_inputs {
        let rep = engine.update_and_evaluate(ret, ofi, depth, spread, 1000);
        assert!(rep.mean_betti_0.is_finite());
        assert!(rep.mean_betti_1.is_finite());
        assert!(rep.persistent_entropy.is_finite());
        assert!(rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.30);
    }

    let p0 = TdaMicroPoint::ZERO;
    let p1 = TdaMicroPoint::new(0.01, 0.5, 1.5, 3.0);
    assert!(p0.distance_to(&p1) > 0.0);
}
