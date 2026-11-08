//! Contrato formal de verificación para Fase 37: Teoría Gauge No Abeliana SU(3) Cromodinámica Multiactivo
//!
//! Invariantes certificados:
//! 1. c1: Antisimetría de constantes de estructura $f^{abc}$ e identidad de Jacobi en $\mathfrak{su}(3)$.
//! 2. c2: Conservación estricta de carga de color: $\sum_{i=0}^2 K_i = 0$.
//! 3. c3: Activación del término no abeliano $\|g [A_0, A_1]\| > 0$ bajo rotaciones cruzadas de liquidez.
//! 4. c4: Coherencia física de señales alfa de arbitraje cromodinámico en la tríada.
//! 5. c5: Cero pánicos (`rb_p1`) y robustez ante entradas no acotadas, NaN o Inf.

use strategy_core::chromodynamic_gauge_su3::{
    AssetColorState, ChromodynamicGaugeSu3Engine, Su3GaugeParams, SU3_GENERATORS,
};
use strategy_core::QuantumStrategy;

#[test]
fn c1_su3_structure_constants_antisymmetry_and_invariants() {
    // 1. Verificar valores canónicos conocidos
    let f123 = ChromodynamicGaugeSu3Engine::structure_constant_f(0, 1, 2);
    assert!(
        (f123 - 1.0).abs() < 1e-12,
        "f^{{123}} debe ser exactamente 1.0"
    );

    let f147 = ChromodynamicGaugeSu3Engine::structure_constant_f(0, 3, 6);
    assert!((f147 - 0.5).abs() < 1e-12, "f^{{147}} debe ser 0.5");

    let f458 = ChromodynamicGaugeSu3Engine::structure_constant_f(3, 4, 7);
    assert!(
        (f458 - (3.0_f64.sqrt() / 2.0)).abs() < 1e-12,
        "f^{{458}} debe ser sqrt(3)/2"
    );

    // 2. Verificar antisimetría total para todas las tripletas
    for a in 0..SU3_GENERATORS {
        for b in 0..SU3_GENERATORS {
            for c in 0..SU3_GENERATORS {
                let fabc = ChromodynamicGaugeSu3Engine::structure_constant_f(a, b, c);
                let fbac = ChromodynamicGaugeSu3Engine::structure_constant_f(b, a, c);
                let facb = ChromodynamicGaugeSu3Engine::structure_constant_f(a, c, b);

                assert!(
                    (fabc + fbac).abs() < 1e-12,
                    "Antisimetría f^{{abc}} = -f^{{bac}} violada para ({}, {}, {})",
                    a,
                    b,
                    c
                );
                assert!(
                    (fabc + facb).abs() < 1e-12,
                    "Antisimetría f^{{abc}} = -f^{{acb}} violada para ({}, {}, {})",
                    a,
                    b,
                    c
                );
            }
        }
    }

    // 3. Verificar identidad de Jacobi en su(3):
    // \sum_e (f^{abe} f^{cde} + f^{bce} f^{ade} + f^{cae} f^{bde}) = 0
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let mut sum_jacobi = 0.0;
                    for e in 0..SU3_GENERATORS {
                        let t1 = ChromodynamicGaugeSu3Engine::structure_constant_f(a, b, e)
                            * ChromodynamicGaugeSu3Engine::structure_constant_f(c, d, e);
                        let t2 = ChromodynamicGaugeSu3Engine::structure_constant_f(b, c, e)
                            * ChromodynamicGaugeSu3Engine::structure_constant_f(a, d, e);
                        let t3 = ChromodynamicGaugeSu3Engine::structure_constant_f(c, a, e)
                            * ChromodynamicGaugeSu3Engine::structure_constant_f(b, d, e);
                        sum_jacobi += t1 + t2 + t3;
                    }
                    assert!(
                        sum_jacobi.abs() < 1e-10,
                        "Identidad de Jacobi violada para ({}, {}, {}, {}): {}",
                        a,
                        b,
                        c,
                        d,
                        sum_jacobi
                    );
                }
            }
        }
    }
}

#[test]
fn c2_color_charge_conservation_zero_net_force() {
    let mut engine = ChromodynamicGaugeSu3Engine::new(Su3GaugeParams::default());

    let assets = [
        AssetColorState {
            normalized_return: 0.025,
            order_flow_imbalance: 0.65,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: -0.015,
            order_flow_imbalance: -0.40,
            relative_weight: 0.35,
        },
        AssetColorState {
            normalized_return: 0.005,
            order_flow_imbalance: -0.25,
            relative_weight: 0.15,
        },
    ];

    // Paso 1 de calentamiento
    let _r1 = engine.update_triad(&assets, 1.0);
    // Paso 2 con dinámica temporal
    let report = engine.update_triad(&assets, 1.0);

    // Suma de fuerzas cromodinámicas sobre los 3 colores debe ser estrictamente cero: \sum K_i = 0
    let force_sum: f64 = report.color_forces.iter().sum();
    assert!(
        force_sum.abs() < 1e-12,
        "Conservación estricta de color violada: suma de fuerzas = {}",
        force_sum
    );

    // Invariantes positivos
    assert!(
        report.yang_mills_action >= 0.0,
        "Acción de Yang-Mills debe ser no negativa"
    );
    assert!(
        report.casimir_invariant >= 0.0,
        "Invariante de Casimir debe ser no negativo"
    );
}

#[test]
fn c3_non_abelian_curl_activation_under_asymmetric_flows() {
    let mut engine = ChromodynamicGaugeSu3Engine::new(Su3GaugeParams {
        gauge_coupling_g: 0.25,
        ..Su3GaugeParams::default()
    });

    // Flujo altamente asimétrico y no conmutativo
    let asymmetric_triad = [
        AssetColorState {
            normalized_return: 0.04,
            order_flow_imbalance: 0.85,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: -0.03,
            order_flow_imbalance: 0.40,
            relative_weight: 0.25,
        },
        AssetColorState {
            normalized_return: -0.01,
            order_flow_imbalance: -0.90,
            relative_weight: 0.15,
        },
    ];

    let report = engine.update_triad(&asymmetric_triad, 0.5);

    // El término no abeliano debe ser estrictamente no nulo
    assert!(
        report.non_abelian_curl_norm > 1e-5,
        "El término no abeliano de auto-interacción gluónica debe activarse"
    );
}

#[test]
fn c4_triad_cross_arbitrage_alpha_signs() {
    let mut engine = ChromodynamicGaugeSu3Engine::new(Su3GaugeParams::default());

    // BTC fuerte al alza con OFI positivo masivo, ETH rezagado y presionado a la baja
    let triad = [
        AssetColorState {
            normalized_return: 0.03,
            order_flow_imbalance: 0.90,
            relative_weight: 0.55,
        },
        AssetColorState {
            normalized_return: -0.02,
            order_flow_imbalance: -0.80,
            relative_weight: 0.30,
        },
        AssetColorState {
            normalized_return: 0.00,
            order_flow_imbalance: 0.00,
            relative_weight: 0.15,
        },
    ];

    let _step1 = engine.update_triad(&triad, 1.0);
    let step2 = engine.update_triad(&triad, 1.0);

    // Las señales alfa deben tener signos opuestos reflejando la atracción restauradora de Yang-Mills
    assert!(
        step2.alpha_signals[0] > 0.0,
        "Color 0 (BTC) debe recibir fuerza positiva"
    );
    assert!(
        step2.alpha_signals[1] < 0.0,
        "Color 1 (ETH) debe recibir fuerza negativa relativa"
    );

    // Verificación de la interfaz QuantumStrategy
    assert_eq!(engine.name(), "ChromodynamicGaugeSu3Engine");
    assert!(engine.is_operable_for_coin(0, "BTCUSDT"));
    assert!((engine.evaluate_for_coin(0, "BTCUSDT") - step2.alpha_signals[0]).abs() < 1e-12);
    assert!((engine.evaluate_for_coin(1, "ETHUSDT") - step2.alpha_signals[1]).abs() < 1e-12);
}

#[test]
fn c5_zero_panic_guarantee() {
    let mut engine = ChromodynamicGaugeSu3Engine::new(Su3GaugeParams::default());

    // Entradas patológicas: ceros, NaNs, Infs, negativos
    let pathological_triad = [
        AssetColorState {
            normalized_return: f64::NAN,
            order_flow_imbalance: f64::INFINITY,
            relative_weight: -100.0,
        },
        AssetColorState {
            normalized_return: f64::NEG_INFINITY,
            order_flow_imbalance: 0.0,
            relative_weight: 0.0,
        },
        AssetColorState {
            normalized_return: 1e308,
            order_flow_imbalance: -1e308,
            relative_weight: f64::NAN,
        },
    ];

    let report = engine.update_triad(&pathological_triad, 0.0);

    // Verificación de que no hay panics y todos los campos numéricos son finitos
    assert!(report.yang_mills_action.is_finite());
    assert!(report.casimir_invariant.is_finite());
    assert!(report.topological_charge.is_finite());
    for f in report.color_forces.iter() {
        assert!(f.is_finite());
    }
    for a in report.alpha_signals.iter() {
        assert!(a.is_finite());
        assert!(*a >= -1.0 && *a <= 1.0);
    }
}
