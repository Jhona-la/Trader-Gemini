//! # Contrato Formal de Certificación: Fase 101 (§101) — FQHE & Chern-Simons $U(1)$
//!
//! ## Modo Profesor: 7 Pilares de Verificación Cuántica
//! - QUÉ: Validación formal del motor `FqheChernSimonsEngine` y sus propiedades topológicas.
//! - POR QUÉ: Garantizar que la cuantización de Hall, las mesetas de Jain, la corriente de borde quiral y el veto de colapso de gap operen con precisión matemática.
//! - PARA QUÉ: Blindar la cota de ruina Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y generar señales topológicas protegidas.
//! - CÓMO: 5 contratos exhaustivos testeando pureza inicial, cuantización de plateaus, corrientes quirales, veto de gap y resiliencia `rb_p1`.
//! - CUÁNDO: En cada ejecución del pipeline de integración continua de `signal-engine`.
//! - DÓNDE: `crates/signal-engine/tests/f101_001_fqhe_chern_simons_contract.rs`.
//! - QUIÉN: Rol 5 (QA Engineer) y Rol 8 (Físico Cuántico de Materia Condensada Topológica).

use signal_engine::fqhe_chern_simons::{
    FqheChernSimonsConfig, FqheChernSimonsEngine, FQHE_PLATEAUS,
};

#[test]
fn test_c1_fqhe_initialization_and_purity() {
    let mut engine = FqheChernSimonsEngine::default();
    // total_flow = 10.0, total_depth = 200.0 -> nu = 2*pi*10/200 = pi/10 ~ 0.31416 -> nearest plateau = 1/3 (0.3333)
    let report = engine.update(5.0, 5.0, 100.0, 100.0, 0.0);

    assert!(
        report.energy_gap >= 0.0,
        "El gap de energía debe ser no negativo"
    );
    assert!(
        !report.fqhe_gap_collapse_veto,
        "No debe haber veto en régimen de balance de inicialización"
    );
    assert!(report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25);
    assert!(
        (report.nearest_plateau_nu - (1.0 / 3.0)).abs() < 0.05,
        "Debe aproximar la meseta 1/3"
    );
}

#[test]
fn test_c2_jain_plateau_localization_and_topological_quantization() {
    // 1. Probar localización de mesetas conocidas
    let (p1, d1) = FqheChernSimonsEngine::find_nearest_plateau(0.34);
    assert!(
        (p1 - (1.0 / 3.0)).abs() < 1e-6,
        "0.34 debe mapear a la meseta Laughlin 1/3"
    );
    assert!(d1 < 0.02);

    let (p2, d2) = FqheChernSimonsEngine::find_nearest_plateau(0.405);
    assert!(
        (p2 - (2.0 / 5.0)).abs() < 1e-6,
        "0.405 debe mapear a la meseta Jain 2/5"
    );
    assert!(d2 < 0.01);

    let (p3, d3) = FqheChernSimonsEngine::find_nearest_plateau(0.505);
    assert!(
        (p3 - (1.0 / 2.0)).abs() < 1e-6,
        "0.505 debe mapear a la meseta 1/2"
    );
    assert!(d3 < 0.01);

    let (p4, d4) = FqheChernSimonsEngine::find_nearest_plateau(0.98);
    assert!(
        (p4 - 1.0).abs() < 1e-6,
        "0.98 debe mapear al estado entero IQHE 1.0"
    );
    assert!(d4 < 0.03);

    // 2. Comprobar que todas las mesetas de FQHE_PLATEAUS tienen propiedades físicas cuantizadas
    for &plateau in &FQHE_PLATEAUS {
        let (found, dist) = FqheChernSimonsEngine::find_nearest_plateau(plateau);
        assert_eq!(found, plateau);
        assert_eq!(dist, 0.0);
    }
}

#[test]
fn test_c3_chiral_edge_current_and_conviction_direction() {
    let mut engine = FqheChernSimonsEngine::default();

    // Régimen de compradores agresivos dominantes
    let rep_buy = engine.update(100.0, 5.0, 500.0, 500.0, 15.0);
    assert!(
        rep_buy.chiral_edge_current > 0.0,
        "Flujo comprador debe inducir corriente quiral positiva"
    );
    assert!(
        rep_buy.signal_conviction > 0.0,
        "Convicción debe ser positiva para compras agresivas"
    );

    // Régimen de vendedores agresivos dominantes
    let rep_sell = engine.update(5.0, 100.0, 500.0, 500.0, -15.0);
    assert!(
        rep_sell.chiral_edge_current < 0.0,
        "Flujo vendedor debe inducir corriente quiral negativa"
    );
    assert!(
        rep_sell.signal_conviction < 0.0,
        "Convicción debe ser negativa para ventas agresivas"
    );
}

#[test]
fn test_c4_gap_collapse_and_veto_activation() {
    let config = FqheChernSimonsConfig {
        coulomb_coupling: 1.0,
        gap_collapse_threshold: 0.20,
        plateau_tolerance: 0.03, // Tolerancia muy estricta para inducir colapso fuera de meseta
        chern_simons_level: 1,
    };
    let mut engine = FqheChernSimonsEngine::new(config);

    // Crear un llenado nu artificialmente situado en la brecha disipativa intermedia (ej: nu = 0.75, lejos de 2/3 y 1.0)
    // 2 * pi * flow / depth ~ 0.75 -> flow = 0.75 * 1000 / (2 * pi) ~ 119.36
    let flow = 0.75 * 1000.0 / (2.0 * std::f64::consts::PI);
    for _ in 0..10 {
        engine.update(flow / 2.0, flow / 2.0, 500.0, 500.0, 0.0);
    }

    let report = engine.update(flow / 2.0, flow / 2.0, 500.0, 500.0, 0.0);

    // En nu ~ 0.75, dist a 2/3 (0.6667) es ~0.083 > tolerance (0.03), gap debe colapsar a 0.0
    assert!(
        report.energy_gap < config.gap_collapse_threshold,
        "El gap ({}) debe colapsar por debajo del umbral ({})",
        report.energy_gap,
        config.gap_collapse_threshold
    );
    assert!(
        report.fqhe_gap_collapse_veto,
        "Debe detonar FQHE_GAP_COLLAPSE_VETO"
    );
    assert!(
        (report.slot_scaling_factor - 0.10).abs() < 1e-6,
        "Slot factor debe colapsar a 0.10 bajo veto"
    );
}

#[test]
fn test_c5_rb_p1_resilience_and_extreme_inputs() {
    let mut engine = FqheChernSimonsEngine::default();

    // Valores extremos, NaN, Inf, negativos
    let rep = engine.update(f64::NAN, f64::INFINITY, -100.0, 0.0, f64::NEG_INFINITY);

    assert!(rep.filling_factor_nu.is_finite(), "nu debe ser finito");
    assert!(rep.energy_gap.is_finite(), "gap debe ser finito");
    assert!(
        rep.hall_conductance.is_finite(),
        "conductancia debe ser finita"
    );
    assert!(
        rep.chiral_edge_current.is_finite(),
        "corriente quiral debe ser finita"
    );
    assert!(
        rep.signal_conviction.is_finite(),
        "convicción debe ser finita"
    );
    assert!(rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25);
}
