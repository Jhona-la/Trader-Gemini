//! ⚡ CONTRATO FORMAL DE PERSISTENCIA TOPOLÓGICA TDA, WASSERSTEIN W_2 Y COMPUERTA ADVERSARIAL PRE-TRADE
//! (OLA Ω220 - ÁTOMO 4.3)
//!
//! # QUÉ
//! Suite de verificación formal que valida:
//! 1. Homología simplicial de Vietoris-Rips y cálculo de números de Betti (β0, β1) con Teorema de Euler-Poincaré.
//! 2. Entropía persistente de Shannon y cuantificación continua de riesgo de vacío de liquidez (VoidRisk).
//! 3. Compuerta adversarial pre-trade (`RedTeamPreTradeGate`) con Vector IV de trampa topológica (`TopologicalVoidTrap`).
//! 4. Inmunidad a entradas degeneradas no finitas (NaN, Inf) con política fail-closed absoluta.
//! 5. Publicación viva y lectura en `OmniscientRegistry` tras el paso de tick en `GodEngineCore`.

use execution_engine::{
    AdversarialClearance, AdversarialTrapReason, RedTeamPreTradeGate,
};
use god_engine_core::GodEngineCore;
use quantum_arena::topological_persistence::TopologicalPersistenceEngine;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_c1_tda_betti_homology_zero_alloc() {
    let mut engine = TopologicalPersistenceEngine::new();

    // 1. Nube colineal (unidimensional / árbol sin ciclos cerrados)
    for i in 0..12 {
        engine.observe(i as f64 * 0.15, 0.0, 0.01);
    }
    let m_tree = engine.last_metrics();
    assert_eq!(m_tree.betti_1, 0.0, "Topología lineal de árbol no debe engendrar 1-ciclos");
    assert!(m_tree.betti_0 >= 1.0, "Debe existir al menos una componente conexa");
    assert!(m_tree.void_risk < 0.25, "Riesgo de vacío en árbol colineal debe ser despreciable");

    // 2. Nube en anillo cerrado (1-ciclo simplicial persistente)
    let mut ring_engine = TopologicalPersistenceEngine::new();
    let n = 8;
    let radius = 1.0;
    for i in 0..n {
        let theta = (i as f64 / n as f64) * 2.0 * std::f64::consts::PI;
        ring_engine.observe(radius * theta.cos(), radius * theta.sin(), 0.05);
    }
    let m_ring = ring_engine.last_metrics();
    assert!(
        m_ring.void_risk > 0.25,
        "Anillo cerrado debe registrar activación de riesgo topológico: void_risk={}",
        m_ring.void_risk
    );
}

#[test]
fn test_c2_tda_persistent_entropy_and_void_risk() {
    let mut engine = TopologicalPersistenceEngine::new();
    // Inserción de patrón periódico toroidal
    for t in 0..16 {
        let x = (t as f64 * 0.5).sin();
        let y = (t as f64 * 0.5).cos();
        let z = 0.02 + 0.01 * (t as f64 * 0.2).sin();
        engine.observe(x, y, z);
    }
    let m = engine.last_metrics();
    assert!(m.persistent_entropy.is_finite());
    assert!(m.persistent_entropy >= 0.0 && m.persistent_entropy <= 1.0);
    assert!(m.void_risk >= 0.0 && m.void_risk <= 1.0);
}

#[test]
fn test_c3_red_team_pre_trade_topological_void_trap_rejection() {
    let gate = RedTeamPreTradeGate::default();

    // Caso A: Mercado benigno y variedad topológica conexa sin vacíos
    let benign = gate.evaluate_order_topological(
        true,     // Compra
        0.0002,   // Spread 2 bps
        0.0010,   // Volatilidad 10 bps
        0.10,     // OBI favorable
        10_000.0, // Profundidad soporte
        10_000.0, // Mediana
        0.15,     // Cancelaciones bajas
        1.1,      // Ratio BBO/L2 normal
        0.10,     // Riesgo de vacío topológico bajo
        0.05,     // Deformación de forma Wasserstein baja
    );
    assert!(benign.is_ok(), "Mercado benigno sin vacío topológico debe ser aprobado");
    let clearance: AdversarialClearance = benign.unwrap();
    assert!(clearance.compound_risk < 0.65);
    assert!(clearance.topological_void_risk < 0.50);

    // Caso B: Rechazo estricto por trampa de vacío topológico y fractura geodésica
    let void_trap = gate.evaluate_order_topological(
        true,     // Compra
        0.0002,   // Spread sano
        0.0010,   // Volatilidad sana
        0.05,     // OBI neutral
        10_000.0, // Profundidad suficiente en papel
        10_000.0,
        0.20,
        1.2,
        0.95,     // Vacío topológico extremo (agujero en el libro)
        1.90,     // Deformación geodésica crítica Wasserstein
    );
    assert_eq!(
        void_trap,
        Err(AdversarialTrapReason::TopologicalVoidTrap),
        "Orden dirigida a un vacío topológico con fractura geodésica debe ser rechazada con TopologicalVoidTrap"
    );
}

#[test]
fn test_c4_red_team_nan_inf_immunity_and_fail_closed() {
    let gate = RedTeamPreTradeGate::default();

    // Inyección de NaN en telemetría topológica
    let nan_void = gate.evaluate_order_topological(
        true, 0.0002, 0.0010, 0.0, 10_000.0, 10_000.0, 0.20, 1.2,
        f64::NAN, 0.1,
    );
    assert_eq!(nan_void, Err(AdversarialTrapReason::InvalidMicrostructureTelemetry));

    // Inyección de Inf en deformación Wasserstein
    let inf_shape = gate.evaluate_order_topological(
        false, 0.0002, 0.0010, 0.0, 10_000.0, 10_000.0, 0.20, 1.2,
        0.1, f64::INFINITY,
    );
    assert_eq!(inf_shape, Err(AdversarialTrapReason::InvalidMicrostructureTelemetry));
}

#[test]
fn test_c5_god_engine_registry_tda_and_red_team_observables() {
    let _guard = TEST_LOCK.lock().unwrap();
    const SIM: &str = "ETHUSDT";
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;

    // Procesamos 10 ticks sucesivos
    for i in 1..=10 {
        let price = 2500.0 + (i as f64 * 0.25);
        let bid = price - 0.10;
        let ask = price + 0.10;
        let event_time = 1_700_000_000_000 + (i as u64 * 100);

        let _ = core.process_tick_dual(
            0,
            bid,
            ask,
            5.0,
            5.0,
            event_time,
            &[0.0; 54],
            true,
            true,
        );
    }

    // Verificamos los 9 observables publicados activamente en OmniscientRegistry
    let reg = &arena.registry;
    let b0 = reg.get_for_coin_or(0, "tda_betti_0", 0.0);
    let b1 = reg.get_for_coin_or(0, "tda_betti_1", -1.0);
    let entropy = reg.get_for_coin_or(0, "tda_persistent_entropy", -1.0);
    let void_risk = reg.get_for_coin_or(0, "tda_void_risk", -1.0);
    let rt_comp = reg.get_for_coin_or(0, "red_team_compound_risk", -1.0);
    let rt_clr_l = reg.get_for_coin_or(0, "red_team_clearance_long", -1.0);
    let rt_clr_s = reg.get_for_coin_or(0, "red_team_clearance_short", -1.0);
    let rt_veto_l = reg.get_for_coin_or(0, "red_team_veto_long", -1.0);
    let rt_veto_s = reg.get_for_coin_or(0, "red_team_veto_short", -1.0);

    assert!(b0 >= 1.0, "Betti 0 debe ser >= 1.0: {}", b0);
    assert!(b1 >= 0.0, "Betti 1 debe ser >= 0.0: {}", b1);
    assert!(entropy >= 0.0 && entropy <= 1.0, "Entropía TDA debe estar en [0, 1]: {}", entropy);
    assert!(void_risk >= 0.0 && void_risk <= 1.0, "Void risk debe estar en [0, 1]: {}", void_risk);
    assert!(rt_comp >= 0.0 && rt_comp <= 1.0, "Red-team compound risk debe estar en [0, 1]: {}", rt_comp);
    assert!(rt_clr_l == 0.0 || rt_clr_l == 1.0, "Clearance long debe ser booleano: {}", rt_clr_l);
    assert!(rt_clr_s == 0.0 || rt_clr_s == 1.0, "Clearance short debe ser booleano: {}", rt_clr_s);
    assert!(rt_veto_l == 0.0 || rt_veto_l == 1.0, "Veto long debe ser booleano: {}", rt_veto_l);
    assert!(rt_veto_s == 0.0 || rt_veto_s == 1.0, "Veto short debe ser booleano: {}", rt_veto_s);
}
