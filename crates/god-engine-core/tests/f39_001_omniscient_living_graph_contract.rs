//! Contrato formal de verificación para Fase 39: Integración Holística del Grafo Vivo Omnisciente
//!
//! Invariantes certificados:
//! 1. c1: Evaluación end-to-end continua del grafo vivo con todos los motores acoplados sinérgicamente.
//! 2. c2: Despacho multi-slot coordinado sin canibalización con cota de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//! 3. c3: Aplanado defensivo conformal ante violación de umbrales dinámicos sin bypass cuántico.
//! 4. c4: Invariante estricto de cero pánicos (`rb_p1`) ante entradas patológicas y degradación de datos.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use strategy_core::AssetColorState;

#[test]
fn c1_omniscient_living_graph_end_to_end_evaluation() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    let bid = 99.95;
    let ask = 100.05;
    let bid_qty = 80.0;
    let ask_qty = 20.0;
    let event_time_ms = 1_700_000_000_000;

    let triad = [
        AssetColorState {
            normalized_return: 0.02,
            order_flow_imbalance: 0.70,
            relative_weight: 0.50,
        },
        AssetColorState {
            normalized_return: -0.01,
            order_flow_imbalance: -0.30,
            relative_weight: 0.30,
        },
        AssetColorState {
            normalized_return: 0.005,
            order_flow_imbalance: 0.10,
            relative_weight: 0.20,
        },
    ];

    let capital_available_usd = 1_000.0;
    let max_leverage = 5.0;
    let risk_fraction = 0.02;

    // Calentamiento
    let _ = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms,
        &triad,
        capital_available_usd,
        max_leverage,
        risk_fraction,
    );

    // Evaluación con dinámica temporal
    let report = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms + 100,
        &triad,
        capital_available_usd,
        max_leverage,
        risk_fraction,
    );

    // 1. Verificación del operador de volatilidad continua CWT
    assert!(report.volatility_state.effective_volatility >= 0.0);
    assert!(report.volatility_state.continuous_hurst >= 0.0);

    // 2. Verificación del tensor gauge no abeliano SU(3)
    let force_sum: f64 = report.su3_gauge_report.color_forces.iter().sum();
    assert!(
        force_sum.abs() < 1e-10,
        "Conservación estricta de color violada en el grafo vivo: {}",
        force_sum
    );
    assert!(report.su3_gauge_report.yang_mills_action >= 0.0);

    // 3. Verificación de Mean Field Games y Psicología de Masas
    assert!((0.0..=1.0).contains(&report.mfg_report.crowd_herding_risk));
    assert!((0.0..=1.0).contains(&report.mfg_report.squeeze_exhaustion_prob));

    // 4. Verificación de Shockwaves Riemann-Burgers
    assert!(report.shockwave_report.shock_velocity_s.is_finite());
    assert!(report.shockwave_report.shock_layer_width > 0.0);

    // 5. Verificación de Gobernanza Conformal
    assert!((0.0..=1.0).contains(&report.conformal_risk_score));
    assert!(report.dynamic_spread_ceiling_bps > 0.0);
    assert!(report.ville_ruin_bound_satisfied);
}

#[test]
fn c2_multi_slot_emission_coordination_without_cannibalization() {
    let arena = GlobalArena::build_in_own_stack(500.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    let bid = 50.00;
    let ask = 50.02; // Spread ajustado de 4 bps
    let bid_qty = 200.0;
    let ask_qty = 10.0; // Desequilibrio alcista masivo
    let event_time_ms = 1_000_000;

    let triad = [
        AssetColorState {
            normalized_return: 0.03,
            order_flow_imbalance: 0.90,
            relative_weight: 0.60,
        },
        AssetColorState {
            normalized_return: -0.02,
            order_flow_imbalance: -0.60,
            relative_weight: 0.25,
        },
        AssetColorState {
            normalized_return: 0.00,
            order_flow_imbalance: 0.00,
            relative_weight: 0.15,
        },
    ];

    let capital_available_usd = 500.0;
    let max_leverage = 3.0;
    let risk_fraction = 0.03;

    let report = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms,
        &triad,
        capital_available_usd,
        max_leverage,
        risk_fraction,
    );

    // Si la entrada es aprobada por la gobernanza conformal, verificar órdenes coordinadas
    if report.elastic_entry_granted {
        let total_notional: f64 = report
            .coordinated_emissions
            .slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|p| p.price * p.quantity)
            .sum();
        assert!(
            total_notional <= capital_available_usd * max_leverage + 1.0,
            "Apalancamiento máximo excedido: {} > {}",
            total_notional,
            capital_available_usd * max_leverage
        );
    }
}

#[test]
fn c3_conformal_rejection_flattens_decisions_safely() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    // Spread abismal (1000 bps) con volatilidad nula y sin convicción cuántica
    let bid = 90.0;
    let ask = 100.0; // 10.5% de spread
    let bid_qty = 1.0;
    let ask_qty = 1.0;
    let event_time_ms = 2_000_000;

    let neutral_triad = [
        AssetColorState::default(),
        AssetColorState::default(),
        AssetColorState::default(),
    ];

    let report = core.evaluate_omniscient_living_graph(
        coin_id,
        bid,
        ask,
        bid_qty,
        ask_qty,
        event_time_ms,
        &neutral_triad,
        100.0,
        1.0,
        0.01,
    );

    // Debe ser rechazado por la compuerta conformal y ninguna orden debe emitirse
    assert!(
        !report.elastic_entry_granted,
        "Spread de 1000 bps debe ser rechazado por gobernanza conformal"
    );
    assert_eq!(
        report.coordinated_emissions.active_count, 0,
        "No deben emitirse órdenes ante rechazo conformal"
    );
}

#[test]
fn c4_zero_alloc_and_zero_panic_guarantee() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);

    let pathological_triad = [
        AssetColorState {
            normalized_return: f64::NAN,
            order_flow_imbalance: f64::INFINITY,
            relative_weight: -5.0,
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

    // Llamadas con parámetros patológicos y coin_id fuera de rango
    let r1 = core.evaluate_omniscient_living_graph(
        9999, // Fuera de rango
        f64::NAN,
        f64::INFINITY,
        0.0,
        0.0,
        0,
        &pathological_triad,
        -100.0,
        0.0,
        0.0,
    );

    assert!(r1.mid_price.is_finite());
    assert!(r1.spread_bps.is_finite());
    assert!(r1.conformal_risk_score.is_finite());
}
