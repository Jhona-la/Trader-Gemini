//! CONTRATOS FORMALES DE VETOS DE LÓGICA (OLA Ω58, FICHA #693)
//!
//! Elimina la deuda contractual de los vetos de lógica en `veto_registry.rs`:
//!  - V-LOGIC-007: `council_confidence_threshold_respects_graceful_cold_modulation`
//!  - V-LOGIC-009: `insufficient_evidence_contract_without_deadlock`

#[path = "support/net_payoff_fixture.rs"]
mod net_payoff_fixture;

use quantum_arena::GlobalArena;
use risk_engine::{
    evidence::win_rate_hierarchical_lcb, ultimo_rechazo, RiskEngine, REJ_CONFIDENCE, REJ_FEE_IMPACT,
    REJ_FLAT_COIN, REJ_INVALID_INPUT, REJ_ORCHESTRATOR, REJ_SPEC, REJ_TP_SL_FLOOR,
};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn fixture() -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    static REGISTRY: std::sync::Once = std::sync::Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
        ]);
    });
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.0, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        net_payoff: Some(net_payoff_fixture::positive(0, SignalType::Long)),
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

#[test]
pub fn council_confidence_threshold_respects_graceful_cold_modulation() {
    let (arena, mut intent) = fixture();

    // 1. Confianza extremadamente baja (0.01): debe ser vetada (Flat)
    intent.confidence = 0.01;
    let out_baja = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_baja.signal, SignalType::Flat, "Confianza ínfima debe ser vetada");
    // QS-REV: la RAZÓN, no sólo el Flat.
    assert_eq!(ultimo_rechazo(), Some(REJ_CONFIDENCE));

    // 2. Confianza robusta (0.85): debe ser admitida sin veto
    intent.confidence = 0.85;
    let out_alta = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_alta.signal, SignalType::Long, "Confianza sólida debe ser admitida: {}", risk_engine::reject_report());
    assert!(out_alta.volume_usd > 0.0);
}

#[test]
pub fn insufficient_evidence_contract_without_deadlock() {
    let (arena, mut intent) = fixture();

    intent.net_payoff = None;

    // 1. Arranque frío total (trade_count == 0): la sonda exploratoria D-750 es admitida sin deadlock
    arena.coins[0].metrics.trade_count.store(0, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.0, Relaxed);
    intent.confidence = 0.85;
    intent.tp_price_target = 0.0;
    intent.sl_price_target = 0.0;

    let out_cold = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_cold.signal, SignalType::Long, "Arranque frío no debe causar deadlock: {}", risk_engine::reject_report());
    assert!(out_cold.volume_usd > 0.0);

    // 2. Fase de sonda tras 1 pérdida (trade_count = 1, win_rate = 0.0):
    // La contracción jerárquica bayesiana contrae hacia 0.55 y no colapsa a 0.0
    arena.coins[0].metrics.trade_count.store(1, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.0, Relaxed);

    let lcb_1_loss = win_rate_hierarchical_lcb(0.0, 1.0, 0.55, 10.0);
    assert!(lcb_1_loss > 0.20, "LCB jerárquico no debe colapsar a 0: {lcb_1_loss}");

    let out_probe = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_probe.signal, SignalType::Long, "Fase de sonda con prior no debe causar veto absorbente: {}", risk_engine::reject_report());
    assert!(out_probe.volume_usd > 0.0);
}

#[test]
pub fn fee_impact_contract_rejects_excessive_friction_and_admits_viable_order() {
    // QS-REV2: la versión anterior usaba un fee de 0,50 por pierna; con esa
    // fricción el stop viable es ≈ 154 %, y la orden salía antes por el
    // suelo de TP/SL (REJ_TP_SL_FLOOR): el contrato seguía verde con el
    // veto de fee_impact borrado. Ahora se usa una geometría que pasa el
    // suelo y el EV (τ = 1 h, stop ancho) y una fricción que, multiplicada
    // por el apalancamiento final, supera el tope (0,035 en micro). Valores
    // medidos con una sonda sobre el risk-engine actual, no supuestos.
    let (arena, mut intent) = fixture();
    intent.expected_duration_ms = 3_600_000;

    // 1. Fricción estándar: admitida.
    let out_viable = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_viable.signal, SignalType::Long, "Fricción viable debe admitirse: {}", risk_engine::reject_report());
    assert!(out_viable.volume_usd > 0.0);
    assert_eq!(ultimo_rechazo(), None);

    // 2. Fee de 0,5 % por pierna: pasa el suelo y el EV, el coste por unidad
    // de margen no.
    arena.config.live_taker_fee.store(0.005, Relaxed);
    let out_friction = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_friction.signal, SignalType::Flat);
    assert_eq!(ultimo_rechazo(), Some(REJ_FEE_IMPACT));

    // 3. Fricción absurda (50 % por pierna): la detiene ANTES el suelo de
    // TP/SL. Es el caso que el contrato anterior confundía con fee_impact.
    arena.config.live_taker_fee.store(0.50, Relaxed);
    let out_absurda = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_absurda.signal, SignalType::Flat);
    assert_eq!(ultimo_rechazo(), Some(REJ_TP_SL_FLOOR));

    // 4. Caso de AGY (#716, Ω78): con el τ por defecto del fixture, ATR 3,0
    // (el stop difusivo pasa el suelo) y 0,6 % por pierna, el impacto en
    // margen (≈ 0,036) supera el tope de 0,035. Fixture propio: el de arriba
    // fija τ = 1 h.
    let (arena_b, intent_b) = fixture();
    arena_b.coins[0].current_atr.store(3.0, Relaxed);
    arena_b.config.live_taker_fee.store(0.0060, Relaxed);
    let out_b = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent_b, &arena_b);
    assert_eq!(out_b.signal, SignalType::Flat, "Fricción excesiva debe ser vetada por fee_impact");
    assert_eq!(ultimo_rechazo(), Some(REJ_FEE_IMPACT));
}

#[test]
pub fn orchestrator_contract_respects_directional_pressure_and_blocks_crash() {
    let (arena, mut intent) = fixture();

    // 1. Crash sistémico agudo (p_crash = 0,95): largos vetados POR EL
    // ORQUESTADOR (la razón, no sólo el Flat).
    arena.regime_p_crash.store(0.95, Relaxed);
    arena.regime_p_bull.store(0.01, Relaxed);
    arena.regime_p_chaos.store(0.04, Relaxed);
    let out_long = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_long.signal, SignalType::Flat, "Compras en crash sistémico deben vetarse por orchestrator");
    assert_eq!(ultimo_rechazo(), Some(REJ_ORCHESTRATOR));

    // 2. Cortos en crash: admitidos.
    intent.signal = SignalType::Short;
    intent.net_payoff = Some(net_payoff_fixture::positive(0, intent.signal));
    let out_short = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_short.signal, SignalType::Short, "Shorts en crash deben admitirse");
    assert!(out_short.volume_usd > 0.0);
    assert_eq!(ultimo_rechazo(), None);

    // 3. QS-REV4 — el ESPEJO: euforia sistémica (p_bull = 0,95). Antes los
    // cortos pasaban aquí (D-403) mientras los largos se vetaban en el
    // crash; ahora cada lado se veta en el régimen que le es adverso.
    arena.regime_p_crash.store(0.01, Relaxed);
    arena.regime_p_bull.store(0.95, Relaxed);
    let out_short_bull = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_short_bull.signal, SignalType::Flat, "Ventas en euforia sistémica deben vetarse");
    assert_eq!(ultimo_rechazo(), Some(REJ_ORCHESTRATOR));
    intent.signal = SignalType::Long;
    intent.net_payoff = Some(net_payoff_fixture::positive(0, intent.signal));
    let out_long_bull = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_long_bull.signal, SignalType::Long, "Compras en euforia deben admitirse");
    assert_eq!(ultimo_rechazo(), None);
}

#[test]
pub fn spec_rejection_contract_handles_unregistered_coin() {
    // QS-REV1: la versión anterior usaba coin_id = 999, fuera del arena (30
    // ranuras): salía por REJ_FLAT_COIN y el veto de spec no se ejecutaba.
    // Ahora: una moneda DENTRO del arena, con precios válidos y sin spec
    // registrado (el fixture sólo registra la moneda 0).
    let (arena, mut intent) = fixture();
    intent.net_payoff = Some(net_payoff_fixture::positive(1, intent.signal));
    arena.coins[1].current_price.store(100.0, Relaxed);
    arena.coins[1].current_atr.store(1.0, Relaxed);
    arena.coins[1].hurst_exponent.store(0.5, Relaxed);
    arena.coins[1].metrics.trade_count.store(100, Relaxed);
    arena.coins[1].metrics.profit_factor.store(2.0, Relaxed);
    arena.coins[1].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[1].metrics.kelly_fraction.store(0.15, Relaxed);
    assert!(quantum_arena::symbol_registry::try_spec(1).is_none());
    let out_unreg = RiskEngine::new(13.0).evaluate_quantum_order(1, &intent, &arena);
    assert_eq!(out_unreg.signal, SignalType::Flat, "Moneda sin spec debe rechazarse como Flat");
    assert_eq!(ultimo_rechazo(), Some(REJ_SPEC));

    // Y fuera del arena la razón es otra: la de V-TECH-001.
    let out_fuera = RiskEngine::new(13.0).evaluate_quantum_order(999, &intent, &arena);
    assert_eq!(out_fuera.signal, SignalType::Flat);
    assert_eq!(ultimo_rechazo(), Some(REJ_FLAT_COIN));
}

#[test]
pub fn invalid_input_contract_rejects_nan_and_infinities() {
    let (arena, mut intent) = fixture();

    // 1. Confianza NaN
    intent.confidence = f64::NAN;
    let out_nan_conf = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_nan_conf.signal, SignalType::Flat, "Confianza NaN debe rechazarse fail-closed");
    assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));

    // 2. Precio no finito
    intent.confidence = 0.85;
    arena.coins[0].current_price.store(f64::NAN, Relaxed);
    let out_nan_price = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_nan_price.signal, SignalType::Flat, "Precio NaN debe rechazarse fail-closed");
    assert_eq!(ultimo_rechazo(), Some(REJ_INVALID_INPUT));
}

#[test]
pub fn flat_coin_contract_rejects_flat_intent() {
    let (arena, mut intent) = fixture();

    // Intención Flat: sin dirección, debe devolver Flat inmediatamente
    intent.signal = SignalType::Flat;
    let out_flat = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_flat.signal, SignalType::Flat, "Señal Flat debe evaluarse como Flat");
    assert_eq!(ultimo_rechazo(), Some(REJ_FLAT_COIN));
}

#[test]
pub fn lh02_valle_capital_friccion_no_veta_orden_viable() {
    let (arena, intent) = fixture();
    // LH-02: En el valle de capital ($35–$65 USD), allocated_capital y unified_capital
    // están en ~$50 USD, donde micro_w_alloc cae a 0.0.
    arena.config.base_capital.store(50.0, Relaxed);
    arena.unified_capital.store(50.0, Relaxed);
    // Taker fee normal de Binance futures (0.05% por pierna = 0.10% roundtrip):
    arena.config.live_taker_fee.store(0.0005, Relaxed);
    let mut engine = RiskEngine::new(50.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_ne!(
        ultimo_rechazo(),
        Some(REJ_FEE_IMPACT),
        "LH-02: La orden en el valle de capital ($50 USD) no debe ser vetada por fee_impact: {}",
        risk_engine::reject_report()
    );
    assert_eq!(out.signal, SignalType::Long, "Orden viable debe admitirse en Long");
    assert!(out.volume_usd > 0.0);
}

