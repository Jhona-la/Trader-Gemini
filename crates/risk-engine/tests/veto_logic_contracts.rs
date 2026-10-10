//! CONTRATOS FORMALES DE VETOS DE LÓGICA (OLA Ω58, FICHA #693)
//!
//! Elimina la deuda contractual de los vetos de lógica en `veto_registry.rs`:
//!  - V-LOGIC-007: `council_confidence_threshold_respects_graceful_cold_modulation`
//!  - V-LOGIC-009: `insufficient_evidence_contract_without_deadlock`

use quantum_arena::GlobalArena;
use risk_engine::{evidence::win_rate_hierarchical_lcb, RiskEngine};
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

    // 2. Confianza robusta (0.85): debe ser admitida sin veto
    intent.confidence = 0.85;
    let out_alta = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_alta.signal, SignalType::Long, "Confianza sólida debe ser admitida: {}", risk_engine::reject_report());
    assert!(out_alta.volume_usd > 0.0);
}

#[test]
pub fn insufficient_evidence_contract_without_deadlock() {
    let (arena, mut intent) = fixture();

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
    let (arena, intent) = fixture();

    // 1. Fricción normal: comisiones estándar viables, orden Long admitida
    let out_viable = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_viable.signal, SignalType::Long, "Fricción viable debe admitirse");
    assert!(out_viable.volume_usd > 0.0);

    // 2. Fricción colosal (50% de comisión por pierna): debe rechazarse limpiamente por fee_impact
    arena.config.live_taker_fee.store(0.50, Relaxed);
    let out_friction = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_friction.signal, SignalType::Flat, "Fricción exorbitante debe ser vetada por fee_impact");
}

#[test]
pub fn orchestrator_contract_respects_directional_pressure_and_blocks_crash() {
    let (arena, mut intent) = fixture();

    // 1. Crash sistémico agudo (p_crash = 0.95): compras (Long) vetadas por orchestrator
    arena.regime_p_crash.store(0.95, Relaxed);
    arena.regime_p_bull.store(0.01, Relaxed);
    arena.regime_p_chaos.store(0.04, Relaxed);
    let out_long = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_long.signal, SignalType::Flat, "Compras en crash sistémico deben vetarse por orchestrator");

    // 2. Operación Short en crash sistémico: admitida simétricamente
    intent.signal = SignalType::Short;
    let out_short = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_short.signal, SignalType::Short, "Shorts en crash deben admitirse");
    assert!(out_short.volume_usd > 0.0);
}

#[test]
pub fn spec_rejection_contract_handles_unregistered_coin() {
    let (arena, intent) = fixture();

    // Símbolo fuera del registro (coin_id = 999): rechazado limpiamente sin pánico
    let out_unreg = RiskEngine::new(13.0).evaluate_quantum_order(999, &intent, &arena);
    assert_eq!(out_unreg.signal, SignalType::Flat, "Moneda sin spec debe rechazarse como Flat");
}

#[test]
pub fn invalid_input_contract_rejects_nan_and_infinities() {
    let (arena, mut intent) = fixture();

    // 1. Confianza NaN
    intent.confidence = f64::NAN;
    let out_nan_conf = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_nan_conf.signal, SignalType::Flat, "Confianza NaN debe rechazarse fail-closed");

    // 2. Precio no finito
    intent.confidence = 0.85;
    arena.coins[0].current_price.store(f64::NAN, Relaxed);
    let out_nan_price = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_nan_price.signal, SignalType::Flat, "Precio NaN debe rechazarse fail-closed");
}

#[test]
pub fn flat_coin_contract_rejects_flat_intent() {
    let (arena, mut intent) = fixture();

    // Intención Flat: sin dirección, debe devolver Flat inmediatamente
    intent.signal = SignalType::Flat;
    let out_flat = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_flat.signal, SignalType::Flat, "Señal Flat debe evaluarse como Flat");
}
