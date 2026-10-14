//! Contrato formal: Desacoplamiento Dinámico del Techo de Fricción Micro (Ola Ω104).
//!
//! F3.1-05: El techo de comisión sobre margen en cuentas micro o rescate de nocional mínimo
//! debe ser configurable y desacoplado a través del registro omnisciente (`"micro_max_fee_pct"`
//! global y escopado por activo), eliminando la constante mágica 0.035 fija y preservando
//! exacta paridad canónica cuando no se define un override específico.

use quantum_arena::GlobalArena;
use risk_engine::{
    resolve_micro_max_fee_pct, ultimo_rechazo, RiskEngine, DEFAULT_MICRO_MAX_FEE_PCT,
    REJ_FEE_IMPACT,
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
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

#[test]
fn test_resolve_micro_max_fee_pct_defaults_and_scoping() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // 1. Sin claves configuradas: rige exactamente el default canónico 0.035
    assert_eq!(resolve_micro_max_fee_pct(&arena, 0), DEFAULT_MICRO_MAX_FEE_PCT);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 1), 0.035);

    // 2. Override global en el registro
    arena.registry.set("micro_max_fee_pct", 0.045);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 0), 0.045);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 1), 0.045);

    // 3. Override escopado por activo (prioridad sobre el global)
    arena.registry.set_for_coin(1, "micro_max_fee_pct", 0.060);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 0), 0.045);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 1), 0.060);

    // 4. Inmunidad a corrupción numérica (NaN, negativos, cero regresan al piso canónico)
    arena.registry.set_for_coin(2, "micro_max_fee_pct", f64::NAN);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 2), 0.045); // cae al global si es finito

    arena.registry.set("micro_max_fee_pct", f64::NAN);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 2), DEFAULT_MICRO_MAX_FEE_PCT);

    arena.registry.set("micro_max_fee_pct", -0.05);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 0), DEFAULT_MICRO_MAX_FEE_PCT);

    arena.registry.set("micro_max_fee_pct", 0.0);
    assert_eq!(resolve_micro_max_fee_pct(&arena, 0), DEFAULT_MICRO_MAX_FEE_PCT);
}

#[test]
fn test_micro_fee_override_controls_admission_in_risk_engine() {
    let (arena, intent) = fixture();
    // LH-04 (GLM 117): este contrato aísla el TECHO DE FRICCIÓN, no el
    // sizing — se desactiva la evidencia madura del fixture para que la
    // orden siga el camino frío (f=0) y el rechazo dependa SÓLO del
    // mecanismo de techo que Ω104 certificó.
    arena.coins[0].metrics.trade_count.store(0, Relaxed);
    // Configurar fricción que arroja ~0.036 de impacto en margen:
    // Con techo default (0.035), esta fricción es rechazada con REJ_FEE_IMPACT.
    arena.coins[0].current_atr.store(3.0, Relaxed);
    arena.config.live_taker_fee.store(0.0060, Relaxed);

    let mut engine = RiskEngine::new(13.0);
    let out_default = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_default.signal, SignalType::Flat);
    assert_eq!(ultimo_rechazo(), Some(REJ_FEE_IMPACT));

    // Techo elevado: el "~0.036" original era del modelo de fricción
    // PRE-unificación (la Ola 72 lo llevó a tp_sl::roundtrip_friction
    // con ATR/fee vivos — el impacto real con fee 0.006 y ATR 3% es ~0.24
    // tras el rescate de apalancamiento). El override a 0.30 cubre el
    // régimen unificado y sigue probando LO MISMO: que el techo dinámico
    // controla la admisión (default veta; override elevado no veta por
    // fee_impact). Test roto desde la Ola 72 — encontrado al certificar
    // LH-04 (GLM 117).
    arena.registry.set_for_coin(0, "micro_max_fee_pct", 0.30);
    let out_elevated = engine.evaluate_quantum_order(0, &intent, &arena);

    // Al haber expandido la tolerancia micro, la orden ya no es vetada por fee_impact
    if out_elevated.signal == SignalType::Flat {
        assert_ne!(
            ultimo_rechazo(),
            Some(REJ_FEE_IMPACT),
            "Con techo elevado a 0.30, no debe ser vetado por fee_impact"
        );
    }
}
