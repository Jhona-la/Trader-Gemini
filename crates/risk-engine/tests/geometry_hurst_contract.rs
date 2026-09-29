//! CL-26 — la ley de escala del TP/SL usa el Hurst muestreado por reloj.
//! Estado sintético local; sin exchange ni genoma activo.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn arena(hurst_por_eventos: f64) -> Arc<GlobalArena> {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
        update_registry(vec![SymbolSpec {
            symbol: "BTCUSDT".into(),
            step_size: 0.001,
            tick_size: 0.01,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            is_shadow: false,
        }])
    });
    let a = GlobalArena::build_in_own_stack(1_000.0);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(0.2, Relaxed);
    // DFA al cierre de la vela de 1 minuto: difusión browniana.
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    // Proxy multifractal por eventos (S-7): con ticks repetidos cae a ~0,2.
    a.coins[0].hurst_scale_matched.store(hurst_por_eventos, Relaxed);
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    a
}

#[test]
fn cl26_el_stop_a_una_hora_no_depende_del_proxy_por_eventos() {
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 3_600_000,
        ..Default::default()
    };
    let referencia = RiskEngine::new(1_000.0).evaluate_quantum_order(0, &intent, &arena(0.5));
    assert_eq!(
        referencia.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    let con_proxy = RiskEngine::new(1_000.0).evaluate_quantum_order(0, &intent, &arena(0.22));
    assert_eq!(
        con_proxy.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    assert_eq!(
        con_proxy.sl_target, referencia.sl_target,
        "el stop a τ = 1 h lo fija la difusión medida por reloj"
    );
    assert_eq!(con_proxy.tp_target, referencia.tp_target);
}
