//! CL-9 — el veto de drawdown juzga la caída de la CUENTA con la tasa de
//! pérdida de la cuenta: su decisión no depende de qué moneda es la candidata.
use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJECT_COUNTERS, REJ_DRAWDOWN};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn vetada_por_drawdown(a: &GlobalArena, coin_id: usize, intent: &SignalIntent) -> bool {
    let antes = REJECT_COUNTERS[REJ_DRAWDOWN].load(Relaxed);
    // Pico 100, capital 90: caída del 10 % de la cuenta.
    let _ = RiskEngine::new(100.0).evaluate_quantum_order(coin_id, intent, a);
    REJECT_COUNTERS[REJ_DRAWDOWN].load(Relaxed) > antes
}

#[test]
fn cl9_la_misma_caida_de_la_cuenta_se_juzga_igual_para_toda_moneda() {
    let a = GlobalArena::build_in_own_stack(90.0);
    a.unified_capital.store(90.0, Relaxed);
    a.config.global_max_drawdown.store(0.95, Relaxed);
    a.riesgo_por_operacion.store(0.005, Relaxed);
    // Moneda 0: 50 cierres con 70 % de acierto. Moneda 1: 2 cierres, 20 %.
    a.coins[0].metrics.trade_count.store(50, Relaxed);
    a.coins[0].metrics.win_rate.store(0.7, Relaxed);
    a.coins[1].metrics.trade_count.store(2, Relaxed);
    a.coins[1].metrics.win_rate.store(0.2, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..Default::default()
    };
    let q = risk_engine::drawdown::q_perdida_cartera([(0.7, 50.0), (0.2, 2.0)]);
    let umbral = risk_engine::drawdown::drawdown_maximo(0.005, q, 0.95);
    assert!(umbral < 0.10, "el fixture debe caer por encima del umbral: {umbral}");
    assert!(vetada_por_drawdown(&a, 0, &intent), "moneda que gana a menudo");
    assert!(
        vetada_por_drawdown(&a, 1, &intent),
        "una moneda que pierde a menudo no afloja el freno de la cuenta"
    );
}
