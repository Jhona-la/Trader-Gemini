//! CL-9 — el veto de drawdown juzga la caída de la CUENTA: su decisión no
//! depende de qué moneda es la candidata. Desde CL-106 es el freno duro
//! (`drawdown::freno_duro`), que sólo lee lo medido de la cuenta (geometría
//! de las órdenes y, desde CL-130, la caída de los cierres en stops de sus
//! posiciones), ni siquiera la tasa de pérdida.
use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJECT_COUNTERS, REJ_DRAWDOWN};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn vetada_por_drawdown(a: &GlobalArena, coin_id: usize, intent: &SignalIntent) -> bool {
    let antes = REJECT_COUNTERS[REJ_DRAWDOWN].load(Relaxed);
    // Pico 100: la caída es la de los 50 SL de la cuenta.
    let _ = RiskEngine::new(100.0).evaluate_quantum_order(coin_id, intent, a);
    REJECT_COUNTERS[REJ_DRAWDOWN].load(Relaxed) > antes
}

#[test]
fn cl9_la_misma_caida_de_la_cuenta_se_juzga_igual_para_toda_moneda() {
    let a = GlobalArena::build_in_own_stack(70.0);
    a.unified_capital.store(70.0, Relaxed);
    a.config.global_max_drawdown.store(0.95, Relaxed);
    // Geometría de una orden con stop del 0,5 %, RR 2,25 y 14,5 pb de fricción.
    let (w, l) = ((2.25 * 0.005 - 0.00145) / 0.005, (0.005 + 0.00145) / 0.005);
    a.ganancia_en_stops.store(w, Relaxed);
    a.perdida_en_stops.store(l, Relaxed);
    // 50 SL de la cuenta (≈ 65 stops de sus posiciones).
    for _ in 0..50 {
        risk_engine::drawdown::registrar_resultado_en_stops(&a, -l);
    }
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
    let k = risk_engine::drawdown::caida_alfa_freno_duro(w, l).unwrap();
    assert!(50.0 * l > k, "el fixture debe caer por encima del umbral: K = {k}");
    assert!(vetada_por_drawdown(&a, 0, &intent), "moneda que gana a menudo");
    assert!(
        vetada_por_drawdown(&a, 1, &intent),
        "una moneda que pierde a menudo no afloja el freno de la cuenta"
    );
}
