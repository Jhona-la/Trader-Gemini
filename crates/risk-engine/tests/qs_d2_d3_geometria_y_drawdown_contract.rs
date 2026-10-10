//! QS-D2 / QS-D3 — decisiones del plan de sincronización §30.5 (ADR-0016).
//!
//! D2: la geometría de la orden (stop y TP) es la de su horizonte τ y NO
//! depende del capital. Antes, en régimen micro, (a) un tope interpolaba el
//! stop hacia 55 pb sin tocar τ y (b) un atajo admitía horizontes cuyo stop
//! difusivo no paga la fricción si el suelo cabía en 55 pb. Acortar el stop
//! sin acortar τ reduce lo que la deriva puede cobrar (E[ganancia] = μ·E[T],
//! E[T] ≈ sl·tp/σ²) y deja la comisión igual; lo que se pierde en dólares lo
//! acota el dimensionado, no la geometría.
//!
//! D3: el umbral del veto de drawdown nunca supera d* = 1 − α^{c/(2−c)}, la
//! caída que un apostador a fracción c de Kelly con la ventaja estimada
//! alcanza con probabilidad α (Thorp/Breiman: P(DD ≥ x) = (1 − x)^{2/c − 1}).
//! Con c = ½ y α = 0,05, d* ≈ 0,632. El literal 0,85 que fijaba el régimen
//! micro no salía de ninguna cuenta.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{RiskEngine, REJECT_COUNTERS_DIR, REJ_DRAWDOWN, REJ_TP_SL_FLOOR};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn fixture(capital: f64, atr: f64) -> (Arc<GlobalArena>, SignalIntent) {
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
    let a = GlobalArena::build_in_own_stack(capital);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(atr, Relaxed);
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    (
        a,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            win_probability: 0.9,
            expected_duration_ms: 60_000,
            ..Default::default()
        },
    )
}

fn stop_de(out: &risk_engine::ValidatedOrder) -> f64 {
    (100.0 - out.sl_target).abs() / 100.0
}

fn tp_de(out: &risk_engine::ValidatedOrder) -> f64 {
    (out.tp_target - 100.0).abs() / 100.0
}

/// D2 — misma moneda, mismo mercado, misma intención: el stop y el TP de la
/// orden son los mismos con 13 USD que con 10 000 USD. Con el ATR del
/// fixture el stop difusivo pasa de 55 pb, así que antes de QS-D2 la cuenta
/// micro salía con el stop recortado a 55 pb.
#[test]
fn qs_d2_la_geometria_de_la_orden_no_depende_del_capital() {
    let (micro, intent) = fixture(13.0, 1.0);
    let (grande, _) = fixture(10_000.0, 1.0);
    let a = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &micro);
    assert_eq!(a.signal, SignalType::Long, "micro: {}", risk_engine::reject_report());
    let b = RiskEngine::new(10_000.0).evaluate_quantum_order(0, &intent, &grande);
    assert_eq!(b.signal, SignalType::Long, "grande: {}", risk_engine::reject_report());
    let (sa, sb) = (stop_de(&a), stop_de(&b));
    assert!(
        sb > 0.0055,
        "el fixture debe pedir un stop difusivo > 55 pb para probar algo: {sb}"
    );
    assert!(
        (sa - sb).abs() <= 1e-9 * sb,
        "stop micro {sa} ≠ stop con capital grande {sb}"
    );
    let (ta, tb) = (tp_de(&a), tp_de(&b));
    assert!(
        (ta - tb).abs() <= 1e-9 * tb,
        "TP micro {ta} ≠ TP con capital grande {tb}"
    );
}

/// D2 — un horizonte cuyo stop difusivo queda bajo el suelo de viabilidad se
/// rechaza con `REJ_TP_SL_FLOOR` también en la cuenta micro (antes la micro
/// lo admitía con el stop elevado al suelo y la τ intacta).
#[test]
fn qs_d2_el_suelo_de_viabilidad_rechaza_en_todo_capital() {
    for capital in [13.0, 10_000.0] {
        // ATR de 0,001 % del precio: el stop difusivo a 60 s queda muy por
        // debajo del suelo que exige la comisión (≈ 20 pb).
        let (a, intent) = fixture(capital, 0.001);
        let antes = REJECT_COUNTERS_DIR[0][REJ_TP_SL_FLOOR].load(Relaxed);
        let out = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &a);
        assert_eq!(out.signal, SignalType::Flat, "capital {capital}: {out:?}");
        assert!(
            REJECT_COUNTERS_DIR[0][REJ_TP_SL_FLOOR].load(Relaxed) > antes,
            "capital {capital}: el rechazo no fue por el suelo de viabilidad\n{}",
            risk_engine::reject_report()
        );
    }
}

/// D3 — con la cota medida laxa (dd máx. 0,99), una caída del 70 % se veta
/// también en la cuenta micro: el umbral efectivo no supera d* ≈ 0,632.
/// Antes la micro interpolaba hacia 0,85 y la dejaba pasar.
#[test]
fn qs_d3_el_veto_de_drawdown_no_pasa_de_la_caida_de_falsacion() {
    let d = risk_engine::drawdown::drawdown_de_falsacion(
        risk_engine::drawdown::FRACCION_KELLY_MAXIMA,
        risk_engine::drawdown::ALFA_FALSACION,
    );
    assert!((d - (1.0 - 0.05f64.powf(1.0 / 3.0))).abs() < 1e-12, "d* = {d}");
    for capital in [13.0, 10_000.0] {
        let (a, intent) = fixture(capital, 1.0);
        a.config.global_max_drawdown.store(0.99, Relaxed);
        let base = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &a);
        assert_eq!(
            base.signal,
            SignalType::Long,
            "capital {capital} sin caída: {}",
            risk_engine::reject_report()
        );
        // La cuenta vale el 30 % del pico: caída del 70 % > d*.
        a.unified_capital.store(capital * 0.30, Relaxed);
        let antes = REJECT_COUNTERS_DIR[0][REJ_DRAWDOWN].load(Relaxed);
        let out = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &a);
        assert_eq!(
            out.signal,
            SignalType::Flat,
            "capital {capital}: una caída del 70 % debe vetarse ({out:?})"
        );
        assert!(
            REJECT_COUNTERS_DIR[0][REJ_DRAWDOWN].load(Relaxed) > antes,
            "capital {capital}: el rechazo no fue el veto de drawdown\n{}",
            risk_engine::reject_report()
        );
    }
}

/// QS-R4b — la evaluación deja en el hilo la razón de su rechazo, y la borra
/// cuando admite: el libro contrafactual sabe qué veto bloqueó cada intención.
#[test]
fn qs_r4b_el_rechazo_deja_su_razon_en_el_hilo() {
    let (a, intent) = fixture(13.0, 0.001);
    let out = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &a);
    assert_eq!(out.signal, SignalType::Flat);
    assert_eq!(risk_engine::ultimo_rechazo(), Some(REJ_TP_SL_FLOOR));
    let (b, intent) = fixture(13.0, 1.0);
    let ok = RiskEngine::new(13.0).evaluate_quantum_order(0, &intent, &b);
    assert_eq!(ok.signal, SignalType::Long, "{}", risk_engine::reject_report());
    assert_eq!(risk_engine::ultimo_rechazo(), None);
}
