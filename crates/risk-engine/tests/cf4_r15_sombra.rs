//! C-F4 · R-15 en sombra (decisión del dueño 2026-10-12; Línea C, CL-100).
//!
//! Teoría. R-15 (R7-R2-A-1, Ola Ω73) vetaba toda orden con P(tocar el SL
//! antes que el TP) > 0,88, con la probabilidad de un movimiento browniano
//! con deriva. Sin deriva, P = TP/(TP + SL) = RR/(1 + RR), así que toda
//! geometría con TP/SL > 7,33 se vetaba por su forma, no por la marea. La
//! deriva sale de la coherencia espectral por el ATR de 1 min (LC-F4-04), un
//! modelo sin validar. El dueño decidió ponerlo en sombra: ya no veta y
//! publica por moneda `p_hit_sl_first` y `r15_vetaria` (1 si habría vetado
//! con el umbral de antes).
//!
//! Fixture. Arena de 1 000 USD, un símbolo de nocional mínimo 5 USD a precio
//! 100 con ATR 1, coherencia espectral 0 (sin deriva), largo con objetivos
//! explícitos.

use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const PRECIO: f64 = 100.0;

fn arena() -> Arc<GlobalArena> {
    static REGISTRO: Once = Once::new();
    REGISTRO.call_once(|| {
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
        ])
    });
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let c = &arena.coins[0];
    c.current_price.store(PRECIO, Relaxed);
    c.current_atr.store(1.0, Relaxed);
    c.hurst_exponent.store(0.5, Relaxed);
    c.spectral_coherence.store(0.0, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.5, Relaxed);
    arena.riesgo_por_operacion.store(1e-4, Relaxed);
    arena
}

/// Largo con TP y SL explícitos (fracciones del precio).
fn larga(tp: f64, sl: f64) -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        tp_price_target: PRECIO * (1.0 + tp),
        sl_price_target: PRECIO * (1.0 - sl),
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

/// (motivo del rechazo o «admitida», p_hit_sl_first, r15_vetaria)
fn evaluar(tp: f64, sl: f64) -> (&'static str, Option<f64>, Option<f64>) {
    let a = arena();
    let orden = RiskEngine::new(1_000.0).evaluate_quantum_order(0, &larga(tp, sl), &a);
    let motivo = if orden.signal == SignalType::Flat {
        risk_engine::ultimo_rechazo().map_or("sin motivo", |i| REJECT_NAMES[i])
    } else {
        "admitida"
    };
    (
        motivo,
        a.registry.get_for_coin_strict(0, "p_hit_sl_first"),
        a.registry.get_for_coin_strict(0, "r15_vetaria"),
    )
}

/// Esperado: una orden con TP/SL = 8 y sin deriva (P = 8/9 ≈ 0,889 > 0,88)
/// ya no se rechaza por geometría, y la telemetría dice que el veto de antes
/// habría disparado. Antes se rechazaba con `geometria_invalida`.
#[test]
fn cl100_r15_ya_no_veta_y_publica_que_habria_vetado() {
    let (motivo, p, vetaria) = evaluar(0.08, 0.01);
    println!("observado: TP/SL = 8 sin deriva ⇒ {motivo}, p = {p:?}, r15_vetaria = {vetaria:?}");
    assert_eq!(motivo, "admitida", "R-15 en sombra no debe rechazar");
    let p = p.expect("publica p_hit_sl_first");
    assert!((p - 8.0 / 9.0).abs() < 1e-9, "sin deriva P = TP/(TP+SL): {p}");
    assert_eq!(vetaria, Some(1.0), "el veto de antes habría disparado");
}

/// Falso positivo: con TP/SL = 2,25 (P = 0,692) la telemetría dice que no
/// habría vetado.
#[test]
fn cl100_r15_geometria_normal_no_habria_vetado() {
    let (motivo, p, vetaria) = evaluar(0.0225, 0.01);
    println!("observado: TP/SL = 2,25 sin deriva ⇒ {motivo}, p = {p:?}, r15_vetaria = {vetaria:?}");
    assert_eq!(motivo, "admitida");
    assert!((p.expect("publica p_hit_sl_first") - 2.25 / 3.25).abs() < 1e-9);
    assert_eq!(vetaria, Some(0.0));
}
