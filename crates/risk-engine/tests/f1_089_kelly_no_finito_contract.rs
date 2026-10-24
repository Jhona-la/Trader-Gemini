//! Contrato F1-089 (Sol, Ola Ω192b): una fracción de Kelly NO FINITA no puede
//! ser «rescatada» por el tope del genoma ni por el nocional mínimo.
//!
//! # QUÉ
//! `evaluate_quantum_order` debe RECHAZAR (Flat, volumen 0, diagnóstico
//! `REJ_INVALID_INPUT`) toda intención cuya `kelly_fraction` publicada no sea
//! finita (NaN, +∞, −∞), sin tocar la EWMA de riesgo ni dejar medida pendiente.
//!
//! # POR QUÉ
//! El camino de sizing terminaba en `raw_kelly.min(clamp_max)`. `f64::min`
//! devuelve el OTRO operando cuando uno es NaN, así que un estadístico corrupto
//! se convertía en el tope del genoma (0,25) y la orden se ADMITÍA con un
//! cuarto de Kelly que nadie calculó; con ±∞ el mismo `min` también lo
//! enmascaraba. El nocional mínimo completaba el rescate: la orden salía con
//! tamaño positivo. Es exactamente el fallo que C07-T03 detectó en main
//! (run 39551358570, shard `riesgo`).
//!
//! # CÓMO
//! Guarda explícita de finitud inmediatamente después de leer la métrica, antes
//! de cualquier clamp que pueda sanearla (fail-closed, con diagnóstico).
//!
//! # DÓNDE
//! `crates/risk-engine/src/lib.rs`, `RiskEngine::evaluate_quantum_order`.
//!
//! # QUIÉN
//! Sol — auditoría de main rojo 2026-10-21.

use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJ_INVALID_INPUT};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{
    atomic::Ordering::Relaxed,
    Arc, Mutex, Once,
};

// El registro de símbolos y los contadores de rechazo son globales al proceso;
// este binario serializa sus propias observaciones.
static SERIAL: Mutex<()> = Mutex::new(());

fn fixture(capital: f64, side: SignalType) -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRY: Once = Once::new();
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
    let arena = GlobalArena::build_in_own_stack(capital);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.win_rate.store(0.75, Relaxed);
    coin.metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.min_confidence_btc.store(0.62, Relaxed);
    let dir = if side == SignalType::Long { 1.0 } else { -1.0 };
    let intent = SignalIntent {
        signal: side,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        tp_price_target: 100.0 + dir * 2.0,
        sl_price_target: 100.0 - dir,
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// Control: con una fracción de Kelly finita y sana el motor sigue admitiendo.
/// Sin este control, un guard que rechazara todo pasaría el resto del contrato.
#[test]
fn f1_089_control_kelly_finito_admite() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for capital in [13.0, 25.0, 100.0] {
        for side in [SignalType::Long, SignalType::Short] {
            let (arena, intent) = fixture(capital, side);
            let mut engine = RiskEngine::new(capital);
            let order = engine.evaluate_quantum_order(0, &intent, &arena);
            assert_eq!(
                order.signal,
                side,
                "C={capital} side={side:?}: {}",
                risk_engine::reject_report()
            );
            assert!(order.volume_usd.is_finite() && order.volume_usd > 0.0);
        }
    }
}

/// El contrato que estaba rojo en main: NaN/±∞ se rechazan con diagnóstico,
/// no se sanean a `clamp_max` ni al nocional mínimo.
#[test]
fn f1_089_kelly_no_finito_se_rechaza_sin_sustituto() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for corrupto in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for capital in [13.0, 25.0, 100.0] {
            for side in [SignalType::Long, SignalType::Short] {
                let (arena, intent) = fixture(capital, side);
                arena.coins[0].metrics.kelly_fraction.store(corrupto, Relaxed);
                let riesgo_antes = arena.riesgo_por_operacion.load(Relaxed).to_bits();
                let mut engine = RiskEngine::new(capital);
                let order = engine.evaluate_quantum_order(0, &intent, &arena);

                assert_eq!(
                    order.signal,
                    SignalType::Flat,
                    "kelly={corrupto} C={capital} side={side:?} fue ADMITIDA"
                );
                assert_eq!(
                    order.volume_usd.to_bits(),
                    0.0_f64.to_bits(),
                    "kelly={corrupto} C={capital} side={side:?} salió con volumen"
                );
                assert_eq!(
                    risk_engine::ultimo_rechazo(),
                    Some(REJ_INVALID_INPUT),
                    "kelly={corrupto} C={capital} side={side:?}: diagnóstico ausente o erróneo ({})",
                    risk_engine::reject_report()
                );
                // Un rechazo no mide: ni EWMA de riesgo ni medida pendiente.
                assert_eq!(
                    arena.riesgo_por_operacion.load(Relaxed).to_bits(),
                    riesgo_antes,
                    "kelly={corrupto}: el rechazo movió la EWMA de riesgo"
                );
                assert!(
                    engine.tomar_medida_validada(0).is_none(),
                    "kelly={corrupto}: el rechazo dejó una medida pendiente"
                );
            }
        }
    }
}
