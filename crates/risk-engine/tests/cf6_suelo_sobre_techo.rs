//! C-F6 · CL-146: el riesgo no entra en pánico cuando la orden mínima del
//! símbolo supera la exposición máxima del envío.
//!
//! Teoría. La conversión de Cramér-Lundberg (Ω152) acota la exposición de la
//! orden, f = nocional / capital, entre la orden mínima que el exchange admite
//! (`min_notional / capital`) y el techo del envío (`0,25·min(L_max, 20)`).
//! Con 13 USD y BTCUSDT (mínimo 100 USD) el suelo es 7,7 y el techo 5:
//! `f64::clamp(7,7, 5)` entra en pánico y, con `panic = "abort"`, el proceso
//! vivo muere con posiciones abiertas. El suelo es la exposición real de la
//! orden más pequeña posible, así que manda él; si la orden cabe lo deciden
//! las puertas de viabilidad, como antes.
//!
//! Fixture. Un símbolo de mínimo 100 USD y apalancamiento máximo 20 a
//! 60 000 USD, ATR del 1 %, cuenta de 13 USD, con y sin historial.

use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const CAPITAL: f64 = 13.0;
const NOCIONAL_MINIMO: f64 = 100.0;
const PRECIO: f64 = 60_000.0;

fn arena() -> Arc<GlobalArena> {
    static REGISTRO: Once = Once::new();
    REGISTRO.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.1,
                min_qty: 0.001,
                min_notional: NOCIONAL_MINIMO,
                max_leverage: 125,
                maker_fee: 0.0002,
                taker_fee: 0.0005,
                is_shadow: false,
            },
        ])
    });
    let arena = GlobalArena::build_in_own_stack(CAPITAL);
    arena.coins[0].current_price.store(PRECIO, Relaxed);
    arena.coins[0].current_atr.store(PRECIO * 0.01, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.global_max_drawdown.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.riesgo_por_operacion.store(1e-4, Relaxed);
    arena
}

fn con_historial(arena: &GlobalArena) {
    let m = &arena.coins[0].metrics;
    m.trade_count.store(100, Relaxed);
    m.win_rate.store(0.75, Relaxed);
    m.gross_wins.store(200.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    m.profit_factor.store(2.0, Relaxed);
    m.kelly_fraction.store(0.15, Relaxed);
}

/// Nocional de la orden admitida, o el nombre del motivo del rechazo.
fn evaluar(arena: &GlobalArena) -> Result<f64, &'static str> {
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    let orden = RiskEngine::new(CAPITAL).evaluate_quantum_order(0, &intent, arena);
    if orden.signal == SignalType::Flat {
        Err(risk_engine::ultimo_rechazo().map_or("sin motivo", |i| REJECT_NAMES[i]))
    } else {
        Ok(orden.volume_usd * orden.leverage)
    }
}

/// Esperado: un veredicto (rechazo o una orden de al menos 100 USD), no un
/// pánico. Antes: `f64::clamp` con el suelo 7,7 sobre el techo 5.
#[test]
fn cl146_la_orden_minima_sobre_el_techo_no_tumba_el_riesgo() {
    for historial in [false, true] {
        let a = arena();
        if historial {
            con_historial(&a);
        }
        let r = evaluar(&a);
        println!("observado (historial: {historial}): {r:?}");
        if let Ok(nocional) = r {
            assert!(
                nocional >= NOCIONAL_MINIMO,
                "orden bajo el mínimo del exchange: {nocional} USD"
            );
        }
    }
}
