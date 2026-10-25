//! CL-172 (fila N12 del censo C-F7b): la viabilidad D-750 cuenta la fricción.
//!
//! La orden mínima del símbolo que toca su stop pierde N_min·(SL + c), con c
//! la fricción de ida y vuelta (`tp_sl::roundtrip_friction`), la misma con la
//! que dimensiona LH-04 (CL-159). `capital_regime::orden_viable` recibía sólo
//! el SL: admitía una orden mínima cuya pérdida real al stop pasaba el tope de
//! riesgo por evento, (SL + c)/SL veces la declarada.
//!
//! El fixture es el de veto_logic_contracts (13 USD, BTCUSDT con paso 0,001,
//! precio 100, ATR 1 %, latencia 0). Las premisas se calculan con las mismas
//! funciones que el gate, sin literales: c, el tope de ruina y la orden mínima
//! del lote.

use quantum_arena::GlobalArena;
use risk_engine::{capital_regime, ruin, tp_sl, ultimo_rechazo, RiskEngine, REJ_VIABILIDAD};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

const CAPITAL: f64 = 13.0;
const PRECIO: f64 = 100.0;

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
    let arena = GlobalArena::build_in_own_stack(CAPITAL);
    arena.coins[0].current_price.store(PRECIO, Relaxed);
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
        tp_price_target: 300.0,
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// c, tope de ruina y orden mínima del lote, con las funciones del gate.
fn premisas(arena: &GlobalArena) -> (f64, f64, f64) {
    let taker = arena.config.fee_rates_for(0).map_or(f64::NAN, |(_, t)| t);
    let piso = arena.config.base_slippage_floor.load(Relaxed).max(0.00001);
    let atr_pct = arena.coins[0].current_atr.load(Relaxed) / PRECIO;
    let c = tp_sl::roundtrip_friction(taker, piso, atr_pct, 0.0);
    let tope = ruin::clamp_ruin(1.0, ruin::conservative_loss_q(0.75, 100.0));
    let n_min = capital_regime::nocional_minimo_ejecutable(
        capital_regime::effective_min_notional(5.0) + 0.1,
        0.001,
        0.001,
        PRECIO,
    )
    .unwrap_or(f64::NAN);
    assert!(c.is_finite() && c > 0.0, "premisa: c = {c}");
    assert_eq!(
        tope, 0.25,
        "premisa: el tope del fixture es el axioma del 25 %"
    );
    assert!(n_min.is_finite() && n_min > 0.0, "premisa: n_min = {n_min}");
    (c, tope, n_min)
}

fn evaluar_con_stop(sl: f64) -> (risk_engine::ValidatedOrder, f64) {
    let (arena, mut intent) = fixture();
    intent.sl_price_target = PRECIO * (1.0 - sl);
    let (c, _, _) = premisas(&arena);
    let out = RiskEngine::new(CAPITAL).evaluate_quantum_order(0, &intent, &arena);
    (out, c)
}

#[test]
fn cl172_la_orden_minima_que_no_paga_la_friccion_al_stop_se_rechaza_por_viabilidad() {
    let (arena, _) = fixture();
    let (c, tope, n_min) = premisas(&arena);
    // El stop con el que la orden mínima arriesga justo el tope, menos c/2:
    // sin c cabe en el tope, con c no.
    let sl = tope * CAPITAL / n_min - c / 2.0;
    assert!(
        n_min * sl / CAPITAL <= tope,
        "premisa: sin c la orden mínima cabe"
    );
    assert!(
        n_min * (sl + c) / CAPITAL > tope,
        "premisa: con c la orden mínima no cabe"
    );
    let (out, _) = evaluar_con_stop(sl);
    println!(
        "SL {sl:.6}, c {c:.6}: {:?}, {} USD ×{}; {}",
        out.signal,
        out.volume_usd,
        out.leverage,
        risk_engine::reject_report()
    );
    assert_eq!(
        out.signal,
        SignalType::Flat,
        "la orden mínima pierde más que el tope al stop"
    );
    assert_eq!(ultimo_rechazo(), Some(REJ_VIABILIDAD));
}

#[test]
fn cl172_la_viabilidad_cuenta_la_friccion_una_sola_vez() {
    let (arena, _) = fixture();
    let (c, tope, n_min) = premisas(&arena);
    // Con c la orden mínima cabe; con 2c, o con c y el MMR, no cabría.
    let sl = tope * CAPITAL / n_min - 1.5 * c;
    assert!(n_min * (sl + c) / CAPITAL <= tope, "premisa: con c cabe");
    assert!(
        n_min * (sl + 2.0 * c) / CAPITAL > tope,
        "premisa: con 2c no cabría"
    );
    let (out, _) = evaluar_con_stop(sl);
    println!(
        "SL {sl:.6}, c {c:.6}: {:?}, {} USD ×{}; {}",
        out.signal,
        out.volume_usd,
        out.leverage,
        risk_engine::reject_report()
    );
    assert_ne!(
        ultimo_rechazo(),
        Some(REJ_VIABILIDAD),
        "c entra una sola vez"
    );
    assert_eq!(out.signal, SignalType::Long, "la orden mínima viable sale");
}
