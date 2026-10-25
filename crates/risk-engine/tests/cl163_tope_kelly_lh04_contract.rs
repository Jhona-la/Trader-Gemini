//! CL-163: el gen 33 (`kelly_clamp_max`) es el techo de la apuesta de LH-04.
//!
//! El gen dice «apuesta Kelly máxima». En el motor legado acotaba
//! W·(1 − 1/PF)·λ (kelly.rs), una fracción del capital que se arriesga: la
//! misma magnitud que la f de LH-04 (pérdida al stop con la fricción, en
//! fracción del capital asignado; CL-159). Desde F1-116 + F1-087 el Kelly
//! legado ya no fija ningún tamaño y el gen había quedado sin efecto.
//! Ahora f = min(½·Kelly-LCB, tope de ruina, κ): sólo aprieta, nunca crea ni
//! agranda una ventaja. Con el κ de la línea base (0,55) el mínimo no cambia
//! nada, porque el axioma de ruina ya acota f en 0,25.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{RiskEngine, ValidatedOrder, REJECT_COUNTERS, REJ_INVALID_INPUT};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Mutex;

/// Los contadores de rechazo son globales del proceso.
static SERIE: Mutex<()> = Mutex::new(());

fn fixture(kappa: f64) -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
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
    }]);
    let arena = GlobalArena::build_in_own_stack(1000.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.001, Relaxed);
    arena.config.kelly_clamp_max.store(kappa, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// La evidencia de `lh04_sizing_riesgo_contract`: n = 100, p̂ = 0,65,
/// PF = 2,6 ⇒ b = 1,4 y f = ½·(0,572 − 0,428/1,4) ≈ 0,1333.
fn evidencia(arena: &GlobalArena, pf: f64) {
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(pf, Relaxed);
}

fn evaluar(arena: &GlobalArena, intent: &SignalIntent) -> ValidatedOrder {
    let mut engine = RiskEngine::new(1000.0);
    engine.evaluate_quantum_order(0, intent, arena)
}

fn f_riesgo(arena: &GlobalArena) -> f64 {
    arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN)
}

fn huella(o: &ValidatedOrder) -> [u64; 4] {
    [o.volume_usd.to_bits(), o.leverage.to_bits(), o.tp_target.to_bits(), o.sl_target.to_bits()]
}

#[test]
fn cl163_el_tope_del_gen_acota_la_f_de_lh04() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (libre, intent) = fixture(1.0);
    evidencia(&libre, 2.6);
    let _ = evaluar(&libre, &intent);
    let f_libre = f_riesgo(&libre);
    assert!(f_libre > 0.13 && f_libre < 0.14, "premisa: f ≈ 0,1333, obtenido {f_libre}");

    let (acotado, intent) = fixture(0.10);
    evidencia(&acotado, 2.6);
    let orden = evaluar(&acotado, &intent);
    assert_eq!(f_riesgo(&acotado), 0.10, "κ = 0,10 acota f");
    assert!(orden.volume_usd > 0.0, "la orden sigue en pie, más pequeña");
}

#[test]
fn cl163_con_el_tope_de_la_linea_base_la_orden_no_cambia() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (base, intent) = fixture(0.55);
    evidencia(&base, 2.6);
    let orden_base = evaluar(&base, &intent);
    let (libre, intent) = fixture(1.0);
    evidencia(&libre, 2.6);
    let orden_libre = evaluar(&libre, &intent);
    assert!(orden_base.volume_usd > 0.0);
    assert_eq!(f_riesgo(&base).to_bits(), f_riesgo(&libre).to_bits());
    assert_eq!(huella(&orden_base), huella(&orden_libre), "κ ≥ f no toca la orden");
}

#[test]
fn cl163_sin_edge_sigue_en_la_orden_minima_con_cualquier_tope() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (alto, intent) = fixture(1.0);
    evidencia(&alto, 0.9);
    let orden_alto = evaluar(&alto, &intent);
    let (bajo, intent) = fixture(0.10);
    evidencia(&bajo, 0.9);
    let orden_bajo = evaluar(&bajo, &intent);
    assert!(orden_alto.volume_usd > 0.0, "F1-087: la moneda sin edge sale a la orden mínima");
    assert_eq!(f_riesgo(&bajo), 0.0);
    assert_eq!(
        (orden_bajo.volume_usd * orden_bajo.leverage).to_bits(),
        (orden_alto.volume_usd * orden_alto.leverage).to_bits(),
        "el tope no cambia el nocional de la orden mínima"
    );
}

#[test]
fn cl163_un_tope_no_finito_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture(f64::NAN);
    evidencia(&arena, 2.6);
    let antes = REJECT_COUNTERS[REJ_INVALID_INPUT].load(Relaxed);
    let orden = evaluar(&arena, &intent);
    assert_eq!(orden.volume_usd, 0.0);
    assert_eq!(REJECT_COUNTERS[REJ_INVALID_INPUT].load(Relaxed) - antes, 1);
}
