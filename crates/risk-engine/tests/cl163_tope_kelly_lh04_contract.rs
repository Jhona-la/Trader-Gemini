//! CL-163: el gen 33 (`kelly_clamp_max`) es el techo de la apuesta de LH-04.
//!
//! El gen dice «apuesta Kelly máxima». En el motor legado acotaba la
//! fracción de Kelly W·(1 − 1/PF)·λ (kelly.rs), que esa cadena aplicaba
//! como fracción de MARGEN (`raw_exposure = dir·conf·kelly·C`), no como
//! pérdida al stop. Desde F1-116 + F1-087 el Kelly legado ya no fija ningún
//! tamaño. Ahora el gen acota la f de LH-04 (pérdida al stop con la
//! fricción, en fracción del capital asignado; CL-159): f = min(½·Kelly-LCB,
//! tope de ruina, κ). Sólo aprieta, nunca crea ni agranda una ventaja. Como
//! el axioma de ruina ya acota f en 0,25, κ sólo muerde en [0,1; 0,25): la
//! línea base (0,55) y los genomas guardados (0,8245) no cambian nada. Y la
//! orden sólo cambia si κ baja del riesgo que los techos de margen ya dejan
//! (en este fixture, con L = 1, ≈ 1,9 % del capital).
//!
//! κ = 0 no sale de un genoma (la banda del gen empieza en 0,1). Fijado a
//! mano por debajo de `kelly_clamp_min` (gen 32, ≥ 0,001), lo rechaza antes
//! la validación de entrada FMT-212 con REJ_INVALID_INPUT.

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
/// PF = 2,6 ⇒ b = 1,4; la cota de Wilson (z = 1,64) da p_LCB ≈ 0,5688 y
/// f = ½·(0,5688 − 0,4312/1,4) ≈ 0,1304.
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

/// El riesgo al stop que la orden emitida realiza (tras los techos), en
/// fracción del capital asignado.
fn f_realizado(arena: &GlobalArena) -> f64 {
    arena.registry.get_for_coin_or(0, "lh04_f_realizado", f64::NAN)
}

fn huella(o: &ValidatedOrder) -> [u64; 4] {
    [o.volume_usd.to_bits(), o.leverage.to_bits(), o.tp_target.to_bits(), o.sl_target.to_bits()]
}

#[test]
fn cl163_el_tope_del_gen_acota_la_f_de_lh04() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (libre, intent) = fixture(1.0);
    evidencia(&libre, 2.6);
    let orden_libre = evaluar(&libre, &intent);
    let f_libre = f_riesgo(&libre);
    assert!(f_libre > 0.13 && f_libre < 0.14, "premisa: f ≈ 0,1304, obtenido {f_libre}");

    let (acotado, intent) = fixture(0.10);
    evidencia(&acotado, 2.6);
    let orden = evaluar(&acotado, &intent);
    assert_eq!(f_riesgo(&acotado), 0.10, "κ = 0,10 acota f");
    assert!(orden.volume_usd > 0.0, "la orden sigue en pie");

    // En este fixture (L = 1) el techo de margen ya acota el riesgo emitido
    // muy por debajo de f: `lh04_f_realizado` ≈ 0,019. κ = 0,10 sólo cambia la
    // f objetivo; κ muerde la orden cuando baja de ese riesgo realizado, y
    // entonces el nocional baja en proporción.
    let realizado_libre = f_realizado(&libre);
    assert!(
        realizado_libre > 0.0 && realizado_libre < f_libre,
        "premisa: el margen acota el riesgo emitido, {realizado_libre} < {f_libre}"
    );
    let kappa = 0.5 * realizado_libre;
    let (bajo, intent) = fixture(kappa);
    evidencia(&bajo, 2.6);
    let orden_baja = evaluar(&bajo, &intent);
    let r = (orden_baja.volume_usd * orden_baja.leverage) / (orden_libre.volume_usd * orden_libre.leverage);
    assert!((r - 0.5).abs() < 0.05 * 0.5, "con κ = ½ del riesgo realizado el nocional baja a la mitad: {r}");
    assert!((f_realizado(&bajo) - kappa).abs() < 1e-9, "el riesgo realizado es κ: {}", f_realizado(&bajo));
}

#[test]
fn cl163_un_tope_bajo_el_minimo_del_gen_32_lo_rechaza_fmt212() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture(0.0);
    evidencia(&arena, 2.6);
    let antes = REJECT_COUNTERS[REJ_INVALID_INPUT].load(Relaxed);
    let orden = evaluar(&arena, &intent);
    assert_eq!(orden.volume_usd, 0.0, "κ = 0 con kelly_clamp_min = 0,001 no sale");
    assert_eq!(
        REJECT_COUNTERS[REJ_INVALID_INPUT].load(Relaxed) - antes,
        1,
        "el rechazo es la validación de entrada (REJ_INVALID_INPUT), no REJ_EXPOSURE_ZERO"
    );
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
