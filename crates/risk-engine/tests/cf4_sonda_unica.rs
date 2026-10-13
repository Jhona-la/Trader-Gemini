//! C-F4 · D4 (ADR-0016, LC-F4-19; Línea C, CL-98): una sola sonda abierta en
//! toda la cartera, y la sonda es la orden mínima del símbolo.
//!
//! Teoría. Una moneda con menos de 5 cierres no tiene evidencia con la que
//! juzgar el valor esperado: sus órdenes son sondas, y se admiten porque
//! están presupuestadas, no por un EV calculado con un prior. El
//! presupuesto de D4 es:
//! - una sonda abierta a la vez en toda la cartera;
//! - de tamaño mínimo;
//! - con sus pérdidas dentro del drawdown de D3.
//! Fuera de la fase de sonda nada de esto aplica.
//!
//! Antes la fase de sonda era por moneda. Con varias monedas sin historial,
//! cada una abría su sonda a la vez, y la propia moneda podía abrir otra en
//! otra ranura. El tamaño salía del Kelly de arranque en frío, no de la
//! orden mínima. Tras un reinicio, `trade_count` vuelve a 0: una posición
//! adoptada del exchange cuenta como sonda abierta.
//!
//! Fixture. Arena de 1 000 USD (lejos del régimen micro, donde el tamaño ya
//! ronda el mínimo), dos símbolos con nocional mínimo de 5 USD, precio 100,
//! ATR 1 y una intención larga con confianza 0,9. Las posiciones abiertas
//! son pequeñas (0,01 unidades, riesgo al stop 1e-5 del capital): el veto
//! de grupo no puede ser el motivo del rechazo.

#[path = "support/net_payoff_fixture.rs"]
mod net_payoff_fixture;

use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const CAPITAL: f64 = 1_000.0;
const NOCIONAL_MINIMO: f64 = 5.0;

fn arena() -> Arc<GlobalArena> {
    static REGISTRO: Once = Once::new();
    REGISTRO.call_once(|| {
        quantum_arena::symbol_registry::update_registry(
            ["BTCUSDT", "ETHUSDT"]
                .iter()
                .map(|s| quantum_arena::symbol_registry::SymbolSpec {
                    symbol: (*s).into(),
                    step_size: 0.001,
                    tick_size: 0.01,
                    min_qty: 0.001,
                    min_notional: NOCIONAL_MINIMO,
                    max_leverage: 20,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        )
    });
    let arena = GlobalArena::build_in_own_stack(CAPITAL);
    for coin in &arena.coins[0..2] {
        coin.current_price.store(100.0, Relaxed);
        coin.current_atr.store(1.0, Relaxed);
        coin.hurst_exponent.store(0.5, Relaxed);
    }
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.global_correlation_threshold.store(0.5, Relaxed);
    arena.riesgo_por_operacion.store(1e-4, Relaxed);
    arena
}

/// Moneda con historial: 100 cierres, acierto 0,75 y PF 2.
fn con_historial(arena: &GlobalArena, coin: usize) {
    let m = &arena.coins[coin].metrics;
    m.trade_count.store(100, Relaxed);
    m.win_rate.store(0.75, Relaxed);
    m.gross_wins.store(200.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    m.profit_factor.store(2.0, Relaxed);
    m.kelly_fraction.store(0.15, Relaxed);
}

/// Posición pequeña abierta (0,01 unidades a 100, stop en 99).
fn abrir(arena: &GlobalArena, coin: usize, slot: usize) {
    arena.coins[coin].positions.slots()[slot].open(true, 100.0, 0.01, 0.05, 1_000, 102.0, 99.0);
    assert!(arena.coins[coin].positions.slots()[slot].is_open());
}

fn larga() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

/// Nocional de la orden admitida, o el nombre del motivo del rechazo.
fn evaluar(arena: &GlobalArena, coin: usize) -> Result<f64, &'static str> {
    let mut intent = larga();
    if arena.coins[coin].metrics.trade_count.load(Relaxed) >= risk_engine::sonda::CIERRES_DE_SONDA {
        intent.net_payoff = Some(net_payoff_fixture::positive(coin, intent.signal));
    }
    let orden = RiskEngine::new(CAPITAL).evaluate_quantum_order(coin, &intent, arena);
    if orden.signal == SignalType::Flat {
        Err(risk_engine::ultimo_rechazo().map_or("sin motivo", |i| REJECT_NAMES[i]))
    } else {
        Ok(orden.volume_usd * orden.leverage)
    }
}

/// Esperado: con una sonda abierta en otra moneda, la moneda sin historial
/// no abre la suya; el motivo es `sonda_ocupada`. Esto vale también tras un
/// reinicio: la posición de la otra moneda está en una moneda con 0 cierres,
/// como una posición adoptada del exchange.
#[test]
fn cl98_una_sola_sonda_abierta_en_la_cartera() {
    let a = arena();
    abrir(&a, 1, 0);
    let r = evaluar(&a, 0);
    println!("observado: con una sonda abierta en ETH, BTC sin historial da {r:?}");
    assert_eq!(r, Err("sonda_ocupada"));
}

/// Esperado: la moneda que ya tiene su sonda abierta tampoco abre otra en
/// otra ranura.
#[test]
fn cl98_la_moneda_de_la_sonda_no_abre_otra() {
    let a = arena();
    abrir(&a, 0, 0);
    let r = evaluar(&a, 0);
    println!("observado: con su sonda abierta, BTC da {r:?}");
    assert_eq!(r, Err("sonda_ocupada"));
}

/// Falso positivo: sin ninguna sonda abierta, la moneda sin historial abre
/// su sonda.
#[test]
fn cl98_sin_sonda_abierta_la_sonda_pasa() {
    let a = arena();
    let r = evaluar(&a, 0);
    println!("observado: sin sondas abiertas, BTC da {r:?}");
    assert!(r.is_ok(), "la sonda sin competencia debe pasar: {r:?}");
}

/// Falso positivo: una moneda con historial no espera a la sonda de otra.
#[test]
fn cl98_una_moneda_con_historial_no_espera_a_la_sonda() {
    let a = arena();
    con_historial(&a, 0);
    abrir(&a, 1, 0);
    let r = evaluar(&a, 0);
    println!("observado: BTC con historial y una sonda en ETH da {r:?}");
    assert!(r.is_ok(), "una moneda fuera de la fase de sonda no se frena: {r:?}");
}

/// Falso positivo: una posición de una moneda con historial no es una sonda.
#[test]
fn cl98_una_posicion_con_historial_no_ocupa_la_sonda() {
    let a = arena();
    con_historial(&a, 1);
    abrir(&a, 1, 0);
    let r = evaluar(&a, 0);
    println!("observado: BTC sin historial con una posición de ETH con historial da {r:?}");
    assert!(r.is_ok(), "sin sonda abierta, la sonda pasa: {r:?}");
}

/// Esperado: la sonda es la orden mínima del símbolo. Su nocional no pasa
/// del mínimo con el margen de seguridad del riesgo (+0,1 USD y 5 pb).
#[test]
fn cl98_la_sonda_es_la_orden_minima() {
    let a = arena();
    let nocional = evaluar(&a, 0).expect("precondición: la sonda pasa");
    println!("observado: nocional de la sonda {nocional:.4} USD (mínimo {NOCIONAL_MINIMO})");
    assert!(nocional >= NOCIONAL_MINIMO, "por debajo del mínimo del exchange: {nocional}");
    assert!(
        nocional <= (NOCIONAL_MINIMO + 0.1) * 1.0005 + 1e-9,
        "la sonda no es la orden mínima: {nocional} USD"
    );
}

/// Falso positivo: fuera de la fase de sonda el tamaño no se reduce al
/// mínimo.
#[test]
fn cl98_con_historial_el_tamano_no_se_reduce_al_minimo() {
    let a = arena();
    con_historial(&a, 0);
    let nocional = evaluar(&a, 0).expect("precondición: la orden pasa");
    println!("observado: nocional con historial {nocional:.4} USD");
    assert!(nocional > (NOCIONAL_MINIMO + 0.1) * 1.01, "nocional {nocional}");
}
