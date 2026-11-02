//! RSK-01 (Claude QSR, 2026-11-01): la guarda de liquidación F1-088 no puede
//! usar una tasa de margen de mantenimiento que nadie mide.
//!
//! # Qué estaba mal
//! F1-088 (Ω192) acota el apalancamiento con `L · (SL + f + mmr) < 1`, la
//! condición para que el stop salte antes que la liquidación. Pero `mmr` se
//! leía de la clave `maintenance_margin_rate`, que ningún código escribe (no
//! se consulta `/fapi/v1/leverageBracket`): valía un 1 % fijo para todo
//! símbolo y, si alguien escribía la clave GLOBAL, la de la última moneda.
//!
//! En el primer tramo de Binance USDⓈ-M la tasa es ≈ 1/(2·L_max): 0,4 % a
//! 125×, 1 % a 50×, 2,5 % a 20×. Con el 1 % fijo, la sonda de una cuenta de
//! 50–100 USD salía a 20× con stop del 3 % en un símbolo de 20×:
//! 20 · (0,03 + f + 0,025) ≈ 1,12 ≥ 1. La posición se liquida antes del stop
//! y se pierde el margen entero más la comisión de liquidación, mientras el
//! riesgo modela una pérdida igual al stop.
//!
//! # Contrato
//! 1. Sin dato medido, la tasa sigue al apalancamiento máximo del símbolo:
//!    max(1 %, 0,5/L_max), acotada a [0,1 %; 5 %]. Para L_max ≥ 50 es el 1 %
//!    de antes (los fixtures del oráculo usan 50×: no cambian).
//! 2. Un dato medido por moneda (`c{id}:maintenance_margin_rate`) manda.
//! 3. La clave global no cuenta: la tasa es de cada símbolo.
//!
//! Los tests comparten el registro global de specs: van serializados.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, PoisonError};

static CERROJO: Mutex<()> = Mutex::new(());

/// Tasa del primer tramo de un símbolo de 20× en Binance: 1/(2·20).
const MMR_20X: f64 = 0.025;

fn fixture(capital: f64, max_leverage: u32, sl: f64) -> (Arc<GlobalArena>, SignalIntent) {
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }]);
    let arena = GlobalArena::build_in_own_stack(capital);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(0.3, Relaxed);
    arena.coins[0].hurst_exponent.store(0.6, Relaxed);
    // Sin historial: la orden es la sonda mínima, la que sale con más
    // apalancamiento (menos margen para el mismo nocional mínimo).
    arena.coins[0].metrics.trade_count.store(0, Relaxed);
    arena.config.global_leverage.store(25.0, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.margin_cushion_pct.store(0.80, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.95,
        win_probability: 0.85,
        expected_duration_ms: 3_600_000,
        sl_price_target: 100.0 * (1.0 - sl),
        tp_price_target: 100.0 * (1.0 + 3.0 * sl),
        ..SignalIntent::default()
    };
    (arena, intent)
}

fn stop_real(sl_target: f64) -> f64 {
    (100.0 - sl_target).abs() / 100.0
}

/// (1) Sin tasa medida, un símbolo de 20× no sale con un apalancamiento que
/// pone la liquidación antes del stop. Antes: L = 20 con stops de 2,5–3,5 %.
#[test]
fn rsk01_sonda_de_20x_no_se_liquida_antes_del_stop() {
    let _turno = CERROJO.lock().unwrap_or_else(PoisonError::into_inner);
    for &capital in &[50.0, 100.0] {
        for &sl in &[0.025, 0.03, 0.035] {
            let (arena, intent) = fixture(capital, 20, sl);
            let out = RiskEngine::new(capital).evaluate_quantum_order(0, &intent, &arena);
            assert_eq!(
                out.signal,
                SignalType::Long,
                "capital {capital}, stop {sl}: la sonda cabe"
            );
            let sl_real = stop_real(out.sl_target);
            assert!(
                out.leverage * (sl_real + MMR_20X) < 1.0,
                "capital {capital}, stop {sl_real}: L = {} liquida antes del stop con la tasa \
                 de 20× (L·(SL+mmr) = {:.3})",
                out.leverage,
                out.leverage * (sl_real + MMR_20X)
            );
        }
    }
}

/// (2) La clave GLOBAL no cuenta: la tasa es de cada símbolo. En un símbolo
/// de 50× con stop del 3 % la guarda deja L = 24 con el 1 %; antes, una clave
/// global de 0,1 % (de otra moneda o de nadie) subía el techo de liquidación a
/// 31 y la orden salía al techo del genoma, 25.
#[test]
fn rsk01_la_clave_global_no_afloja_la_guarda() {
    let _turno = CERROJO.lock().unwrap_or_else(PoisonError::into_inner);
    for &max_leverage in &[50u32, 20] {
        let (arena_sin, intent) = fixture(100.0, max_leverage, 0.03);
        let sin_clave = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena_sin);
        let (arena_con, intent) = fixture(100.0, max_leverage, 0.03);
        arena_con.registry.set("maintenance_margin_rate", 0.001);
        let con_clave = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena_con);
        assert_eq!(sin_clave.signal, SignalType::Long, "{max_leverage}×");
        assert_eq!(sin_clave.signal, con_clave.signal, "{max_leverage}×");
        assert_eq!(
            sin_clave.leverage.to_bits(),
            con_clave.leverage.to_bits(),
            "{max_leverage}×: la clave global movió el apalancamiento ({} → {})",
            sin_clave.leverage,
            con_clave.leverage
        );
        assert_eq!(
            sin_clave.volume_usd.to_bits(),
            con_clave.volume_usd.to_bits(),
            "{max_leverage}×"
        );
    }
}

/// (3) Una tasa medida para la moneda manda: con 0,4 % (la de BTC) el stop
/// del 3 % admite el techo del símbolo, 20×; sin dato, la tasa de 20× (2,5 %)
/// lo baja a 17.
#[test]
fn rsk01_la_tasa_medida_por_moneda_manda() {
    let _turno = CERROJO.lock().unwrap_or_else(PoisonError::into_inner);
    let (arena_sin, intent) = fixture(100.0, 20, 0.03);
    let sin_dato = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena_sin);
    let (arena_con, intent) = fixture(100.0, 20, 0.03);
    arena_con
        .registry
        .set_for_coin(0, "maintenance_margin_rate", 0.004);
    let con_dato = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena_con);
    assert_eq!(sin_dato.signal, SignalType::Long);
    assert_eq!(con_dato.signal, SignalType::Long);
    assert_eq!(
        con_dato.leverage, 20.0,
        "con la tasa medida de 0,4 % cabe el techo del símbolo"
    );
    assert!(
        sin_dato.leverage < con_dato.leverage,
        "sin dato la guarda debe usar la tasa de 20× y bajar el apalancamiento          (sin dato {}, medida {})",
        sin_dato.leverage,
        con_dato.leverage
    );
}

/// (4) Para un símbolo de 50× sin dato, la tasa es el 1 % de antes: la orden
/// es idéntica a la que sale con el 1 % medido.
#[test]
fn rsk01_simbolo_de_50x_sin_dato_no_cambia() {
    let _turno = CERROJO.lock().unwrap_or_else(PoisonError::into_inner);
    for &sl in &[0.005, 0.02, 0.03] {
        let (arena_sin, intent) = fixture(100.0, 50, sl);
        let sin_dato = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena_sin);
        let (arena_con, intent) = fixture(100.0, 50, sl);
        arena_con
            .registry
            .set_for_coin(0, "maintenance_margin_rate", 0.01);
        let con_dato = RiskEngine::new(100.0).evaluate_quantum_order(0, &intent, &arena_con);
        assert_eq!(sin_dato.signal, con_dato.signal, "stop {sl}");
        assert_eq!(
            sin_dato.leverage.to_bits(),
            con_dato.leverage.to_bits(),
            "stop {sl}"
        );
        assert_eq!(
            sin_dato.volume_usd.to_bits(),
            con_dato.volume_usd.to_bits(),
            "stop {sl}"
        );
        assert_eq!(
            sin_dato.sl_target.to_bits(),
            con_dato.sl_target.to_bits(),
            "stop {sl}"
        );
    }
}

/// (5) Sin dato, la tasa sigue al apalancamiento máximo: 2,5 % a 20×, el 1 %
/// de antes a 50× y más (piso), y el techo con un L_max inválido.
#[test]
fn rsk01_tasa_sin_dato_sigue_al_apalancamiento_maximo() {
    use risk_engine::tasa_mantenimiento_efectiva as tasa;
    assert!((tasa(None, 20.0) - 0.025).abs() < 1e-12);
    assert!((tasa(None, 10.0) - 0.05).abs() < 1e-12);
    assert!((tasa(None, 50.0) - 0.01).abs() < 1e-12);
    assert!((tasa(None, 125.0) - 0.01).abs() < 1e-12);
    for l_max in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -5.0, 1.0] {
        assert!((tasa(None, l_max) - 0.05).abs() < 1e-12, "L_max = {l_max}");
    }
    let mut previa = f64::INFINITY;
    for l_max in 2..=200 {
        let actual = tasa(None, l_max as f64);
        assert!(
            actual <= previa,
            "la tasa no sube con el apalancamiento máximo ({l_max})"
        );
        assert!((0.01..=0.05).contains(&actual));
        previa = actual;
    }
}

/// (6) Una tasa medida válida manda (acotada); una corrupta no cuenta como
/// dato y se usa la del apalancamiento máximo.
#[test]
fn rsk01_tasa_medida_valida_manda_y_la_corrupta_no_cuenta() {
    use risk_engine::tasa_mantenimiento_efectiva as tasa;
    assert!((tasa(Some(0.004), 125.0) - 0.004).abs() < 1e-12);
    assert!((tasa(Some(0.004), 20.0) - 0.004).abs() < 1e-12);
    assert!((tasa(Some(0.0001), 20.0) - 0.001).abs() < 1e-12);
    assert!((tasa(Some(0.5), 20.0) - 0.05).abs() < 1e-12);
    for corrupta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.01] {
        assert!(
            (tasa(Some(corrupta), 20.0) - 0.025).abs() < 1e-12,
            "medida = {corrupta}"
        );
    }
}
