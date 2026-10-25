//! CL-159 — LH-04 dimensiona con la pérdida al stop QUE INCLUYE LA FRICCIÓN.
//!
//! Teoría que se prueba. LH-04 calcula f = ½·max(0, p_LCB − (1 − p_LCB)/b)
//! con b = PF·(1 − p̂)/p̂, y el PF es NETO: god-engine-core acumula
//! `net_trade_pnl` (bruto menos las comisiones de entrada y de salida) en
//! gross_wins/gross_losses. Así f es la fracción de Kelly que la cuenta
//! PIERDE cuando la operación pierde. Una orden de nocional N que toca su
//! stop pierde N·(SL + c), con c = `tp_sl::roundtrip_friction` (dos
//! comisiones taker y dos deslizamientos): es la l = (SL + c)/SL del freno
//! duro y de la medida D-744. Con N = f·C/SL la pérdida al stop era
//! f·C·(1 + c/SL) —1,29·f con SL 0,5 % y c 14,5 pb, un 0,645 de Kelly en
//! vez del ½ declarado— y la caída de falsación d* (drawdown.rs, válida con
//! c ≤ ½) se rebasaba con probabilidad ≈ 12 % en vez del 5 %.
//!
//! Contrato: (a) N = f·C/(SL + c) cuando ningún tope muerde; (b)
//! `lh04_f_realizado` publica N·(SL + c)/C, que sin tope es f, y coincide con
//! la orden emitida; (c) la pérdida al stop con la fricción nunca supera f·C.
//!
//! Fixture: el de `fable_lh04_stop_real_contract.rs` (1 000 USD, precio 100,
//! 100 cierres, p̂ 0,55, PF 1,45 ⇒ f ≈ 0,99 %; τ 1 h, k = 1). El stop que
//! lleva la orden, medido bajo wine, es ≈ 5,9·ATR% (5,9 % con ATR 1).
//! Latencia 0 ⇒ c = 2·0,0004 + 2·0,00001 = 8,2 pb. Antes de CL-159 los tres
//! fallan por el factor 1 + c/SL: 1,0139 con ATR 1 y 1,0273 con ATR 0,5.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex, Once};

static SERIAL: Mutex<()> = Mutex::new(());

const CAPITAL: f64 = 1_000.0;
const PRECIO: f64 = 100.0;
const TAKER: f64 = 0.0004;
const PISO: f64 = 0.00001;
/// Tolerancia relativa: el nocional pasa por margen = N/L y vuelve por
/// margen·L (unos pocos ULP). El defecto es un factor 1 + c/SL ≥ 1,004.
const TOL: f64 = 1e-6;

fn arena(atr: f64) -> Arc<GlobalArena> {
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
            taker_fee: TAKER,
            is_shadow: false,
        }])
    });
    let a = GlobalArena::build_in_own_stack(CAPITAL);
    a.coins[0].current_price.store(PRECIO, Relaxed);
    a.coins[0].current_atr.store(atr, Relaxed);
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    let m = &a.coins[0].metrics;
    m.trade_count.store(100, Relaxed);
    m.win_rate.store(0.55, Relaxed);
    m.gross_wins.store(145.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    m.profit_factor.store(1.45, Relaxed);
    // Geometría fijada: stop = 1·ATR%·(τ/τ_ref)^H, TP = 2·stop.
    a.config.sl_atr_mult_btc.store(1.0, Relaxed);
    a.config.tp_rr_ratio_btc.store(2.0, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(TAKER, Relaxed);
    a.config.base_slippage_floor.store(PISO, Relaxed);
    a
}

fn intencion() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.60,
        expected_duration_ms: 3_600_000,
        ..Default::default()
    }
}

struct Medida {
    f_objetivo: f64,
    f_realizado: f64,
    capital: f64,
    stop: f64,
    friccion: f64,
    /// volume_usd · leverage de la orden EMITIDA.
    nocional: f64,
    recortada: bool,
}

impl Medida {
    /// Pérdida al stop de la orden emitida, con la fricción, en USD.
    fn perdida_al_stop(&self) -> f64 {
        self.nocional * (self.stop + self.friccion)
    }
}

fn medir(atr: f64) -> Medida {
    let a = arena(atr);
    let recortes_antes = a.registry.get_for_coin_or(0, "sol_a2_margen_reducido", 0.0);
    let o = RiskEngine::new(CAPITAL).evaluate_quantum_order(0, &intencion(), &a);
    assert_eq!(
        o.signal,
        SignalType::Long,
        "ATR {atr}: {}",
        risk_engine::reject_report()
    );
    let f_objetivo = a.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let f_realizado = a.registry.get_for_coin_or(0, "lh04_f_realizado", f64::NAN);
    let capital = a
        .registry
        .get_for_coin_or(0, "lh04_capital_asignado", f64::NAN);
    let stop = a.registry.get_for_coin_or(0, "lh04_stop_real", f64::NAN);
    let recortes = a.registry.get_for_coin_or(0, "sol_a2_margen_reducido", 0.0);
    assert!(
        f_objetivo.is_finite() && f_objetivo > 0.0,
        "ATR {atr}: f con evidencia: {f_objetivo}"
    );
    assert!(
        capital.is_finite() && capital > 0.0,
        "ATR {atr}: capital asignado: {capital}"
    );
    assert!(
        stop.is_finite() && stop > 0.0,
        "ATR {atr}: stop publicado: {stop}"
    );
    let stop_orden = (PRECIO - o.sl_target) / PRECIO;
    assert!(
        (stop - stop_orden).abs() < 1e-9,
        "ATR {atr}: el stop publicado es el de la orden: {stop} vs {stop_orden}"
    );
    // La MISMA función y las MISMAS entradas que el gate (risk lib.rs:988):
    // taker en las dos piernas, piso de deslizamiento, ATR/precio, latencia 0.
    let friccion = risk_engine::tp_sl::roundtrip_friction(TAKER, PISO, atr / PRECIO, 0.0);
    assert!(
        friccion.is_finite() && friccion > 0.0,
        "fricción de ida y vuelta: {friccion}"
    );
    Medida {
        f_objetivo,
        f_realizado,
        capital,
        stop,
        friccion,
        nocional: o.volume_usd * o.leverage,
        recortada: recortes > recortes_antes,
    }
}

/// (a) Sin tope, el nocional es f·C/(SL + c), no f·C/SL.
#[test]
fn cl159_el_nocional_divide_por_el_stop_mas_la_friccion() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for atr in [1.0, 3.0] {
        let m = medir(atr);
        assert!(
            !m.recortada,
            "ATR {atr}: el fixture no recorta el margen (Sol A2)"
        );
        let esperado = m.f_objetivo * m.capital / (m.stop + m.friccion);
        assert!(
            (m.nocional - esperado).abs() <= TOL * esperado,
            "ATR {atr}: nocional {} ≠ f·C/(SL + c) = {esperado} (f·C/SL = {}; c = {}, SL = {})",
            m.nocional,
            m.f_objetivo * m.capital / m.stop,
            m.friccion,
            m.stop
        );
    }
}

/// (b) Lo publicado es la pérdida al stop con la fricción: sin tope vale f
/// y coincide con la orden emitida.
#[test]
fn cl159_el_realizado_publicado_es_la_perdida_al_stop_con_friccion() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let m = medir(1.0);
    assert!(!m.recortada);
    assert!(
        (m.f_realizado - m.f_objetivo).abs() < 1e-9,
        "sin tope, realizado = f: {} vs {}",
        m.f_realizado,
        m.f_objetivo
    );
    let emitida = m.perdida_al_stop() / m.capital;
    assert!(
        (emitida - m.f_realizado).abs() <= TOL * m.f_realizado,
        "N·(SL + c)/C de la orden = {emitida}, publicado {}",
        m.f_realizado
    );
}

/// (c) En todo ATR del fixture la pérdida al stop con la fricción cabe en f·C.
#[test]
fn cl159_la_perdida_al_stop_con_friccion_nunca_supera_f_por_c() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for atr in [0.5, 1.0, 2.0, 3.0] {
        let m = medir(atr);
        let tope = m.f_objetivo * m.capital;
        assert!(
            m.perdida_al_stop() <= tope * (1.0 + TOL),
            "ATR {atr}: pérdida al stop {} > f·C = {tope} (c/SL = {})",
            m.perdida_al_stop(),
            m.friccion / m.stop
        );
    }
}
