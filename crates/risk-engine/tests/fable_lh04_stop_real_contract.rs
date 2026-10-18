//! F-A5 (Fable, 2026-10-17; hallazgo F1-013) — el dimensionado en espacio de
//! riesgo LH-04 convierte la fracción f a nocional con el stop QUE LA ORDEN
//! LLEVA, no con la curva genómica `sl_at_tau`.
//!
//! Teoría que se prueba. LH-04 define el tamaño por el riesgo al stop:
//! nocional = f·C/SL, de modo que si el stop se toca se pierde exactamente
//! f·C (hasta el tope de ruina). Hasta F-A5 el divisor era `sl_at_tau(τ)`,
//! una curva del genoma sin relación con el stop ATR-difusivo que la orden
//! lleva (`expected_loss` = k·ATR·(τ/τ_ref)^H), así que el riesgo realizado
//! era f·C·(stop real / stop genómico): con ATR alto se arriesgaba varias
//! veces f y con ATR bajo una fracción. El contrato: (a) el riesgo realizado
//! al stop (nocional · stop real / capital asignado) es f cuando el tope de
//! ruina no muerde, y nunca mayor que f; (b) es invariante al ATR (el
//! nocional se adapta al stop) mientras el stop sí crece con el ATR; (c) la
//! telemetría publica objetivo, realizado, capital y stop, y coinciden con la
//! orden emitida.
//!
//! Fixture (familia de `lh04_sizing_riesgo_contract.rs`): 1 000 USD, precio
//! 100, 100 cierres con p̂ = 0,55 y PF = 1,45 ⇒ b = 1,186, Wilson-LCB(1,64)
//! ≈ 0,469, f = ½·max(0, 0,469 − 0,531/1,186) ≈ 1,1 %. Horizonte 1 h con
//! k = 1 y H = 0,5: stop = ATR%·√60 ≈ 7,7 × ATR%, de modo que el nocional
//! objetivo (f·C/stop ≈ 140 USD con ATR 1 %) cabe en el tope de ruina (25 %
//! del capital) con CUALQUIER apalancamiento ≥ 1: el tope no muerde y el
//! riesgo realizado debe ser exactamente f.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex, Once};

static SERIAL: Mutex<()> = Mutex::new(());

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
            taker_fee: 0.0004,
            is_shadow: false,
        }])
    });
    let a = GlobalArena::build_in_own_stack(1_000.0);
    a.coins[0].current_price.store(100.0, Relaxed);
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
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
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
    /// nocional·stop/C de la orden EMITIDA.
    f_emitido: f64,
    stop: f64,
    capital: f64,
    margen: f64,
    /// La orden sufrió el recorte de margen del tope de exposición (Sol A2):
    /// entonces lo emitido queda por debajo de lo publicado por LH-04.
    recortada: bool,
}

fn medir(atr: f64) -> Medida {
    let a = arena(atr);
    let recortes_antes = a.registry.get_for_coin_or(0, "sol_a2_margen_reducido", 0.0);
    let o = RiskEngine::new(1_000.0).evaluate_quantum_order(0, &intencion(), &a);
    assert_eq!(o.signal, SignalType::Long, "ATR {atr}: {}", risk_engine::reject_report());
    let f_objetivo = a.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let f_realizado = a.registry.get_for_coin_or(0, "lh04_f_realizado", f64::NAN);
    let capital = a.registry.get_for_coin_or(0, "lh04_capital_asignado", f64::NAN);
    let stop_pub = a.registry.get_for_coin_or(0, "lh04_stop_real", f64::NAN);
    let recortes = a.registry.get_for_coin_or(0, "sol_a2_margen_reducido", 0.0);
    assert!(capital.is_finite() && capital > 0.0, "capital asignado publicado: {capital}");
    let stop = (100.0 - o.sl_target) / 100.0;
    assert!(
        (stop - stop_pub).abs() < 1e-9,
        "el stop publicado es el de la orden: {stop_pub} vs {stop}"
    );
    let nocional = o.volume_usd * o.leverage;
    Medida {
        f_objetivo,
        f_realizado,
        f_emitido: nocional * stop / capital,
        stop,
        capital,
        margen: o.volume_usd,
        recortada: recortes > recortes_antes,
    }
}

#[test]
fn con_evidencia_moderada_el_riesgo_realizado_es_f_y_la_orden_lo_lleva() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let m = medir(1.0);
    assert!(
        m.f_objetivo.is_finite() && m.f_objetivo > 0.0 && m.f_objetivo < 0.05,
        "f moderada (≈1,1 %): {}",
        m.f_objetivo
    );
    assert!(m.stop > 0.03 && m.stop < 0.30, "stop ATR-difusivo a 1 h en fracción: {}", m.stop);
    assert!(
        m.margen < 0.9 * 0.25 * m.capital,
        "el tope de ruina (25 % del capital asignado) no muerde en este fixture: margen {}, capital {}",
        m.margen,
        m.capital
    );
    assert!(
        (m.f_realizado - m.f_objetivo).abs() < 1e-9,
        "sin tope, el riesgo realizado publicado es f: {} vs {}",
        m.f_realizado,
        m.f_objetivo
    );
    assert!(!m.recortada, "con ≈14 % del capital en margen no hay recorte de exposición");
    // Lo emitido coincide con lo publicado salvo la cuantización entera del
    // apalancamiento y el redondeo al lote (±5 %).
    assert!(
        (m.f_emitido - m.f_objetivo).abs() <= 0.05 * m.f_objetivo,
        "nocional·stop/C = f ± 5 %: emitido {}, f {}",
        m.f_emitido,
        m.f_objetivo
    );
}

#[test]
fn el_riesgo_realizado_no_depende_del_atr_aunque_el_stop_si() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // Con el divisor genómico el riesgo realizado escalaba con el ATR (×3 al
    // triplicar el ATR). Con el stop real, el nocional se adapta al stop.
    let a = medir(1.0);
    let b = medir(3.0);
    assert!((a.f_objetivo - b.f_objetivo).abs() < 1e-12, "la misma evidencia da la misma f");
    assert!(
        b.stop > 2.5 * a.stop && b.stop < 3.5 * a.stop,
        "el stop ATR-difusivo escala con el ATR: {} vs {}",
        a.stop,
        b.stop
    );
    assert!(
        (a.f_realizado - a.f_objetivo).abs() < 1e-9 && (b.f_realizado - b.f_objetivo).abs() < 1e-9,
        "sin tope, realizado = f en los dos: {} / {} vs {}",
        a.f_realizado,
        b.f_realizado,
        a.f_objetivo
    );
    assert!(!a.recortada && !b.recortada);
    assert!(
        (a.f_emitido - b.f_emitido).abs() <= 0.08 * a.f_objetivo,
        "el riesgo emitido es invariante al ATR (±8 % por cuantización): {} vs {} (f {})",
        a.f_emitido,
        b.f_emitido,
        a.f_objetivo
    );
}

#[test]
fn el_riesgo_realizado_nunca_supera_el_objetivo() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // Con ATR 0,5 el nocional objetivo (≈280 USD) puede exceder el tope de
    // ruina a apalancamiento bajo: entonces el realizado queda por debajo de
    // f y así se publica; jamás por encima.
    for atr in [0.5, 1.0, 2.0, 3.0] {
        let m = medir(atr);
        assert!(
            m.f_realizado <= m.f_objetivo + 1e-12,
            "ATR {atr}: realizado {} > objetivo {}",
            m.f_realizado,
            m.f_objetivo
        );
        assert!(
            m.f_emitido <= m.f_objetivo * 1.05 + 1e-12,
            "ATR {atr}: emitido {} > objetivo {}",
            m.f_emitido,
            m.f_objetivo
        );
        if !m.recortada {
            assert!(
                (m.f_emitido - m.f_realizado).abs() <= 0.05 * m.f_realizado,
                "ATR {atr}: emitido {} = realizado publicado {} ± 5 %",
                m.f_emitido,
                m.f_realizado
            );
        }
    }
}
