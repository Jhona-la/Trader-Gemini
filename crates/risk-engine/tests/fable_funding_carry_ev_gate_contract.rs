//! F-A1 (Fable, 2026-10-17) — el coste de carry esperado del funding entra en
//! la barrera de expectativa del risk-engine.
//!
//! Teoría que se prueba. La barrera D-751 exige EV > fricción·k. Hasta F-A1
//! la fricción era sólo la de ejecución (2·taker + 2·deslizamiento) y el
//! funding per-símbolo no llegaba a ninguna decisión. Ahora la fricción es
//! `fricción_ejecución + E[carry]`, con E[carry] = d·f·τ/T_F (d = +1 largo,
//! −1 corto; f tasa prevista; T_F = 8 h), cobrando sólo el coste: un largo
//! con funding positivo paga, un corto con el mismo funding no cobra el
//! ingreso previsto, un dato caducado no cuenta y sin poller (replay) nada
//! cambia respecto al comportamiento anterior.
//!
//! Fixture de `cf3_confianza_tamano_contract.rs` (capital 1 000, moneda con
//! historial para salir de la fase de sonda), horizonte 1 h y ATR 0,2 % como
//! en `geometry_hurst_contract.rs`.
//!
//! El reloj del risk-engine es el del registro omnisciente (el del último
//! evento procesado); cada prueba lo fija y las pruebas se serializan.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{RiskEngine, ValidatedOrder, REJ_EV};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex, Once};

const AHORA_MS: u64 = 1_700_000_000_000;
const UNA_HORA_MS: u64 = 3_600_000;

static SERIAL: Mutex<()> = Mutex::new(());

fn arena() -> Arc<GlobalArena> {
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
    a.coins[0].current_atr.store(0.2, Relaxed);
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    a.coins[0].hurst_scale_matched.store(0.5, Relaxed);
    // CL-98 (D4): con historial la moneda sale de la fase de sonda y la
    // barrera de EV completa (D-751) es la que decide.
    let m = &a.coins[0].metrics;
    m.trade_count.store(100, Relaxed);
    m.win_rate.store(0.75, Relaxed);
    m.gross_wins.store(200.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    m.profit_factor.store(2.0, Relaxed);
    m.kelly_fraction.store(0.15, Relaxed);
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    a.config.min_confidence_btc.store(0.70, Relaxed);
    a.registry.update_event_time(AHORA_MS as i64);
    a
}

fn intencion(signal: SignalType, p: f64) -> SignalIntent {
    SignalIntent {
        signal,
        confidence: 0.9,
        win_probability: p,
        expected_duration_ms: UNA_HORA_MS,
        ..Default::default()
    }
}

fn evaluar(a: &GlobalArena, signal: SignalType, p: f64) -> ValidatedOrder {
    RiskEngine::new(1_000.0).evaluate_quantum_order(0, &intencion(signal, p), a)
}

/// Menor probabilidad calibrada (malla de 1e-3) con la que la orden sale.
/// La malla arranca en 0,30: con TP ≈ 2·SL el punto de equilibrio de la
/// barrera D-751 es p* = (SL + f)/(TP + SL) ≈ 0,35, por debajo de 0,50.
fn p_minima_admitida(a: &GlobalArena, signal: SignalType) -> Option<f64> {
    (300..=990)
        .map(|k| k as f64 / 1000.0)
        .find(|&p| evaluar(a, signal, p).signal == signal)
}

#[test]
fn sin_funding_la_orden_sale_como_antes() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena();
    let o = evaluar(&a, SignalType::Long, 0.9);
    assert_eq!(o.signal, SignalType::Long, "{}", risk_engine::reject_report());
}

#[test]
fn un_largo_con_funding_positivo_extremo_se_veta_por_ev() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena();
    // f = 1,0 por asentamiento (tope del multiplexer) sobre τ = 1 h:
    // E[carry] = 1,0 · 1/8 = 12,5 % del nocional. Ninguna geometría de 1 h
    // con ATR 0,2 % paga eso.
    a.coins[0].publish_funding_next(1.0, AHORA_MS);
    let o = evaluar(&a, SignalType::Long, 0.9);
    assert_eq!(o.signal, SignalType::Flat, "debería vetarse: {}", risk_engine::reject_report());
    assert_eq!(
        risk_engine::ultimo_rechazo(),
        Some(REJ_EV),
        "el veto es por expectativa, no por otra puerta: {}",
        risk_engine::reject_report()
    );
}

#[test]
fn un_corto_con_el_mismo_funding_positivo_no_paga_ni_cobra() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let sin = arena();
    let base = evaluar(&sin, SignalType::Short, 0.9);
    assert_eq!(base.signal, SignalType::Short, "{}", risk_engine::reject_report());
    let con = arena();
    con.coins[0].publish_funding_next(1.0, AHORA_MS);
    let o = evaluar(&con, SignalType::Short, 0.9);
    assert_eq!(o.signal, SignalType::Short, "{}", risk_engine::reject_report());
    // El ingreso previsto se acota en 0: la orden es idéntica a la de sin funding.
    assert_eq!(o.volume_usd, base.volume_usd);
    assert_eq!(o.leverage, base.leverage);
    assert_eq!(o.tp_target, base.tp_target);
    assert_eq!(o.sl_target, base.sl_target);
}

#[test]
fn un_largo_con_funding_negativo_no_cobra_el_ingreso_previsto() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let sin = arena();
    let base = evaluar(&sin, SignalType::Long, 0.9);
    let con = arena();
    con.coins[0].publish_funding_next(-1.0, AHORA_MS);
    let o = evaluar(&con, SignalType::Long, 0.9);
    assert_eq!(o.signal, SignalType::Long, "{}", risk_engine::reject_report());
    assert_eq!(o.volume_usd, base.volume_usd);
    assert_eq!(o.leverage, base.leverage);
    assert_eq!(o.tp_target, base.tp_target);
    assert_eq!(o.sl_target, base.sl_target);
}

#[test]
fn un_funding_caducado_no_cuenta() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena();
    // Publicado hace más del TTL (3 ciclos del poller = 360 s).
    a.coins[0].publish_funding_next(1.0, AHORA_MS - 400_000);
    let o = evaluar(&a, SignalType::Long, 0.9);
    assert_eq!(
        o.signal,
        SignalType::Long,
        "un dato viejo es «sin dato», no un cobro: {}",
        risk_engine::reject_report()
    );
}

#[test]
fn en_replay_sin_poller_nada_cambia() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena();
    // El replay lleva el reloj del registro a 2024 y el poller no corre:
    // una publicación con marca de hoy no es dato para ese instante.
    a.coins[0].publish_funding_next(1.0, AHORA_MS);
    a.registry.update_event_time(1_700_000_000_000 - 365 * 24 * 3_600_000);
    let o = evaluar(&a, SignalType::Long, 0.9);
    assert_eq!(o.signal, SignalType::Long, "{}", risk_engine::reject_report());
}

#[test]
fn con_un_funding_realista_la_barrera_de_ev_sube_de_forma_monotona() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // f = 0,20 % por asentamiento (zona de squeeze, observada en Binance),
    // τ = 1 h ⇒ E[carry] = 0,025 % del nocional, frente a una fricción de
    // ejecución de 0,082 %: la barrera D-751 sube un 30 % y p* se desplaza
    // ≈ 0,025 %·k/(TP + SL) ≈ 6 pasos de la malla. La probabilidad mínima
    // admitida de un largo debe subir; la de un corto no.
    let sin = arena();
    let p0_largo = p_minima_admitida(&sin, SignalType::Long).expect("algún largo sale sin funding");
    let p0_corto = p_minima_admitida(&sin, SignalType::Short).expect("algún corto sale sin funding");
    assert!(p0_largo > 0.30, "la malla debe cortar por debajo del mínimo admitido: {p0_largo}");
    let con = arena();
    con.coins[0].publish_funding_next(0.002, AHORA_MS);
    let p1_largo = p_minima_admitida(&con, SignalType::Long).expect("algún largo sale con funding");
    let p1_corto = p_minima_admitida(&con, SignalType::Short).expect("algún corto sale con funding");
    assert!(
        p1_largo > p0_largo,
        "el carry debe exigir más probabilidad al largo: sin {p0_largo} con {p1_largo}"
    );
    assert_eq!(p1_corto, p0_corto, "el corto no cobra el ingreso previsto");
    // Y justo en la frontera anterior, el veto es por expectativa.
    let o = evaluar(&con, SignalType::Long, p0_largo);
    assert_eq!(o.signal, SignalType::Flat, "{}", risk_engine::reject_report());
    assert_eq!(risk_engine::ultimo_rechazo(), Some(REJ_EV), "{}", risk_engine::reject_report());
}
