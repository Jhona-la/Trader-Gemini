//! CAPITAL-UNICO (prioridad 1 del mapa de pérdidas, 2026-11-11) — la RED.
//!
//! Corridas Demo a3d/a3f/a3g: REST (`/fapi/v2/account`) daba
//! `totalWalletBalance` = 7 299,88 (todos los activos de la Testnet) y el
//! user-data stream publicaba sólo el saldo de USDT = 1 655,81. El freno del
//! inmune mide `min(arena, exchange)` contra el pico de arranque: una caída
//! falsa del 77,4 % ≥ d* armaba el kill-switch y aplanaba a los 3-6 s.
//!
//! Este archivo sólo usa la API que ya existía en `main` (`parse_account_equity`
//! y `UserDataStreamer`), para que la RED se reproduzca con
//! `git restore --source=main` de las fuentes de producción (revisión
//! independiente, punto 5). Las dos primeras fallan en main por la razón
//! correcta; la guarda pasa en los dos.

use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::parse_account_equity;
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use quantum_arena::GlobalArena;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

/// `/fapi/v2/account` de una cuenta Testnet con varios activos (USDT, USDC, BTC).
const CUENTA_MULTI: &str = r#"{
  "feeTier":0,"canTrade":true,"canDeposit":true,"canWithdraw":true,"updateTime":0,
  "multiAssetsMargin":false,"tradeGroupId":-1,
  "totalInitialMargin":"0.00000000","totalMaintMargin":"0.00000000",
  "totalWalletBalance":"7299.88000000","totalUnrealizedProfit":"0.00000000",
  "totalMarginBalance":"7299.88000000","availableBalance":"7299.88000000",
  "maxWithdrawAmount":"7299.88000000",
  "assets":[
    {"asset":"USDT","walletBalance":"1655.81000000","unrealizedProfit":"0.00000000","marginBalance":"1655.81000000","availableBalance":"1655.81000000"},
    {"asset":"USDC","walletBalance":"5000.00000000","unrealizedProfit":"0.00000000","marginBalance":"5000.00000000","availableBalance":"5000.00000000"},
    {"asset":"BTC","walletBalance":"0.01000000","unrealizedProfit":"0.00000000","marginBalance":"0.01000000","availableBalance":"0.01000000"}
  ],
  "positions":[]
}"#;

fn account_update(usdt_wb: f64) -> String {
    format!(
        r#"{{"e":"ACCOUNT_UPDATE","E":1789000000000,"T":1789000000000,"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{usdt_wb:.8}","cw":"{usdt_wb:.8}","bc":"0"}},{{"a":"USDC","wb":"5000.00000000","cw":"5000.00000000","bc":"0"}}],"P":[]}}}}"#
    )
}

struct Captura(AtomicU64);
impl AccountSink for Captura {
    fn on_capital(&self, usdt: f64) {
        self.0.store(usdt.to_bits(), Relaxed);
    }
    fn on_positions(&self, _p: &[RemotePosition]) {}
    fn on_bracket_close(&self, _c: &BracketClose) {}
}

/// El capital que el stream publica por `on_capital` para un ACCOUNT_UPDATE.
fn capital_del_stream(evento: &str) -> f64 {
    let sink = Arc::new(Captura(AtomicU64::new(f64::NAN.to_bits())));
    let streamer = UserDataStreamer::new(
        BinanceClient::new("k".into(), true),
        Arc::new(OrderRegistry::new()),
    )
    .with_sink(sink.clone());
    streamer.dispatch(evento.as_bytes());
    f64::from_bits(sink.0.load(Relaxed))
}

#[test]
fn capital_unico_rest_y_stream_miden_el_mismo_capital() {
    let rest = parse_account_equity(CUENTA_MULTI).expect("REST válido");
    let stream = capital_del_stream(&account_update(1655.81));
    assert!(
        (rest - stream).abs() <= 0.01,
        "REST ({rest}) y stream ({stream}) deben medir el mismo capital"
    );
    assert!((rest - 1655.81).abs() <= 0.01, "el capital es el saldo de USDT: {rest}");
}

/// Freno con los dos planos como los cablea el host (god_engine.rs, lazo
/// inmune): arena = REST del arranque, exchange = stream, pico = arranque.
fn freno_tras_el_evento(evento: &str) -> risk_engine::drawdown::FrenoDuro {
    let rest = parse_account_equity(CUENTA_MULTI).expect("REST válido");
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(rest);
    arena.unified_capital.store(rest, Relaxed);
    let pico = risk_engine::drawdown::registrar_pico(&arena.pico_capital, rest);
    arena.capital_exchange.store(capital_del_stream(evento), Relaxed);
    let cap = risk_engine::drawdown::capital_del_freno(&arena);
    let pico = risk_engine::drawdown::registrar_pico(&arena.pico_capital, cap).max(pico);
    risk_engine::drawdown::freno_duro(&arena, pico, cap)
}

#[test]
fn capital_unico_sin_perdida_real_el_freno_no_dispara() {
    let freno = freno_tras_el_evento(&account_update(1655.81));
    assert!(
        !freno.echado(),
        "un ACCOUNT_UPDATE sin pérdida real no puede disparar el freno: {freno}"
    );
}

/// Guarda: el freno sigue vivo. Lejos de la frontera de d* (revisión,
/// punto 5): una caída real del 55 % dispara y una del 45 % no.
#[test]
fn capital_unico_con_caida_real_del_55_el_freno_dispara_y_del_45_no() {
    let freno = freno_tras_el_evento(&account_update(1655.81 * 0.45));
    assert!(
        freno.por_capital(),
        "una caída real del 55 % del mismo capital debe disparar el freno: {freno}"
    );
    let freno = freno_tras_el_evento(&account_update(1655.81 * 0.55));
    assert!(
        !freno.echado(),
        "una caída real del 45 % no llega a d*: {freno}"
    );
}
