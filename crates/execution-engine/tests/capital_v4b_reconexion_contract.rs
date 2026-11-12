//! CAPITAL-UNICO V4B: dictamen de Codex QRA sobre la propuesta de N1
//! (`qra_humo_registry_n1.md`) y contrato de aceptación `qra_capital_v4b.md`.
//!
//! La generación del socket del stream es distinta de la era de la cuenta. La
//! procedencia del stream vale sólo para el socket que la puso; la pérdida y
//! la reapertura del socket vetan las entradas; dentro de un socket `E`
//! desempata dos eventos con la misma `T`; un empate sin orden demostrable
//! (REST contra stream, o tras reconectar) veta y sólo lo resuelve una REST
//! con ticket posterior; una REST de otro socket no suelta el veto.
//!
//! Las dos pruebas de Codex QRA, tal como están escritas, en
//! `capital_v4b_qra_red.rs`. No tocan d*, `capital_del_freno` ni ningún umbral.

use execution_engine::client::BinanceClient;
use execution_engine::puente_cuenta::{
    DatoPosicion, EventoStream, LecturaCuenta, Observacion, PosicionRest, PuenteCuenta, Rechazo,
};
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use std::sync::Arc;

fn k(s: &str) -> (String, String) {
    (s.to_string(), "BOTH".to_string())
}

fn ab(cantidad: f64, upnl: f64) -> DatoPosicion {
    DatoPosicion::Abierta { cantidad, upnl }
}

fn rest(saldo: f64, t: Option<u64>, pos: &[(&str, DatoPosicion, Option<u64>)]) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: t,
        posiciones: Some(
            pos.iter()
                .map(|(s, d, t)| (k(s), PosicionRest { dato: *d, tiempo_ms: *t }))
                .collect(),
        ),
        rechazo_arranque: None,
        upnl_del_activo: None,
    }
}

fn ev(t: u64, saldo: Option<f64>, pos: &[(&str, DatoPosicion)]) -> EventoStream {
    EventoStream {
        tiempo_ms: Some(t),
        saldo,
        posiciones: pos.iter().map(|(s, d)| (k(s), *d)).collect(),
    }
}

/// Evento del socket `socket` con `T = t` y `E = e`.
fn ws(
    p: &PuenteCuenta,
    socket: u64,
    t: u64,
    e: u64,
    saldo: Option<f64>,
    pos: &[(&str, DatoPosicion)],
) -> Result<Option<Observacion>, Rechazo> {
    p.observar_stream_del_socket(p.era(), socket, Some(e), &ev(t, saldo, pos), |_| {})
}

fn saldo(p: &PuenteCuenta) -> f64 {
    p.observacion().unwrap().saldo
}

/// Puente arrancado con una lectura REST aceptada (equity completa, sin veto).
fn arrancado(s: f64, t: u64, pos: &[(&str, DatoPosicion, Option<u64>)]) -> PuenteCuenta {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(s, Some(t), pos)).unwrap();
    assert!(!p.veta_entradas());
    p
}

// ── N1 dentro de un mismo socket: T igual, E distinta ────────────────────

#[test]
fn v4b_mismo_socket_dos_eventos_misma_t_aplica_e_mayor_deduplica_y_descarta_la_regresion() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -10.0), Some(100))]);
    let s = p.socket();
    assert_eq!(ws(&p, s, 200, 201, Some(95.0), &[("BTCUSDT", ab(0.5, -5.0))]).unwrap().unwrap().saldo, 95.0);
    let r = ws(&p, s, 200, 202, Some(89.9), &[("BTCUSDT", DatoPosicion::Cerrada)]);
    assert_eq!(r.unwrap().unwrap(), Observacion { saldo: 89.9, equity: 89.9 }, "E mayor aplica");
    assert_eq!(p.posiciones_en_cache(), 0);
    assert!(!p.veta_entradas());
    // El idéntico (mismas T y E, mismo contenido) se deduplica sin veto.
    assert!(ws(&p, s, 200, 202, Some(89.9), &[("BTCUSDT", DatoPosicion::Cerrada)]).is_ok());
    assert!(!p.veta_entradas());
    // La regresión (misma T, E menor) se descarta: ni rebobina ni reabre ni veta.
    let r = ws(&p, s, 200, 201, Some(95.0), &[("BTCUSDT", ab(0.5, -5.0))]);
    assert!(matches!(r, Err(Rechazo::Atrasada { .. })), "{r:?}");
    assert_eq!(saldo(&p), 89.9);
    assert_eq!(p.posiciones_en_cache(), 0);
    assert!(!p.veta_entradas());
    // El diagnóstico dice fuente, T, E y motivo.
    let d = p.tomar_diagnostico();
    assert_eq!(d.rechazos, vec![("stream", "Atrasada", 1)]);
    assert!(d.ultimo_rechazo.unwrap().starts_with("fuente=stream T=200 E=201 motivo=Atrasada"));
}

#[test]
fn v4b_mismo_socket_tres_eventos_misma_t() {
    let p = arrancado(100.0, 100, &[]);
    let s = p.socket();
    for (e, v) in [(301, 99.0), (302, 98.0), (303, 97.0)] {
        assert_eq!(ws(&p, s, 200, e, Some(v), &[]).unwrap().unwrap().saldo, v);
    }
    assert!(!p.veta_entradas());
    assert!(matches!(ws(&p, s, 200, 302, Some(98.0), &[]), Err(Rechazo::Atrasada { .. })));
    assert!(ws(&p, s, 200, 303, Some(97.0), &[]).is_ok(), "duplicado");
    assert_eq!(saldo(&p), 97.0);
    assert!(!p.veta_entradas());
    // REST con la verdad y la misma marca: coherente.
    assert_eq!(p.observar_rest(p.ticket(), &rest(97.0, Some(200), &[])).unwrap().saldo, 97.0);
    assert!(!p.veta_entradas());
}

#[test]
fn v4b_mismas_marcas_con_otro_contenido_pide_reconciliacion() {
    let p = arrancado(100.0, 100, &[]);
    let s = p.socket();
    ws(&p, s, 200, 201, Some(99.0), &[]).unwrap();
    let antes = p.ticket();
    let r = ws(&p, s, 200, 201, Some(98.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(saldo(&p), 99.0);
    assert!(p.veta_entradas(), "la contradicción veta las entradas");
    // Una foto pedida ANTES de la contradicción no la resuelve ni suelta el veto.
    assert!(matches!(p.observar_rest(antes, &rest(98.0, Some(200), &[])), Err(Rechazo::Contradiccion { .. })));
    assert!(p.observar_rest(antes, &rest(99.0, Some(200), &[])).is_ok());
    assert!(p.veta_entradas(), "la foto previa no suelta el veto");
    assert_eq!(p.tomar_diagnostico().motivo_veto, Some("Contradiccion"));
    // Una REST pedida después la resuelve (aunque la marca sea la misma).
    assert_eq!(p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[])).unwrap().saldo, 98.0);
    assert!(!p.veta_entradas());
}

// ── Reconexión: un socket nuevo no hereda el orden del anterior ──────────

/// A y B: dos actualizaciones con la misma T en el socket viejo; A abre la
/// posición con saldo 100, B la cierra con saldo 0.
fn a_y_b(p: &PuenteCuenta, s: u64) {
    ws(p, s, 200, 201, Some(100.0), &[("VETUSDT", ab(-1.0, -20.0))]).unwrap();
    ws(p, s, 200, 202, Some(0.0), &[("VETUSDT", DatoPosicion::Cerrada)]).unwrap();
    assert_eq!(saldo(p), 0.0);
    assert_eq!(p.posiciones_en_cache(), 0);
}

#[test]
fn v4b_reemision_tras_reconexion_viejo_y_luego_nuevo() {
    let p = arrancado(120.0, 100, &[]);
    let s0 = p.socket();
    a_y_b(&p, s0);
    let s1 = p.abrir_socket(p.era()).unwrap();
    assert_ne!(s1, s0);
    assert!(p.veta_entradas());
    let r = ws(&p, s1, 200, 201, Some(100.0), &[("VETUSDT", ab(-1.0, -20.0))]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert!(ws(&p, s1, 200, 202, Some(0.0), &[("VETUSDT", DatoPosicion::Cerrada)]).is_ok());
    assert_eq!(saldo(&p), 0.0, "el saldo publicado no retrocede");
    assert_eq!(p.posiciones_en_cache(), 0, "la posición cerrada no se reabre");
    assert!(p.veta_entradas());
    let r = p.observar_rest(p.ticket(), &rest(0.0, Some(200), &[("VETUSDT", DatoPosicion::Cerrada, Some(200))]));
    assert_eq!(r.unwrap(), Observacion { saldo: 0.0, equity: 0.0 });
    assert!(!p.veta_entradas());
}

#[test]
fn v4b_reemision_tras_reconexion_nuevo_y_luego_viejo() {
    let p = arrancado(120.0, 100, &[]);
    a_y_b(&p, p.socket());
    let s1 = p.abrir_socket(p.era()).unwrap();
    assert!(ws(&p, s1, 200, 202, Some(0.0), &[("VETUSDT", DatoPosicion::Cerrada)]).is_ok());
    let r = ws(&p, s1, 200, 201, Some(100.0), &[("VETUSDT", ab(-1.0, -20.0))]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(saldo(&p), 0.0);
    assert_eq!(p.posiciones_en_cache(), 0);
    assert!(p.veta_entradas());
    // Sólo la posición (sin fila de saldo): tampoco reabre.
    let r = ws(&p, s1, 200, 201, None, &[("VETUSDT", ab(-1.0, -20.0))]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.posiciones_en_cache(), 0);
    // La foto completa sin la posición confirma el cierre y suelta el veto.
    assert!(p.observar_rest(p.ticket(), &rest(0.0, Some(200), &[])).is_ok());
    assert!(!p.veta_entradas());
}

#[test]
fn v4b_evento_tardio_del_socket_anterior_tras_abrir_el_nuevo() {
    let p = arrancado(100.0, 100, &[]);
    let s0 = p.socket();
    let s1 = p.abrir_socket(p.era()).unwrap();
    // Aunque traiga una T más nueva, no tiene autoridad sobre el socket nuevo.
    let r = ws(&p, s0, 300, 301, Some(5.0), &[]);
    assert_eq!(r, Err(Rechazo::OtroSocket { socket: s0, vigente: s1 }));
    assert_eq!(saldo(&p), 100.0);
    assert!(p.veta_entradas());
    assert_eq!(ws(&p, s1, 300, 301, Some(5.0), &[]).unwrap().unwrap().saldo, 5.0);
}

#[test]
fn v4b_segundo_fill_de_la_misma_t_tras_reconexion_lo_resuelve_una_rest_posterior() {
    // El residual de V4 (veto pegado) queda cerrado: veta y reconcilia.
    let p = arrancado(100.0, 100, &[]);
    ws(&p, p.socket(), 200, 201, Some(99.0), &[]).unwrap();
    let s1 = p.abrir_socket(p.era()).unwrap();
    let tras_abrir = p.ticket();
    let r = ws(&p, s1, 200, 202, Some(98.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "sin orden entre sockets: {r:?}");
    assert_eq!(saldo(&p), 99.0);
    // Una REST pedida antes del empate no lo resuelve por llegada.
    assert!(matches!(p.observar_rest(tras_abrir, &rest(98.0, Some(200), &[])), Err(Rechazo::Contradiccion { .. })));
    assert!(p.veta_entradas());
    // La pedida después sí.
    assert_eq!(p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[])).unwrap().saldo, 98.0);
    assert!(!p.veta_entradas());
}

// ── Pérdida del socket: el veto empieza en el corte ──────────────────────

#[test]
fn v4b_veto_desde_la_perdida_cero_y_cierre_en_el_hueco() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -10.0), Some(100))]);
    let s0 = p.socket();
    p.perdida_socket(p.era()).unwrap();
    assert!(p.veta_entradas(), "el veto se activa al perder el socket, no al reconectar");
    assert_eq!(p.tomar_diagnostico().motivo_veto, Some("PerdidaSocket"));
    // Cero explícito y cierre durante el hueco, por REST: llegan al freno, el veto sigue.
    let hueco = p.ticket();
    let r = p.observar_rest(hueco, &rest(0.0, Some(300), &[("BTCUSDT", DatoPosicion::Cerrada, Some(300))]));
    assert_eq!(r.unwrap(), Observacion { saldo: 0.0, equity: 0.0 });
    assert_eq!(p.equity_del_freno(), Some(0.0));
    assert!(p.veta_entradas(), "una REST del hueco no suelta el veto");
    // Nada del socket perdido entra en el hueco.
    assert!(matches!(ws(&p, s0, 400, 401, Some(50.0), &[]), Err(Rechazo::OtroSocket { .. })));
    // Al abrir el socket nuevo, el ticket del hueco queda viejo.
    p.abrir_socket(p.era()).unwrap();
    assert!(matches!(p.observar_rest(hueco, &rest(0.0, Some(300), &[])), Err(Rechazo::OtroSocket { .. })));
    assert!(p.veta_entradas());
    // La lectura pedida después del anclaje confirma la cuenta quieta (mismo
    // saldo y mismo updateTime): no se compara con la hora local.
    assert!(p.observar_rest(p.ticket(), &rest(0.0, Some(300), &[("BTCUSDT", DatoPosicion::Cerrada, Some(300))])).is_ok());
    assert!(!p.veta_entradas());
    assert_eq!(p.posiciones_en_cache(), 0);
}

#[test]
fn v4b_cero_y_cierre_del_hueco_por_el_socket_nuevo() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -10.0), Some(100))]);
    p.perdida_socket(p.era()).unwrap();
    let s1 = p.abrir_socket(p.era()).unwrap();
    // El socket nuevo trae el cierre y el cero (T más nueva): se aplican, y siguen vetando.
    let r = ws(&p, s1, 300, 301, Some(0.0), &[("BTCUSDT", DatoPosicion::Cerrada)]);
    assert_eq!(r.unwrap().unwrap(), Observacion { saldo: 0.0, equity: 0.0 });
    assert!(p.veta_entradas());
    assert!(p.observar_rest(p.ticket(), &rest(0.0, Some(300), &[])).is_ok());
    assert!(!p.veta_entradas());
}

#[test]
fn v4b_un_corte_durante_la_lectura_invalida_el_anclaje() {
    for perdida in [false, true] {
        let p = arrancado(100.0, 100, &[]);
        p.abrir_socket(p.era()).unwrap();
        let t1 = p.ticket();
        if perdida {
            p.perdida_socket(p.era()).unwrap();
        } else {
            p.abrir_socket(p.era()).unwrap();
        }
        assert!(matches!(p.observar_rest(t1, &rest(100.0, Some(100), &[])), Err(Rechazo::OtroSocket { .. })));
        assert!(p.veta_entradas(), "perdida={perdida}");
        if perdida {
            p.abrir_socket(p.era()).unwrap();
        }
        assert!(p.observar_rest(p.ticket(), &rest(100.0, Some(100), &[])).is_ok());
        assert!(!p.veta_entradas());
    }
}

// ── REST concurrente por componente ──────────────────────────────────────

#[test]
fn v4b_rest_concurrente_por_componente_saldo_y_cada_posicion() {
    let p = arrancado(
        100.0,
        100,
        &[("BTCUSDT", ab(1.0, -10.0), Some(100)), ("ETHUSDT", ab(2.0, -4.0), Some(100))],
    );
    let s = p.socket();
    let pedida = p.ticket();
    // El stream mueve el saldo y BTC antes de que conteste la REST.
    ws(&p, s, 200, 201, Some(90.0), &[("BTCUSDT", ab(1.0, -30.0))]).unwrap();
    assert_eq!(p.observacion().unwrap().equity, 56.0);
    // Saldo: la foto vieja (150) se rechaza entera.
    let r = p.observar_rest(pedida, &rest(100.0, Some(150), &[("ETHUSDT", ab(2.0, -9.0), Some(300))]));
    assert!(matches!(r, Err(Rechazo::Atrasada { ref componente, .. }) if componente == "saldo"), "{r:?}");
    assert_eq!(p.observacion().unwrap().equity, 56.0, "entera o nada: ETH tampoco entra");
    // BTC con la misma marca y la misma cantidad, pedida antes del evento: no pisa
    // el PnL más nuevo; ETH con marca más nueva sí entra.
    let r = p.observar_rest(
        pedida,
        &rest(90.0, Some(200), &[("BTCUSDT", ab(1.0, -10.0), Some(200)), ("ETHUSDT", ab(2.0, -8.0), Some(250))]),
    );
    assert_eq!(r.unwrap().equity, 52.0);
    // BTC con otra cantidad y la misma marca: contradicción, veto.
    let r = p.observar_rest(
        pedida,
        &rest(90.0, Some(200), &[("BTCUSDT", ab(0.5, -15.0), Some(200)), ("ETHUSDT", ab(2.0, -8.0), Some(250))]),
    );
    assert!(matches!(r, Err(Rechazo::Contradiccion { ref componente, .. }) if componente.contains("BTCUSDT")), "{r:?}");
    assert!(p.veta_entradas());
    assert_eq!(p.observacion().unwrap().equity, 52.0);
    // Una REST pedida después la resuelve y suelta el veto.
    let r = p.observar_rest(
        p.ticket(),
        &rest(90.0, Some(200), &[("BTCUSDT", ab(0.5, -15.0), Some(200)), ("ETHUSDT", ab(2.0, -8.0), Some(250))]),
    );
    assert_eq!(r.unwrap().equity, 67.0);
    assert!(!p.veta_entradas());
}

// ── El streamer: E conservada, socket propio, pérdida al salir del lazo ──

struct Nada;
impl AccountSink for Nada {
    fn on_capital(&self, _u: f64) {}
    fn on_positions(&self, _p: &[RemotePosition]) {}
    fn on_bracket_close(&self, _c: &BracketClose) {}
}

fn cuenta_ws(t: u64, e: u64, wb: &str) -> String {
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{e},"T":{t},"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{wb}","cw":"{wb}","bc":"0"}}],"P":[]}}}}"#)
}

#[test]
fn v4b_el_streamer_conserva_e_y_desempata_con_ella() {
    let p = Arc::new(arrancado(100.0, 100, &[]));
    let s = UserDataStreamer::new(BinanceClient::new("k".into(), true), Arc::new(OrderRegistry::new()))
        .with_sink(Arc::new(Nada))
        .with_puente(Arc::clone(&p));
    s.dispatch(cuenta_ws(200, 201, "99").as_bytes());
    s.dispatch(cuenta_ws(200, 202, "98").as_bytes());
    assert_eq!(saldo(&p), 98.0);
    s.dispatch(cuenta_ws(200, 201, "99").as_bytes());
    assert_eq!(saldo(&p), 98.0, "la regresión de E se descarta");
    assert!(!p.veta_entradas());
    let d = p.tomar_diagnostico();
    assert!(d.ultimo_rechazo.unwrap().contains("T=200 E=201 motivo=Atrasada"));
}

#[test]
fn v4b_contrato_de_fuente_del_streamer() {
    let src = include_str!("../src/user_data_stream.rs").replace("\r\n", "\n");
    // El veto empieza al perder el socket: antes de la pausa de reconexión.
    let lazo = &src[src.find("while let Some(msg) = read.next().await").unwrap()..];
    let perdida = lazo.find("self.puente.perdida_socket(self.era)").expect("pérdida al salir del lazo");
    let pausa = lazo.find("Desconectado. Reanudando").unwrap();
    assert!(perdida < pausa, "la pérdida se marca antes de la pausa");
    // La reconexión abre un socket nuevo y el streamer lo adopta.
    let rec = &src[src.find("fn reconcile_after_reconnect").unwrap()..];
    let rec = &rec[..rec.find("pub fn dispatch").unwrap()];
    assert!(rec.contains("self.puente.reconexion(self.era)"));
    assert!(rec.contains("self.socket.store(self.puente.socket()"));
    // T y E se conservan; el evento va con su socket.
    assert!(src.contains("let hora_evento = ev.event_time"));
    assert!(src.contains("self.puente.observar_stream_del_socket(self.era, socket, hora_evento, &evento"));
    // Un evento ilegible no se confunde con un corte.
    assert!(src.contains("self.puente.evidencia_perdida(self.era)"));
    assert!(!src.contains("eprintln!(\"⚠️ [CAPITAL] ACCOUNT_UPDATE no aplicado"));
}
