//! CAPITAL-UNICO V4D: pruebas del integrador para los puntos de la lectura
//! estatica de Codex QRA sobre V4B (`qra_capital_v4b_lectura.md`) que V4C no
//! cubria, y para el pedido V4D del coordinador. Las RED de Codex QRA, tal como
//! estan escritas, en `capital_v4d_qra_red.rs`.
//!
//! P1-a: primer socket por el arranque real del streamer. P1-b: los dos ceros.
//! P2 (ii) a (v): E ausente o cero, GET en vuelo con dos productores, cierre y
//! parser por el JSON real del streamer y de REST, variantes de F3.
//!
//! No tocan d*, `capital_del_freno` ni ningun umbral.

use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::parse_account_lectura;
use execution_engine::puente_cuenta::{
    DatoPosicion, EventoStream, LecturaCuenta, PosicionRest, PuenteCuenta, Rechazo,
};
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use std::collections::HashMap;
use std::sync::Arc;

struct Nada;
impl AccountSink for Nada {
    fn on_capital(&self, _u: f64) {}
    fn on_positions(&self, _p: &[RemotePosition]) {}
    fn on_bracket_close(&self, _c: &BracketClose) {}
}

fn streamer(p: &Arc<PuenteCuenta>) -> UserDataStreamer {
    UserDataStreamer::new(BinanceClient::new("k".into(), true), Arc::new(OrderRegistry::new()))
        .with_sink(Arc::new(Nada))
        .with_puente(Arc::clone(p))
}

fn btc() -> (String, String) {
    ("BTCUSDT".to_string(), "BOTH".to_string())
}

fn eth() -> (String, String) {
    ("ETHUSDT".to_string(), "BOTH".to_string())
}

fn abierta(cantidad: f64, upnl: f64) -> DatoPosicion {
    DatoPosicion::Abierta { cantidad, upnl }
}

/// Lectura REST a mano: saldo con su marca y posiciones con la suya.
fn rest(saldo: f64, t: u64, pos: &[((String, String), DatoPosicion, Option<u64>)]) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(
            pos.iter()
                .map(|(k, d, tp)| (k.clone(), PosicionRest { dato: *d, tiempo_ms: *tp }))
                .collect::<HashMap<_, _>>(),
        ),
        rechazo_arranque: None,
        upnl_del_activo: None,
    }
}

/// `ACCOUNT_UPDATE` por el camino real del streamer. `e`: `None` omite `E`.
fn cuenta_ws(t: u64, e: Option<u64>, wb: Option<&str>, pos: &[(&str, &str, &str)]) -> String {
    let b = wb.map_or(String::new(), |w| format!(r#"{{"a":"USDT","wb":"{w}","cw":"{w}","bc":"0"}}"#));
    let p: Vec<String> = pos
        .iter()
        .map(|(s, pa, up)| format!(r#"{{"s":"{s}","pa":"{pa}","ep":"60000","up":"{up}","iw":"0","ps":"BOTH"}}"#))
        .collect();
    let e = e.map_or(String::new(), |e| format!(r#""E":{e},"#));
    format!(r#"{{"e":"ACCOUNT_UPDATE",{e}"T":{t},"a":{{"m":"ORDER","B":[{b}],"P":[{}]}}}}"#, p.join(","))
}

/// `/fapi/v2/account` por el parser real. Filas: (simbolo, positionAmt,
/// unrealizedProfit, updateTime como texto JSON o `None` para omitirlo).
fn cuenta_rest(wb: &str, t: u64, filas: &[(&str, &str, &str, Option<&str>)]) -> String {
    let p: Vec<String> = filas
        .iter()
        .map(|(s, pa, up, ut)| {
            let ut = ut.map_or(String::new(), |u| format!(r#","updateTime":{u}"#));
            format!(r#"{{"symbol":"{s}","positionSide":"BOTH","positionAmt":"{pa}","unrealizedProfit":"{up}"{ut}}}"#)
        })
        .collect();
    format!(
        r#"{{"totalWalletBalance":"{wb}","multiAssetsMargin":false,"assets":[{{"asset":"USDT","walletBalance":"{wb}","unrealizedProfit":"0","updateTime":{t}}}],"positions":[{}]}}"#,
        p.join(",")
    )
}

fn lectura_json(json: &str) -> LecturaCuenta {
    parse_account_lectura(json).expect("JSON de prueba valido").1
}

/// Arranque como en produccion: REST del arranque, el streamer arranca sin
/// socket, abre el primero y el resincronizador pide una REST posterior.
fn arrancado(pos: &[((String, String), DatoPosicion, Option<u64>)]) -> (Arc<PuenteCuenta>, UserDataStreamer) {
    let p = Arc::new(PuenteCuenta::new());
    p.observar_rest(p.ticket(), &rest(100.0, 100, pos)).expect("REST del arranque");
    let s = streamer(&p);
    s.sin_socket_todavia();
    s.primer_socket_abierto();
    p.observar_rest(p.ticket(), &rest(100.0, 100, pos)).expect("REST tras el primer socket");
    assert!(!p.veta_entradas());
    (p, s)
}

fn saldo(p: &PuenteCuenta) -> Option<f64> {
    p.observacion().map(|o| o.saldo)
}

// ---- P1-a: primer socket ---------------------------------------------------

#[test]
fn v4d_p1a_rest_de_arranque_sin_socket_veta_y_llega_al_freno() {
    let p = Arc::new(PuenteCuenta::new());
    let s = streamer(&p);
    s.sin_socket_todavia();
    let obs = p.observar_rest(p.ticket(), &rest(100.0, 100, &[])).expect("se aplica al freno");
    assert_eq!(obs.saldo, 100.0);
    assert_eq!(p.equity_del_freno(), Some(100.0), "la REST del hueco llega al freno");
    assert!(p.veta_entradas(), "sin primer socket no hay entradas nuevas");
    assert!(p.necesita_resincronizar(), "el resincronizador sigue pidiendo");
    assert_eq!(p.tomar_diagnostico().motivo_veto, Some("SinSocket"));
}

#[test]
fn v4d_p1a_foto_previa_a_la_apertura_mantiene_el_veto_y_la_posterior_admite() {
    let p = Arc::new(PuenteCuenta::new());
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[])).unwrap();
    let s = streamer(&p);
    s.sin_socket_todavia();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[])).unwrap();
    let previa = p.ticket();
    s.primer_socket_abierto();
    assert!(matches!(p.observar_rest(previa, &rest(100.0, 100, &[])), Err(Rechazo::OtroSocket { .. })));
    assert!(p.veta_entradas(), "la foto pedida antes de abrir el socket no suelta el veto");
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[])).unwrap();
    assert!(!p.veta_entradas(), "la foto completa posterior admite");
    assert_eq!(saldo(&p), Some(100.0), "con el saldo vigente");
}

#[test]
fn v4d_p1a_ws_retenido_o_fallido_mantiene_el_veto() {
    let p = Arc::new(PuenteCuenta::new());
    let s = streamer(&p);
    s.sin_socket_todavia();
    // Handshake retenido o fallido: nunca llega `primer_socket_abierto`.
    for _ in 0..3 {
        p.observar_rest(p.ticket(), &rest(100.0, 100, &[])).unwrap();
        assert!(p.veta_entradas());
    }
    // Sin socket no hay eventos del stream con autoridad.
    s.dispatch(cuenta_ws(200, Some(200), Some("50"), &[]).as_bytes());
    assert_eq!(saldo(&p), Some(100.0));
    assert!(p.veta_entradas());
}

#[test]
fn v4d_p1a_era_nueva_de_la_transicion_espera_su_primer_socket() {
    let (p, _demo) = arrancado(&[]);
    let era = p.nueva_era();
    let mainnet = streamer(&p);
    mainnet.sin_socket_todavia();
    p.observar_rest(p.ticket_de_era(era).unwrap(), &rest(80.0, 300, &[])).unwrap();
    assert_eq!(p.equity_del_freno(), Some(80.0));
    assert!(p.veta_entradas(), "la REST de la transicion sin socket de mainnet no admite");
    mainnet.primer_socket_abierto();
    p.observar_rest(p.ticket_de_era(era).unwrap(), &rest(80.0, 300, &[])).unwrap();
    assert!(!p.veta_entradas());
}

#[test]
fn v4d_p1a_fuente_start_marca_sin_socket_antes_de_esperar_y_el_handshake_lo_abre() {
    let src = include_str!("../src/user_data_stream.rs").replace("\r\n", "\n");
    let start = &src[src.find("pub async fn start(&self)").unwrap()..];
    let marca = start.find("self.sin_socket_todavia();").expect("start marca sin socket");
    let espera = start.find(".await").unwrap();
    assert!(marca < espera, "antes de la primera espera");
    let rama = &start[start.find("if connected_once {").unwrap()..];
    let rama = &rama[..rama.find("connected_once = true;").unwrap()];
    assert!(rama.contains("self.reconcile_after_reconnect();"));
    assert!(rama.contains("self.primer_socket_abierto();"), "el primer handshake abre el socket en el puente");
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert_eq!(host.matches("streamer.start().await").count(), 2, "demo y mainnet arrancan por start()");
}

#[test]
fn v4d_p1a_y_p1b_los_cierres_no_leen_el_veto() {
    let oms = include_str!("../../../src/host_oms.rs").replace("\r\n", "\n");
    assert_eq!(oms.matches("capital_incompleto").count(), 1, "solo una lectura del veto en el OMS");
    let entrada = &oms[oms.find("pub fn check_entry(").unwrap()..];
    let entrada = &entrada[..entrada.find("pub fn begin_refresh(").unwrap()];
    assert!(entrada.contains("capital_incompleto"), "y es la de la admision de entradas");
}

/// Deuda documentada (P1-a, socket abierto y mudo): `user_data_stream.rs`
/// solo marca la perdida cuando sale el lector (`while let Some(msg) =
/// read.next().await` y `perdida_socket` despues). No hay senal de vida
/// privada con plazo en el codigo (el keepalive del listenKey es un PUT por
/// REST, no prueba que el socket entregue) y el pedido prohibe inventar uno.
#[test]
#[ignore = "V4D: DEUDA. Socket abierto y mudo: sin senal de vida privada con plazo existente en el codigo; no se inventa"]
fn v4d_p1a_deuda_socket_abierto_y_mudo_deberia_vetar() {
    let (p, _s) = arrancado(&[]);
    // Aqui pasaria el tiempo sin ningun mensaje del socket privado.
    assert!(p.veta_entradas(), "un socket mudo deberia contar como perdido");
}

// ---- P1-b: los dos ceros ------------------------------------------------------

#[test]
fn v4d_p1b_cero_de_cuenta_vacia_llega_al_freno_por_rest_y_por_stream() {
    let (p, s) = arrancado(&[]);
    let vacia = lectura_json(&cuenta_rest("0", 200, &[]));
    assert!(vacia.rechazo_arranque.is_none(), "cuenta vacia de verdad: no es 3a");
    p.observar_rest(p.ticket(), &vacia).unwrap();
    assert_eq!(p.equity_del_freno(), Some(0.0), "el cero de REST llega al freno");
    s.dispatch(cuenta_ws(300, Some(300), Some("50"), &[]).as_bytes());
    s.dispatch(cuenta_ws(400, Some(400), Some("0"), &[]).as_bytes());
    assert_eq!(p.equity_del_freno(), Some(0.0), "el cero del stream llega al freno");
    assert!(!p.necesita_resincronizar(), "un cero completo no pide REST");
}

#[test]
fn v4d_p1b_cero_3a_por_el_parser_real_retiene_el_veto_hasta_un_saldo_positivo() {
    let (p, s) = arrancado(&[]);
    let json = r#"{"totalWalletBalance":"50","multiAssetsMargin":true,"assets":[{"asset":"USDT","walletBalance":"0","unrealizedProfit":"0","updateTime":300}],"positions":[]}"#;
    let l = lectura_json(json);
    assert!(l.rechazo_arranque.is_some(), "el parser clasifica 3a");
    p.observar_rest(p.ticket(), &l).unwrap();
    assert_eq!(p.equity_del_freno(), Some(0.0), "el cero de 3a llega al freno");
    assert!(p.veta_entradas(), "3a conserva el veto de admision");
    assert!(!p.necesita_resincronizar(), "3a es un dato completo: no martillea REST");
    assert_eq!(p.tomar_diagnostico().motivo_veto, Some("RechazoArranque"));
    // Otra REST con 3a: sigue vetando.
    p.observar_rest(p.ticket(), &l).unwrap();
    assert!(p.veta_entradas());
    // Llega USDT a la cuenta por el stream: deja de ser 3a.
    s.dispatch(cuenta_ws(400, Some(400), Some("50"), &[]).as_bytes());
    assert_eq!(saldo(&p), Some(50.0));
    assert!(!p.veta_entradas());
    // La foto 3a atrasada no vuelve a vetar.
    p.observar_rest(p.ticket(), &l).unwrap_err();
    assert!(!p.veta_entradas());
    // Control: una REST sin rechazo tambien lo levanta.
    let (p, _s) = arrancado(&[]);
    p.observar_rest(p.ticket(), &l).unwrap();
    assert!(p.veta_entradas());
    p.observar_rest(p.ticket(), &lectura_json(&cuenta_rest("40", 500, &[]))).unwrap();
    assert!(!p.veta_entradas());
}

// ---- P2 (ii): E ausente o cero ---------------------------------------------------

fn e_ausente_o_cero(intermedio: Option<u64>) {
    let (p, s) = arrancado(&[(btc(), abierta(1.0, 0.0), Some(100))]);
    s.dispatch(cuenta_ws(200, Some(202), Some("98"), &[("BTCUSDT", "1", "-2")]).as_bytes());
    assert_eq!(saldo(&p), Some(98.0));
    s.dispatch(cuenta_ws(200, intermedio, Some("97"), &[("BTCUSDT", "1", "-3")]).as_bytes());
    let tras_intermedio = p.observacion();
    // E=201: regresion del mismo socket; no retrocede saldo ni posicion.
    s.dispatch(cuenta_ws(200, Some(201), Some("99"), &[("BTCUSDT", "1", "-1")]).as_bytes());
    assert_eq!(p.observacion(), tras_intermedio, "E=201 no retrocede el saldo");
    s.dispatch(cuenta_ws(200, Some(201), None, &[("BTCUSDT", "1", "5")]).as_bytes());
    assert_eq!(p.observacion(), tras_intermedio, "E=201 no retrocede la posicion");
    assert!(!p.veta_entradas());
    // E=203: avance verificable.
    s.dispatch(cuenta_ws(200, Some(203), Some("96"), &[("BTCUSDT", "1", "-4")]).as_bytes());
    assert_eq!(p.observacion().map(|o| (o.saldo, o.equity)), Some((96.0, 92.0)));
    assert!(!p.veta_entradas());
}

#[test]
fn v4d_p2_e_ausente_no_deja_retroceder_y_e_mayor_avanza() {
    e_ausente_o_cero(None);
}

#[test]
fn v4d_p2_e_cero_no_deja_retroceder_y_e_mayor_avanza() {
    e_ausente_o_cero(Some(0));
}

// ---- P2 (iii): GET en vuelo tras el nuevo anclaje ----------------------------------

#[test]
fn v4d_p2_get_en_vuelo_tras_rest_incompleta_con_dos_productores() {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[])).unwrap();
    let viejo = p.socket();
    let s = p.abrir_socket(p.era()).unwrap();
    // REST incompleta (sin positions[]): no suelta el veto.
    let mut incompleta = rest(100.0, 100, &[]);
    incompleta.posiciones = None;
    p.observar_rest(p.ticket(), &incompleta).unwrap();
    assert!(p.veta_entradas());
    let a = p.ticket(); // la transicion
    let ev = EventoStream { tiempo_ms: Some(200), saldo: Some(99.0), posiciones: vec![] };
    p.observar_stream_del_socket(p.era(), s, Some(201), &ev, |_| {}).unwrap();
    let b = p.ticket(); // el resincronizador
    let _ = p.observar_rest(a, &rest(99.0, 200, &[]));
    assert!(p.veta_entradas(), "ticket A anterior al WS aceptado: sigue vetado");
    let viejo_ev = EventoStream { tiempo_ms: Some(999), saldo: Some(1.0), posiciones: vec![] };
    assert!(p.observar_stream_del_socket(p.era(), viejo, Some(999), &viejo_ev, |_| {}).is_err());
    p.observar_rest(b, &rest(99.0, 200, &[])).unwrap();
    assert!(!p.veta_entradas(), "ticket B posterior autoriza con el mismo updateTime");
    assert_eq!(saldo(&p), Some(99.0));
}

// ---- P2 (iv): cierre y parser ---------------------------------------------------------

#[test]
fn v4d_p2_cierre_explicito_por_el_stream() {
    let (p, s) = arrancado(&[(btc(), abierta(1.0, -10.0), Some(100))]);
    assert_eq!(p.equity_del_freno(), Some(90.0));
    s.dispatch(cuenta_ws(200, Some(200), Some("95"), &[("BTCUSDT", "0", "0")]).as_bytes());
    assert_eq!(p.posiciones_en_cache(), 0);
    assert_eq!(p.observacion().map(|o| (o.saldo, o.equity)), Some((95.0, 95.0)));
    assert!(!p.veta_entradas());
}

#[test]
fn v4d_p2_fila_ausente_no_cierra_sin_marca_y_la_fila_con_cero_si() {
    let (p, _s) = arrancado(&[(btc(), abierta(1.0, -10.0), Some(100))]);
    p.reconexion(p.era()).unwrap();
    let sin_fila = lectura_json(&cuenta_rest("100", 100, &[]));
    assert!(p.observar_rest(p.ticket(), &sin_fila).is_err(), "fila ausente: no es un cierre con orden");
    assert_eq!(p.posiciones_en_cache(), 1);
    assert!(p.veta_entradas());
    assert_eq!(p.equity_del_freno(), Some(90.0), "la perdida abierta sigue en el freno");
    // Control de cobertura: la fila con cantidad cero y marca cierra.
    let con_cero = lectura_json(&cuenta_rest("100", 100, &[("BTCUSDT", "0", "0", Some("300"))]));
    p.observar_rest(p.ticket(), &con_cero).unwrap();
    assert_eq!(p.posiciones_en_cache(), 0);
    assert!(!p.veta_entradas());
}

#[test]
fn v4d_p2_update_time_ausente_y_cero_no_son_posicion_cerrada() {
    for ut in [None, Some("0")] {
        let l = lectura_json(&cuenta_rest("100", 100, &[("BTCUSDT", "1", "-20", ut)]));
        let fila = l.posiciones.as_ref().unwrap()[&btc()];
        assert_eq!(fila.tiempo_ms, None, "marca desconocida");
        assert_eq!(fila.dato, abierta(1.0, -20.0), "no es un cierre");
        let (p, _s) = arrancado(&[(btc(), abierta(1.0, -10.0), Some(100))]);
        p.observar_rest(p.ticket(), &l).unwrap();
        assert_eq!(p.posiciones_en_cache(), 1);
        assert_eq!(p.equity_del_freno(), Some(80.0), "sin orden, el freno ve la perdida mayor");
        // Cantidad cero con marca desconocida frente a una abierta conocida: no la cierra.
        let cero = lectura_json(&cuenta_rest("100", 100, &[("BTCUSDT", "0", "0", ut)]));
        assert!(p.observar_rest(p.ticket(), &cero).is_err());
        assert_eq!(p.posiciones_en_cache(), 1);
    }
}

#[test]
fn v4d_p2_frame_ilegible_con_posiciones_validas_activa_la_proteccion() {
    let (p, s) = arrancado(&[(btc(), abierta(1.0, -10.0), Some(100))]);
    let ilegible = r#"{"e":"ACCOUNT_UPDATE","E":300,"T":300,"a":{"m":"ORDER","B":"x","P":[{"s":"BTCUSDT","pa":"1","ep":"60000","up":"-50","iw":"0","ps":"BOTH"}]}}"#;
    s.dispatch(ilegible.as_bytes());
    assert!(p.veta_entradas());
    assert!(p.necesita_resincronizar(), "dispara la recuperacion por REST");
}

#[test]
fn v4d_p2_frame_con_dos_filas_usdt_y_posiciones_validas_activa_la_proteccion() {
    let (p, s) = arrancado(&[(btc(), abierta(1.0, -10.0), Some(100))]);
    let doble = r#"{"e":"ACCOUNT_UPDATE","E":300,"T":300,"a":{"m":"ORDER","B":[{"a":"USDT","wb":"90","cw":"90","bc":"0"},{"a":"USDT","wb":"95","cw":"95","bc":"0"}],"P":[{"s":"BTCUSDT","pa":"1","ep":"60000","up":"-50","iw":"0","ps":"BOTH"}]}}"#;
    s.dispatch(doble.as_bytes());
    assert_eq!(saldo(&p), Some(100.0), "un saldo ambiguo no se aplica");
    assert!(p.veta_entradas(), "saldo ambiguo: la proteccion se activa");
    assert!(p.necesita_resincronizar(), "y dispara la recuperacion por REST");
}

// ---- P2 (v): variantes de F3 --------------------------------------------------------------

#[test]
fn v4d_f3_marcas_iguales_el_freno_ve_la_perdida_de_cualquier_fuente() {
    // REST primero, el stream trae la perdida con la misma marca.
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[(btc(), abierta(1.0, -10.0), Some(100))])).unwrap();
    let ev = EventoStream { tiempo_ms: Some(100), saldo: None, posiciones: vec![(btc(), abierta(1.0, -50.0))] };
    p.observar_stream_del_socket(p.era(), p.socket(), Some(100), &ev, |_| {}).unwrap();
    assert_eq!(p.equity_del_freno(), Some(50.0));
    // Al reves: el stream primero y la REST trae la perdida con la misma marca.
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[(btc(), abierta(1.0, 0.0), Some(100))])).unwrap();
    let ev = EventoStream { tiempo_ms: Some(200), saldo: None, posiciones: vec![(btc(), abierta(1.0, -10.0))] };
    p.observar_stream_del_socket(p.era(), p.socket(), Some(200), &ev, |_| {}).unwrap();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[(btc(), abierta(1.0, -50.0), Some(200))])).unwrap();
    assert_eq!(p.equity_del_freno(), Some(50.0));
}

#[test]
fn v4d_f3_marcas_ausentes_las_perdidas_pasan_y_las_ganancias_esperan() {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[(btc(), abierta(1.0, -10.0), Some(100))])).unwrap();
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[(btc(), abierta(1.0, 20.0), None)])).unwrap();
    assert_eq!(p.equity_del_freno(), Some(90.0), "la ganancia sin marca espera");
    p.observar_rest(p.ticket(), &rest(100.0, 100, &[(btc(), abierta(1.0, -50.0), None)])).unwrap();
    assert_eq!(p.equity_del_freno(), Some(50.0), "la perdida sin marca pasa");
    let ev = EventoStream { tiempo_ms: None, saldo: None, posiciones: vec![(btc(), abierta(1.0, -70.0))] };
    p.observar_stream_del_socket(p.era(), p.socket(), None, &ev, |_| {}).unwrap();
    assert_eq!(p.equity_del_freno(), Some(30.0), "tambien sin T del stream");
}

#[test]
fn v4d_f3_contradiccion_en_otra_fila_conserva_las_perdidas_para_el_freno() {
    let p = PuenteCuenta::new();
    p.observar_rest(
        p.ticket(),
        &rest(100.0, 100, &[(btc(), abierta(1.0, 0.0), Some(100)), (eth(), abierta(1.0, 0.0), Some(100))]),
    )
    .unwrap();
    let ev = EventoStream {
        tiempo_ms: Some(300),
        saldo: Some(100.0),
        posiciones: vec![(btc(), abierta(1.0, -10.0)), (eth(), abierta(1.0, -5.0))],
    };
    p.observar_stream_del_socket(p.era(), p.socket(), Some(301), &ev, |_| {}).unwrap();
    p.reconexion(p.era()).unwrap();
    // BTC: misma cantidad, marca vieja, perdida mayor. ETH: otra cantidad con la misma marca.
    let r = p.observar_rest(
        p.ticket(),
        &rest(100.0, 300, &[(btc(), abierta(1.0, -50.0), Some(299)), (eth(), abierta(2.0, -30.0), Some(300))]),
    );
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })));
    assert!(p.veta_entradas());
    assert_eq!(saldo(&p), Some(100.0));
    assert_eq!(p.equity_del_freno(), Some(20.0), "100 + min(-10, -50) + min(-5, -30)");
}
