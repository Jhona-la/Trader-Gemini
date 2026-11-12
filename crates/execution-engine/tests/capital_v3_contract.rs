//! CAPITAL-UNICO V3 — respuesta al dictamen de Codex QRA sobre 1757709c4
//! (cuatro P1) con las decisiones del coordinador del 2026-11-12. Las dos RED
//! de Codex QRA, tal como están escritas, en `capital_v2_audit_red_codex.rs`.
//! La RED de estas pruebas contra 1757709c4 (con la API de ese SHA) está en
//! `logs/m2-calidad-datos-20261111/CAPITAL-V3.red-1757709c4.{rs,log}`.
//!
//! No tocan d*, `capital_del_freno` ni ningún umbral.

use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::{parse_account_capital, parse_account_lectura};
use execution_engine::puente_cuenta::{
    DatoPosicion, EventoStream, LecturaCuenta, PosicionRest, PuenteCuenta, Rechazo,
};
use execution_engine::resincronizar_cuenta::{resincronizar_una_vez, Pausa};
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use quantum_arena::GlobalArena;
use risk_engine::drawdown::{self, CargaPico, RutasPico};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Duration;

const BTC: (&str, &str) = ("BTCUSDT", "BOTH");

fn clave(k: (&str, &str)) -> (String, String) {
    (k.0.to_string(), k.1.to_string())
}

/// Lectura V3 a mano: saldo con su marca y posiciones con la suya.
fn lectura(saldo: f64, t_saldo: Option<u64>, pos: &[((&str, &str), DatoPosicion, Option<u64>)]) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: t_saldo,
        posiciones: Some(
            pos.iter()
                .map(|(k, dato, t)| (clave(*k), PosicionRest { dato: *dato, tiempo_ms: *t }))
                .collect(),
        ),
        rechazo_arranque: None,
        upnl_del_activo: None,
    }
}

fn abierta(cantidad: f64, upnl: f64) -> DatoPosicion {
    DatoPosicion::Abierta { cantidad, upnl }
}

fn ws(p: &PuenteCuenta, t: Option<u64>, saldo: Option<f64>, pos: &[((&str, &str), DatoPosicion)]) -> Result<Option<execution_engine::puente_cuenta::Observacion>, Rechazo> {
    p.observar_stream(
        p.era(),
        &EventoStream {
            tiempo_ms: t,
            saldo,
            posiciones: pos.iter().map(|(k, d)| (clave(*k), *d)).collect(),
        },
    )
}

/// `/fapi/v2/account` con la fila de USDT (marca `t_saldo`) y posiciones
/// (símbolo, cantidad, uPnL opcional, marca).
fn cuenta(wb: &str, t_saldo: u64, pos: &[(&str, &str, Option<&str>, u64)]) -> String {
    let filas: Vec<String> = pos
        .iter()
        .map(|(s, pa, up, t)| match up {
            Some(u) => format!(r#"{{"symbol":"{s}","positionSide":"BOTH","positionAmt":"{pa}","unrealizedProfit":"{u}","updateTime":{t}}}"#),
            None => format!(r#"{{"symbol":"{s}","positionSide":"BOTH","positionAmt":"{pa}","updateTime":{t}}}"#),
        })
        .collect();
    format!(
        r#"{{"multiAssetsMargin":false,"totalWalletBalance":"{wb}","assets":[{{"asset":"USDT","walletBalance":"{wb}","unrealizedProfit":"0","updateTime":{t_saldo}}}],"positions":[{}]}}"#,
        filas.join(",")
    )
}

fn freno(arena_cap: f64, exchange: f64) -> drawdown::FrenoDuro {
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(arena_cap);
    arena.unified_capital.store(arena_cap, Relaxed);
    let pico = drawdown::registrar_pico(&arena.pico_capital, arena_cap);
    arena.capital_exchange.store(exchange, Relaxed);
    let cap = drawdown::capital_del_freno(&arena);
    drawdown::freno_duro(&arena, pico.max(drawdown::registrar_pico(&arena.pico_capital, cap)), cap)
}

// ── P1.1 · frescura por componente ───────────────────────────────────────

#[test]
fn p11_el_parser_da_marcas_separadas_para_saldo_y_posicion() {
    let (_, l) = parse_account_lectura(&cuenta("100", 100, &[("BTCUSDT", "0.01", Some("-1"), 201), ("ETHUSDT", "0", Some("0"), 50)])).unwrap();
    assert_eq!(l.tiempo_saldo_ms, Some(100), "la marca del saldo es la de SU fila, no el máximo");
    let m = l.posiciones.unwrap();
    assert_eq!(m[&clave(BTC)], PosicionRest { dato: abierta(0.01, -1.0), tiempo_ms: Some(201) });
    assert_eq!(m[&clave(("ETHUSDT", "BOTH"))].dato, DatoPosicion::Cerrada, "cantidad cero = cierre con su marca");
}

#[test]
fn p11_rest_anterior_al_ticket_no_revierte_el_cero() {
    let p = PuenteCuenta::new();
    ws(&p, Some(200), Some(0.0), &[]).unwrap();
    let ticket = p.ticket(); // el ticket se toma DESPUÉS del evento: misma generación
    let r = p.observar_rest(ticket, &lectura(100.0, Some(150), &[]));
    assert!(matches!(r, Err(Rechazo::Atrasada { ref componente, ultimo_ms: 200, .. }) if componente == "saldo"), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
}

#[test]
fn p11_rest_simultaneo_empate_contradictorio_no_sobrescribe_y_coherente_si() {
    let p = PuenteCuenta::new();
    ws(&p, Some(200), Some(0.0), &[]).unwrap();
    let r = p.observar_rest(p.ticket(), &lectura(100.0, Some(200), &[]));
    assert!(matches!(r, Err(Rechazo::Contradiccion { tiempo_ms: 200, .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
    // El empate coherente sí vale (y completa la equity).
    assert_eq!(p.observar_rest(p.ticket(), &lectura(0.0, Some(200), &[])).unwrap().saldo, 0.0);
    assert!(!p.veta_entradas());
    // Dos eventos del stream con la misma marca y otro saldo: el segundo no pisa
    // y deja la equity incompleta (hace falta REST).
    let r = ws(&p, Some(200), Some(7.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
    assert!(p.veta_entradas());
}

#[test]
fn p11_marca_ausente_no_modifica_lo_observado() {
    let p = PuenteCuenta::new();
    // Sin nada observado, una lectura sin marca vale (arranque).
    assert_eq!(p.observar_rest(p.ticket(), &lectura(50.0, None, &[])).unwrap().saldo, 50.0);
    ws(&p, Some(200), Some(0.0), &[]).unwrap();
    let r = p.observar_rest(p.ticket(), &lectura(100.0, None, &[]));
    assert!(matches!(r, Err(Rechazo::SinOrden { .. })), "{r:?}");
    // Sin marca pero sin modificar nada (mismo saldo): vale.
    assert!(p.observar_rest(p.ticket(), &lectura(0.0, None, &[])).is_ok());
    // En el stream, un evento sin marca con otro saldo tampoco modifica.
    assert!(matches!(ws(&p, None, Some(9.0), &[]), Err(Rechazo::SinOrden { .. })));
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
}

#[test]
fn p11_posicion_mas_reciente_no_acredita_una_fila_de_saldo_vieja() {
    let p = PuenteCuenta::new();
    let ticket = p.ticket();
    ws(&p, Some(200), Some(0.0), &[]).unwrap();
    let (_, l) = parse_account_lectura(&cuenta("100", 100, &[("BTCUSDT", "0.01", Some("0"), 201)])).unwrap();
    let r = p.observar_rest(ticket, &l);
    assert!(matches!(r, Err(Rechazo::Atrasada { ref componente, .. }) if componente == "saldo"), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
    assert_eq!(p.posiciones_en_cache(), 0, "entera o nada: la posición tampoco entra");
}

#[test]
fn p11_dos_respuestas_rest_invertidas() {
    // B (saldo t=150, posición t=300) llega antes que A (saldo t=200, misma
    // posición t=200): el saldo más nuevo es el de A y gana; la posición de A
    // es más vieja pero no la modifica (misma cantidad), así que A entra entera.
    let p = PuenteCuenta::new();
    let (ta, tb) = (p.ticket(), p.ticket());
    let (_, b) = parse_account_lectura(&cuenta("50", 150, &[("BTCUSDT", "0.01", Some("-1"), 300)])).unwrap();
    let (_, a) = parse_account_lectura(&cuenta("80", 200, &[("BTCUSDT", "0.01", Some("-2"), 200)])).unwrap();
    p.observar_rest(tb, &b).unwrap();
    assert_eq!(p.observar_rest(ta, &a).unwrap().saldo, 80.0);
    // Y al revés: la respuesta vieja después de la nueva se rechaza.
    let q = PuenteCuenta::new();
    let (t1, t2) = (q.ticket(), q.ticket());
    q.observar_rest(t2, &lectura(80.0, Some(300), &[])).unwrap();
    assert!(matches!(q.observar_rest(t1, &lectura(50.0, Some(250), &[])), Err(Rechazo::Atrasada { .. })));
    assert_eq!(q.observacion().unwrap().saldo, 80.0);
    // Una posición más vieja que la observada, con OTRA cantidad, se rechaza.
    let (_, vieja) = parse_account_lectura(&cuenta("80", 300, &[("BTCUSDT", "0.02", Some("-1"), 250)])).unwrap();
    q.observar_rest(q.ticket(), &lectura(80.0, Some(300), &[(BTC, abierta(0.01, -1.0), Some(260))])).unwrap();
    assert!(matches!(q.observar_rest(q.ticket(), &vieja), Err(Rechazo::Atrasada { .. })));
}

// ── P1.2 · era, transición y reconexión ──────────────────────────────────

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap()
}

#[test]
fn p12_era_nueva_veta_entradas_hasta_su_saldo_y_reintenta_get_lento_o_fallido() {
    let p = Arc::new(PuenteCuenta::new());
    let bandera = Arc::new(AtomicBool::new(false));
    {
        let b = Arc::clone(&bandera);
        p.vincular_veto(move |v| b.store(v, Relaxed));
    }
    assert!(bandera.load(Relaxed), "sin saldo observado nace vetando");
    p.observar_rest(p.ticket(), &lectura(100.0, Some(1), &[])).unwrap();
    assert!(!bandera.load(Relaxed));

    let era = p.nueva_era();
    assert!(bandera.load(Relaxed) && p.necesita_resincronizar(), "la era nueva veta y pide resincronizar");
    assert_eq!(p.observacion(), None);
    assert_eq!(p.equity_del_freno(), Some(100.0), "el freno conserva la evidencia de la era anterior");

    let rt = rt();
    // GET fallido.
    let r = rt.block_on(resincronizar_una_vez(&p, async { Err("HTTP 503".to_string()) }, Duration::from_millis(50), |_| {}));
    assert!(r.unwrap_err().contains("503"));
    assert!(bandera.load(Relaxed));
    // GET lento: el tope lo corta y el puente no cambia.
    let r = rt.block_on(resincronizar_una_vez(&p, std::future::pending(), Duration::from_millis(20), |_| {}));
    assert!(r.unwrap_err().contains("sin respuesta"));
    assert!(bandera.load(Relaxed));
    // Reintento con éxito: publica bajo la era vigente y suelta el veto.
    let mut publicado = None;
    let r = rt.block_on(resincronizar_una_vez(&p, async { Ok(lectura(13.0, Some(10), &[])) }, Duration::from_millis(50), |o| publicado = Some(o.saldo)));
    assert_eq!(r.unwrap().saldo, 13.0);
    assert_eq!(publicado, Some(13.0));
    assert!(!bandera.load(Relaxed));
    assert_eq!(p.ticket_de_era(era).map(|t| t.era()), Some(era));
    assert_eq!(p.ticket_de_era(era - 1), None);

    // La espera del reintento: 5 s, doblando hasta 30 s; reposo 1 s.
    let mut pausa = Pausa::nueva();
    assert_eq!(pausa.actual(), Pausa::REPOSO);
    let esperas: Vec<u64> = (0..5).map(|_| { pausa.fallo(); pausa.actual().as_secs() }).collect();
    assert_eq!(esperas, vec![5, 10, 20, 30, 30]);
}

#[test]
fn p12_evento_y_reconexion_tardios_de_la_era_anterior_no_tocan_la_vigente() {
    let p = Arc::new(PuenteCuenta::new());
    let demo = streamer(&p);
    demo.dispatch(evento_ws(1, Some("100"), &[("BTCUSDT", Some("0.01"), Some("-5"))]).as_bytes());
    let era_vieja = p.era();
    let era = p.nueva_era();
    p.observar_rest(p.ticket(), &lectura(13.0, Some(10), &[(BTC, abierta(0.02, -1.0), Some(10))])).unwrap();
    assert!(!p.veta_entradas());
    // Evento tardío del streamer viejo: rechazado.
    demo.dispatch(evento_ws(20, Some("5000"), &[]).as_bytes());
    assert_eq!(p.observacion().unwrap().saldo, 13.0);
    // Reconexión tardía del streamer viejo: no borra ni marca la vigente.
    assert_eq!(p.reconexion(era_vieja), Err(Rechazo::OtraEra));
    assert_eq!(p.posiciones_en_cache(), 1);
    assert!(!p.veta_entradas());
    // La de la era vigente sí la marca incompleta (sin borrar la evidencia).
    p.reconexion(era).unwrap();
    assert!(p.veta_entradas());
    assert_eq!(p.posiciones_en_cache(), 1);
    // Publicar sólo bajo la era vigente.
    assert_eq!(p.con_era_vigente(era_vieja, |_| ()), None);
    assert!(p.con_era_vigente(era, |o| o.map(|o| o.saldo)).flatten() == Some(13.0));
    let stream = include_str!("../src/user_data_stream.rs");
    assert!(stream.contains("self.puente.reconexion(self.era)"));
    assert!(stream.contains("self.puente.observar_stream_y_publicar(self.era, &evento"));
}

#[test]
fn p12_cero_confirmado_durante_la_transicion() {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &lectura(100.0, Some(1), &[])).unwrap();
    let era = p.nueva_era();
    // El stream de la cuenta nueva confirma 0 en t=500 mientras el GET de la
    // transición se reintenta; la respuesta trae una fila vieja de 100 (t=400).
    p.observar_stream(era, &EventoStream { tiempo_ms: Some(500), saldo: Some(0.0), posiciones: vec![] }).unwrap();
    let r = p.observar_rest(p.ticket(), &lectura(100.0, Some(400), &[]));
    assert!(matches!(r, Err(Rechazo::Atrasada { .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
    assert!(p.veta_entradas(), "sin una lectura completa de la era, sigue vetando");
    // La lectura coherente (misma marca, mismo cero) la completa; el cero gobierna el freno.
    p.observar_rest(p.ticket(), &lectura(0.0, Some(500), &[])).unwrap();
    assert!(!p.veta_entradas());
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.capital_exchange.store(p.equity_del_freno().unwrap(), Relaxed);
    assert!(drawdown::capital_del_freno(&arena) <= 0.0);
}

#[test]
fn p12_contrato_de_fuente_de_la_transicion() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    let tr = host.split("let era_mainnet = puente_cuenta.nueva_era();").nth(1).expect("transición");
    let refresco = tr.find("oms_transition.refresh(&arena_transition)").expect("refresco OMS");
    let antes = &host[..host.find("let era_mainnet = puente_cuenta.nueva_era();").unwrap()];
    assert!(!antes[antes.rfind("ResultadoTransicion::Publicada").unwrap()..].contains("oms_transition.refresh"), "la era cambia ANTES del refresco del OMS");
    assert!(refresco > 0);
    assert!(tr.contains("puente_transicion.ticket_de_era(era_mainnet)"), "el GET va atado a la era");
    assert!(tr.contains("puente_transicion.con_era_vigente(era_mainnet,"), "la publicación también");
    assert!(tr.contains("publish_current_balance(&balance_token,"), "y al testigo de la sesión OMS");
    assert!(tr.contains("tokio::time::sleep(std::time::Duration::from_secs(5))"), "los fallos se reintentan");
}

// ── P1.3 · equity incompleta ─────────────────────────────────────────────

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

fn evento_ws(t: u64, wb: Option<&str>, pos: &[(&str, Option<&str>, Option<&str>)]) -> String {
    let b = wb.map_or(String::new(), |w| format!(r#"{{"a":"USDT","wb":"{w}"}}"#));
    let p: Vec<String> = pos
        .iter()
        .map(|(s, pa, up)| {
            let mut campos = vec![format!(r#""s":"{s}""#), r#""ps":"BOTH""#.to_string()];
            if let Some(pa) = pa {
                campos.push(format!(r#""pa":"{pa}""#));
            }
            if let Some(up) = up {
                campos.push(format!(r#""up":"{up}""#));
            }
            format!("{{{}}}", campos.join(","))
        })
        .collect();
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{t},"T":{t},"a":{{"B":[{b}],"P":[{}]}}}}"#, p.join(","))
}

/// Saldo 100 con pérdida abierta de −60 (equity 40): ni REST ni el stream
/// incompletos la borran; con la reconexión y el GET en tope, la evidencia se
/// conserva, las entradas se vetan y el freno sigue viendo la caída (los
/// cierres del inmune no dependen de nada de esto).
#[test]
fn p13_saldo_100_perdida_abierta_60_con_datos_incompletos_y_reconexion_en_tope() {
    let p = Arc::new(PuenteCuenta::new());
    let s = streamer(&p);
    let (_, l) = parse_account_lectura(&cuenta("100", 1, &[("BTCUSDT", "0.01", Some("-60"), 1)])).unwrap();
    assert_eq!(p.observar_rest(p.ticket(), &l).unwrap().equity, 40.0);
    assert!(!p.veta_entradas());

    // REST sin uPnL y REST con cantidad inválida: equity 40, incompleta.
    for fila in [
        cuenta("100", 2, &[("BTCUSDT", "0.01", None, 2)]),
        cuenta("100", 3, &[("BTCUSDT", "abc", Some("-60"), 3)]),
    ] {
        let (_, l) = parse_account_lectura(&fila).unwrap();
        assert_eq!(l.posiciones.as_ref().unwrap()[&clave(BTC)].dato, DatoPosicion::Invalida);
        assert_eq!(p.observar_rest(p.ticket(), &l).unwrap().equity, 40.0, "{fila}");
        assert!(p.veta_entradas());
    }
    // Stream sin `up`, sin `pa` o con `pa` ilegible: no cierra ni borra la pérdida.
    for ev in [
        evento_ws(10, None, &[("BTCUSDT", Some("0.01"), None)]),
        evento_ws(11, None, &[("BTCUSDT", None, Some("-60"))]),
        evento_ws(12, None, &[("BTCUSDT", Some("abc"), Some("0"))]),
    ] {
        s.dispatch(ev.as_bytes());
        assert_eq!(p.observacion().unwrap().equity, 40.0, "{ev}");
        assert_eq!(p.posiciones_en_cache(), 1);
    }
    // Un ACCOUNT_UPDATE ilegible entero: evidencia perdida = incompleta.
    let q = Arc::new(PuenteCuenta::new());
    q.observar_rest(q.ticket(), &lectura(100.0, Some(1), &[])).unwrap();
    streamer(&q).dispatch(br#"{"e":"ACCOUNT_UPDATE","E":2,"T":2,"a":5}"#);
    assert!(q.veta_entradas());

    // Reconexión con el REST en tope: la evidencia se conserva.
    p.reconexion(p.era()).unwrap();
    let r = rt().block_on(resincronizar_una_vez(&p, std::future::pending(), Duration::from_millis(20), |_| {}));
    assert!(r.is_err());
    assert_eq!(p.equity_del_freno(), Some(40.0));
    assert!(p.veta_entradas());
    // Una ganancia sin evidencia completa no sube la equity; una pérdida sí baja.
    s.dispatch(evento_ws(20, Some("100"), &[("BTCUSDT", Some("0.01"), Some("+30"))]).as_bytes());
    assert_eq!(p.observacion().unwrap().equity, 40.0);
    s.dispatch(evento_ws(21, Some("100"), &[("BTCUSDT", Some("0.01"), Some("-70"))]).as_bytes());
    assert_eq!(p.observacion().unwrap().equity, 30.0);
    // El freno ve la caída abierta con el umbral de siempre.
    assert!(freno(100.0, p.equity_del_freno().unwrap()).por_capital());
    // Una lectura completa la resincroniza.
    let (_, l) = parse_account_lectura(&cuenta("100", 30, &[("BTCUSDT", "0.01", Some("-10"), 30)])).unwrap();
    assert_eq!(p.observar_rest(p.ticket(), &l).unwrap().equity, 90.0);
    assert!(!p.veta_entradas());
}

/// Primera lectura con una posición inválida nunca observada: no cuenta 0; la
/// equity se acota con el `unrealizedProfit` de la fila de USDT y se vetan
/// las entradas.
#[test]
fn p13_posicion_nunca_observada_invalida_se_acota_con_el_upnl_del_activo() {
    let json = r#"{"multiAssetsMargin":false,"totalWalletBalance":"100","assets":[{"asset":"USDT","walletBalance":"100","unrealizedProfit":"-60","updateTime":1}],"positions":[{"symbol":"BTCUSDT","positionSide":"BOTH","positionAmt":"0.01","updateTime":1}]}"#;
    let (_, l) = parse_account_lectura(json).unwrap();
    assert_eq!(l.upnl_del_activo, Some(-60.0));
    let p = PuenteCuenta::new();
    let obs = p.observar_rest(p.ticket(), &l).unwrap();
    assert_eq!((obs.saldo, obs.equity), (100.0, 40.0));
    assert!(p.veta_entradas());
}

/// 3a: USDT 0 con fondos en otros activos. El arranque y el tamaño lo
/// rechazan (fallo cerrado), pero el cero explícito llega al plano de
/// seguridad del puente y el freno lo recibe.
#[test]
fn p13_cero_explicito_con_otros_activos_llega_al_plano_de_seguridad() {
    let usdc = r#"{"multiAssetsMargin":true,"totalWalletBalance":"5000","assets":[{"asset":"USDT","walletBalance":"0","updateTime":5},{"asset":"USDC","walletBalance":"5000"}],"positions":[]}"#;
    assert!(parse_account_capital(usdc).is_err(), "el arranque sigue fallando cerrado");
    let (_, l) = parse_account_lectura(usdc).unwrap();
    assert!(l.rechazo_arranque.as_deref().is_some_and(|r| r.contains("otros activos")));
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &lectura(100.0, Some(1), &[])).unwrap();
    assert_eq!(p.observar_rest(p.ticket(), &l).unwrap().saldo, 0.0);
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.capital_exchange.store(p.equity_del_freno().unwrap(), Relaxed);
    assert!(drawdown::capital_del_freno(&arena) <= 0.0, "el inmune lo ve");
    // Ausente sigue sin ser cero: sin la fila de USDT no hay lectura.
    assert!(parse_account_lectura(r#"{"assets":[{"asset":"USDC","walletBalance":"5"}],"positions":[]}"#).is_err());
}

#[test]
fn p13_el_lazo_inmune_no_espera_un_get() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    let inicio = host.find("let mut latched = false;").expect("lazo inmune");
    let fin = inicio + host[inicio..].find("freno_duro(&arena_imm").expect("freno");
    assert!(!host[inicio..fin].contains("fetch_account_lectura()"), "el lazo del inmune espera un GET");
    assert!(host[inicio..fin].contains("puente_imm.equity_del_freno()"));
    // El resincronizador es otra tarea, con tope por GET.
    let rs = host.split("RESINCRONIZADOR del puente").nth(1).expect("resincronizador");
    let rs = &rs[..rs.find("let mut latched = false;").unwrap()];
    assert!(rs.contains("rt_handle.spawn(async move {"));
    assert!(rs.contains("resincronizar_una_vez("));
    assert!(rs.contains("std::time::Duration::from_secs(15)"));
}

// ── P1.4 · pico, opción A pura ───────────────────────────────────────────

const PICO_VIEJO: &str = "7323.93139463\n";

fn dir_temporal(nombre: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("capital_v3_{nombre}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rutas(dir: &std::path::Path, testnet: bool, clave_api: &[u8]) -> RutasPico {
    let espacio = drawdown::espacio_pico(testnet, &drawdown::huella_cuenta(clave_api));
    RutasPico::del_espacio(&espacio, |n| dir.join(n))
}

/// Una cuenta que fue multi-activo (pico agregado 7 323,93 en la definición
/// anterior) y hoy sólo tiene USDT no lo hereda: el pico nuevo nace vacío.
#[test]
fn p14_cuenta_antes_multi_activo_hoy_solo_usdt_no_hereda_7323_93() {
    let dir = dir_temporal("multi_hoy_usdt");
    let r = rutas(&dir, true, b"clave-a3g");
    for ajeno in &r.ajenos {
        std::fs::write(ajeno, PICO_VIEJO).unwrap();
    }
    let antes: Vec<Vec<u8>> = r.ajenos.iter().map(|a| std::fs::read(a).unwrap()).collect();
    // Hoy sólo USDT: la foto del arranque «diría» unidades iguales. Da igual.
    let solo_usdt = r#"{"multiAssetsMargin":false,"totalWalletBalance":"1655.81","assets":[{"asset":"USDT","walletBalance":"1655.81"}]}"#;
    assert_eq!(parse_account_capital(solo_usdt).unwrap().aviso(), None);

    let a = drawdown::decidir_arranque_pico(&r);
    assert_eq!(a.carga, CargaPico::Ausente, "el pico nuevo nace vacío SIEMPRE");
    assert!(!a.rearmar_stops, "espacio nuevo: los stops se conservan");
    assert_eq!(a.avisos.len(), 1, "una sola línea de aviso: {:?}", a.avisos);
    assert!(a.avisos[0].contains("7323.9314") && a.avisos[0].contains("IGNORADOS"), "{}", a.avisos[0]);
    // Nada se borra, se renombra ni se crea.
    let despues: Vec<Vec<u8>> = r.ajenos.iter().map(|a| std::fs::read(a).unwrap()).collect();
    assert_eq!(antes, despues);
    assert!(!r.pico.exists() && !r.marca.exists());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), r.ajenos.len());
    // Con 1 655,81 y el pico nuevo no dispara; con el viejo, caída falsa del 77 %.
    assert!(!freno(1655.81, 1655.81).echado());
    let arena = GlobalArena::build_in_own_stack(1655.81);
    let pico = drawdown::registrar_pico(&arena.pico_capital, 7323.93139463);
    assert!(drawdown::freno_duro(&arena, pico, 1655.81).echado(), "adoptarlo dispararía");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Espacio sin procedencia (migración): ni el viejo legible ni el ilegible se
/// adoptan; tras crear la marca, retirar el pico es un retiro explícito.
#[test]
fn p14_migracion_sin_procedencia_y_retiro_explicito_tras_la_marca() {
    let dir = dir_temporal("migracion");
    let r = rutas(&dir, false, b"clave-mainnet");
    std::fs::write(&r.ajenos[0], "20\n").unwrap();
    assert_eq!(drawdown::decidir_arranque_pico(&r).carga, CargaPico::Ausente, "unidades «iguales»: tampoco se adopta");
    std::fs::write(&r.ajenos[0], "veinte\n").unwrap();
    let a = drawdown::decidir_arranque_pico(&r);
    assert_eq!(a.carga, CargaPico::Ausente, "un ajeno ilegible no bloquea ni se adopta");
    assert!(a.avisos[0].contains("ilegible"));

    drawdown::guardar_pico(&r.pico, 1700.0).unwrap();
    drawdown::guardar_marca_definicion(&r.marca).unwrap();
    assert_eq!(drawdown::decidir_arranque_pico(&r).carga, CargaPico::Leido(1700.0));
    std::fs::remove_file(&r.pico).unwrap();
    let a = drawdown::decidir_arranque_pico(&r);
    assert_eq!(a.carga, CargaPico::Ausente);
    assert!(a.rearmar_stops, "con marca y sin pico: lo retiró el operador");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Transición de cuenta: otra cuenta u otro venue = otro espacio del pico.
#[test]
fn p14_transicion_de_cuenta_o_de_venue_cambia_el_espacio_del_pico() {
    let dir = dir_temporal("transicion");
    let demo = rutas(&dir, true, b"clave-demo");
    let otra_cuenta = rutas(&dir, true, b"clave-otra");
    let mainnet = rutas(&dir, false, b"clave-demo");
    assert_ne!(demo.pico, otra_cuenta.pico);
    assert_ne!(demo.pico, mainnet.pico);
    assert_ne!(demo.marca, otra_cuenta.marca);
    drawdown::guardar_pico(&demo.pico, 5000.0).unwrap();
    drawdown::guardar_marca_definicion(&demo.marca).unwrap();
    let a = drawdown::decidir_arranque_pico(&otra_cuenta);
    assert_eq!(a.carga, CargaPico::Ausente, "la otra cuenta no hereda el pico de 5 000");
    assert!(!a.rearmar_stops, "la marca de otra cuenta no cuenta como retiro");
    assert_eq!(rutas(&dir, true, b"clave-demo"), demo, "la misma cuenta, el mismo espacio");
    // La huella no contiene la clave.
    let h = drawdown::huella_cuenta(b"clave-demo");
    assert_eq!(h.len(), 12);
    assert!(!h.contains("clave"));
    // El host: espacio al arrancar y, en la transición, buzón al inmune.
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert!(host.contains("RutasPico::del_espacio(&espacio_pico,"));
    assert!(host.contains("if espacio != espacio_pico {"));
    assert!(host.contains("buzon_imm.lock()"));
    assert!(!host.contains("unidades_coinciden"), "la inferencia de unidades salió");
    let dd = include_str!("../../risk-engine/src/drawdown.rs");
    assert!(!dd.contains("unidades_coinciden"));
    assert!(!dd.contains("ADOPTADO"));
    let _ = std::fs::remove_dir_all(&dir);
}

