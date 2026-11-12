//! CAPITAL-UNICO (prioridad 1 del mapa de pérdidas, 2026-11-11) — lo que usa
//! la API nueva. La RED contra `main` está en `capital_unico_red_contract.rs`.
//!
//! - Aviso de otros activos y del modo multi-activo; fallo cerrado sin USDT.
//! - Revisión independiente 3a: saldo de USDT 0 con fondos en otros activos = `Err`.
//! - Opción A del pico (V3, pura): el pico del espacio (venue + cuenta +
//!   definición) manda; la marca distingue espacio nuevo (stops conservados,
//!   7a) de retiro del operador; ningún pico ajeno se adopta nunca; nada viejo
//!   se toca. Más en `capital_v3_contract.rs`.
//!
//! No tocan d*, `capital_del_freno` ni ningún umbral.

use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::{parse_account_capital, parse_account_equity};
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use quantum_arena::GlobalArena;
use risk_engine::drawdown::{self, CargaPico, FrenoDuro, RutasPico};
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

const SALDO: f64 = 1655.81;
/// El pico guardado en la definición anterior (todos los activos), a3g.
const PICO_VIEJO: &str = "7323.93139463\n";

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

/// El saldo que el stream publica por `on_capital`.
fn stream(evento: &str) -> f64 {
    let sink = Arc::new(Captura(AtomicU64::new(f64::NAN.to_bits())));
    let streamer = UserDataStreamer::new(
        BinanceClient::new("k".into(), true),
        Arc::new(OrderRegistry::new()),
    )
    .with_sink(sink.clone());
    streamer.dispatch(evento.as_bytes());
    f64::from_bits(sink.0.load(Relaxed))
}

/// Freno como lo cablea el host (lazo inmune): arena = REST del arranque, pico
/// = el guardado (si se restaura) y el arranque, exchange = observación del exchange.
fn freno(arena_cap: f64, pico_restaurado: Option<f64>, exchange: f64) -> FrenoDuro {
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(arena_cap);
    arena.unified_capital.store(arena_cap, Relaxed);
    let mut pico = drawdown::registrar_pico(&arena.pico_capital, arena_cap);
    if let Some(p) = pico_restaurado {
        pico = drawdown::registrar_pico(&arena.pico_capital, p);
    }
    arena.capital_exchange.store(exchange, Relaxed);
    let cap = drawdown::capital_del_freno(&arena);
    let pico = drawdown::registrar_pico(&arena.pico_capital, cap).max(pico);
    drawdown::freno_duro(&arena, pico, cap)
}

fn dir_temporal(nombre: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("capital_unico_{nombre}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Rutas del pico de Testnet dentro de `dir` (V3: espacio venue + cuenta).
fn rutas_testnet(dir: &std::path::Path) -> RutasPico {
    let espacio = drawdown::espacio_pico(true, &drawdown::huella_cuenta(b"clave-testnet"));
    let r = RutasPico::del_espacio(&espacio, |n| dir.join(n));
    assert!(r.pico.ends_with(format!("pico_capital_{espacio}_usdt.txt")));
    assert!(r.ajenos.iter().any(|a| a.ends_with("pico_capital_prueba.txt")));
    assert!(r.ajenos.iter().any(|a| a.ends_with(drawdown::ARCHIVO_PICO_SIN_VENUE)));
    r
}

// ── Aviso y fallo cerrado ────────────────────────────────────────────────

#[test]
fn capital_unico_avisa_de_otros_activos_y_del_modo_multi_activo() {
    let c = parse_account_capital(CUENTA_MULTI).unwrap();
    assert_eq!(c.multi_activo, Some(false));
    let aviso = c.aviso().expect("hay saldo fuera de USDT: debe avisar");
    assert!(aviso.contains("NO cuentan"), "{aviso}");

    let multi = CUENTA_MULTI.replace(r#""multiAssetsMargin":false"#, r#""multiAssetsMargin":true"#);
    let aviso = parse_account_capital(&multi).unwrap().aviso().unwrap();
    assert!(aviso.contains("MULTI-ACTIVO"), "{aviso}");

    let solo_usdt = r#"{"totalWalletBalance":"13.0","assets":[{"asset":"USDT","walletBalance":"13.0"}]}"#;
    assert_eq!(parse_account_capital(solo_usdt).unwrap().aviso(), None);
    // Números JSON (no texto) también valen.
    let numero = r#"{"totalWalletBalance":13.0,"assets":[{"asset":"USDT","walletBalance":13.0}]}"#;
    assert_eq!(parse_account_equity(numero).unwrap(), 13.0);
}

#[test]
fn capital_unico_sin_el_activo_de_margen_no_inventa_capital() {
    for malo in [
        r#"{"totalWalletBalance":"7299.88"}"#,
        r#"{"totalWalletBalance":"7299.88","assets":[]}"#,
        r#"{"totalWalletBalance":"5000","assets":[{"asset":"USDC","walletBalance":"5000"}]}"#,
        r#"{"assets":[{"asset":"USDT","walletBalance":"NaN"}]}"#,
        r#"{"assets":[{"asset":"USDT","walletBalance":"-1"}]}"#,
        r#"{"assets":[{"asset":"USDT"}]}"#,
        r#"{"assets":[{"asset":"USDT","walletBalance":"1"},{"asset":"USDT","walletBalance":"2"}]}"#,
    ] {
        assert!(parse_account_equity(malo).is_err(), "{malo}");
    }
}

/// Revisión independiente, 3a: USDT 0 con colateral en otros activos no es
/// una cuenta liquidada; antes arrancaba con 0 y el inmune aplanaba lo adoptado.
#[test]
fn capital_unico_usdt_cero_con_fondos_en_otros_activos_falla_cerrado() {
    let usdc = r#"{"multiAssetsMargin":true,"totalWalletBalance":"5000","assets":[{"asset":"USDT","walletBalance":"0"},{"asset":"USDC","walletBalance":"5000"}]}"#;
    let e = parse_account_equity(usdc).unwrap_err();
    assert!(e.contains("ACCOUNT_EVIDENCE_INVALID") && e.contains("otros activos"), "{e}");
    let solo_multi = r#"{"multiAssetsMargin":true,"assets":[{"asset":"USDT","walletBalance":"0"}]}"#;
    assert!(parse_account_equity(solo_multi).is_err());
    let negativo = r#"{"multiAssetsMargin":true,"totalWalletBalance":"5000","assets":[{"asset":"USDT","walletBalance":"-20"}]}"#;
    assert!(parse_account_equity(negativo).unwrap_err().contains("préstamo"));
    // Una cuenta vacía de verdad sigue dando 0 (no hay nada que aplanar).
    let vacia = r#"{"multiAssetsMargin":false,"totalWalletBalance":"0","assets":[{"asset":"USDT","walletBalance":"0"}]}"#;
    assert_eq!(parse_account_equity(vacia).unwrap(), 0.0);
}

// ── Opción A del pico (V3: pura) ─────────────────────────────────────────

/// Testnet de a3g: sin pico del espacio y sin marca, con el pico viejo de
/// 7 323,93 en los archivos ajenos: se ignora (una línea) y nace vacío.
#[test]
fn pico_espacio_nuevo_ignora_los_viejos_y_conserva_los_stops() {
    let dir = dir_temporal("migracion_distintas");
    let r = rutas_testnet(&dir);
    for ajeno in &r.ajenos {
        std::fs::write(ajeno, PICO_VIEJO).unwrap();
    }
    let antes = std::fs::read(&r.ajenos[0]).unwrap();

    let a = drawdown::decidir_arranque_pico(&r);
    assert_eq!(a.carga, CargaPico::Ausente, "el pico nuevo nace vacío");
    assert!(!a.rearmar_stops, "7a: el espacio nuevo no rearma el freno en stops");
    let todo = a.avisos.join(" | ");
    assert_eq!(a.avisos.len(), 1);
    assert!(todo.contains("7323.9314") && todo.contains("IGNORADOS"), "{todo}");

    // Intactos byte a byte, sin renombrar (ni `.adoptado`), y leer no crea nada.
    for ajeno in &r.ajenos {
        assert_eq!(std::fs::read(ajeno).unwrap(), antes);
    }
    assert!(!dir.join("pico_capital.txt.adoptado").exists());
    assert!(!r.pico.exists() && !r.marca.exists());

    // Con el pico nuevo (vacío) y saldo 1655.81 el inmune no dispara...
    let rest = parse_account_equity(CUENTA_MULTI).unwrap();
    assert!(!freno(rest, None, stream(&account_update(SALDO))).echado());
    // ...con una caída real del 55 % sí (lejos de la frontera de d*)...
    assert!(freno(rest, None, stream(&account_update(SALDO * 0.45))).por_capital());
    // ...y adoptar el pico viejo disparaba con una caída falsa del 77 %.
    assert!(freno(rest, Some(7323.93139463), stream(&account_update(SALDO))).echado());
    let _ = std::fs::remove_dir_all(&dir);
}

/// V3 (decisión 4 del coordinador): se acabó la adopción condicional de V2
/// (7d). Con sólo USDT y sin multi-activo el viejo del venue TAMPOCO se
/// adopta; ilegible, tampoco bloquea (es ajeno).
#[test]
fn pico_nunca_adopta_ni_con_unidades_en_apariencia_iguales() {
    let dir = dir_temporal("migracion_iguales");
    let r = rutas_testnet(&dir);
    let solo_usdt = r#"{"multiAssetsMargin":false,"totalWalletBalance":"13.0","assets":[{"asset":"USDT","walletBalance":"13.0"}]}"#;
    assert_eq!(parse_account_capital(solo_usdt).unwrap().aviso(), None);
    for contenido in ["20\n", "veinte\n"] {
        for ajeno in &r.ajenos {
            std::fs::write(ajeno, contenido).unwrap();
        }
        let a = drawdown::decidir_arranque_pico(&r);
        assert_eq!(a.carga, CargaPico::Ausente, "{contenido:?}");
        assert!(!a.rearmar_stops);
    }
    assert!(!r.pico.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 7a: con la marca, el pico ausente es un retiro del operador (rearme de los
/// stops, F1-011); el pico del espacio, si existe, manda sobre los viejos.
#[test]
fn pico_marca_distingue_retiro_del_operador_del_espacio_nuevo() {
    let dir = dir_temporal("marca");
    let r = rutas_testnet(&dir);
    std::fs::write(&r.ajenos[0], PICO_VIEJO).unwrap();

    drawdown::guardar_marca_definicion(&r.marca).unwrap();
    assert_eq!(std::fs::read_to_string(&r.marca).unwrap(), drawdown::MARCA_DEFINICION);
    let a = drawdown::decidir_arranque_pico(&r);
    assert_eq!(a.carga, CargaPico::Ausente);
    assert!(a.rearmar_stops, "con marca y sin pico, lo retiró el operador");

    drawdown::guardar_pico(&r.pico, 1700.0).unwrap();
    let a = drawdown::decidir_arranque_pico(&r);
    assert_eq!(a.carga, CargaPico::Leido(1700.0), "el pico del espacio manda");
    assert!(!a.rearmar_stops);

    std::fs::write(&r.pico, "1,7\n").unwrap();
    let a = drawdown::decidir_arranque_pico(&r);
    assert!(matches!(a.carga, CargaPico::Ilegible(_)), "CL-133 B2");
    assert!(!a.rearmar_stops);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pico_contrato_de_fuente_del_host() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert!(!host.contains("txt.adoptado"), "el host ya no adopta ni renombra el pico sin venue");
    assert!(host.contains("risk_engine::drawdown::decidir_arranque_pico(rutas)"));
    assert!(host.contains("risk_engine::drawdown::guardar_marca_definicion(&ruta_marca)"));
    assert!(!host.contains("unidades_coinciden"), "V3: la inferencia de unidades salió");
    assert!(!host.contains("capital_operativo"), "el tope salió de esta rama");
    // 7e: el aplanado nombra el archivo real, no `pico_capital.txt`.
    assert!(!host.contains("retirar pico_capital.txt"));
    assert!(!host.contains("retirar `pico_capital.txt`"));
    let aplanado = host
        .split("Rearme manual: si fue por drawdown")
        .nth(1)
        .expect("mensaje del aplanado");
    let args = &aplanado[..aplanado.find("),").expect("fin del mensaje")];
    assert!(args.contains("drawdown::archivo_pico()"), "{args}");
}

// ── V2: puente de la cuenta (decisión 1; P1.1 y P1.4 de Codex QRA) ───────

use execution_engine::execution_evidence::parse_account_lectura;
use execution_engine::puente_cuenta::{DatoPosicion, EventoStream, PuenteCuenta, Rechazo};

/// Saldo y equity que publica el stream (todas las llamadas, en orden).
#[derive(Default)]
struct Dos {
    saldo: std::sync::Mutex<Vec<f64>>,
    equity: std::sync::Mutex<Vec<f64>>,
}
impl AccountSink for Dos {
    fn on_capital(&self, usdt: f64) {
        self.saldo.lock().unwrap().push(usdt);
    }
    fn on_equity(&self, usdt: f64) {
        self.equity.lock().unwrap().push(usdt);
    }
    fn on_positions(&self, _p: &[RemotePosition]) {}
    fn on_bracket_close(&self, _c: &BracketClose) {}
}

fn streamer_con(puente: &Arc<PuenteCuenta>) -> (UserDataStreamer, Arc<Dos>) {
    let sink = Arc::new(Dos::default());
    let s = UserDataStreamer::new(BinanceClient::new("k".into(), true), Arc::new(OrderRegistry::new()))
        .with_sink(sink.clone())
        .with_puente(Arc::clone(puente));
    (s, sink)
}

fn evento(t: u64, wb: Option<f64>, posiciones: &[(&str, f64, f64)]) -> String {
    let b = wb.map_or(String::new(), |w| format!(r#"{{"a":"USDT","wb":"{w:.8}","cw":"{w:.8}","bc":"0"}}"#));
    let p: Vec<String> = posiciones
        .iter()
        .map(|(s, pa, up)| format!(r#"{{"s":"{s}","pa":"{pa}","ep":"60000","up":"{up:.8}","iw":"0","ps":"BOTH"}}"#))
        .collect();
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{t},"T":{t},"a":{{"m":"ORDER","B":[{b}],"P":[{}]}}}}"#, p.join(","))
}

fn ultima(v: &std::sync::Mutex<Vec<f64>>) -> Option<f64> {
    v.lock().unwrap().last().copied()
}

/// `/fapi/v2/account` con una posición abierta en USDT (y otra en USDC).
fn cuenta_con_posicion(wb: f64, up_usdt: f64, t: u64) -> String {
    format!(
        r#"{{"multiAssetsMargin":false,"totalWalletBalance":"{wb}","assets":[{{"asset":"USDT","walletBalance":"{wb}","unrealizedProfit":"{up_usdt}","marginBalance":"999999","updateTime":{t}}}],
        "positions":[{{"symbol":"BTCUSDT","positionSide":"BOTH","positionAmt":"0.010","unrealizedProfit":"{up_usdt}","updateTime":{t}}},
                     {{"symbol":"BTCUSDC","positionSide":"BOTH","positionAmt":"0.010","unrealizedProfit":"-500","updateTime":{t}}},
                     {{"symbol":"ETHUSDT","positionSide":"BOTH","positionAmt":"0","unrealizedProfit":"0","updateTime":0}}]}}"#
    )
}

/// REST: el SALDO siembra el arena (sin uPnL: el arena adopta las posiciones
/// al precio de entrada y suma su PnL al cerrar, CL-8); la EQUITY es saldo +
/// uPnL de las posiciones con margen en USDT. `marginBalance` no se usa.
#[test]
fn v2_rest_saldo_sin_upnl_y_equity_aparte_con_upnl_positivo_y_negativo() {
    for up in [40.0, -60.0] {
        let (c, l) = parse_account_lectura(&cuenta_con_posicion(100.0, up, 1789000000000)).unwrap();
        assert_eq!(c.capital, 100.0);
        assert_eq!(l.saldo, 100.0, "el arena nace del saldo, no de wb + uPnL");
        let p = PuenteCuenta::new();
        assert_eq!(p.observar_rest(p.ticket(), &l).unwrap().equity, 100.0 + up, "la posición en USDC no entra");
        assert_eq!(l.tiempo_saldo_ms, Some(1789000000000));
        let abiertas = l.posiciones.as_ref().unwrap().values().filter(|p| matches!(p.dato, DatoPosicion::Abierta { .. })).count();
        assert_eq!(abiertas, 1, "sólo la abierta con margen en USDT");
    }
}

/// Stream: el saldo es `wb` y la equity va aparte, con uPnL positivo y negativo.
/// La pérdida abierta del 55 % (sin cambiar el umbral) dispara el freno por
/// la equity; la del 45 % no; el cero real llega al freno por capital ≤ 0.
#[test]
fn v2_stream_saldo_y_equity_separados_y_el_freno_ve_la_perdida_abierta() {
    let puente = Arc::new(PuenteCuenta::new());
    let (s, sink) = streamer_con(&puente);
    s.dispatch(evento(1, Some(100.0), &[("BTCUSDT", 0.01, -60.0)]).as_bytes());
    assert_eq!(ultima(&sink.saldo), Some(100.0));
    assert_eq!(ultima(&sink.equity), Some(40.0));
    s.dispatch(evento(2, Some(0.0), &[("BTCUSDT", 0.01, 10.0)]).as_bytes());
    assert_eq!(ultima(&sink.saldo), Some(0.0));
    assert_eq!(ultima(&sink.equity), Some(10.0));

    let p = Arc::new(PuenteCuenta::new());
    let (s, sink) = streamer_con(&p);
    s.dispatch(evento(1, Some(SALDO), &[("BTCUSDT", 0.01, -0.55 * SALDO)]).as_bytes());
    let f = freno(SALDO, None, ultima(&sink.equity).unwrap());
    assert!(f.por_capital(), "pérdida abierta del 55 %: {f}");
    assert!(!freno(SALDO, None, ultima(&sink.saldo).unwrap()).echado(), "con el saldo no se vería");
    s.dispatch(evento(2, Some(SALDO), &[("BTCUSDT", 0.01, -0.45 * SALDO)]).as_bytes());
    assert!(!freno(SALDO, None, ultima(&sink.equity).unwrap()).echado(), "45 %: no");
    // Pérdida realizada del 50 % + cero: el saldo baja y llega a 0.
    s.dispatch(evento(3, Some(0.5 * SALDO), &[("BTCUSDT", 0.0, 0.0)]).as_bytes());
    assert_eq!(ultima(&sink.equity), Some(0.5 * SALDO));
    s.dispatch(evento(4, Some(0.0), &[]).as_bytes());
    let arena = GlobalArena::build_in_own_stack(SALDO);
    arena.capital_exchange.store(ultima(&sink.equity).unwrap(), Relaxed);
    assert!(drawdown::capital_del_freno(&arena) <= 0.0, "el inmune lo ve insolvente");
}

/// Caché anterior de otro activo; USDT ausente o duplicado; sólo posiciones.
#[test]
fn v2_otro_activo_ausente_duplicado_y_solo_posiciones() {
    let puente = Arc::new(PuenteCuenta::new());
    let (s, sink) = streamer_con(&puente);
    // Posición con margen en USDC (por sufijo y por `ma`): no entra.
    s.dispatch(evento(1, None, &[("BTCUSDC", 0.01, -50.0)]).as_bytes());
    s.dispatch(br#"{"e":"ACCOUNT_UPDATE","E":2,"T":2,"a":{"B":[],"P":[{"s":"ETHUSDC","pa":"1","up":"-7","ps":"BOTH","ma":"USDC"}]}}"#);
    assert_eq!(puente.posiciones_en_cache(), 0);
    assert!(sink.saldo.lock().unwrap().is_empty() && sink.equity.lock().unwrap().is_empty(), "sin saldo no hay nada que publicar");
    s.dispatch(evento(3, Some(100.0), &[]).as_bytes());
    assert_eq!((ultima(&sink.saldo), ultima(&sink.equity)), (Some(100.0), Some(100.0)));
    // Sólo posiciones: el saldo no cambia (sin on_capital), la equity sí.
    s.dispatch(evento(4, None, &[("BTCUSDT", 0.01, -30.0)]).as_bytes());
    assert_eq!(sink.saldo.lock().unwrap().len(), 1);
    assert_eq!(ultima(&sink.equity), Some(70.0));
    // USDT sin `wb`: ausente, no cero.
    s.dispatch(br#"{"e":"ACCOUNT_UPDATE","E":5,"T":5,"a":{"B":[{"a":"USDT"}],"P":[]}}"#);
    assert_eq!(sink.saldo.lock().unwrap().len(), 1);
    assert_eq!(puente.observacion().unwrap().saldo, 100.0);
    // USDT duplicado: descartado entero, como en REST.
    s.dispatch(br#"{"e":"ACCOUNT_UPDATE","E":6,"T":6,"a":{"B":[{"a":"USDT","wb":"1"},{"a":"USDT","wb":"9"}],"P":[{"s":"BTCUSDT","pa":"0.01","up":"5","ps":"BOTH"}]}}"#);
    assert_eq!(puente.observacion().unwrap().saldo, 100.0);
    assert_eq!(puente.observacion().unwrap().equity, 70.0);
    assert!(parse_account_capital(r#"{"assets":[{"asset":"USDT","walletBalance":"1"},{"asset":"USDT","walletBalance":"9"}]}"#).is_err());
}

/// Frescura: un evento del stream más viejo que el último se rechaza; una
/// respuesta REST pedida antes de un evento más nuevo no lo pisa.
#[test]
fn v2_rest_atrasado_no_pisa_un_estado_mas_nuevo() {
    let puente = Arc::new(PuenteCuenta::new());
    let (s, sink) = streamer_con(&puente);
    let l = |wb: f64, up: f64, t: u64| parse_account_lectura(&cuenta_con_posicion(wb, up, t)).unwrap().1;

    // Arranque: sin nada observado, una lectura sin marca se acepta.
    let t0 = puente.ticket();
    let sin_tiempo = execution_engine::puente_cuenta::LecturaCuenta { tiempo_saldo_ms: None, ..l(100.0, 0.0, 1) };
    assert_eq!(puente.observar_rest(t0, &sin_tiempo).unwrap().saldo, 100.0);

    // Se pide REST (ticket) y, antes de que responda, el stream publica 80.
    let ticket = puente.ticket();
    s.dispatch(evento(2_000, Some(80.0), &[("BTCUSDT", 0.01, -5.0)]).as_bytes());
    assert_eq!(ultima(&sink.saldo), Some(80.0));
    // La respuesta llega tarde con el estado de t = 1 500: no pisa.
    assert!(matches!(puente.observar_rest(ticket, &l(100.0, 0.0, 1_500)), Err(Rechazo::Atrasada { .. })));
    // V3: sin marca y con otro saldo, no modifica lo observado.
    assert!(matches!(puente.observar_rest(ticket, &sin_tiempo), Err(Rechazo::SinOrden { .. })));
    assert_eq!(puente.observacion().unwrap().saldo, 80.0);
    // Con un tiempo propio no anterior al último aceptado, sí vale.
    assert_eq!(puente.observar_rest(ticket, &l(79.0, -4.0, 2_500)).unwrap().equity, 75.0);
    // Un evento del stream más viejo que lo aceptado también se rechaza.
    s.dispatch(evento(2_100, Some(500.0), &[]).as_bytes());
    assert_eq!(puente.observacion().unwrap().saldo, 79.0);
    assert_eq!(ultima(&sink.saldo), Some(80.0), "no se publicó el atrasado");
}

/// Reconexión: la evidencia se conserva (V3), la equity queda incompleta y no
/// sube sobre la última completa hasta que una lectura REST la completa. Cero
/// tras saldo positivo.
#[test]
fn v2_reconexion_no_olvida_la_perdida_abierta_hasta_resincronizar() {
    let puente = Arc::new(PuenteCuenta::new());
    let (s, sink) = streamer_con(&puente);
    s.dispatch(evento(1, Some(100.0), &[("BTCUSDT", 0.01, -60.0)]).as_bytes());
    assert_eq!(ultima(&sink.equity), Some(40.0));
    puente.reconexion(puente.era()).unwrap();
    assert!(puente.necesita_resincronizar());
    assert_eq!(puente.posiciones_en_cache(), 1, "V3: la evidencia se conserva");
    // Tras el hueco llega sólo el saldo: la equity queda en 40 (la pérdida
    // abierta sigue a la vista).
    s.dispatch(evento(2, Some(100.0), &[]).as_bytes());
    assert_eq!(ultima(&sink.saldo), Some(100.0));
    assert_eq!(ultima(&sink.equity), Some(40.0));
    // Resincronización REST (lo que hace el inmune): equity completa otra vez.
    let ticket = puente.ticket();
    let lectura = parse_account_lectura(&cuenta_con_posicion(100.0, -10.0, 3)).unwrap().1;
    let obs = puente.observar_rest(ticket, &lectura).unwrap();
    assert_eq!((obs.saldo, obs.equity), (100.0, 90.0));
    assert!(!puente.necesita_resincronizar());
    // Cero tras saldo positivo: se publica y gobierna.
    s.dispatch(evento(4, Some(0.0), &[("BTCUSDT", 0.0, 0.0)]).as_bytes());
    assert_eq!((ultima(&sink.saldo), ultima(&sink.equity)), (Some(0.0), Some(0.0)));
}

/// Otra cuenta (transición): era nueva; el streamer viejo y los tickets
/// viejos se rechazan.
#[test]
fn v2_era_nueva_rechaza_el_streamer_y_el_ticket_de_la_cuenta_anterior() {
    let puente = Arc::new(PuenteCuenta::new());
    let (demo, sink_demo) = streamer_con(&puente);
    demo.dispatch(evento(1, Some(SALDO), &[]).as_bytes());
    let ticket_viejo = puente.ticket();
    puente.nueva_era();
    assert_eq!(puente.observacion(), None, "la cuenta nueva aún no se observó");
    demo.dispatch(evento(9, Some(5_000.0), &[]).as_bytes());
    assert_eq!(puente.observacion(), None);
    assert_eq!(sink_demo.saldo.lock().unwrap().len(), 1);
    let l = parse_account_lectura(&cuenta_con_posicion(13.0, 0.0, 10)).unwrap().1;
    assert_eq!(puente.observar_rest(ticket_viejo, &l), Err(Rechazo::OtraEra));
    let (mainnet, sink) = streamer_con(&puente);
    mainnet.dispatch(evento(11, Some(13.0), &[]).as_bytes());
    assert_eq!(ultima(&sink.saldo), Some(13.0));
    let _ = EventoStream::default();
}

/// Recibo: tras el primer fill, la lectura REST y el ACCOUNT_UPDATE del mismo
/// fill dan las dos observaciones congruentes (mismo saldo, misma equity).
#[test]
fn v2_recibo_rest_y_stream_congruentes_tras_el_primer_fill() {
    let wb = 1655.15; // 1655.81 menos la comisión del fill
    let up = -3.2;
    let rest = PuenteCuenta::new();
    let por_rest = rest
        .observar_rest(rest.ticket(), &parse_account_lectura(&cuenta_con_posicion(wb, up, 1789000000500)).unwrap().1)
        .unwrap();
    let puente = Arc::new(PuenteCuenta::new());
    let (s, _sink) = streamer_con(&puente);
    s.dispatch(evento(1789000000500, Some(wb), &[("BTCUSDT", 0.01, up), ("BTCUSDC", 0.01, -500.0)]).as_bytes());
    let por_stream = puente.observacion().unwrap();
    assert_eq!(por_rest, por_stream, "REST {por_rest:?} y stream {por_stream:?}");
}

/// Fixture multi-activo (sintético, esquema V2): USDT 0 con colateral en
/// USDC/BNB es `Err`, no `Ok(0)` (decisión 5 / 3a); USDT negativo también.
#[test]
fn v2_fixture_multi_activo_falla_cerrado() {
    let fixture = include_str!("fixtures/cuenta_multi_activo_v2.json");
    let e = parse_account_capital(fixture).unwrap_err();
    assert!(e.contains("ACCOUNT_EVIDENCE_INVALID") && e.contains("otros activos"), "{e}");
    // V3 (P1.3): el cero explícito llega al puente (plano de seguridad) con el
    // rechazo para el arranque.
    let (_, l) = parse_account_lectura(fixture).unwrap();
    assert_eq!(l.saldo, 0.0);
    assert!(l.rechazo_arranque.as_deref().is_some_and(|r| r.contains("otros activos")));
    let negativo = fixture.replacen(r#""walletBalance": "0.00000000""#, r#""walletBalance": "-20.00000000""#, 1);
    assert!(parse_account_capital(&negativo).unwrap_err().contains("préstamo"));
}

/// Contrato de fuente del host: un solo puente, sembrado por REST, compartido
/// con los dos streamers; el freno mide con la equity del puente.
#[test]
fn v2_contrato_de_fuente_del_host() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert_eq!(host.matches("PuenteCuenta::new()").count(), 1, "un solo puente por proceso");
    assert_eq!(host.matches(".with_puente(Arc::clone(&puente_cuenta))").count(), 2, "demo y mainnet");
    assert!(host.contains("puente_imm.equity_del_freno()"), "el freno usa la equity");
    assert!(host.contains("puente_cuenta.nueva_era()"));
    assert!(host.contains("puente_rs.necesita_resincronizar()"));
    assert!(!host.contains(".fetch_account_balance()"), "el host lee la cuenta por fetch_account_lectura");
    assert!(host.matches("observar_rest(ticket,").count() >= 2, "arranque y transición");
    assert!(host.contains("resincronizar_cuenta::resincronizar_una_vez("), "y el resincronizador");
    let stream = include_str!("../src/user_data_stream.rs");
    assert!(stream.contains("self.puente.observar_stream_y_publicar(self.era, &evento"));
}
