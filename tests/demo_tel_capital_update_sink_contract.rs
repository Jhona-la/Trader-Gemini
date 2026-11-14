//! Prueba B reforzada (QRA-TELEMETRIA-DEMO, fila T3): la EMISION real de `CAPITAL_UPDATE`
//! por el sink del host (`quantum_engine::demo_tel::CapitalBridgeSink`, el mismo tipo que usa
//! `god_engine.rs`), una linea por revision aceptada, con el streamer y el puente reales.
//!
//! Secuencia: REST inicial (T=100) -> reserva 0,04 -> WS aceptado (T=200) -> WS viejo (T=150)
//! rechazado -> resincronizacion REST (T=300) -> WS viejo para la REST (T=250) rechazado -> WS
//! aceptado (T=400). Se afirma, sobre las lineas que el sink EMITE (testigo de solo lectura):
//! exactamente dos lineas, con el importe exacto del saldo aceptado (no el dimensionable con la
//! reserva), con su revision (fuente WS, era, socket, marca T y hora E), ninguna por los WS
//! rechazados. La REST (inicial y resincronizacion) no pasa por el sink: no emite linea aqui (su saldo
//! llega al espejo por el publicador). T3: el host la registra con `demo_tel::capital_rest_aceptado`;
//! la prueba T3-1 de abajo la sigue hasta el destino real del logger.
//! Si el sink deja de emitir, la prueba falla (cuenta exacta de lineas).

use execution_engine::client::BinanceClient;
use execution_engine::order_registry::OrderRegistry;
use execution_engine::puente_cuenta::{LecturaCuenta, Publicacion, PuenteCuenta};
use execution_engine::user_data_stream::UserDataStreamer;
use quantum_arena::dimensionable::{ClaveReserva, LibroDimensionable};
use quantum_arena::{symbol_registry, GlobalArena};
use quantum_engine::demo_tel::{vincular_testigo, CapitalBridgeSink};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

const SIMBOLO: &str = "TELCUSDT";
const SESION: u64 = 4;
const COMISION: f64 = 0.04;
const CLAVE: ClaveReserva = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
static LINEAS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn lineas() -> Vec<String> {
    LINEAS.lock().unwrap().iter().filter(|l| l.contains("DEMO_TEL event=CAPITAL_UPDATE")).cloned().collect()
}

fn libro(arena: &GlobalArena) -> MutexGuard<'_, LibroDimensionable> {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner())
}

/// `publicar_capital_de_sesion` del host (publicador del puente, como en la prueba B de T1).
fn publicar(oms: &AtomicU64, arena: &GlobalArena, espejo: &AtomicU64, p: &Publicacion) -> bool {
    if p.veta {
        libro(arena).veto_del_puente();
    }
    let suelto = match (p.veta, p.vigente) {
        (true, _) | (false, None) => false,
        (false, Some((saldo, marca))) => {
            arena.capital_incompleto.store(true, Ordering::Release);
            oms.load(Ordering::SeqCst) == p.sesion && {
                let mut l = libro(arena);
                l.fijar_alcance(p.era, p.sesion);
                l.saldo_aceptado(arena, saldo, marca);
                drop(l);
                espejo.store(saldo.to_bits(), Ordering::Relaxed);
                true
            }
        }
    };
    arena.capital_incompleto.store(!suelto, Ordering::Release);
    suelto
}

fn lectura(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

fn cuenta_ws(t: u64, e: u64, wb: &str) -> String {
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{e},"T":{t},"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{wb}","cw":"{wb}","bc":"0"}}],"P":[]}}}}"#)
}

/// `campo=valor` de una linea.
fn campo<'a>(linea: &'a str, k: &str) -> &'a str {
    let pref = format!(" {k}=");
    let i = linea.find(&pref).unwrap_or_else(|| panic!("sin {k}: {linea}")) + pref.len();
    linea[i..].split_whitespace().next().unwrap()
}

#[test]
fn capital_update_sale_del_sink_real_una_vez_por_revision_aceptada() {
    assert!(vincular_testigo(|l| LINEAS.lock().unwrap().push(l.to_string())), "testigo ya vinculado");
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
    let arena = GlobalArena::build_in_own_stack(0.0);
    // El espejo del host: el MISMO `unified_capital` lo escriben el publicador y el sink.
    let espejo = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let equity = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let oms = Arc::new(AtomicU64::new(SESION));
    let puente = Arc::new(PuenteCuenta::con_sesion(SESION));
    {
        let oms = Arc::clone(&oms);
        puente.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    libro(&arena).activar(&arena);
    {
        let (arena, espejo, oms) = (Arc::clone(&arena), Arc::clone(&espejo), Arc::clone(&oms));
        puente.vincular_publicador(move |p| publicar(&oms, &arena, &espejo, p));
    }
    let streamer = UserDataStreamer::new(BinanceClient::new("k".into(), true), Arc::new(OrderRegistry::new()))
        .with_sink(Arc::new(CapitalBridgeSink {
            unified_capital: Arc::clone(&espejo),
            equity_exchange: Arc::clone(&equity),
        }))
        .with_puente(Arc::clone(&puente));

    // (1) REST inicial: el puente acepta 100 (T=100). No pasa por el sink: sin linea.
    puente.observar_rest_de_sesion(puente.ticket(), puente.sesion(), &lectura(100.0, 100)).expect("REST inicial");
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 100.0);
    assert!(lineas().is_empty(), "la REST no emite CAPITAL_UPDATE: {:?}", lineas());

    // (2) Reserva de una entrada: el dimensionable queda por debajo del saldo aceptado.
    libro(&arena).reservar(&arena, CLAVE, COMISION);
    let dimensionable = arena.unified_capital.load(Ordering::Relaxed);
    assert!(dimensionable < 100.0, "el libro no derivo el dimensionable ({dimensionable})");

    // (3) WS aceptado T=200: UNA linea, importe exacto y revision.
    let (era, socket) = (puente.era(), puente.socket());
    streamer.dispatch(cuenta_ws(200, 201, "101.5").as_bytes());
    let l = lineas();
    assert_eq!(l.len(), 1, "el sink no emitio CAPITAL_UPDATE para el WS aceptado: {l:?}");
    // T3 (T3-1): el saldo con su representacion exacta (antes `{:.4}`: "101.5000").
    assert_eq!(campo(&l[0], "wallet_balance"), "101.5");
    assert_eq!(campo(&l[0], "fuente"), "WS");
    assert_eq!(campo(&l[0], "era"), era.to_string());
    assert_eq!(campo(&l[0], "socket"), socket.to_string());
    assert_eq!(campo(&l[0], "marca_ms"), "200");
    assert_eq!(campo(&l[0], "hora_evento_ms"), "201");
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 101.5);
    let dimensionable = arena.unified_capital.load(Ordering::Relaxed);
    assert_ne!(campo(&l[0], "wallet_balance"), format!("{dimensionable:.4}"), "publica el dimensionable con la reserva");

    // (4) WS viejo T=150: rechazado por el puente, sin linea y sin tocar el espejo.
    streamer.dispatch(cuenta_ws(150, 151, "90").as_bytes());
    assert_eq!(lineas().len(), 1, "un WS rechazado emitio linea: {:?}", lineas());
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 101.5);

    // (5) Resincronizacion REST T=300 (la pasada real): aceptada, sin linea.
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let mut publicado = None;
    let r = rt.block_on(execution_engine::resincronizar_cuenta::resincronizar_una_vez(
        &puente,
        puente.sesion(),
        async { Ok(lectura(102.0, 300)) },
        std::time::Duration::from_secs(5),
        |obs| publicado = Some(obs.saldo),
    ));
    assert_eq!(r.map(|o| o.saldo), Ok(102.0));
    assert_eq!(publicado, Some(102.0));
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 102.0);
    assert_eq!(lineas().len(), 1, "la resincronizacion REST no pasa por el sink");

    // (6) WS viejo para la REST (T=250 < 300): rechazado, sin linea.
    streamer.dispatch(cuenta_ws(250, 251, "95").as_bytes());
    assert_eq!(lineas().len(), 1, "{:?}", lineas());
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 102.0);

    // (7) WS aceptado T=400: segunda linea, con su revision.
    streamer.dispatch(cuenta_ws(400, 401, "103.25").as_bytes());
    let l = lineas();
    assert_eq!(l.len(), 2, "{l:?}");
    assert_eq!(campo(&l[1], "wallet_balance"), "103.25");
    assert_eq!(campo(&l[1], "marca_ms"), "400");
    assert_eq!(campo(&l[1], "hora_evento_ms"), "401");
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 103.25);

    // Control: la linea sin revision (camino `on_capital` directo) declara N/D, no inventa.
    let sin = quantum_engine::demo_tel::linea_capital_update(1, 7.0, None);
    assert!(sin.ends_with("fuente=N/D era=N/D socket=N/D marca_ms=N/D hora_evento_ms=N/D"), "{sin}");
}

// ===========================================================================
// DEMO_TEL T3-1 (qra_t3_criterios.md:5-15): B observa el DESTINO REAL del logger.
//
// Proceso hijo aislado (este mismo binario, `--exact`, con T3_CAPITAL_HIJO): inicializa el
// logger como el host (`Rotacion::desde_entorno` + `init_telemetry_logger_rotativo(16_384,
// rot)`, god_engine.rs:1392-1395) con un TG_LOG_FILE temporal, corre la secuencia con el
// puente, el streamer, el sink y el resincronizador reales y fixtures offline, y vacía la
// cola (`vaciar_telemetria`). El padre lee el archivo que consume M1, lo parsea con el lector
// real (`parse_demo_tel`) y compara cada revisión y sus decimales con el fixture. La ruta
// REST de cada pin la conecta `t3_montaje/adaptador.rs` (la misma que llama su host).
// ===========================================================================

#[path = "t3_montaje/adaptador.rs"]
mod adaptador;

/// Para el adaptador del lines contract (no se usa aquí).
#[allow(dead_code)]
pub struct CasoInmune {
    pub ts_ms: u64,
    pub motivo: &'static str,
    pub freno_capital: f64,
    pub peak_capital: f64,
    pub equity_del_freno: Option<f64>,
    pub caida_capital_pct: f64,
    pub tope_capital_pct: f64,
    pub operator_lock: bool,
    pub insolvent: bool,
    pub dd_breach: bool,
    pub lat_breach: bool,
    pub latency_strikes: u32,
}

const T3_HIJO: &str = "T3_CAPITAL_HIJO";
const T3_NOMBRE: &str = "t3_capital_update_pasa_por_logger_real_por_revision";
const HOST_SRC: &str = include_str!("../src/bin/god_engine.rs");

/// (fuente, marca T, hora E, saldo) de cada revisión ACEPTADA de la secuencia, en orden.
/// Los saldos tienen más de cuatro decimales; los dos últimos coinciden a cuatro.
const ESPERADAS: &[(&str, &str, &str, f64)] = &[
    ("REST", "100", "N/D", 100.123456),
    ("WS", "200", "201", 101.50005),
    ("REST", "300", "N/D", 102.000001),
    ("WS", "400", "401", 103.25001),
    ("WS", "500", "501", 103.25004),
];

fn iniciar_logger_como_el_host() {
    match telemetry_server::macros::Rotacion::desde_entorno() {
        Some(rot) => assert!(telemetry_server::macros::init_telemetry_logger_rotativo(16_384, rot), "logger ya iniciado"),
        None => panic!("el hijo necesita TG_LOG_FILE"),
    }
    assert!(telemetry_server::macros::LOG_QUEUE.get().is_some(), "la cola del logger no está activa");
}

/// El cuerpo del hijo: la secuencia del criterio sobre las piezas reales.
fn hijo_capital() {
    iniciar_logger_como_el_host();
    telemetry_server::telemetry_log!("T3_CENTINELA antes de la secuencia");
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
    let arena = GlobalArena::build_in_own_stack(0.0);
    let espejo = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let equity = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let oms = Arc::new(AtomicU64::new(SESION));
    let puente = Arc::new(PuenteCuenta::con_sesion(SESION));
    {
        let oms = Arc::clone(&oms);
        puente.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    libro(&arena).activar(&arena);
    {
        let (arena, espejo, oms) = (Arc::clone(&arena), Arc::clone(&espejo), Arc::clone(&oms));
        puente.vincular_publicador(move |p| publicar(&oms, &arena, &espejo, p));
    }
    let streamer = UserDataStreamer::new(BinanceClient::new("k".into(), true), Arc::new(OrderRegistry::new()))
        .with_sink(Arc::new(CapitalBridgeSink { unified_capital: Arc::clone(&espejo), equity_exchange: Arc::clone(&equity) }))
        .with_puente(Arc::clone(&puente));
    let mirror = || f64::from_bits(espejo.load(Ordering::Relaxed));
    let mut esperado = Vec::new();

    // (1) REST inicial válida (T=100).
    let ticket = puente.ticket();
    adaptador::rest_inicial(&puente, ticket, puente.sesion(), &lectura(ESPERADAS[0].3, 100)).expect("REST inicial");
    assert_eq!(mirror(), ESPERADAS[0].3, "el espejo no avanzó con la REST inicial");
    esperado.push(format!("REST {} {}", ticket.era(), ticket.socket()));
    // (2) Reserva viva: cambia el dimensionable, no el saldo publicado.
    libro(&arena).reservar(&arena, CLAVE, COMISION);
    let dimensionable = arena.unified_capital.load(Ordering::Relaxed);
    assert!(dimensionable < ESPERADAS[0].3, "el libro no derivó el dimensionable ({dimensionable})");
    // (3) WS aceptado (T=200).
    esperado.push(format!("WS {} {}", puente.era(), puente.socket()));
    streamer.dispatch(cuenta_ws(200, 201, "101.50005").as_bytes());
    assert_eq!(mirror(), ESPERADAS[1].3);
    // (4) WS obsoleto (T=150): rechazado.
    streamer.dispatch(cuenta_ws(150, 151, "90").as_bytes());
    assert_eq!(mirror(), ESPERADAS[1].3);
    // (5) Resincronización REST aceptada (T=300), por el resincronizador real.
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let ticket = puente.ticket();
    let r = rt.block_on(adaptador::resincronizar(&puente, puente.sesion(), async { Ok(lectura(ESPERADAS[2].3, 300)) }, |_| {}));
    assert_eq!(r.map(|o| o.saldo), Ok(ESPERADAS[2].3));
    assert_eq!(mirror(), ESPERADAS[2].3, "el espejo no avanzó con la resincronización");
    esperado.push(format!("REST {} {}", ticket.era(), ticket.socket()));
    // (6) WS obsoleto para la REST (T=250): rechazado.
    streamer.dispatch(cuenta_ws(250, 251, "95").as_bytes());
    assert_eq!(mirror(), ESPERADAS[2].3);
    // (7) WS aceptado (T=400) y su repetición exacta.
    esperado.push(format!("WS {} {}", puente.era(), puente.socket()));
    streamer.dispatch(cuenta_ws(400, 401, "103.25001").as_bytes());
    assert_eq!(mirror(), ESPERADAS[3].3);
    streamer.dispatch(cuenta_ws(400, 401, "103.25001").as_bytes());
    assert_eq!(mirror(), ESPERADAS[3].3);
    // (8) WS aceptado (T=500) con un saldo distinto que coincide a cuatro decimales.
    esperado.push(format!("WS {} {}", puente.era(), puente.socket()));
    streamer.dispatch(cuenta_ws(500, 501, "103.25004").as_bytes());
    assert_eq!(mirror(), ESPERADAS[4].3);
    assert_ne!(format!("{dimensionable}"), format!("{}", ESPERADAS[4].3));

    telemetry_server::telemetry_log!("T3_CENTINELA despues de la secuencia");
    telemetry_server::macros::vaciar_telemetria();
    for e in esperado {
        println!("T3_ESPERADO {e}");
    }
}

#[test]
fn t3_capital_update_pasa_por_logger_real_por_revision() {
    if std::env::var(T3_HIJO).is_ok() {
        hijo_capital();
        return;
    }
    adaptador::ruta_rest_del_host(HOST_SRC).unwrap_or_else(|e| panic!("[{}] {e}", adaptador::PIN));
    let host = HOST_SRC.replace("\r\n", "\n");
    assert!(host.contains("Some(rot) => telemetry_server::macros::init_telemetry_logger_rotativo(16_384, rot),"));
    let dir = std::env::temp_dir().join(format!("t3_capital_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("demo_t3.log");
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", T3_NOMBRE, "--nocapture", "--test-threads=1"])
        .env(T3_HIJO, "1")
        .env("TG_LOG_FILE", &log)
        .env("TG_LOG_TS", "1")
        .output()
        .expect("lanza el hijo");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "[{}] el hijo falló:\n{stdout}\n{}", adaptador::PIN, String::from_utf8_lossy(&out.stderr));
    let texto = std::fs::read_to_string(&log).expect("el destino del logger existe");
    let esperado: Vec<&str> = stdout.lines().filter_map(|l| l.strip_prefix("T3_ESPERADO ")).collect();
    assert_eq!(esperado.len(), ESPERADAS.len(), "el hijo no declaró las revisiones:\n{stdout}");

    // Control contra vacuidad: el destino trae las centinelas ajenas, en orden, con prefijo.
    let antes = texto.find("T3_CENTINELA antes").expect("centinela previa en el destino");
    let despues = texto.find("T3_CENTINELA despues").expect("centinela final en el destino (cola drenada)");
    assert!(antes < despues);
    assert!(texto.lines().all(|l| l.trim().is_empty() || l.starts_with("[ts=")), "líneas sin el prefijo del logger:\n{texto}");
    let en_medio = &texto[antes..despues];
    assert_eq!(texto.matches("event=CAPITAL_UPDATE").count(), en_medio.matches("event=CAPITAL_UPDATE").count());

    // Las líneas YA ESCRITAS, por el lector real de M1.
    let lectura = telemetry_server::report_demo_run::parse_demo_tel(
        &texto,
        "segmento",
        &telemetry_server::report_demo_run::Ambito::new("binance_usdm", "A"),
    );
    assert!(lectura.diagnosticos.is_empty(), "[{}] líneas truncadas o inválidas: {:?}", adaptador::PIN, lectura.diagnosticos);
    let cap: Vec<_> = lectura.eventos.iter().filter(|e| e.tipo == "CAPITAL_UPDATE").collect();
    assert_eq!(
        cap.len(),
        ESPERADAS.len(),
        "[{}] una línea por revisión aceptada (REST inicial y periódica incluidas; ninguna por la reserva, la repetición ni los WS obsoletos):\n{texto}",
        adaptador::PIN
    );
    let mut balances = Vec::new();
    for ((e, (fuente, marca, hora, saldo)), decl) in cap.iter().zip(ESPERADAS).zip(&esperado) {
        let c = &e.campos;
        let (f, era, socket) = {
            let mut p = decl.split(' ');
            (p.next().unwrap(), p.next().unwrap(), p.next().unwrap())
        };
        assert_eq!(f, *fuente);
        assert_eq!(
            (c["fuente"].as_str(), c["era"].as_str(), c["socket"].as_str(), c["marca_ms"].as_str(), c["hora_evento_ms"].as_str()),
            (*fuente, era, socket, *marca, *hora),
            "identidad de la revisión: {c:?}"
        );
        let wb: f64 = c["wallet_balance"].parse().expect("saldo decimal");
        assert_eq!(wb.to_bits(), saldo.to_bits(), "[{}] saldo exacto de la revisión T={marca}: {}", adaptador::PIN, c["wallet_balance"]);
        balances.push(c["wallet_balance"].clone());
    }
    assert_ne!(balances[3], balances[4], "dos saldos distintos que coinciden a cuatro decimales salen distintos");
    let _ = std::fs::remove_dir_all(&dir);
}
