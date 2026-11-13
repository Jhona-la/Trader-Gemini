//! Prueba B reforzada (QRA-TELEMETRIA-DEMO, fila T3): la EMISION real de `CAPITAL_UPDATE`
//! por el sink del host (`quantum_engine::demo_tel::CapitalBridgeSink`, el mismo tipo que usa
//! `god_engine.rs`), una linea por revision aceptada, con el streamer y el puente reales.
//!
//! Secuencia: REST inicial (T=100) -> reserva 0,04 -> WS aceptado (T=200) -> WS viejo (T=150)
//! rechazado -> resincronizacion REST (T=300) -> WS viejo para la REST (T=250) rechazado -> WS
//! aceptado (T=400). Se afirma, sobre las lineas que el sink EMITE (testigo de solo lectura):
//! exactamente dos lineas, con el importe exacto del saldo aceptado (no el dimensionable con la
//! reserva), con su revision (fuente WS, era, socket, marca T y hora E), ninguna por los WS
//! rechazados. La REST (inicial y resincronizacion) no pasa por el sink: no emite linea (su saldo
//! llega al espejo por el publicador); la prueba lo fija para que el lector lo sepa (N/D).
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
    assert_eq!(campo(&l[0], "wallet_balance"), "101.5000");
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
    assert_eq!(campo(&l[1], "wallet_balance"), "103.2500");
    assert_eq!(campo(&l[1], "marca_ms"), "400");
    assert_eq!(campo(&l[1], "hora_evento_ms"), "401");
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 103.25);

    // Control: la linea sin revision (camino `on_capital` directo) declara N/D, no inventa.
    let sin = quantum_engine::demo_tel::linea_capital_update(1, 7.0, None);
    assert!(sin.ends_with("fuente=N/D era=N/D socket=N/D marca_ms=N/D hora_evento_ms=N/D"), "{sin}");
}
