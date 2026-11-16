//! Prueba B de Qoder (`qoder_pruebas_que_faltan.md`): el balance que llega a la telemetria es
//! el SALDO ACEPTADO del puente, no el dimensionable (prerregistro v7, §2.3.A).
//!
//! Piezas reales: el `PuenteCuenta` de produccion, el `GlobalArena` con su libro ACTIVADO
//! (modo derivado), el sink financiero `telemetry_server::profiler::update_financials_atomic`
//! y su lector. Lo que vive en linea en el host (el publicador `publicar_capital_de_sesion`,
//! el espejo `unified_capital` que alimenta el sink y `on_capital` del puente del stream) va en
//! el ADAPTADOR, atado a sus lineas con anclas de fuente (mismo patron que capital_v4i_contract).

use execution_engine::puente_cuenta::{EventoStream, LecturaCuenta, Publicacion, PuenteCuenta};
use quantum_arena::dimensionable::{ClaveReserva, LibroDimensionable};
use quantum_arena::{symbol_registry, GlobalArena};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, MutexGuard};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const STREAM: &str = include_str!("../../execution-engine/src/user_data_stream.rs");
const DEMO_TEL_MOD: &str = include_str!("../../../src/demo_tel.rs");
const SIMBOLO: &str = "TELBUSDT";
const SESION: u64 = 3;
const COMISION: f64 = 0.04;
const CLAVE: ClaveReserva = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
const FALLO: &str = "TELEMETRIA: el balance publicado es el dimensionable (derivado) y no el saldo aceptado del puente — la Identidad 1 del prerregistro v7 fallaría por diseño";

fn texto(fuente: &str) -> String {
    fuente.replace("\r\n", "\n")
}

fn orden(fuente: &str, pasos: &[&str]) {
    let fuente = texto(fuente);
    let mut resto = fuente.as_str();
    for paso in pasos {
        resto = resto.split_once(paso).unwrap_or_else(|| panic!("ancla ausente o fuera de orden: {paso}")).1;
    }
}

// ============================ ADAPTADOR ============================

fn anclas_del_adaptador() {
    // El publicador del host escribe el espejo con el saldo aceptado, despues del libro.
    orden(HOST, &[
        "fn publicar_capital_de_sesion(",
        "libro.saldo_aceptado(arena, balance, marca);",
        "espejo.store(balance.to_bits(), Ordering::Relaxed);",
    ]);
    // El espejo es `unified_capital` del host, y es lo que recibe el sink financiero.
    assert_eq!(texto(HOST).matches("let unified_capital = ").count(), 1, "un solo `unified_capital` en el host");
    orden(HOST, &[
        "let unified_capital = Arc::new(AtomicU64::new(initial_capital.to_bits()));",
        "let espejo_veto = Arc::clone(&unified_capital);",
        "publicar_capital_de_sesion(&oms_veto, &arena_veto, &espejo_veto, p)",
        "telemetry_server::profiler::update_financials_atomic(",
        "f64::from_bits(unified_capital.load(std::sync::atomic::Ordering::Relaxed))",
    ]);
    // `on_capital` (CAPITAL_UPDATE y update_dynamic_capital) recibe la observacion del puente.
    // T2: el stream la entrega con su revision (`on_capital_aceptado`) y el sink del host vive en
    // `src/demo_tel.rs`; su emision real la prueba `tests/demo_tel_capital_update_sink_contract.rs`.
    orden(STREAM, &[
        "let r = self.puente.observar_stream_del_socket(self.era, socket, hora_evento, &evento, |obs| {",
        "sink.on_capital_aceptado(obs.saldo, &revision);",
    ]);
    orden(DEMO_TEL_MOD, &[
        "fn capital(&self, usdt: f64, revision: Option<&RevisionSaldo>) {",
        "audit_engine::telemetry::update_dynamic_capital(usdt);",
        // DEMO_TEL T3 (T3-1, ancla adaptada): la línea sale por `registrar_revision`, que
        // escribe con el mismo `emitir` y no repite una revisión idéntica.
        "registrar_revision(usdt, revision);",
    ]);
    orden(DEMO_TEL_MOD, &[
        "fn registrar_revision(saldo: f64, revision: Option<&RevisionSaldo>) {",
        "emitir(linea_capital_update(ts_ahora_ms(), saldo, revision));",
    ]);
    assert!(texto(HOST).contains("use quantum_engine::demo_tel::CapitalBridgeSink;"));
}

fn libro(arena: &GlobalArena) -> MutexGuard<'_, LibroDimensionable> {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner())
}

/// `publicar_capital_de_sesion` del host.
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

/// El bloque de cada 100 mensajes del host: el sink financiero con el espejo.
fn publicar_telemetria(espejo: &AtomicU64) {
    telemetry_server::profiler::update_financials_atomic(1.0, 1.0, 50.0, f64::from_bits(espejo.load(Ordering::Relaxed)));
}
// ========================== FIN DEL ADAPTADOR ==========================

/// El balance publicado, leido del sink: ROI = neto / capital * 100 con neto = 1.
fn roi_publicado() -> f64 {
    telemetry_server::profiler::read_financials_snapshot().expect("lectura coherente").roi_net
}

fn roi_de(capital: f64) -> f64 {
    (1.0 / capital) * 100.0
}

#[test]
fn telemetria_publica_saldo_aceptado_no_el_dimensionable() {
    anclas_del_adaptador();
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
    let arena = GlobalArena::build_in_own_stack(0.0);
    let espejo = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let oms = Arc::new(AtomicU64::new(SESION));
    let puente = PuenteCuenta::con_sesion(SESION);
    {
        let oms = Arc::clone(&oms);
        puente.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    libro(&arena).activar(&arena);
    {
        let (arena, espejo, oms) = (Arc::clone(&arena), Arc::clone(&espejo), Arc::clone(&oms));
        puente.vincular_publicador(move |p| publicar(&oms, &arena, &espejo, p));
    }
    let socket = puente.abrir_socket(puente.era()).unwrap();

    // (1) saldo aceptado 100,0 con marca T=100 por el puente real.
    let lectura = LecturaCuenta {
        saldo: 100.0,
        tiempo_saldo_ms: Some(100),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    };
    let _ = puente.observar_rest_de_sesion(puente.ticket(), puente.sesion(), &lectura);
    assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 100.0, "el puente no acepto el saldo");

    // (2) reserva de una entrada con comision 0,04: el dimensionable queda en 99,96.
    libro(&arena).reservar(&arena, CLAVE, COMISION);
    let dimensionable = arena.unified_capital.load(Ordering::Relaxed);
    assert_eq!(dimensionable, 100.0 - COMISION, "el libro no derivo el dimensionable");

    // (3) el sink de telemetria con lo que el host le pasa.
    publicar_telemetria(&espejo);
    let roi = roi_publicado();
    // Asercion 1: el balance publicado es el saldo aceptado, bit a bit.
    assert_eq!(roi.to_bits(), roi_de(100.0).to_bits(), "{FALLO} (roi {roi})");
    assert_eq!(espejo.load(Ordering::Relaxed), 100.0f64.to_bits(), "{FALLO}");
    // Asercion 2: distinto del dimensionable.
    assert_ne!(roi.to_bits(), roi_de(dimensionable).to_bits(), "{FALLO}");
    assert_ne!(espejo.load(Ordering::Relaxed), dimensionable.to_bits(), "{FALLO}");

    // `on_capital` del stream (CAPITAL_UPDATE / update_dynamic_capital): la observacion del puente.
    // T2: el callback TIENE que producirse (antes se comprobaba solo si ocurria): un evento con
    // marca posterior (T=150) al saldo aceptado.
    let mut on_capital = None;
    let evento = EventoStream { tiempo_ms: Some(150), saldo: Some(100.0), posiciones: Vec::new() };
    let _ = puente.observar_stream_del_socket(puente.era(), socket, Some(150), &evento, |obs| {
        on_capital = Some(obs.saldo);
    });
    let usdt = on_capital.expect("TELEMETRIA: el puente no entrego la observacion del stream a on_capital");
    assert_eq!(usdt.to_bits(), 100.0f64.to_bits(), "{FALLO} (on_capital {usdt})");
    assert_ne!(usdt.to_bits(), dimensionable.to_bits(), "{FALLO} (on_capital {usdt})");

    // Asercion 3: tras cancelar la reserva, el publicado sigue siendo el saldo aceptado (INV-C).
    libro(&arena).cancelar(&arena, CLAVE, COMISION);
    publicar_telemetria(&espejo);
    assert_eq!(roi_publicado().to_bits(), roi_de(100.0).to_bits(), "{FALLO} (tras cancelar)");
    assert_eq!(espejo.load(Ordering::Relaxed), 100.0f64.to_bits(), "{FALLO} (tras cancelar)");
}
