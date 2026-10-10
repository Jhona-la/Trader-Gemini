//! CL-50 / CL-51 / CL-52: la ingesta del WS y su reinicio.
//!
//! El bucle de eventos reinicia el estado de mercado cuando el lector del WS
//! se reconecta. Tres defectos: la primera conexión también se anunciaba
//! como reconexión (borraba el calentamiento con K-lines REST de cada
//! arranque), el reinicio devolvía el pico del drawdown a la base de la
//! cuenta, y la cola llena podía descartar el centinela sin dejar rastro.
//!
//! Las guardias sobre la fuente siguen el estilo de
//! `kill_switch_host_contract.rs`: sin comentarios de línea completa y con el
//! espacio compactado.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;

fn host() -> String {
    include_str!("../../../src/bin/god_engine.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let i = h.find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..]
        .find(hasta)
        .unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

/// El reinicio por reconexión es del feed. Antes `reset_engines` también
/// hacía `risk_engine.reset(base_capital)`: con un máximo de 20 USD y la
/// cuenta en 15, tras la reconexión el veto medía la caída desde 13.
#[test]
fn cl51_una_reconexion_no_olvida_el_pico_del_drawdown() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    core.swing_nn = None;
    core.scalp_forest = None;
    core.risk_engine.peak_capital = 20.0;
    core.reset_engines();
    assert_eq!(core.risk_engine.peak_capital, 20.0);
}

/// La primera conexión no envía el centinela: el calentamiento con K-lines
/// REST se inyecta antes del bucle y el centinela lo borraba en cada arranque.
#[test]
fn cl50_solo_una_reconexion_anuncia_el_reinicio() {
    let h = host();
    let t = tramo(&h, "WebSocketTLSConnectedwithTCP_NODELAY", "ws_stream.split()");
    assert!(
        t.contains("ifconexiones.establecida(){data_ingest::cola_ws::encolar(&tx_events,&rx_events_dropper,data_ingest::cola_ws::CENTINELA_RECONEXION.to_vec(),"),
        "el centinela sólo sale en una reconexión"
    );
}

/// Todo envío del lector pasa por la cola contada: ningún `try_send` suelto
/// que descarte sin contar ni pierda el centinela.
#[test]
fn cl52_el_lector_encola_siempre_por_la_cola_contada() {
    let h = host();
    let t = tramo(&h, "WebSocketTLSConnectedwithTCP_NODELAY", "[WS]Reconnecting...");
    assert!(!t.contains("try_send("), "sin envíos que eludan la cola contada");
    assert!(!t.contains("rx_events_dropper.try_recv()"));
    assert!(t.contains("data_ingest::cola_ws::encolar(&tx_events,&rx_events_dropper,msg.into_data(),&estado_cola,)"));
}

/// Si la cola se llevó el centinela, el bucle reinicia igual, antes de
/// consumir el mensaje en la mano (que ya es posterior a la reconexión).
#[test]
fn cl52_el_bucle_reinicia_aunque_el_centinela_se_perdiera() {
    let h = host();
    let t = tramo(&h, "whileletOk(mutmsg_bytes)=rx_events.recv(){", "decode_liquidation_snapshots(");
    let i = t
        .find("letcentinela_perdido=estado_cola.tomar_reconexion_perdida();ifes_centinela||centinela_perdido{")
        .expect("el bucle consulta la marca de la cola antes de consumir el mensaje");
    let reinicio = &t[i..];
    assert!(reinicio.contains("engine_real.reset_engines();"));
    assert!(reinicio.contains("book_seq_guard=parsers::BookSequenceGuard::new("));
    assert!(reinicio.contains("ifes_centinela{continue;}"));
}
