//! CL-50 / CL-50b / CL-51 / CL-52 / CL-54: la ingesta del WS y su reinicio.
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
        t.contains("ifconexiones.establecida(){data_ingest::cola_ws::encolar(&tx_events,&rx_events_dropper,data_ingest::cola_ws::MensajeWs::recibido_ahora(data_ingest::cola_ws::CENTINELA_RECONEXION.to_vec(),),"),
        "el centinela sólo sale en una reconexión"
    );
}

/// CL-50b — la transición del calentamiento no re-suscribe el WS. Releía
/// el mismo USE_TESTNET que eligió el host al arrancar, así que sólo forzaba
/// una reconexión (con su reinicio de motores) y en mainnet abandonaba el
/// host elegido por latencia o por BEST_WS_ENDPOINT. Desde CL-38 nadie más
/// escribe la URL del WS.
#[test]
fn cl50b_la_transicion_no_reconecta_el_ws() {
    let h = host();
    assert!(!h.contains("tx_ws_control.try_send("), "nadie fuerza la reconexión del WS");
    assert!(!h.contains("ws_url.store("), "la URL del WS es la del arranque");
}

/// CL-54 — el selector de host por latencia sólo ofrece hosts USDⓈ-M. Con
/// `dstream.binance.com` (COIN-M) ganando por latencia no llegaba ningún
/// tick de los pares USDT y el calentamiento no terminaba.
#[test]
fn cl54_el_selector_de_host_solo_ofrece_usdm() {
    let h = host();
    let t = tramo(&h, "letinitial_ws_host=", "letws_url=");
    assert!(t.contains("\"fstream.binance.com\""));
    assert!(!t.contains("dstream.binance.com"), "COIN-M no sirve pares USDT-M");
}

/// Todo envío del lector pasa por la cola contada: ningún `try_send` suelto
/// que descarte sin contar ni pierda el centinela.
#[test]
fn cl52_el_lector_encola_siempre_por_la_cola_contada() {
    let h = host();
    let t = tramo(&h, "WebSocketTLSConnectedwithTCP_NODELAY", "[WS]Reconnecting...");
    assert!(!t.contains("try_send("), "sin envíos que eludan la cola contada");
    assert!(!t.contains("rx_events_dropper.try_recv()"));
    assert!(t.contains("data_ingest::cola_ws::encolar(&tx_events,&rx_events_dropper,data_ingest::cola_ws::MensajeWs::recibido_ahora(msg.into_data()),&estado_cola,)"));
}

/// Si la cola se llevó el centinela, el bucle reinicia igual, antes de
/// consumir el mensaje en la mano (que ya es posterior a la reconexión).
#[test]
fn cl52_el_bucle_reinicia_aunque_el_centinela_se_perdiera() {
    let h = host();
    let t = tramo(&h, "whileletOk(mensaje_ws)=rx_events.recv(){", "decode_liquidation_snapshots(");
    let i = t
        .find("letcentinela_perdido=estado_cola.tomar_reconexion_perdida();ifes_centinela||centinela_perdido{")
        .expect("el bucle consulta la marca de la cola antes de consumir el mensaje");
    let reinicio = &t[i..];
    assert!(reinicio.contains("engine_real.reset_engines();"));
    assert!(reinicio.contains("book_seq_guard=parsers::BookSequenceGuard::new("));
    assert!(reinicio.contains("ifes_centinela{continue;}"));
}

/// CL-156 (G021 de Línea G) — el calentamiento REST pide las K-lines al
/// venue que el host eligió para el cliente REST y el ejecutor (`is_testnet`,
/// de USE_TESTNET), no a mainnet fijo. Antes `execute_phase_3_warmup`
/// ignoraba su argumento y fijaba `fapi.binance.com`: en testnet sembraba
/// Hurst, FFT, multifractal y σ̂ con la serie de otro mercado, y el empalme
/// con el WS de testnet saltaba. El modo demo (`is_demo_mode`) no sirve:
/// SHADOW_MODE y BINANCE_USE_DEMO operan papel contra mainnet.
#[test]
fn cl156_el_calentamiento_rest_usa_el_venue_del_ejecutor() {
    let b: String = include_str!("../src/bootloader.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect();
    let cuerpo = tramo(
        &b,
        "pubasyncfnexecute_phase_3_warmup(",
        "pubasyncfnexecute_phase_4_training(",
    );
    assert!(
        !cuerpo.contains("fapi.binance.com"),
        "el calentamiento vivo no fija el host de mainnet"
    );
    assert!(cuerpo.starts_with("pubasyncfnexecute_phase_3_warmup(usa_testnet:bool,"));
    assert!(
        cuerpo.contains("warmup_klines_url(usa_testnet,"),
        "la URL sale del venue recibido"
    );

    let h = host();
    let t = tramo(
        &h,
        "let(active_key,active_secret,is_testnet)=",
        "SystemDiagnostics::execute_phase_3_warmup(",
    );
    assert!(t.contains("OrderExecutor::new(active_key,active_secret,is_testnet);"));
    assert!(
        !t.contains("letis_testnet") && !t.contains("letmutis_testnet"),
        "nadie redefine el venue entre el ejecutor y el calentamiento"
    );
    assert!(
        h.contains("SystemDiagnostics::execute_phase_3_warmup(is_testnet,&symbols"),
        "el host pasa el venue del ejecutor, no is_demo_mode"
    );
}
