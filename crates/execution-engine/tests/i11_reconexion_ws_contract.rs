//! CONTRATO I-11 (Línea I, fila N56 del censo C-F7b): los clientes WS de
//! Binance del proceso esperan cada vez más entre reconexiones y no salen del
//! límite de conexiones (300 cada 5 min por IP en futuros,
//! docs/BINANCE_API_MAP.md; el spot tiene el mismo límite, contado aparte).
//!
//! Antes (desde I-6b) cada fallo de conexión del lector del host esperaba 1 s
//! fijo, el único `retry_count += 1` estaba en un brazo inalcanzable (rustc:
//! «unreachable pattern») y el contador volvía a cero al completar el
//! handshake. El user-data stream volvía a 500 ms al obtener el listenKey, antes
//! de abrir el WS: con el REST sano y el WS fallando se reabría cada 500 ms.
//! El feed spot, tras el corte de una conexión hecha, reconectaba sin esperar.
//!
//! (1) a (4) leen el fuente del lector del host (la función vive en
//! `src/bin/god_engine.rs`; su conducta la prueba
//! `i11_un_servidor_que_acepta_y_corta_no_se_reintenta_sin_espera` en ese bin);
//! (5) lee el user-data stream; (6) cuenta el peor caso de los clientes de
//! futuros; (7) lee el feed spot y cuenta el suyo.

use execution_engine::reconexion_ws::{
    intentos_en_ventana, CONEXIONES_POR_VENTANA_BINANCE, ESPERA_BASE_LECTOR, ESPERA_BASE_SPOT,
    ESPERA_BASE_USUARIO, ESPERA_TOPE_LECTOR, ESPERA_TOPE_SPOT, ESPERA_TOPE_USUARIO,
    VENTANA_CONEXIONES_BINANCE,
};
use std::path::PathBuf;

fn leer(relativa: &str) -> String {
    let ruta = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relativa);
    std::fs::read_to_string(&ruta).unwrap_or_else(|e| panic!("no pude leer {:?}: {}", ruta, e))
}

/// Fuente sin espacios ni saltos de línea.
fn compacto(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn bucle_lector() -> String {
    let fuente = leer("../../src/bin/god_engine.rs");
    let inicio = fuente
        .find("async fn bucle_lector_ws(")
        .expect("bucle_lector_ws debe existir");
    // La función acaba en la primera llave de cierre en la columna 0.
    let fin = fuente[inicio..]
        .find("\n}\n")
        .map(|i| inicio + i)
        .expect("fin de bucle_lector_ws");
    compacto(&fuente[inicio..fin])
}

fn bucle_usuario() -> String {
    let fuente = leer("src/user_data_stream.rs");
    let inicio = fuente
        .find("let base_ws = if is_testnet")
        .expect("el bucle del user-data stream debe existir");
    let fin = fuente[inicio..]
        .find("fn reconcile_after_reconnect")
        .map(|i| inicio + i)
        .expect("reconcile_after_reconnect sigue al bucle");
    compacto(&fuente[inicio..fin])
}

/// Tramo del texto compacto entre `desde` (incluido) y la primera `hasta`
/// posterior; vacío si falta alguno.
fn tramo<'a>(s: &'a str, desde: &str, hasta: &str) -> &'a str {
    let Some(i) = s.find(desde) else { return "" };
    match s[i..].find(hasta) {
        Some(j) => &s[i..i + j],
        None => "",
    }
}

#[test]
fn i11_1_ningun_fallo_de_conexion_del_lector_espera_fijo() {
    let b = bucle_lector();
    assert!(!b.contains("retry_count"), "el contador suelto desaparece");
    for fijo in [
        "Duration::from_millis(1000)).await;continue;",
        "Duration::from_secs(1)).await;continue;",
    ] {
        assert!(
            !b.contains(fijo),
            "queda una espera fija antes de reintentar: {fijo}"
        );
    }
    // DNS sin IPs, TCP fallido, TCP sin completar, par cortado, handshake sin
    // completar, handshake rechazado y el corte de una conexión hecha.
    let fallos = b.matches("espera.fallo(").count();
    assert_eq!(
        fallos, 7,
        "cada salida de la fase de conexión pasa por la escalera"
    );
}

#[test]
fn i11_2_el_lector_vuelve_a_cero_con_el_primer_mensaje_no_con_el_handshake() {
    let b = bucle_lector();
    let handshake = tramo(&b, "Ok(Ok((ws_stream,_)))=>{", "loop{");
    assert!(!handshake.is_empty(), "brazo del handshake completado");
    assert!(
        !handshake.contains("datos_recibidos"),
        "un servidor que acepta y corta sin enviar nada sigue contando como fallo"
    );
    let mensaje = tramo(&b, "Ok(Some(Ok(msg)))=>{", "Ok(Some(Err(e)))=>{");
    let reinicia = mensaje
        .find("if!msg.is_close(){espera.datos_recibidos();}")
        .unwrap_or(usize::MAX);
    let encola = mensaje.find("cola_ws::encolar(").unwrap_or(usize::MAX);
    assert!(
        reinicia < encola && encola != usize::MAX,
        "el primer mensaje que no es un Close reinicia la escalera"
    );
}

#[test]
fn i11_3_un_rechazo_429_418_del_handshake_respeta_retry_after() {
    let b = bucle_lector();
    let rechazo = tramo(
        &b,
        "Ok(Err(e))=>{//I-11:un429/418",
        "Ok(Ok((ws_stream,_)))=>{",
    );
    assert!(
        rechazo
            .contains("espera.fallo(execution_engine::reconexion_ws::espera_pedida_por_error(&e))"),
        "el rechazo HTTP del handshake pasa su Retry-After a la escalera"
    );
}

#[test]
fn i11_4_no_queda_el_brazo_inalcanzable_del_handshake() {
    let b = bucle_lector();
    let fase = tramo(&b, "client_async_tls(url,tcp_stream)", "//I-11:traselcorte");
    assert!(!fase.is_empty(), "fase de handshake del lector");
    assert!(
        !fase.contains("TLSHandshakefailed"),
        "el `Err(e)` inalcanzable se retiró"
    );
}

#[test]
fn i11_5_el_user_stream_reinicia_con_datos_no_con_el_listen_key() {
    let u = bucle_usuario();
    assert!(
        !u.contains("backoff_ms"),
        "la espera vieja del user stream desaparece"
    );
    assert!(
        u.contains("letmutespera=crate::reconexion_ws::EsperaReconexion::usuario();"),
        "el user stream usa su escalera"
    );
    let listen_key = tramo(&u, "Ok(k)if!k.is_empty()=>", "Ok(_)=>{");
    assert!(
        !listen_key.is_empty() && !listen_key.contains("datos_recibidos"),
        "obtener el listenKey (REST) no prueba que el WS funcione"
    );
    let texto = tramo(&u, "Message::Text(text))=>{", "self.route_event(&text);");
    assert!(
        texto.contains("espera.datos_recibidos();"),
        "un evento reinicia la escalera"
    );
    let ping = tramo(&u, "Message::Ping(_))=>{", "Message::Close(frame))=>{");
    assert!(
        ping.contains("espera.datos_recibidos();"),
        "el ping del servidor también"
    );
    let cierre = tramo(&u, "Message::Close(frame))=>{", "Err(e)=>{");
    assert!(
        !cierre.contains("datos_recibidos"),
        "un Close no la reinicia"
    );
    // listenKey vacío, listenKey fallido, WS fallido y el corte.
    assert_eq!(
        u.matches("espera.fallo(").count(),
        4,
        "cada salida pasa por la escalera"
    );
    assert!(
        u.contains("espera.fallo(crate::reconexion_ws::espera_pedida_por_error(&e))"),
        "el rechazo HTTP del WS pasa su Retry-After"
    );
    assert!(
        !u.contains("Duration::from_secs(1)).await;"),
        "tras el corte ya no espera 1 s fijo"
    );
}

#[test]
fn i11_6_los_clientes_ws_de_futuros_caben_en_el_limite_de_binance() {
    // Lectores del host: el público y el de mercado (sin contar las pruebas
    // del bin, que construyen el suyo).
    let fuente = leer("../../src/bin/god_engine.rs");
    let produccion = fuente.split("#[cfg(test)]").next().unwrap_or("");
    let host = compacto(produccion);
    let lectores = host.matches("=LectorWs{").count() as u32;
    assert_eq!(lectores, 2, "lectores WS que construye el host");
    assert_eq!(
        host.matches("reconexion_ws::EsperaReconexion::lector()")
            .count(),
        1,
        "los lectores usan la escalera del lector"
    );

    let por_lector = intentos_en_ventana(
        VENTANA_CONEXIONES_BINANCE,
        ESPERA_BASE_LECTOR,
        ESPERA_TOPE_LECTOR,
    );
    let por_usuario = intentos_en_ventana(
        VENTANA_CONEXIONES_BINANCE,
        ESPERA_BASE_USUARIO,
        ESPERA_TOPE_USUARIO,
    );
    let total = lectores * por_lector + por_usuario;
    assert!(
        total <= CONEXIONES_POR_VENTANA_BINANCE,
        "peor caso en 5 min: {lectores}×{por_lector} + {por_usuario} = {total}"
    );
}

fn bucle_spot() -> String {
    let fuente = leer("../data-pipeline/src/spot_feed.rs");
    let inicio = fuente
        .find("async fn run_spot_ws_streamer(")
        .expect("run_spot_ws_streamer debe existir");
    let fin = fuente[inicio..]
        .find(
            "
}
",
        )
        .map(|i| inicio + i)
        .expect("fin de run_spot_ws_streamer");
    compacto(&fuente[inicio..fin])
}

#[test]
fn i11_7_el_feed_spot_espera_tras_un_corte_y_cabe_en_su_limite() {
    let sp = bucle_spot();
    assert!(
        !sp.contains("backoff_ms"),
        "la espera vieja del spot desaparece"
    );
    assert!(
        sp.contains("letmutespera=quantum_arena::reconexion_ws::EsperaReconexion::spot();"),
        "el spot usa su escalera"
    );
    let conectado = tramo(&sp, "Ok((mutws_stream,_))=>{", "Message::Text(text))=>{");
    assert!(
        !conectado.is_empty() && !conectado.contains("datos_recibidos"),
        "conectar no reinicia la escalera"
    );
    let texto = tramo(
        &sp,
        "Message::Text(text))=>{",
        "parse_ws_book_ticker(&text)",
    );
    assert!(
        texto.contains("espera.datos_recibidos();"),
        "un mensaje la reinicia"
    );
    let cierre = tramo(&sp, "Message::Close(_))=>", "Err(_)=>break");
    assert!(
        !cierre.contains("datos_recibidos"),
        "un Close no la reinicia"
    );
    // El rechazo del handshake y el corte de una conexión hecha.
    assert_eq!(
        sp.matches("espera.fallo(").count(),
        2,
        "cada salida pasa por la escalera"
    );
    assert!(
        sp.contains("espera.fallo(None)).await;}"),
        "tras el corte de una conexión hecha espera (antes reconectaba al instante)"
    );
    assert!(
        !sp.contains("let_=ws_stream"),
        "el Pong fallido ya no se descarta"
    );

    let por_spot = intentos_en_ventana(
        VENTANA_CONEXIONES_BINANCE,
        ESPERA_BASE_SPOT,
        ESPERA_TOPE_SPOT,
    );
    assert!(
        por_spot <= CONEXIONES_POR_VENTANA_BINANCE,
        "peor caso del spot en 5 min: {por_spot}"
    );
}
