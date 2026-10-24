//! I-2 (Línea I) — contrato de fuente del lado consumidor de la edad del dato.
//!
//! El host medía `latency_ms = ahora − E` tras sacar el evento de la cola, sin
//! saber cuánto había esperado en ella. I-2 sella cada mensaje en el lector
//! (`cola_ws::MensajeWs`) y descompone la edad en red+reloj y espera local
//! (`edad_feed::EdadEvento`). Este contrato fija que el bucle usa el sello del
//! lector y un solo instante para ambas medidas, y que las decisiones siguen
//! con la edad TOTAL: el núcleo bloquea entradas con datos viejos aunque la
//! red esté sana (premisa de I-3a).

const HOST: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../src/bin/god_engine.rs"));

fn compacto() -> String {
    HOST.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let i = h.find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..].find(hasta).unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

#[test]
fn i2_el_bucle_toma_el_sello_del_lector_al_sacar_el_mensaje() {
    let h = compacto();
    assert!(
        h.contains("whileletOk(mensaje_ws)=rx_events.recv(){letrecibido_ws=mensaje_ws.recibido;letmutmsg_bytes=mensaje_ws.datos;"),
        "el sello de recepción debe leerse del mensaje, no volver a sellarse en el bucle"
    );
}

#[test]
fn i2_la_edad_se_descompone_con_un_solo_instante_y_decide_la_total() {
    let h = compacto();
    let t = tramo(&h, "letahora_mono=Instant::now();", "letmutlatency_panic=false;");
    assert!(t.contains("ahora_mono.saturating_duration_since(instant_baseline)"), "now_ms con el mismo instante");
    assert!(!t.contains("instant_baseline.elapsed()"), "un segundo instante desalinearía las dos medidas");
    assert!(
        t.contains("letedad=data_ingest::edad_feed::EdadEvento::medir(event_time,now_ms,ahora_mono.saturating_duration_since(recibido_ws),);letlatency_ms=edad.total_ms;"),
        "la espera local se mide desde el sello y las decisiones siguen con la edad total"
    );
    assert!(t.contains("estado_cola.registrar_espera_local("));
}

#[test]
fn i2_los_avisos_de_latencia_llevan_la_descomposicion() {
    assert!(HOST.contains("red+reloj {} ms, cola local {} ms, deriva del reloj {} ms). Skiping"));
    assert!(HOST.contains("(última: red+reloj {} ms, cola local {} ms, deriva del reloj {} ms)"));
    let h = compacto();
    assert!(h.contains(".deriva_reloj_pared_ms.store(deriva_reloj_ms,Ordering::Relaxed);"));
}

#[test]
fn i2c_el_monitor_informa_la_contrapresion_por_periodo() {
    let h = compacto();
    let t = tramo(&h, "\"ws-backpressure-monitor\"", "letmuttestnet_key=");
    assert!(t.contains("estado_cola.tomar_descartes_del_periodo()"));
    assert!(t.contains("estado_cola.tomar_espera_max()"));
    assert!(t.contains("telemetry_server::telemetry_log!("), "por la cola de telemetría, no por stdout");
    assert!(!t.contains("println!("));
}
