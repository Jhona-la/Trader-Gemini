//! CONTRATO I-6b (Línea I, rescate GLM-168): la política de espera del feed
//! (watchdog) gobierna también la FASE DE CONEXIÓN, no sólo la lectura.
//!
//! QUÉ: DNS, carrera TCP (Happy Eyeballs) y handshake TLS de
//! `bucle_lector_ws` quedan acotados por `watchdog_secs`.
//! POR QUÉ: antes `TcpStream::connect`/`client_async_tls` no tenían timeout:
//! un handshake TLS colgado (TCP aceptado, servidor mudo) dejaba el feed
//! MUDO para siempre sin disparar el stall — el watchdog sólo vigilaba
//! DESPUÉS de conectar (docs/PLAN_LINEA_I_CLAUDE_2026-10-24.md, fila I-6b).
//! PARA QUÉ: el arranque y cada reconexión son auto-limitados; un endpoint
//! negro se abandona en watchdog_secs y se reintenta, en vez de colgar el
//! lector para siempre.
//! CÓMO: (1-4) verificación estructural del fuente del bin (la función vive
//! en `src/bin/god_engine.rs`, como en `arranque_feed_contract`); (5) la
//! semántica del timeout sobre un futuro que nunca termina.

use std::path::PathBuf;

fn fuente_god_engine() -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ruta = manifest.join("../../src/bin/god_engine.rs");
    std::fs::read_to_string(&ruta).unwrap_or_else(|e| panic!("no pude leer {:?}: {}", ruta, e))
}

fn cuerpo_bucle(fuente: &str) -> &str {
    let inicio = fuente
        .find("async fn bucle_lector_ws(")
        .expect("bucle_lector_ws debe existir");
    // Hasta la siguiente función async del bin.
    let fin = fuente[inicio + 20..]
        .find("\nasync fn ")
        .map(|i| inicio + 20 + i)
        .unwrap_or(fuente.len());
    &fuente[inicio..fin]
}

#[test]
fn i6b_1_el_watchdog_se_calcula_antes_del_bucle_y_gobierna_toda_la_fn() {
    let fuente = fuente_god_engine();
    let cuerpo = cuerpo_bucle(&fuente);
    let pos_watchdog = cuerpo
        .find("let watchdog_secs: u64")
        .expect("watchdog_secs debe existir");
    let pos_loop = cuerpo
        .find("    loop {")
        .expect("el bucle de conexión debe existir");
    assert!(
        pos_watchdog < pos_loop,
        "I-6b: watchdog_secs debe calcularse ANTES del loop (hoisted), no dentro de la rama de lectura"
    );
}

#[test]
fn i6b_2_dns_acotado_por_la_politica_de_espera() {
    let fuente = fuente_god_engine();
    let cuerpo = cuerpo_bucle(&fuente);
    let dns = cuerpo
        .find("tokio::net::lookup_host(&host_port)")
        .expect("la resolución DNS debe existir");
    let ventana = cuerpo[..dns]
        .rfind("tokio::time::timeout(")
        .unwrap_or(usize::MAX);
    // El timeout más reciente ANTES del lookup debe estar a menos de 200
    // chars (es decir, envolviéndolo, no en otra parte de la función).
    assert!(
        dns - ventana < 200,
        "I-6b: lookup_host debe quedar envuelto por tokio::time::timeout"
    );
}

#[test]
fn i6b_3_la_carrera_tcp_tiene_timeout_de_conexion() {
    let fuente = fuente_god_engine();
    let cuerpo = cuerpo_bucle(&fuente);
    let carrera = cuerpo
        .find("futures_util::future::select_ok(tasks)")
        .expect("la carrera Happy Eyeballs debe existir");
    let ventana = cuerpo[..carrera]
        .rfind("tokio::time::timeout(")
        .unwrap_or(usize::MAX);
    assert!(
        carrera - ventana < 200,
        "I-6b: select_ok debe quedar envuelto por tokio::time::timeout (timeout de conexión)"
    );
}

#[test]
fn i6b_4_el_handshake_tls_tiene_timeout_propio() {
    let fuente = fuente_god_engine();
    let cuerpo = cuerpo_bucle(&fuente);
    let tls = cuerpo
        .find("tokio_tungstenite::client_async_tls(url, tcp_stream)")
        .expect("el handshake TLS debe existir");
    let ventana = cuerpo[..tls]
        .rfind("tokio::time::timeout(")
        .unwrap_or(usize::MAX);
    assert!(
        tls - ventana < 200,
        "I-6b: client_async_tls debe quedar envuelto por tokio::time::timeout (timeout de handshake)"
    );
}

#[test]
fn i6b_5_un_futuro_que_nunca_termina_es_abandonado_por_la_ventana() {
    // Semántica de la política de espera: el timeout vence aunque el futuro
    // interior jamás resuelva — exactamente el caso del TLS colgado.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime de prueba");
    let colgado = std::future::pending::<()>();
    let inicio = std::time::Instant::now();
    let resultado = rt.block_on(async {
        tokio::time::timeout(std::time::Duration::from_millis(50), colgado).await
    });
    assert!(resultado.is_err(), "el timeout debe vencer");
    assert!(
        inicio.elapsed() < std::time::Duration::from_secs(2),
        "el abandono ocurre en la ventana, no espera al futuro"
    );
}
