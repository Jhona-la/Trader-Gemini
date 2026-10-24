//! Contrato F1-090 (Sol): un solo cliente HTTP por proceso, con transporte real.
//!
//! # QUÉ
//! `cliente_compartido()` devuelve **la misma instancia** en cada llamada
//! (`Arc::ptr_eq`) y ese cliente completa una petición HTTP real sobre un
//! socket real.
//!
//! # POR QUÉ
//! Cada módulo construía su propio `reqwest::Client` —tres pollers del
//! multiplexor con `Client::new()` y seis módulos más con
//! `.unwrap_or_else(Client::new)`—, de modo que el proceso mantenía varios
//! pools de conexión con `tcp_nodelay`, keep-alive y timeouts distintos o sin
//! declarar: un handshake TLS por módulo y cero reutilización de conexión entre
//! pollers. Es una de las fuentes del coste que el operador percibe como
//! «latencia de red elevada».
//!
//! # CÓMO
//! `OnceLock<Arc<Client>>` construido con `tcp_nodelay(true)`, pool acotado
//! (`pool_max_idle_per_host`), keep-alive de 90 s y timeouts explícitos de
//! conexión y de petición. Si el backend TLS no se puede construir se devuelve
//! `None` (fail-closed): ni pánico ni cliente por defecto en silencio.
//!
//! # DÓNDE
//! `crates/data-pipeline/src/http_cliente.rs`; consumido por los pollers REST de
//! `omni_multiplexer`.
//!
//! # QUIÉN
//! Sol — átomo 2.2 del plan maestro por fases (2026-10-22).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use data_pipeline::http_cliente::{cliente_compartido, cliente_ya_construido};

/// Servidor HTTP/1.1 mínimo sobre un socket real: acepta una conexión, lee la
/// petición y responde `200 OK` con cuerpo `ok`. No hay mock ni intercepción:
/// el cliente habla TCP de verdad.
fn servidor_de_una_respuesta() -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local");
    let addr = listener.local_addr().expect("addr local");
    let handle = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let cuerpo = "ok";
            let respuesta = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{cuerpo}",
                cuerpo.len()
            );
            let _ = stream.write_all(respuesta.as_bytes());
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}/ping"), handle)
}

/// La instancia es ÚNICA: dos llamadas devuelven el mismo `Arc`.
#[test]
fn f1_090_cliente_es_una_sola_instancia() {
    let primero = cliente_compartido().expect("el transporte debe construirse");
    let segundo = cliente_compartido().expect("el transporte debe construirse");
    assert!(
        Arc::ptr_eq(&primero, &segundo),
        "dos llamadas deben devolver el MISMO cliente (pool compartido), no dos pools"
    );
    assert!(
        cliente_ya_construido(),
        "tras la primera llamada el cliente debe quedar instalado en el proceso"
    );
}

/// El cliente compartido completa una petición HTTP real contra un socket real.
#[tokio::test]
async fn f1_090_cliente_compartido_completa_una_peticion_real() {
    let (url, servidor) = servidor_de_una_respuesta();
    let cliente = cliente_compartido().expect("el transporte debe construirse");
    let respuesta = cliente
        .get(&url)
        .send()
        .await
        .expect("la petición al servidor local debe completarse");
    assert_eq!(respuesta.status().as_u16(), 200);
    let cuerpo = respuesta.text().await.expect("el cuerpo debe leerse");
    assert_eq!(cuerpo, "ok");
    let _ = servidor.join();
}

/// Dos peticiones seguidas usan el MISMO cliente (y por tanto el mismo pool):
/// la segunda no reconstruye nada.
#[tokio::test]
async fn f1_090_dos_peticiones_seguidas_reutilizan_el_pool() {
    let (url_uno, servidor_uno) = servidor_de_una_respuesta();
    let (url_dos, servidor_dos) = servidor_de_una_respuesta();
    let cliente = cliente_compartido().expect("el transporte debe construirse");
    for url in [url_uno, url_dos] {
        let respuesta = cliente.get(&url).send().await.expect("petición local");
        assert_eq!(respuesta.status().as_u16(), 200);
    }
    let _ = servidor_uno.join();
    let _ = servidor_dos.join();
}
