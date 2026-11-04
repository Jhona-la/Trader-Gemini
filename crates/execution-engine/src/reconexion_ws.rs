//! I-11 (Línea I, fila N56 del censo C-F7b): la regla de espera entre
//! reconexiones vive en `quantum_arena::reconexion_ws` (la usan también los
//! lectores del host y el feed spot); aquí se reexporta y se añade la lectura
//! del rechazo HTTP de un handshake de tungstenite.

pub use quantum_arena::reconexion_ws::*;

use std::time::Duration;

/// Espera pedida por un error de handshake de tungstenite: la de
/// [`espera_pedida`] si el error es un rechazo HTTP (429/418 con
/// `Retry-After`).
pub fn espera_pedida_por_error(e: &tokio_tungstenite::tungstenite::Error) -> Option<Duration> {
    match e {
        tokio_tungstenite::tungstenite::Error::Http(resp) => espera_pedida(
            resp.status().as_u16(),
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn espera_pedida_por_error_lee_el_rechazo_http() {
        use tokio_tungstenite::tungstenite::http::Response;
        let resp = Response::builder()
            .status(429)
            .header("Retry-After", "12")
            .body(None)
            .expect("respuesta de prueba");
        let e = tokio_tungstenite::tungstenite::Error::Http(resp);
        assert_eq!(espera_pedida_por_error(&e), Some(Duration::from_secs(12)));
        let sin_pedir = Response::builder()
            .status(503)
            .header("Retry-After", "12")
            .body(None)
            .expect("respuesta de prueba");
        let e = tokio_tungstenite::tungstenite::Error::Http(sin_pedir);
        assert_eq!(espera_pedida_por_error(&e), None);
        let otro = tokio_tungstenite::tungstenite::Error::ConnectionClosed;
        assert_eq!(espera_pedida_por_error(&otro), None);
    }
}
