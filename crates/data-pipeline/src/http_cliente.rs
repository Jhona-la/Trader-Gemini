//! F1-090 (Sol) — UN SOLO CLIENTE HTTP PARA TODO EL PROCESO.
//!
//! # QUÉ
//! `cliente_compartido()` devuelve **el mismo** `Arc<reqwest::Client>` en cada
//! llamada: se construye una sola vez por proceso con tuning explícito de
//! transporte.
//!
//! # POR QUÉ
//! Cada módulo construía el suyo —`reqwest::Client::new()` en los tres pollers
//! del multiplexor, `Client::builder()…unwrap_or_else(Client::new)` en seis
//! módulos más—, así que el proceso mantenía varios pools de conexión, cada uno
//! con `tcp_nodelay`, keep-alive y timeouts distintos o sin declarar. Un pool
//! por módulo significa un handshake TLS nuevo por cada ciclo de vida de ese
//! módulo y ninguna reutilización de conexión entre pollers: parte del coste
//! que el operador percibe como «latencia de red elevada».
//!
//! # CÓMO
//! `OnceLock<Arc<Client>>`: la primera llamada construye, las demás comparten.
//! El `Arc` permite entregarlo a tareas `'static` sin clonar el cliente (clonar
//! el `Arc` es un incremento atómico, no una copia del pool).
//!
//! # FAIL-CLOSED, SIN PÁNICO NI DEFAULT SILENCIOSO
//! Si el builder falla (backend TLS no disponible, por ejemplo) se devuelve
//! `None`: **no** se cae a un cliente por defecto en silencio —el patrón que el
//! mandato prohíbe— y **no** se panica. El llamador registra el fallo y salta
//! el ciclo, que es el comportamiento correcto cuando no hay transporte.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

static CLIENTE: OnceLock<Arc<reqwest::Client>> = OnceLock::new();

/// Conexiones ociosas que se conservan por host. Un poller hace una petición
/// cada 60–3600 s; ocho es margen de sobra para los picos y evita que el pool
/// crezca sin techo.
const MAX_IDLE_POR_HOST: usize = 8;
/// Vida de una conexión ociosa: mayor que el intervalo del poller más lento que
/// usa REST (60 s) para que la conexión siga viva entre ciclos.
const IDLE_TIMEOUT: Duration = Duration::from_secs(90);
/// Corte de establecimiento: si el handshake TCP/TLS no termina en 5 s, la ruta
/// está rota y no tiene sentido esperar al timeout global.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Techo de una petición completa. Los endpoints de datos responden en
/// milisegundos; 20 s es el fallo lento, no la latencia esperada.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// Cliente HTTP compartido del proceso.
///
/// Devuelve `None` sólo si el backend de transporte no se pudo construir; el
/// error se registra aquí, una sola vez, para que el operador lo vea.
pub fn cliente_compartido() -> Option<Arc<reqwest::Client>> {
    if let Some(ya) = CLIENTE.get() {
        return Some(ya.clone());
    }
    match reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(MAX_IDLE_POR_HOST)
        .pool_idle_timeout(IDLE_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
    {
        Ok(cliente) => {
            let nuevo = Arc::new(cliente);
            match CLIENTE.set(nuevo) {
                // Ganó esta hebra: se devuelve el que acaba de instalar.
                Ok(()) => CLIENTE.get().cloned(),
                // Otra hebra ganó la carrera; se usa el suyo, que ya está
                // construido con la misma configuración.
                Err(_) => CLIENTE.get().cloned(),
            }
        }
        Err(error) => {
            eprintln!(
                "🚨 [HTTP] No se pudo construir el cliente compartido ({error}). \
                 Los pollers REST no arrancan este ciclo; NO se usa un cliente \
                 por defecto en silencio."
            );
            None
        }
    }
}

/// `true` si el cliente compartido ya fue construido. Para diagnósticos y para
/// el contrato de instancia única.
#[inline]
pub fn cliente_ya_construido() -> bool {
    CLIENTE.get().is_some()
}
