//! I-11 (Línea I, fila N56 del censo C-F7b): espera entre reconexiones de los
//! clientes WS de Binance del proceso (los dos lectores del host y el
//! user-data stream, en futuros; el feed spot de data-pipeline).
//!
//! Antes, en el lector del host (desde I-6b) cada fallo de la fase de
//! conexión (DNS, TCP, handshake) esperaba 1 s fijo y hacía `continue`, y el
//! único `retry_count += 1` quedó en un brazo inalcanzable: la espera
//! exponencial de después no crecía nunca. Además el contador volvía a cero al
//! completar el handshake, así que un servidor que acepta y corta sin enviar
//! nada se reintentaba cada 100 ms. El user-data stream tenía el mismo
//! defecto: su espera volvía a 500 ms al obtener el listenKey (REST), antes de
//! abrir el WS, de modo que un WS que falla con el REST sano se reabría cada
//! 500 ms (unas 600 conexiones en 5 min). Y el feed spot, tras el corte de
//! una conexión hecha, reconectaba sin esperar nada.
//!
//! Ahora cada intento fallido o terminado pasa por [`EsperaReconexion::fallo`]
//! y el contador sólo vuelve a cero con el primer mensaje recibido
//! ([`EsperaReconexion::datos_recibidos`]): una conexión que no entregó nada
//! cuenta como fallo.
//!
//! Cada cliente conserva su escalera (lector: 100 ms·2^n con tope 5 s;
//! user-data stream: 500 ms·2^n con tope 30 s; spot: 500 ms·2^n con tope 10 s).
//! [`intentos_en_ventana`] cuenta los intentos de un fallo continuo dentro de
//! la ventana del límite de Binance (300 conexiones cada 5 min por IP en
//! futuros, docs/BINANCE_API_MAP.md; el spot tiene el mismo límite, contado
//! aparte) y el contrato `i11_reconexion_ws_contract` exige que quepan.
//!
//! La lectura del rechazo HTTP de un handshake de tungstenite vive en
//! `execution_engine::reconexion_ws`; aquí sólo la regla, sin red.
//!
//! Un rechazo 429/418 del handshake que trae `Retry-After` se respeta: la
//! espera es la mayor de las dos.

use std::time::Duration;

/// Conexiones WS permitidas por IP en la ventana de Binance.
pub const CONEXIONES_POR_VENTANA_BINANCE: u32 = 300;
/// Ventana del límite de conexiones WS de Binance (5 min).
pub const VENTANA_CONEXIONES_BINANCE: Duration = Duration::from_secs(300);

/// Escalera del lector WS del host: espera tras el primer fallo.
pub const ESPERA_BASE_LECTOR: Duration = Duration::from_millis(100);
/// Escalera del lector WS del host: tope.
pub const ESPERA_TOPE_LECTOR: Duration = Duration::from_millis(5_000);
/// Escalera del user-data stream: espera tras el primer fallo.
pub const ESPERA_BASE_USUARIO: Duration = Duration::from_millis(500);
/// Escalera del user-data stream: tope.
pub const ESPERA_TOPE_USUARIO: Duration = Duration::from_millis(30_000);
/// Escalera del feed spot: espera tras el primer fallo.
pub const ESPERA_BASE_SPOT: Duration = Duration::from_millis(500);
/// Escalera del feed spot: tope.
pub const ESPERA_TOPE_SPOT: Duration = Duration::from_millis(10_000);

/// Estado de la espera entre reconexiones de un cliente WS.
#[derive(Debug, Clone)]
pub struct EsperaReconexion {
    base: Duration,
    tope: Duration,
    fallos_seguidos: u32,
}

impl EsperaReconexion {
    /// Escalera del lector WS del host.
    pub fn lector() -> Self {
        Self::con_escalera(ESPERA_BASE_LECTOR, ESPERA_TOPE_LECTOR)
    }

    /// Escalera del user-data stream.
    pub fn usuario() -> Self {
        Self::con_escalera(ESPERA_BASE_USUARIO, ESPERA_TOPE_USUARIO)
    }

    /// Escalera del feed spot.
    pub fn spot() -> Self {
        Self::con_escalera(ESPERA_BASE_SPOT, ESPERA_TOPE_SPOT)
    }

    /// Escalera `base`·2^n con `tope`.
    pub fn con_escalera(base: Duration, tope: Duration) -> Self {
        Self {
            base,
            tope,
            fallos_seguidos: 0,
        }
    }

    /// Fallos seguidos sin un mensaje recibido.
    pub fn fallos_seguidos(&self) -> u32 {
        self.fallos_seguidos
    }

    /// Registra un intento que falló o una conexión que terminó, y devuelve
    /// cuánto esperar antes del siguiente. `rechazo` es la espera que pidió el
    /// exchange (`Retry-After` de un 429/418), si la hubo.
    pub fn fallo(&mut self, rechazo: Option<Duration>) -> Duration {
        let espera = escalon_con(self.base, self.tope, self.fallos_seguidos);
        self.fallos_seguidos = self.fallos_seguidos.saturating_add(1);
        match rechazo {
            Some(pedida) => espera.max(pedida),
            None => espera,
        }
    }

    /// La conexión entregó un mensaje: el servidor está sano y la escalera
    /// vuelve a empezar.
    pub fn datos_recibidos(&mut self) {
        self.fallos_seguidos = 0;
    }
}

/// Espera pedida por el exchange al rechazar el handshake: sólo para 429 y
/// 418, y sólo si `Retry-After` trae segundos enteros (la forma que usa
/// Binance). Otra cosa no pide espera.
pub fn espera_pedida(status: u16, retry_after: Option<&str>) -> Option<Duration> {
    if status != 429 && status != 418 {
        return None;
    }
    let segundos: u64 = retry_after?.trim().parse().ok()?;
    Some(Duration::from_secs(segundos))
}

/// Escalera exponencial: `base`·2^fallos con `tope`.
pub fn escalon_con(base: Duration, tope: Duration, fallos: u32) -> Duration {
    let mut espera = base;
    for _ in 0..fallos {
        if espera >= tope || espera.is_zero() {
            break;
        }
        espera = espera.saturating_mul(2);
    }
    espera.min(tope)
}

/// Intentos que hace, dentro de `ventana`, un cliente que falla sin parar al
/// instante y espera según la escalera (`base`, `tope`). Es el peor caso del
/// límite de conexiones: un intento que tarda en fallar abre menos.
/// Sin espera (base o tope en cero) no hay cota: devuelve `u32::MAX`.
pub fn intentos_en_ventana(ventana: Duration, base: Duration, tope: Duration) -> u32 {
    if base.is_zero() || tope.is_zero() {
        return u32::MAX;
    }
    let mut t = Duration::ZERO;
    let mut intentos = 0u32;
    let mut fallos = 0u32;
    while t < ventana {
        intentos += 1;
        t += escalon_con(base, tope, fallos);
        fallos = fallos.saturating_add(1);
    }
    intentos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_escalera_del_lector_es_la_del_bucle() {
        let ms: Vec<u64> = (0..9)
            .map(|n| escalon_con(ESPERA_BASE_LECTOR, ESPERA_TOPE_LECTOR, n).as_millis() as u64)
            .collect();
        assert_eq!(
            ms,
            vec![100, 200, 400, 800, 1_600, 3_200, 5_000, 5_000, 5_000]
        );
        assert_eq!(
            escalon_con(ESPERA_BASE_LECTOR, ESPERA_TOPE_LECTOR, u32::MAX),
            ESPERA_TOPE_LECTOR
        );
    }

    #[test]
    fn la_escalera_del_usuario_es_la_del_user_stream() {
        let ms: Vec<u64> = (0..8)
            .map(|n| escalon_con(ESPERA_BASE_USUARIO, ESPERA_TOPE_USUARIO, n).as_millis() as u64)
            .collect();
        assert_eq!(
            ms,
            vec![500, 1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000]
        );
    }

    #[test]
    fn cada_fallo_sube_un_escalon_y_un_mensaje_la_reinicia() {
        let mut e = EsperaReconexion::lector();
        assert_eq!(e.fallo(None), Duration::from_millis(100));
        assert_eq!(e.fallo(None), Duration::from_millis(200));
        assert_eq!(e.fallo(None), Duration::from_millis(400));
        assert_eq!(e.fallos_seguidos(), 3);
        e.datos_recibidos();
        assert_eq!(e.fallos_seguidos(), 0);
        assert_eq!(e.fallo(None), Duration::from_millis(100));
    }

    #[test]
    fn el_retry_after_manda_si_es_mayor() {
        let mut e = EsperaReconexion::lector();
        assert_eq!(
            e.fallo(Some(Duration::from_secs(30))),
            Duration::from_secs(30)
        );
        // El escalón siguió subiendo (200 ms) y el rechazo corto no lo baja.
        assert_eq!(
            e.fallo(Some(Duration::from_millis(1))),
            Duration::from_millis(200)
        );
    }

    #[test]
    fn espera_pedida_solo_para_429_y_418_con_segundos() {
        assert_eq!(espera_pedida(429, Some("7")), Some(Duration::from_secs(7)));
        assert_eq!(
            espera_pedida(418, Some(" 120 ")),
            Some(Duration::from_secs(120))
        );
        assert_eq!(espera_pedida(429, None), None);
        assert_eq!(
            espera_pedida(429, Some("Wed, 21 Oct 2015 07:28:00 GMT")),
            None
        );
        assert_eq!(espera_pedida(503, Some("7")), None);
        assert_eq!(espera_pedida(400, Some("7")), None);
    }

    #[test]
    fn la_escalera_no_se_cuelga_sin_espera() {
        assert_eq!(
            escalon_con(Duration::ZERO, ESPERA_TOPE_LECTOR, u32::MAX),
            Duration::ZERO
        );
        assert_eq!(
            intentos_en_ventana(
                VENTANA_CONEXIONES_BINANCE,
                Duration::ZERO,
                ESPERA_TOPE_LECTOR
            ),
            u32::MAX
        );
    }

    #[test]
    fn intentos_en_5_min_de_cada_escalera() {
        // Lector: 6 escalones (100 ms..3,2 s, 6,3 s en total) y luego uno
        // cada 5 s.
        let lector = intentos_en_ventana(
            VENTANA_CONEXIONES_BINANCE,
            ESPERA_BASE_LECTOR,
            ESPERA_TOPE_LECTOR,
        );
        assert_eq!(lector, 65);
        // Usuario: 6 escalones (500 ms..16 s, 31,5 s) y luego uno cada 30 s.
        let usuario = intentos_en_ventana(
            VENTANA_CONEXIONES_BINANCE,
            ESPERA_BASE_USUARIO,
            ESPERA_TOPE_USUARIO,
        );
        assert_eq!(usuario, 15);
        // Spot: 5 escalones (500 ms..8 s, 15,5 s) y luego uno cada 10 s.
        let spot = intentos_en_ventana(
            VENTANA_CONEXIONES_BINANCE,
            ESPERA_BASE_SPOT,
            ESPERA_TOPE_SPOT,
        );
        assert_eq!(spot, 34);
    }
}
