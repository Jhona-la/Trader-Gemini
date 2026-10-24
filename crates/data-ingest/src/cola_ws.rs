//! CL-50 / CL-52 — COLA ENTRE EL LECTOR DEL WS Y EL BUCLE DE EVENTOS.
//!
//! La cola es acotada y descarta el mensaje más antiguo cuando se llena
//! (un tick viejo vale menos que uno nuevo). Dos defectos vivían en el host:
//! - Sólo la ruta del centinela contaba los descartes; los del flujo normal,
//!   que son los que ocurren, se perdían sin evidencia (CERT-M1-H01 a medias).
//! - El descarte podía tirar el propio centinela de reconexión: el bucle no
//!   reiniciaba el libro ni la guardia de secuencia tras reconectar.
//!
//! Y la primera conexión se anunciaba como reconexión: el bucle reiniciaba
//! los motores y borraba el calentamiento con K-lines REST de cada arranque.
//!
//! I-2a (Línea I): cada mensaje viaja con el instante en que el lector lo
//! sacó del socket (`MensajeWs::recibido`, reloj monótono). Sin ese sello,
//! la espera en esta cola se confundía con latencia de red (ver
//! `edad_feed`). I-2c: la cola informa por periodo sus descartes y la
//! espera máxima observada, para ver la contrapresión antes de que llene.

use crossbeam::channel::{Receiver, Sender, TrySendError};
use crossbeam::utils::CachePadded;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Mensaje en banda con el que el lector anuncia una reconexión.
pub const CENTINELA_RECONEXION: &[u8] = b"[SYSTEM:RECONNECT]";

/// Capacidad de la cola entre el lector y el bucle de eventos.
pub const CAPACIDAD_COLA_WS: usize = 5_000;

/// I-2a: un mensaje del feed con el instante en que el lector lo recibió.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MensajeWs {
    /// Bytes tal como llegaron del socket.
    pub datos: Vec<u8>,
    /// Instante de recepción en el lector (reloj monótono).
    pub recibido: Instant,
}

impl MensajeWs {
    /// Sella `datos` con el instante actual.
    #[inline]
    pub fn recibido_ahora(datos: Vec<u8>) -> Self {
        Self { datos, recibido: Instant::now() }
    }

    /// `true` si es el centinela de reconexión.
    #[inline]
    pub fn es_centinela(&self) -> bool {
        self.datos == CENTINELA_RECONEXION
    }
}

/// Contabilidad compartida entre el lector (escribe) y el bucle (lee).
#[derive(Debug, Default)]
pub struct EstadoCola {
    descartados: AtomicU64,
    reconexion_perdida: AtomicBool,
    /// I-2c: descartes desde la última lectura del monitor.
    descartes_periodo: AtomicU64,
    /// I-2c: espera local máxima (ms) desde la última lectura del monitor.
    /// La escribe el bucle en cada evento: va en su propia línea de caché,
    /// lejos de los contadores que escriben los lectores al descartar.
    espera_max_ms: CachePadded<AtomicU64>,
}

impl EstadoCola {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mensajes descartados desde el arranque.
    pub fn descartados(&self) -> u64 {
        self.descartados.load(Ordering::Relaxed)
    }

    /// `true` una sola vez si el descarte se llevó un centinela: el bucle
    /// debe reiniciar como si lo hubiera recibido. Todo lo anterior al
    /// centinela ya salió de la cola (era el más antiguo), así que el
    /// mensaje que el bucle tiene en la mano es posterior a la reconexión.
    ///
    /// CL-52b — carrera acotada: el lector saca el centinela y DESPUÉS
    /// publica la marca. Si el bucle toma un mensaje justo en ese hueco, lo
    /// procesa con el estado anterior y reinicia en el siguiente. El
    /// reinicio llega unos mensajes tarde, nunca se pierde. Evitarlo exigiría
    /// mirar la cabeza de la cola sin sacarla, y el canal no lo permite.
    pub fn tomar_reconexion_perdida(&self) -> bool {
        self.reconexion_perdida.swap(false, Ordering::AcqRel)
    }

    /// I-2c: descartes desde la lectura anterior (la reinicia).
    pub fn tomar_descartes_del_periodo(&self) -> u64 {
        self.descartes_periodo.swap(0, Ordering::Relaxed)
    }

    /// I-2c: el bucle anota la espera local de cada evento. En estado estable
    /// la espera no supera el máximo del periodo y basta una carga: el
    /// `fetch_max` (una RMW con lock en x86-64) sólo corre con un máximo nuevo.
    #[inline]
    pub fn registrar_espera_local(&self, ms: u64) {
        if ms > self.espera_max_ms.load(Ordering::Relaxed) {
            self.espera_max_ms.fetch_max(ms, Ordering::Relaxed);
        }
    }

    /// I-2c: espera local máxima desde la lectura anterior (la reinicia).
    pub fn tomar_espera_max(&self) -> u64 {
        self.espera_max_ms.swap(0, Ordering::Relaxed)
    }
}

/// Encola `msg` descartando los más antiguos mientras la cola esté llena.
/// Cuenta cada descarte efectivo y marca la reconexión si el descartado era
/// el centinela. Devuelve `false` si el bucle ya no escucha. El sello de
/// recepción viaja intacto: la cola no vuelve a sellar.
pub fn encolar(
    tx: &Sender<MensajeWs>,
    rx: &Receiver<MensajeWs>,
    msg: MensajeWs,
    estado: &EstadoCola,
) -> bool {
    let mut dato = msg;
    loop {
        match tx.try_send(dato) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(devuelto)) => {
                dato = devuelto;
                if let Ok(viejo) = rx.try_recv() {
                    estado.descartados.fetch_add(1, Ordering::Relaxed);
                    estado.descartes_periodo.fetch_add(1, Ordering::Relaxed);
                    if viejo.es_centinela() {
                        estado.reconexion_perdida.store(true, Ordering::Release);
                    }
                }
            }
        }
    }
}

/// Conexiones del lector. La primera no es una reconexión: anunciarla
/// reiniciaba los motores recién calentados.
///
/// I-1b (Línea I): el host abre un lector por categoría de streams. Sólo el
/// público (libro y trades) anuncia sus reconexiones: el reinicio que
/// provoca el centinela (motores y guardia de secuencia del libro, D-610)
/// depende de la continuidad del libro. Una reconexión del lector de
/// mercado (velas y liquidaciones) no rompe esa continuidad.
#[derive(Debug)]
pub struct ConexionesWs {
    establecidas: u64,
    anuncia: bool,
}

impl Default for ConexionesWs {
    fn default() -> Self {
        Self::new()
    }
}

impl ConexionesWs {
    /// Lector que anuncia sus reconexiones (el público).
    pub fn new() -> Self {
        Self { establecidas: 0, anuncia: true }
    }

    /// Lector que nunca anuncia reconexiones (el de mercado).
    pub fn sin_anuncio() -> Self {
        Self { establecidas: 0, anuncia: false }
    }

    /// Conexiones establecidas desde el arranque.
    pub fn establecidas(&self) -> u64 {
        self.establecidas
    }

    /// Registra una conexión nueva; `true` si hubo otra antes (reconexión) y
    /// este lector anuncia sus reconexiones.
    pub fn establecida(&mut self) -> bool {
        self.establecidas = self.establecidas.saturating_add(1);
        self.anuncia && self.establecidas > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel::bounded;
    use std::time::Duration;

    fn m(datos: &[u8]) -> MensajeWs {
        MensajeWs::recibido_ahora(datos.to_vec())
    }

    #[test]
    fn cl52_cada_descarte_cuenta() {
        let (tx, rx) = bounded::<MensajeWs>(2);
        let estado = EstadoCola::new();
        for i in 0..5u8 {
            assert!(encolar(&tx, &rx, m(&[i]), &estado));
        }
        assert_eq!(estado.descartados(), 3);
        // Sobreviven los dos más recientes, en orden.
        assert_eq!(rx.try_recv().unwrap().datos, vec![3]);
        assert_eq!(rx.try_recv().unwrap().datos, vec![4]);
        assert!(!estado.tomar_reconexion_perdida());
    }

    #[test]
    fn cl52_un_centinela_descartado_no_se_pierde() {
        let (tx, rx) = bounded::<MensajeWs>(2);
        let estado = EstadoCola::new();
        encolar(&tx, &rx, m(CENTINELA_RECONEXION), &estado);
        encolar(&tx, &rx, m(b"a"), &estado);
        // Se llena: cae el centinela, que queda anotado.
        encolar(&tx, &rx, m(b"b"), &estado);
        assert_eq!(estado.descartados(), 1);
        assert!(estado.tomar_reconexion_perdida());
        assert!(!estado.tomar_reconexion_perdida());
        assert_eq!(rx.try_recv().unwrap().datos, b"a".to_vec());
    }

    #[test]
    fn cl52_sin_receptor_no_bloquea() {
        let (tx, rx) = bounded::<MensajeWs>(1);
        let estado = EstadoCola::new();
        drop(rx);
        let (_tx_otro, rx_otro) = bounded::<MensajeWs>(1);
        assert!(!encolar(&tx, &rx_otro, m(&[1]), &estado));
    }

    #[test]
    fn cl50_la_primera_conexion_no_es_reconexion() {
        let mut c = ConexionesWs::new();
        assert!(!c.establecida());
        assert!(c.establecida());
        assert!(c.establecida());
    }

    #[test]
    fn i1b_el_lector_de_mercado_nunca_anuncia_reconexion() {
        let mut c = ConexionesWs::sin_anuncio();
        assert!(!c.establecida());
        assert!(!c.establecida());
        assert!(!c.establecida());
        assert_eq!(c.establecidas(), 3);
        // Default conserva la conducta de `new` (anuncia).
        let mut d = ConexionesWs::default();
        assert!(!d.establecida());
        assert!(d.establecida());
    }

    #[test]
    fn i2_el_sello_de_recepcion_sobrevive_a_la_cola() {
        let (tx, rx) = bounded::<MensajeWs>(2);
        let estado = EstadoCola::new();
        let Some(t0) = Instant::now().checked_sub(Duration::from_secs(5)) else {
            return; // reloj recién arrancado: sin pasado de 5 s que sellar
        };
        assert!(encolar(&tx, &rx, MensajeWs { datos: b"x".to_vec(), recibido: t0 }, &estado));
        let sacado = rx.try_recv().unwrap();
        assert_eq!(sacado.recibido, t0, "la cola no vuelve a sellar");
        assert!(sacado.recibido.elapsed() >= Duration::from_secs(5));
    }

    #[test]
    fn i2_el_centinela_se_reconoce_por_sus_datos() {
        assert!(m(CENTINELA_RECONEXION).es_centinela());
        assert!(!m(b"[SYSTEM:RECONNECT]x").es_centinela());
        assert!(!m(&[0xFF]).es_centinela());
    }

    #[test]
    fn i2c_los_descartes_se_informan_por_periodo() {
        let (tx, rx) = bounded::<MensajeWs>(2);
        let estado = EstadoCola::new();
        for i in 0..5u8 {
            assert!(encolar(&tx, &rx, m(&[i]), &estado));
        }
        assert_eq!(estado.tomar_descartes_del_periodo(), 3);
        assert_eq!(estado.tomar_descartes_del_periodo(), 0);
        assert_eq!(estado.descartados(), 3, "el total no se reinicia");
    }

    #[test]
    fn i2c_la_espera_maxima_es_del_periodo() {
        let estado = EstadoCola::new();
        for ms in [10, 300, 20] {
            estado.registrar_espera_local(ms);
        }
        assert_eq!(estado.tomar_espera_max(), 300);
        assert_eq!(estado.tomar_espera_max(), 0);
    }
}
