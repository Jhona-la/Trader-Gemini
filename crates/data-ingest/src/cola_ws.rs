//! CL-52 — COLA ENTRE EL LECTOR DEL WS Y EL BUCLE DE EVENTOS.
//!
//! La cola es acotada y descarta el mensaje más antiguo cuando se llena
//! (un tick viejo vale menos que uno nuevo). Dos defectos vivían en el host:
//! - Sólo la ruta del centinela contaba los descartes; los del flujo normal,
//!   que son los que ocurren, se perdían sin evidencia (CERT-M1-H01 a medias).
//! - El descarte podía tirar el propio centinela de reconexión: el bucle no
//!   reiniciaba el libro ni la guardia de secuencia tras reconectar.

use crossbeam::channel::{Receiver, Sender, TrySendError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Mensaje en banda con el que el lector anuncia una reconexión.
pub const CENTINELA_RECONEXION: &[u8] = b"[SYSTEM:RECONNECT]";

/// Contabilidad compartida entre el lector (escribe) y el bucle (lee).
#[derive(Debug, Default)]
pub struct EstadoCola {
    descartados: AtomicU64,
    reconexion_perdida: AtomicBool,
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
    /// centinela también se descartó (era el más antiguo), así que el
    /// mensaje que el bucle tiene en la mano es el primero posterior.
    pub fn tomar_reconexion_perdida(&self) -> bool {
        self.reconexion_perdida.swap(false, Ordering::AcqRel)
    }
}

/// Encola `dato` descartando los más antiguos mientras la cola esté llena.
/// Cuenta cada descarte efectivo y marca la reconexión si el descartado era
/// el centinela. Devuelve `false` si el bucle ya no escucha.
pub fn encolar(
    tx: &Sender<Vec<u8>>,
    rx: &Receiver<Vec<u8>>,
    dato: Vec<u8>,
    estado: &EstadoCola,
) -> bool {
    let mut dato = dato;
    loop {
        match tx.try_send(dato) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(devuelto)) => {
                dato = devuelto;
                if let Ok(viejo) = rx.try_recv() {
                    estado.descartados.fetch_add(1, Ordering::Relaxed);
                    if viejo == CENTINELA_RECONEXION {
                        estado.reconexion_perdida.store(true, Ordering::Release);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel::bounded;

    #[test]
    fn cl52_cada_descarte_cuenta() {
        let (tx, rx) = bounded::<Vec<u8>>(2);
        let estado = EstadoCola::new();
        for i in 0..5u8 {
            assert!(encolar(&tx, &rx, vec![i], &estado));
        }
        assert_eq!(estado.descartados(), 3);
        // Sobreviven los dos más recientes, en orden.
        assert_eq!(rx.try_recv().unwrap(), vec![3]);
        assert_eq!(rx.try_recv().unwrap(), vec![4]);
        assert!(!estado.tomar_reconexion_perdida());
    }

    #[test]
    fn cl52_un_centinela_descartado_no_se_pierde() {
        let (tx, rx) = bounded::<Vec<u8>>(2);
        let estado = EstadoCola::new();
        encolar(&tx, &rx, CENTINELA_RECONEXION.to_vec(), &estado);
        encolar(&tx, &rx, b"a".to_vec(), &estado);
        // Se llena: cae el centinela, que queda anotado.
        encolar(&tx, &rx, b"b".to_vec(), &estado);
        assert_eq!(estado.descartados(), 1);
        assert!(estado.tomar_reconexion_perdida());
        assert!(!estado.tomar_reconexion_perdida());
        assert_eq!(rx.try_recv().unwrap(), b"a".to_vec());
    }

    #[test]
    fn cl52_sin_receptor_no_bloquea() {
        let (tx, rx) = bounded::<Vec<u8>>(1);
        let estado = EstadoCola::new();
        drop(rx);
        let (_tx_otro, rx_otro) = bounded::<Vec<u8>>(1);
        assert!(!encolar(&tx, &rx_otro, vec![1], &estado));
    }
}
