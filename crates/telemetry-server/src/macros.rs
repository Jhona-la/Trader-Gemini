//! Wait-Free Telemetry Macros
//! FASE 16: Pervasive Wait-Free Telemetry
//! These macros allow printing information and warnings without blocking the Hot Path.
//! They submit logs into a global lock-free ArrayQueue.

use std::fmt::{self, Write};
use std::sync::{Arc, OnceLock};

/// Limit the max string length to avoid large dynamic allocations in hot paths.
pub const MAX_LOG_LEN: usize = 512;

#[derive(Clone, Copy)]
pub struct FixedLogBuffer {
    pub data: [u8; MAX_LOG_LEN],
    pub len: usize,
}

impl FixedLogBuffer {
    pub fn new() -> Self {
        Self {
            data: [0; MAX_LOG_LEN],
            len: 0,
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len]).unwrap_or("UTF8_ERR")
    }
}

impl Write for FixedLogBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let remaining = MAX_LOG_LEN - self.len;
        // LG-19: cortar en frontera de carácter. Un corte dentro de un
        // carácter multibyte (un emoji son 4 B) dejaba el buffer como UTF-8
        // inválido y `as_str` tiraba el mensaje entero («UTF8_ERR»).
        let mut to_copy = std::cmp::min(bytes.len(), remaining);
        while to_copy > 0 && !s.is_char_boundary(to_copy) {
            to_copy -= 1;
        }
        if to_copy > 0 {
            self.data[self.len..self.len + to_copy].copy_from_slice(&bytes[..to_copy]);
            self.len += to_copy;
        }
        Ok(())
    }
}

// FASE VII: Zero-Latency RingBuffer (Lock-Free)
// Usamos ArrayQueue en lugar de channels MPMC para garantizar latencia O(1) determinística.
pub static LOG_QUEUE: OnceLock<Arc<crossbeam::queue::ArrayQueue<FixedLogBuffer>>> = OnceLock::new();

/// LG-11 (Línea G): capacidad por defecto de la cola de logs. Con
/// 1 000 000 entradas de 520 B la cola reservaba ~520 MB; 65 536 son ~34 MB.
pub const CAPACIDAD_LOG_POR_DEFECTO: usize = 65_536;

/// Inicializa la cola global de logs asíncronos en modo RingBuffer Lock-Free
pub fn init_telemetry_logger() {
    init_telemetry_logger_con_capacidad(CAPACIDAD_LOG_POR_DEFECTO);
}

/// LG-11 (Línea G) — REGISTRO ASÍNCRONO ACOTADO Y SIN GIRO ACTIVO.
///
/// QUÉ ESTABA MAL: el host nunca inicializaba la cola, así que cada
/// `telemetry_log!` del bucle vivo hacía un `println!` síncrono (en Windows,
/// una selección en la consola con QuickEdit congela la escritura y con ella
/// el bucle). Y la cola, cuando se usaba, reservaba ~520 MB y su hilo giraba
/// con `yield_now` ocupando un núcleo entero.
///
/// QUÉ GARANTIZA: capacidad acotada (`force_push` descarta lo más antiguo si
/// se llena), el consumidor drena por lotes con un único bloqueo de stdout y
/// duerme 200 µs cuando la cola está vacía. Idempotente: una segunda llamada
/// no crea otro hilo. Devuelve `true` si inicializó.
pub fn init_telemetry_logger_con_capacidad(capacidad: usize) -> bool {
    let queue = Arc::new(crossbeam::queue::ArrayQueue::new(capacidad.max(1)));
    if LOG_QUEUE.set(queue.clone()).is_err() {
        return false;
    }
    std::thread::Builder::new()
        .name("GhostLogger".to_string())
        .spawn(move || loop {
            if vaciar_cola(&queue) == 0 {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        })
        .expect("Fallo al crear hilo GhostLogger");
    true
}

/// Escribe en stdout todo lo que hay en la cola, con un solo bloqueo.
fn vaciar_cola(queue: &crossbeam::queue::ArrayQueue<FixedLogBuffer>) -> usize {
    use std::io::Write as _;
    if queue.is_empty() {
        return 0;
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut n = 0;
    while let Some(msg) = queue.pop() {
        let _ = out.write_all(msg.as_str().as_bytes());
        let _ = out.write_all(b"\n");
        n += 1;
    }
    let _ = out.flush();
    n
}

/// LG-11: vacía la cola de logs de forma síncrona (apagado, pruebas).
/// Devuelve cuántos mensajes escribió.
pub fn vaciar_telemetria() -> usize {
    LOG_QUEUE.get().map(|q| vaciar_cola(q)).unwrap_or(0)
}

#[cfg(test)]
mod tests_lg11 {
    use super::*;

    #[test]
    fn lg11_la_cola_es_acotada_e_idempotente_y_se_vacia() {
        assert!(init_telemetry_logger_con_capacidad(1_024));
        assert!(!init_telemetry_logger_con_capacidad(8), "la segunda llamada no reinicializa");
        let q = LOG_QUEUE.get().expect("cola");
        assert_eq!(q.capacity(), 1_024);
        for i in 0..2_000 {
            crate::telemetry_log!("mensaje {}", i);
        }
        assert!(q.len() <= 1_024, "force_push descarta lo más antiguo: {}", q.len());
        let _ = vaciar_telemetria();
        assert!(q.len() <= 1_024);
    }

    #[test]
    fn lg11_la_capacidad_por_defecto_no_reserva_cientos_de_megas() {
        let bytes = CAPACIDAD_LOG_POR_DEFECTO * std::mem::size_of::<FixedLogBuffer>();
        assert!(bytes < 64 * 1024 * 1024, "{bytes} bytes");
    }
}

#[macro_export]
macro_rules! telemetry_log {
    ($($arg:tt)*) => {{
        if let Some(queue) = $crate::macros::LOG_QUEUE.get() {
            use std::fmt::Write;
            let mut buf = $crate::macros::FixedLogBuffer::new();
            let _ = write!(&mut buf, $($arg)*);
            let _ = queue.force_push(buf);
        } else {
            println!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! telemetry_err {
    ($($arg:tt)*) => {{
        if let Some(queue) = $crate::macros::LOG_QUEUE.get() {
            use std::fmt::Write;
            let mut buf = $crate::macros::FixedLogBuffer::new();
            let _ = write!(&mut buf, "⚠️ ERROR: ");
            let _ = write!(&mut buf, $($arg)*);
            let _ = queue.force_push(buf);
        } else {
            eprintln!($($arg)*);
        }
    }};
}


#[cfg(test)]
mod tests_lg19 {
    use super::*;

    #[test]
    fn lg19_un_corte_dentro_de_un_emoji_conserva_el_mensaje() {
        let mut buf = FixedLogBuffer::new();
        // 510 B de ASCII y un emoji de 4 B: el corte a 512 cae dentro del emoji.
        let relleno = "a".repeat(MAX_LOG_LEN - 2);
        let _ = write!(&mut buf, "{}🚀 resto", relleno);
        assert_eq!(buf.as_str(), relleno);
        assert_eq!(buf.len, MAX_LOG_LEN - 2);
    }

    #[test]
    fn lg19_un_mensaje_corto_queda_intacto() {
        let mut buf = FixedLogBuffer::new();
        let _ = write!(&mut buf, "🚪 [CLOSE TRACE] #{} ok", 7);
        assert_eq!(buf.as_str(), "🚪 [CLOSE TRACE] #7 ok");
    }
}
