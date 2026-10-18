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
    /// LG-19b: el mensaje se truncó; no se escribe nada más.
    pub cerrado: bool,
}

impl FixedLogBuffer {
    pub fn new() -> Self {
        Self {
            data: [0; MAX_LOG_LEN],
            len: 0,
            cerrado: false,
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.data[..self.len]).unwrap_or("UTF8_ERR")
    }
}

impl Write for FixedLogBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.cerrado {
            return Ok(());
        }
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
        // LG-19b: tras un corte el mensaje queda cerrado. Si no, las piezas
        // siguientes del mismo `write!` rellenaban los 1–3 bytes libres y la
        // línea dejaba de ser un prefijo del original (p. ej. un ts truncado
        // que parece válido).
        if to_copy < bytes.len() {
            self.cerrado = true;
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
    // CL-144: sin hilo la cola sigue acotada (force_push) y el apagado la
    // vacía con `vaciar_telemetria`; no se aborta el proceso por el log.
    if let Err(error) = std::thread::Builder::new()
        .name("GhostLogger".to_string())
        .spawn(move || loop {
            if vaciar_cola(&queue) == 0 {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        })
    {
        eprintln!("⚠️ [LOG] no se pudo crear el hilo GhostLogger: {error}");
    }
    true
}

/// CL-144 (C-F6) — LOG DEL MOTOR CON ROTACIÓN Y COMPRESIÓN.
///
/// QUÉ ESTABA MAL: todo `telemetry_log!`/`telemetry!` del proceso acaba en
/// stdout, que el lanzador redirige a un único `logs/demo_vNN.log`. Con el
/// handle abierto por el proceso, nada externo puede rotarlo, y el archivo
/// crece mientras dure la sesión.
///
/// QUÉ GARANTIZA: con `TG_LOG_FILE` definido, el hilo GhostLogger escribe en
/// ese archivo y, al pasar `tope_bytes`, lo comprime a `<ruta>.1.gz` y
/// desplaza los anteriores hasta `<ruta>.<conservar>.gz`; el más viejo se
/// borra. En disco hay como mucho el activo, `conservar` segmentos
/// comprimidos y, si una compresión falla, un `<ruta>.1.sin_comprimir`. La
/// rotación corre en el hilo del log, nunca en el camino caliente. Sin la
/// variable, todo sigue yendo a stdout como antes. Los `println!` directos
/// (fuera de las macros) siguen yendo a stdout.
#[derive(Debug, Clone)]
pub struct Rotacion {
    pub ruta: std::path::PathBuf,
    pub tope_bytes: u64,
    pub conservar: usize,
}

impl Rotacion {
    /// `TG_LOG_FILE` (ruta del log), `TG_LOG_MAX_MB` (tope del activo, 64 por
    /// defecto) y `TG_LOG_KEEP` (segmentos comprimidos, 8 por defecto).
    pub fn desde_entorno() -> Option<Self> {
        let ruta = std::env::var("TG_LOG_FILE").ok().filter(|r| !r.trim().is_empty())?;
        let entero = |clave: &str, defecto: u64| {
            std::env::var(clave).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(defecto)
        };
        Some(Self {
            ruta: ruta.trim().into(),
            tope_bytes: entero("TG_LOG_MAX_MB", 64).clamp(1, 4_096) * 1024 * 1024,
            conservar: entero("TG_LOG_KEEP", 8).clamp(1, 100) as usize,
        })
    }

    fn segmento(&self, i: usize, sufijo: &str) -> std::path::PathBuf {
        let mut s = self.ruta.clone().into_os_string();
        s.push(format!(".{i}{sufijo}"));
        s.into()
    }
}

/// Archivo activo del log rotativo.
struct Sumidero {
    rot: Rotacion,
    archivo: Option<std::io::BufWriter<std::fs::File>>,
    escritos: u64,
}

impl Sumidero {
    fn abrir(rot: Rotacion) -> std::io::Result<Self> {
        if let Some(dir) = rot.ruta.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let f = std::fs::OpenOptions::new().create(true).append(true).open(&rot.ruta)?;
        let escritos = f.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self { rot, archivo: Some(std::io::BufWriter::new(f)), escritos })
    }

    fn escribir_lote(&mut self, queue: &crossbeam::queue::ArrayQueue<FixedLogBuffer>) -> usize {
        use std::io::Write as _;
        let mut n = 0;
        while let Some(msg) = queue.pop() {
            let linea = msg.as_str().as_bytes();
            let largo = linea.len() as u64 + 1;
            if self.escritos > 0 && self.escritos.saturating_add(largo) > self.rot.tope_bytes {
                self.rotar();
            }
            match self.archivo.as_mut() {
                Some(w) => {
                    if w.write_all(linea).and_then(|_| w.write_all(b"\n")).is_ok() {
                        self.escritos = self.escritos.saturating_add(largo);
                    }
                }
                None => {
                    let mut out = std::io::stdout().lock();
                    let _ = out.write_all(linea);
                    let _ = out.write_all(b"\n");
                }
            }
            n += 1;
        }
        if let Some(w) = self.archivo.as_mut() {
            let _ = w.flush();
        }
        n
    }

    fn rotar(&mut self) {
        use std::io::Write as _;
        if let Some(mut w) = self.archivo.take() {
            let _ = w.flush();
        }
        if let Err(error) = comprimir_y_desplazar(&self.rot) {
            eprintln!("⚠️ [LOG] no se pudo comprimir {}: {error}", self.rot.ruta.display());
            let _ = std::fs::remove_file(self.rot.segmento(1, ".gz.tmp"));
            let _ = std::fs::rename(&self.rot.ruta, self.rot.segmento(1, ".sin_comprimir"));
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.rot.ruta)
        {
            Ok(f) => {
                self.archivo = Some(std::io::BufWriter::new(f));
                self.escritos = 0;
            }
            Err(error) => {
                eprintln!("⚠️ [LOG] no se pudo reabrir {}: {error}; sigo en stdout", self.rot.ruta.display());
            }
        }
    }
}

/// Desplaza `<ruta>.i.gz` a `<ruta>.(i+1).gz`, borra el que sobra y comprime
/// el activo a `<ruta>.1.gz` (vía `.gz.tmp`, para no dejar un .gz a medias).
fn comprimir_y_desplazar(rot: &Rotacion) -> std::io::Result<()> {
    use std::io::Write as _;
    let conservar = rot.conservar.max(1);
    let tmp = rot.segmento(1, ".gz.tmp");
    {
        let mut entrada = std::fs::File::open(&rot.ruta)?;
        let salida = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut gz = flate2::write::GzEncoder::new(salida, flate2::Compression::fast());
        std::io::copy(&mut entrada, &mut gz)?;
        gz.finish()?.flush()?;
    }
    match std::fs::remove_file(rot.segmento(conservar, ".gz")) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    for i in (1..conservar).rev() {
        let viejo = rot.segmento(i, ".gz");
        if viejo.exists() {
            std::fs::rename(&viejo, rot.segmento(i + 1, ".gz"))?;
        }
    }
    std::fs::rename(&tmp, rot.segmento(1, ".gz"))?;
    let _ = std::fs::remove_file(rot.segmento(1, ".sin_comprimir"));
    Ok(())
}

static SUMIDERO: OnceLock<std::sync::Mutex<Sumidero>> = OnceLock::new();

/// CL-144: como `init_telemetry_logger_con_capacidad`, pero el hilo escribe
/// en el log rotativo. Si el archivo no se puede abrir, avisa por stderr y
/// sigue en stdout.
pub fn init_telemetry_logger_rotativo(capacidad: usize, rot: Rotacion) -> bool {
    match Sumidero::abrir(rot.clone()) {
        Ok(s) => {
            let _ = SUMIDERO.set(std::sync::Mutex::new(s));
        }
        Err(error) => eprintln!("⚠️ [LOG] no se pudo abrir {}: {error}; sigo en stdout", rot.ruta.display()),
    }
    init_telemetry_logger_con_capacidad(capacidad)
}

/// Escribe todo lo que hay en la cola en el sumidero (log rotativo o stdout).
fn vaciar_cola(queue: &crossbeam::queue::ArrayQueue<FixedLogBuffer>) -> usize {
    if queue.is_empty() {
        return 0;
    }
    match SUMIDERO.get() {
        Some(m) => m.lock().unwrap_or_else(|p| p.into_inner()).escribir_lote(queue),
        None => vaciar_a_stdout(queue),
    }
}

/// Escribe en stdout todo lo que hay en la cola, con un solo bloqueo.
fn vaciar_a_stdout(queue: &crossbeam::queue::ArrayQueue<FixedLogBuffer>) -> usize {
    use std::io::Write as _;
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
mod tests_cl144 {
    use super::*;

    fn dir_temporal(etiqueta: &str) -> std::path::PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let d = std::env::temp_dir().join(format!("cl144_{etiqueta}_{}_{id}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn lote(q: &crossbeam::queue::ArrayQueue<FixedLogBuffer>, desde: usize, n: usize) {
        for i in desde..desde + n {
            let mut b = FixedLogBuffer::new();
            let _ = write!(&mut b, "linea {i:06} {}", "x".repeat(90));
            let _ = q.force_push(b);
        }
    }

    /// Esperado: 2 000 líneas de ~100 B con tope de 8 KiB y 3 segmentos dejan
    /// el activo bajo el tope, exactamente 3 .gz y ninguno más; el .gz más
    /// reciente acaba justo antes de la primera línea del activo. Antes, un
    /// único archivo de ~200 KB que sólo crece.
    #[test]
    fn cl144_el_log_rota_comprime_y_conserva_los_ultimos() {
        use std::io::Read as _;
        let d = dir_temporal("rota");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 8 * 1024, conservar: 3 };
        let mut s = Sumidero::abrir(rot.clone()).unwrap();
        let q = crossbeam::queue::ArrayQueue::new(64);
        for k in 0..50 {
            lote(&q, k * 40, 40);
            assert_eq!(s.escribir_lote(&q), 40);
        }
        let activo = std::fs::read_to_string(&rot.ruta).unwrap();
        assert!(activo.len() as u64 <= rot.tope_bytes, "activo {} B", activo.len());
        for i in 1..=3 {
            assert!(rot.segmento(i, ".gz").exists(), "falta el segmento {i}");
        }
        assert!(!rot.segmento(4, ".gz").exists(), "sólo se conservan 3");
        let mut gz = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(rot.segmento(1, ".gz")).unwrap())
            .read_to_string(&mut gz)
            .unwrap();
        let ultima_gz: usize = gz.lines().last().unwrap()[6..12].parse().unwrap();
        let primera_activo: usize = activo.lines().next().unwrap()[6..12].parse().unwrap();
        assert_eq!(ultima_gz + 1, primera_activo, "sin huecos entre el .gz y el activo");
        assert!(activo.lines().last().unwrap().starts_with("linea 001999"));
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: un archivo previo se continúa y su tamaño cuenta para el tope.
    #[test]
    fn cl144_un_log_previo_cuenta_para_el_tope() {
        let d = dir_temporal("previo");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 4 * 1024, conservar: 2 };
        std::fs::write(&rot.ruta, vec![b'a'; 4_000]).unwrap();
        let mut s = Sumidero::abrir(rot.clone()).unwrap();
        let q = crossbeam::queue::ArrayQueue::new(8);
        lote(&q, 0, 2);
        s.escribir_lote(&q);
        assert!(rot.segmento(1, ".gz").exists(), "4 000 B previos + 2 líneas pasan el tope");
        assert!(std::fs::metadata(&rot.ruta).unwrap().len() < 4 * 1024);
        let _ = std::fs::remove_dir_all(d);
    }
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

    /// LG-19b: con piezas separadas (cada argumento es un `write_str`), lo
    /// que sigue al corte no se pega en los bytes libres.
    #[test]
    fn lg19b_tras_el_corte_no_se_pega_texto_posterior() {
        let mut buf = FixedLogBuffer::new();
        let relleno = "a".repeat(MAX_LOG_LEN - 2);
        // Variables, no literales: `format_args!` incrusta los literales en la
        // plantilla y llegarían como una sola pieza.
        let cohete = String::from("🚀");
        let resto = String::from("xy");
        let _ = write!(&mut buf, "{}{}{}", relleno, cohete, resto);
        assert_eq!(buf.as_str(), relleno);
    }

    #[test]
    fn lg19_un_mensaje_corto_queda_intacto() {
        let mut buf = FixedLogBuffer::new();
        let _ = write!(&mut buf, "🚪 [CLOSE TRACE] #{} ok", 7);
        assert_eq!(buf.as_str(), "🚪 [CLOSE TRACE] #7 ok");
    }
}
