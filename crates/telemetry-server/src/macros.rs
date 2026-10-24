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
pub static LOG_QUEUE: OnceLock<Arc<ColaDeLog>> = OnceLock::new();

type ColaDeLog = crossbeam::queue::ArrayQueue<FixedLogBuffer>;

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
    instalar_en(&LOG_QUEUE, capacidad, |queue| {
        std::thread::Builder::new()
            .name("GhostLogger".to_string())
            .spawn(move || loop {
                if vaciar_cola(&queue) == 0 {
                    // Si otra inicialización publicó antes su cola, ésta no
                    // la tiene nadie más y el hilo termina.
                    if Arc::strong_count(&queue) == 1 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            })
            .map(|_| ())
    })
}

/// CL-144b (C-F6) — LA COLA SÓLO SE PUBLICA CON SU CONSUMIDOR VIVO.
///
/// QUÉ ESTABA MAL: la cola se publicaba antes de crear el hilo que la vacía.
/// Si el hilo no se podía crear, todo `telemetry_log!` entraba en una cola
/// que nadie leía y el motor corría sin log hasta el apagado.
///
/// QUÉ GARANTIZA: `lanzar` recibe la cola y crea su consumidor; la cola sólo
/// se publica en `celda` si lo consigue. Si no, las macros siguen escribiendo
/// de forma síncrona en stdout y stderr. Devuelve `true` si publicó.
fn instalar_en(
    celda: &OnceLock<Arc<ColaDeLog>>,
    capacidad: usize,
    lanzar: impl FnOnce(Arc<ColaDeLog>) -> std::io::Result<()>,
) -> bool {
    if celda.get().is_some() {
        return false;
    }
    let queue = Arc::new(ColaDeLog::new(capacidad.max(1)));
    if let Err(error) = lanzar(Arc::clone(&queue)) {
        avisar(format_args!(
            "⚠️ [LOG] no se pudo crear el hilo del log ({error}); sigo escribiendo sin cola"
        ));
        return false;
    }
    celda.set(queue).is_ok()
}

/// CL-144b: aviso del propio log por stderr. `eprintln!` entra en pánico si
/// stderr falla y, con `panic = "abort"`, tumbaría el motor por el log.
fn avisar(args: fmt::Arguments<'_>) {
    use std::io::Write as _;
    let mut err = std::io::stderr().lock();
    let _ = err.write_fmt(args);
    let _ = err.write_all(b"\n");
}

/// CL-144b: espera entre dos intentos de abrir el log activo.
const REINTENTO_APERTURA_MS: u64 = 30_000;

/// CL-144 (C-F6) — LOG DEL MOTOR CON ROTACIÓN Y COMPRESIÓN.
///
/// QUÉ ESTABA MAL: todo `telemetry_log!`/`telemetry!` del proceso acaba en
/// stdout, que el lanzador redirige a un único `logs/demo_vNN.log`. Con el
/// handle abierto por el proceso, nada externo puede rotarlo, y el archivo
/// crece mientras dure la sesión.
///
/// QUÉ GARANTIZA: con `TG_LOG_FILE` definido, el hilo GhostLogger escribe en
/// ese archivo. Al pasar `tope_bytes` lo renombra a `<ruta>.<n>` (`n` crece
/// en cada rotación y sigue tras un reinicio) y abre uno nuevo al momento; un
/// hilo aparte lo comprime a `<ruta>.<n>.gz` y borra los segmentos más viejos
/// hasta dejar `conservar`. En disco hay como mucho el activo y `conservar`
/// segmentos, comprimidos o por comprimir, más un `.gz.tmp` mientras se
/// comprime y el cerrojo `<ruta>.lock`. Sin la variable, todo sigue yendo a
/// stdout como antes. Los `println!` directos (fuera de las macros) siguen
/// yendo a stdout, que por eso no debe redirigirse al mismo archivo.
///
/// CL-144b: un fallo de E/S no pierde segmentos ni repite la rotación. Si el
/// activo no se deja renombrar (otro proceso lo tiene abierto sin compartir
/// el borrado), se archiva una copia comprimida y sólo después se vacía; si
/// tampoco se puede, se sigue escribiendo en él y se reintenta cuando vuelva
/// a llenarse. Si el activo no se puede abrir, las líneas van a stdout y la
/// apertura se reintenta cada `REINTENTO_APERTURA_MS`. Un segmento que no se
/// pudo comprimir se comprime en la pasada siguiente o tras un reinicio.
///
/// CL-144c: un solo escritor y nada se borra antes de ser durable.
/// - El dueño de la ruta tiene el cerrojo exclusivo `<ruta>.lock` mientras
///   vive; un segundo proceso con la misma ruta sigue en stdout sin tocar
///   los segmentos y la toma cuando el primero termina. En Windows el activo
///   se abre sin compartir la escritura, así que tampoco se abre si otro
///   handle ya escribe en él (stdout redirigido al mismo archivo).
/// - El `.gz` se sincroniza a disco antes de su renombre, y el crudo sólo se
///   borra (o el activo se vacía) después. Un `.gz` que un corte de luz dejó
///   vacío o truncado no cuenta como hecho: se vuelve a comprimir el crudo.
#[derive(Debug, Clone)]
pub struct Rotacion {
    pub ruta: std::path::PathBuf,
    pub tope_bytes: u64,
    pub conservar: usize,
}

/// Forma de un segmento en disco.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pieza {
    /// `<ruta>.<n>`: archivado y pendiente de comprimir.
    Crudo,
    /// `<ruta>.<n>.gz`: sólo aparece completo (se escribe vía `.gz.tmp`).
    Comprimido,
    /// `<ruta>.<n>.gz.tmp`: compresión en curso o cortada por una caída.
    Temporal,
}

/// Cómo se abre el activo: para añadir, o vaciado tras archivar una copia.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Apertura {
    Anadir,
    Vaciar,
}

impl Rotacion {
    /// `TG_LOG_FILE` (ruta del log), `TG_LOG_MAX_MB` (tope del activo, 64 por
    /// defecto) y `TG_LOG_KEEP` (segmentos conservados, 8 por defecto).
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

    fn pieza(&self, n: u64, pieza: Pieza) -> std::path::PathBuf {
        let sufijo = match pieza {
            Pieza::Crudo => "",
            Pieza::Comprimido => ".gz",
            Pieza::Temporal => ".gz.tmp",
        };
        let mut s = self.ruta.clone().into_os_string();
        s.push(format!(".{n:06}{sufijo}"));
        s.into()
    }

    fn cerrojo(&self) -> std::path::PathBuf {
        let mut s = self.ruta.clone().into_os_string();
        s.push(".lock");
        s.into()
    }

    /// Los segmentos que hay en disco, en cualquier orden. CL-144c: sólo los
    /// nombres que `pieza` construye (`<ruta>.000007`, no `<ruta>.7`); otro
    /// nombre no se comprime, no se borra ni cuenta para la numeración.
    fn piezas(&self) -> Vec<(u64, Pieza)> {
        let Some(nombre) = self.ruta.file_name().and_then(|n| n.to_str()) else {
            return Vec::new();
        };
        let prefijo = format!("{nombre}.");
        let dir = match self.ruta.parent().filter(|d| !d.as_os_str().is_empty()) {
            Some(d) => d.to_path_buf(),
            None => std::path::PathBuf::from("."),
        };
        let Ok(entradas) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        entradas
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|archivo| {
                let resto = archivo.strip_prefix(prefijo.as_str())?;
                let (digitos, pieza) = if let Some(d) = resto.strip_suffix(".gz.tmp") {
                    (d, Pieza::Temporal)
                } else if let Some(d) = resto.strip_suffix(".gz") {
                    (d, Pieza::Comprimido)
                } else {
                    (resto, Pieza::Crudo)
                };
                if digitos.is_empty() || !digitos.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                let n: u64 = digitos.parse().ok()?;
                (format!("{n:06}") == digitos).then_some((n, pieza))
            })
            .collect()
    }
}

/// Abre el activo. CL-144c: en Windows sin compartir la escritura (lectura y
/// borrado sí, para los visores y el renombre), así que falla si otro handle
/// ya escribe en él.
fn abrir_archivo(ruta: &std::path::Path, modo: Apertura) -> std::io::Result<std::fs::File> {
    let mut opciones = std::fs::OpenOptions::new();
    match modo {
        Apertura::Anadir => opciones.create(true).append(true),
        Apertura::Vaciar => opciones.write(true).truncate(true),
    };
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        // FILE_SHARE_READ | FILE_SHARE_DELETE.
        opciones.share_mode(0x1 | 0x4);
    }
    opciones.open(ruta)
}

/// Comprime `origen` en `destino` pasando por `tmp`, sin tocar `origen`. Un
/// `.gz` sólo aparece completo; si algo falla, el temporal se borra.
/// CL-144c: el `.gz` está en disco (`sync_all`) antes de su renombre, así que
/// quien borre el crudo después no pierde nada ante un corte de luz.
fn comprimir(origen: &std::path::Path, tmp: &std::path::Path, destino: &std::path::Path) -> std::io::Result<()> {
    let resultado = (|| {
        let mut entrada = std::fs::File::open(origen)?;
        let salida = std::io::BufWriter::new(std::fs::File::create(tmp)?);
        let mut gz = flate2::write::GzEncoder::new(salida, flate2::Compression::fast());
        std::io::copy(&mut entrada, &mut gz)?;
        let archivo = gz.finish()?.into_inner().map_err(|e| e.into_error())?;
        archivo.sync_all()?;
        drop(archivo);
        std::fs::rename(tmp, destino)?;
        #[cfg(unix)]
        if let Some(dir) = destino.parent().filter(|d| !d.as_os_str().is_empty()) {
            let _ = std::fs::File::open(dir).and_then(|d| d.sync_all());
        }
        Ok(())
    })();
    if resultado.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    resultado
}

/// CL-144c: `true` si `gz` es un gzip completo de `crudo`: empieza con la
/// firma y su cola (ISIZE, el tamaño sin comprimir módulo 2³²) coincide con
/// la longitud del crudo. Un `.gz` vacío o a ceros tras un corte no pasa.
fn gz_completo_de(gz: &std::path::Path, crudo: &std::path::Path) -> bool {
    use std::io::{Read as _, Seek as _, SeekFrom};
    let (Ok(largo_crudo), Ok(mut f)) = (std::fs::metadata(crudo).map(|m| m.len()), std::fs::File::open(gz)) else {
        return false;
    };
    let mut firma = [0u8; 2];
    let mut cola = [0u8; 4];
    let leido = f
        .read_exact(&mut firma)
        .and_then(|_| f.seek(SeekFrom::End(-4)))
        .and_then(|_| f.read_exact(&mut cola));
    leido.is_ok() && firma == [0x1f, 0x8b] && u64::from(u32::from_le_bytes(cola)) == largo_crudo & 0xffff_ffff
}

/// Comprime los segmentos pendientes y deja los `conservar` más nuevos. Corre
/// en el hilo de compresión, o en el del log si ese hilo no existe.
fn mantener_segmentos(rot: &Rotacion) {
    let piezas = rot.piezas();
    let mut pendientes: Vec<u64> =
        piezas.iter().filter(|p| p.1 == Pieza::Crudo).map(|p| p.0).collect();
    pendientes.sort_unstable();
    for n in pendientes {
        let crudo = rot.pieza(n, Pieza::Crudo);
        let gz = rot.pieza(n, Pieza::Comprimido);
        // Con su .gz ya completo, el crudo es el resto de una caída entre el
        // renombre del .gz y el borrado del crudo.
        let ya_comprimido = piezas.contains(&(n, Pieza::Comprimido)) && gz_completo_de(&gz, &crudo);
        let hecho = ya_comprimido
            || match comprimir(&crudo, &rot.pieza(n, Pieza::Temporal), &gz) {
                Ok(()) => true,
                Err(error) => {
                    avisar(format_args!(
                        "⚠️ [LOG] no se pudo comprimir {}: {error}; queda sin comprimir",
                        crudo.display()
                    ));
                    false
                }
            };
        if hecho {
            let _ = std::fs::remove_file(&crudo);
        }
    }
    let mut numeros: Vec<u64> = rot
        .piezas()
        .into_iter()
        .filter(|p| p.1 != Pieza::Temporal)
        .map(|p| p.0)
        .collect();
    numeros.sort_unstable_by(|a, b| b.cmp(a));
    numeros.dedup();
    for &n in numeros.iter().skip(rot.conservar.max(1)) {
        for pieza in [Pieza::Crudo, Pieza::Comprimido] {
            let ruta = rot.pieza(n, pieza);
            match std::fs::remove_file(&ruta) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => avisar(format_args!(
                    "⚠️ [LOG] no se pudo borrar {}: {e}; se reintenta en la próxima rotación",
                    ruta.display()
                )),
                _ => {}
            }
        }
    }
}

/// Hilo que comprime y recorta los segmentos fuera del hilo del log. Cada
/// aviso pide una pasada; uno que llega con otra pendiente se funde con ella.
fn lanzar_compresor(rot: Rotacion) -> Option<std::sync::mpsc::SyncSender<()>> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
    match std::thread::Builder::new()
        .name("GhostLogger-gzip".to_string())
        .spawn(move || {
            while rx.recv().is_ok() {
                mantener_segmentos(&rot);
            }
        }) {
        Ok(_) => Some(tx),
        Err(error) => {
            avisar(format_args!(
                "⚠️ [LOG] no se pudo crear el hilo de compresión ({error}); comprimo en el hilo del log"
            ));
            None
        }
    }
}

/// CL-144c: qué tiene este proceso sobre la ruta del log.
enum Posesion {
    /// Cerrojo exclusivo de `<ruta>.lock`; se libera al cerrar el handle,
    /// también si el proceso muere.
    Exclusiva { _cerrojo: std::fs::File },
    /// El sistema de archivos no admite cerrojos: se escribe sin garantía de
    /// escritor único (se avisó una vez).
    SinCerrojo,
}

/// Archivo activo del log rotativo.
struct Sumidero {
    rot: Rotacion,
    archivo: Option<std::io::BufWriter<std::fs::File>>,
    escritos: u64,
    /// Número del último segmento archivado; el siguiente es `ultimo + 1`.
    ultimo: u64,
    reapertura: crate::estrangulador::Estrangulador,
    inicio: std::time::Instant,
    /// Aviso al hilo de compresión; sin él se comprime en el hilo del log.
    compresor: Option<std::sync::mpsc::SyncSender<()>>,
    /// Si se quiere hilo de compresión cuando se tome la ruta.
    con_hilo: bool,
    /// `None` mientras otro proceso es el dueño de la ruta.
    posesion: Option<Posesion>,
    /// Apertura del activo (las pruebas inyectan fallos).
    abrir: fn(&std::path::Path, Apertura) -> std::io::Result<std::fs::File>,
    /// Último aviso emitido, para no repetirlo en cada reintento.
    ultimo_aviso: Option<String>,
}

impl Sumidero {
    /// Continúa el activo si existe. No falla: sin activo, las líneas van a
    /// stdout y la apertura se reintenta. Comprime lo que un arranque anterior
    /// dejó pendiente y borra los temporales de una compresión cortada.
    fn abrir(rot: Rotacion, hilo_de_compresion: bool) -> Self {
        if let Some(dir) = rot.ruta.parent().filter(|d| !d.as_os_str().is_empty()) {
            if let Err(error) = std::fs::create_dir_all(dir) {
                avisar(format_args!("⚠️ [LOG] no se pudo crear {}: {error}", dir.display()));
            }
        }
        let mut s = Self {
            rot,
            archivo: None,
            escritos: 0,
            ultimo: 0,
            reapertura: crate::estrangulador::Estrangulador::new(REINTENTO_APERTURA_MS),
            inicio: std::time::Instant::now(),
            compresor: None,
            con_hilo: hilo_de_compresion,
            posesion: None,
            abrir: abrir_archivo,
            ultimo_aviso: None,
        };
        s.reintentar(0);
        s
    }

    fn ahora_ms(&self) -> u64 {
        u64::try_from(self.inicio.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// CL-144c: avisa por stderr salvo que sea el mismo aviso que el último.
    fn avisar_una_vez(&mut self, aviso: String) {
        if self.ultimo_aviso.as_deref() != Some(aviso.as_str()) {
            avisar(format_args!("{aviso}"));
            self.ultimo_aviso = Some(aviso);
        }
    }

    /// Toma la ruta si no es suya y abre el activo. Cuenta la espera hasta el
    /// próximo intento desde `ahora_ms` si algo falla.
    fn reintentar(&mut self, ahora_ms: u64) {
        if self.tomar_posesion(ahora_ms) {
            self.abrir_activo(ahora_ms);
        }
    }

    /// CL-144c: el cerrojo de la ruta. Al tomarla borra los temporales de una
    /// compresión cortada, sigue la numeración de lo que hay en disco y pide
    /// una pasada de compresión. Sin la ruta, nada de eso: los segmentos son
    /// del otro proceso.
    fn tomar_posesion(&mut self, ahora_ms: u64) -> bool {
        if self.posesion.is_some() {
            return true;
        }
        let ruta_cerrojo = self.rot.cerrojo();
        let cerrojo = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&ruta_cerrojo);
        let posesion = match cerrojo {
            Ok(f) => match f.try_lock() {
                Ok(()) => Posesion::Exclusiva { _cerrojo: f },
                Err(std::fs::TryLockError::WouldBlock) => {
                    let _ = self.reapertura.admitir(ahora_ms);
                    self.avisar_una_vez(format!(
                        "⚠️ [LOG] otro proceso escribe en {} (cerrojo {}); sigo en stdout y reintento cada {} s",
                        self.rot.ruta.display(),
                        ruta_cerrojo.display(),
                        REINTENTO_APERTURA_MS / 1_000
                    ));
                    return false;
                }
                Err(std::fs::TryLockError::Error(error)) => {
                    self.avisar_una_vez(format!(
                        "⚠️ [LOG] sin cerrojo en {} ({error}); escribo sin garantía de escritor único",
                        ruta_cerrojo.display()
                    ));
                    Posesion::SinCerrojo
                }
            },
            Err(error) => {
                let _ = self.reapertura.admitir(ahora_ms);
                self.avisar_una_vez(format!(
                    "⚠️ [LOG] no se pudo crear {}: {error}; sigo en stdout y reintento cada {} s",
                    ruta_cerrojo.display(),
                    REINTENTO_APERTURA_MS / 1_000
                ));
                return false;
            }
        };
        self.posesion = Some(posesion);
        let piezas = self.rot.piezas();
        for &(n, pieza) in &piezas {
            if pieza == Pieza::Temporal {
                let _ = std::fs::remove_file(self.rot.pieza(n, Pieza::Temporal));
            }
        }
        self.ultimo = piezas.iter().map(|p| p.0).max().unwrap_or(0);
        if self.con_hilo && self.compresor.is_none() {
            self.compresor = lanzar_compresor(self.rot.clone());
        }
        self.mantener();
        true
    }

    /// Abre el activo para añadir. Si falla, sigue en stdout y cuenta la
    /// espera hasta el próximo intento desde ahora.
    fn abrir_activo(&mut self, ahora_ms: u64) {
        match (self.abrir)(&self.rot.ruta, Apertura::Anadir) {
            Ok(f) => {
                self.escritos = f.metadata().map(|m| m.len()).unwrap_or(0);
                self.archivo = Some(std::io::BufWriter::new(f));
                self.ultimo_aviso = None;
            }
            Err(error) => {
                self.archivo = None;
                self.escritos = 0;
                let _ = self.reapertura.admitir(ahora_ms);
                // 32 = ERROR_SHARING_VIOLATION: otro handle escribe en él.
                let causa = if cfg!(windows) && error.raw_os_error() == Some(32) {
                    " (otro handle lo tiene abierto para escribir: ¿stdout redirigido al mismo archivo?)"
                } else {
                    ""
                };
                self.avisar_una_vez(format!(
                    "⚠️ [LOG] no se pudo abrir {}: {error}{causa}; sigo en stdout y reintento cada {} s",
                    self.rot.ruta.display(),
                    REINTENTO_APERTURA_MS / 1_000
                ));
            }
        }
    }

    /// Pide una pasada de compresión y recorte.
    fn mantener(&mut self) {
        if let Some(tx) = &self.compresor {
            match tx.try_send(()) {
                // Lleno: ya hay una pasada pendiente y verá este segmento.
                Ok(()) | Err(std::sync::mpsc::TrySendError::Full(())) => return,
                Err(std::sync::mpsc::TrySendError::Disconnected(())) => {
                    avisar(format_args!(
                        "⚠️ [LOG] el hilo de compresión terminó; comprimo en el hilo del log"
                    ));
                    self.compresor = None;
                }
            }
        }
        mantener_segmentos(&self.rot);
    }

    fn escribir_lote(&mut self, queue: &ColaDeLog, ahora_ms: u64) -> usize {
        use std::io::Write as _;
        if self.archivo.is_none() && self.reapertura.admitir(ahora_ms).is_some() {
            self.reintentar(ahora_ms);
        }
        let mut n = 0;
        while let Some(msg) = queue.pop() {
            let linea = msg.as_str().as_bytes();
            let largo = linea.len() as u64 + 1;
            if self.archivo.is_some()
                && self.escritos > 0
                && self.escritos.saturating_add(largo) > self.rot.tope_bytes
            {
                self.rotar(ahora_ms);
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

    /// Archiva el activo como segmento `ultimo + 1` y sigue en uno vacío.
    fn rotar(&mut self, ahora_ms: u64) {
        use std::io::Write as _;
        let Some(n) = self.ultimo.checked_add(1) else {
            self.avisar_una_vez(format!(
                "⚠️ [LOG] numeración de segmentos agotada en {}; sigo escribiendo sin rotar",
                self.rot.ruta.display()
            ));
            self.escritos = 0;
            return;
        };
        if let Some(mut w) = self.archivo.take() {
            let _ = w.flush();
        }
        let error_renombre = match std::fs::rename(&self.rot.ruta, self.rot.pieza(n, Pieza::Crudo)) {
            Ok(()) => {
                self.ultimo = n;
                self.abrir_activo(ahora_ms);
                self.mantener();
                return;
            }
            Err(error) => error,
        };
        // No se deja renombrar: se archiva una copia comprimida y sólo
        // después se vacía el activo.
        let copia = comprimir(
            &self.rot.ruta,
            &self.rot.pieza(n, Pieza::Temporal),
            &self.rot.pieza(n, Pieza::Comprimido),
        );
        match copia {
            Ok(()) => {
                self.ultimo = n;
                match (self.abrir)(&self.rot.ruta, Apertura::Vaciar) {
                    Ok(f) => self.archivo = Some(std::io::BufWriter::new(f)),
                    // Sin vaciar: lo archivado se repite en el activo, no se pierde.
                    Err(_) => self.abrir_activo(ahora_ms),
                }
                self.mantener();
            }
            Err(error) => {
                self.avisar_una_vez(format!(
                    "⚠️ [LOG] no se pudo archivar {} (renombrar: {error_renombre}; comprimir: {error}); sigo escribiendo en él",
                    self.rot.ruta.display()
                ));
                self.abrir_activo(ahora_ms);
            }
        }
        // El siguiente intento, cuando se escriba otro tope entero.
        self.escritos = 0;
    }
}

static SUMIDERO: OnceLock<std::sync::Mutex<Sumidero>> = OnceLock::new();

/// CL-144: como `init_telemetry_logger_con_capacidad`, pero el hilo escribe
/// en el log rotativo. Si el archivo no se puede abrir, avisa por stderr,
/// sigue en stdout y reintenta la apertura.
pub fn init_telemetry_logger_rotativo(capacidad: usize, rot: Rotacion) -> bool {
    let _ = SUMIDERO.set(std::sync::Mutex::new(Sumidero::abrir(rot, true)));
    init_telemetry_logger_con_capacidad(capacidad)
}

/// Escribe todo lo que hay en la cola en el sumidero (log rotativo o stdout).
fn vaciar_cola(queue: &ColaDeLog) -> usize {
    if queue.is_empty() {
        return 0;
    }
    vaciar_en(SUMIDERO.get(), queue)
}

/// CL-144b: sin el atajo de la cola vacía. Con el sumidero toma su cerrojo,
/// así espera al lote que GhostLogger tenga a medias, y vacía el búfer al
/// disco aunque no quede nada en la cola.
fn vaciar_en(sumidero: Option<&std::sync::Mutex<Sumidero>>, queue: &ColaDeLog) -> usize {
    match sumidero {
        Some(m) => {
            let mut s = m.lock().unwrap_or_else(|p| p.into_inner());
            let ahora = s.ahora_ms();
            s.escribir_lote(queue, ahora)
        }
        None => vaciar_a_stdout(queue),
    }
}

/// Escribe en stdout todo lo que hay en la cola, con un solo bloqueo.
fn vaciar_a_stdout(queue: &ColaDeLog) -> usize {
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
    vaciar_telemetria_en(LOG_QUEUE.get().map(|q| &**q), SUMIDERO.get())
}

/// CL-144c: `vaciar_telemetria` con la cola y el sumidero explícitos. Con
/// sumidero no hay atajo por cola vacía: el apagado espera el lote que
/// GhostLogger tenga a medias y deja en disco su búfer.
fn vaciar_telemetria_en(queue: Option<&ColaDeLog>, sumidero: Option<&std::sync::Mutex<Sumidero>>) -> usize {
    queue.map(|q| vaciar_en(sumidero, q)).unwrap_or(0)
}

/// RET-51b (CL-160, censo C-F7b, F155): destino que el host conecta con
/// `telemetry_engine::conectar`. Escribe la línea de `telemetry!` o
/// `telemetry_err!` en la cola del log, en el hilo que la emite, como
/// `telemetry_log!` (mismo corte en frontera de carácter, LG-19); sin cola,
/// en stdout. Así el apagado (`vaciar_telemetria`) también la vacía y sale
/// en el orden en que se emitió.
pub fn linea_de_telemetria(args: fmt::Arguments<'_>) {
    crate::telemetry_log!("{}", args);
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

    fn lote(q: &ColaDeLog, desde: usize, n: usize) {
        for i in desde..desde + n {
            let mut b = FixedLogBuffer::new();
            let _ = write!(&mut b, "linea {i:06} {}", "x".repeat(90));
            let _ = q.force_push(b);
        }
    }

    fn leer_gz(ruta: &std::path::Path) -> String {
        use std::io::Read as _;
        let mut s = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(ruta).unwrap())
            .read_to_string(&mut s)
            .unwrap();
        s
    }

    fn numero(linea: &str) -> usize {
        linea[6..12].parse().unwrap()
    }

    fn numeros(rot: &Rotacion, pieza: Pieza) -> Vec<u64> {
        let mut v: Vec<u64> = rot.piezas().into_iter().filter(|p| p.1 == pieza).map(|p| p.0).collect();
        v.sort_unstable();
        v
    }

    /// Esperado: 2 000 líneas de ~100 B con tope de 8 KiB y 3 segmentos dejan
    /// el activo bajo el tope y exactamente los 3 segmentos más nuevos, todos
    /// comprimidos y sin huecos entre ellos ni con el activo. Antes, un único
    /// archivo de ~200 KB que sólo crece.
    #[test]
    fn cl144_el_log_rota_comprime_y_conserva_los_ultimos() {
        let d = dir_temporal("rota");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 8 * 1024, conservar: 3 };
        let mut s = Sumidero::abrir(rot.clone(), false);
        let q = ColaDeLog::new(64);
        for k in 0..50 {
            lote(&q, k * 40, 40);
            assert_eq!(s.escribir_lote(&q, 0), 40);
        }
        let activo = std::fs::read_to_string(&rot.ruta).unwrap();
        assert!(activo.len() as u64 <= rot.tope_bytes, "activo {} B", activo.len());
        assert_eq!(numeros(&rot, Pieza::Crudo), Vec::<u64>::new(), "todo comprimido");
        let gz = numeros(&rot, Pieza::Comprimido);
        assert_eq!(gz, vec![s.ultimo - 2, s.ultimo - 1, s.ultimo], "los 3 más nuevos");
        let mut siguiente = None;
        for n in gz {
            let texto = leer_gz(&rot.pieza(n, Pieza::Comprimido));
            if let Some(esperado) = siguiente {
                assert_eq!(numero(texto.lines().next().unwrap()), esperado, "hueco antes del segmento {n}");
            }
            siguiente = Some(numero(texto.lines().last().unwrap()) + 1);
        }
        assert_eq!(numero(activo.lines().next().unwrap()), siguiente.unwrap(), "hueco antes del activo");
        assert!(activo.lines().last().unwrap().starts_with("linea 001999"));
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: un archivo previo se continúa y su tamaño cuenta para el tope.
    #[test]
    fn cl144_un_log_previo_cuenta_para_el_tope() {
        let d = dir_temporal("previo");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 4 * 1024, conservar: 2 };
        std::fs::write(&rot.ruta, vec![b'a'; 4_000]).unwrap();
        let mut s = Sumidero::abrir(rot.clone(), false);
        let q = ColaDeLog::new(8);
        lote(&q, 0, 2);
        s.escribir_lote(&q, 0);
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![1], "4 000 B previos + 2 líneas pasan el tope");
        assert_eq!(leer_gz(&rot.pieza(1, Pieza::Comprimido)), "a".repeat(4_000));
        assert!(std::fs::metadata(&rot.ruta).unwrap().len() < 4 * 1024);
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: lo que una sesión anterior dejó a medias no se pierde. Un
    /// segmento sin comprimir se comprime, el crudo sobrante de una caída
    /// tras su .gz se borra, el temporal cortado se descarta, se recorta a
    /// `conservar` y la numeración sigue sin pisar nada. Antes, la siguiente
    /// rotación borraba el `.1.sin_comprimir` sin archivarlo y desplazaba la
    /// cadena borrando el más viejo antes de saber si el resto salía bien.
    #[test]
    fn cl144b_lo_que_dejo_un_arranque_anterior_se_archiva() {
        let d = dir_temporal("previo_b");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 3 };
        for n in 1..=3 {
            std::fs::write(rot.pieza(n, Pieza::Crudo), format!("segmento {n}\n")).unwrap();
            mantener_segmentos(&Rotacion { conservar: 100, ..rot.clone() });
        }
        std::fs::write(rot.pieza(3, Pieza::Crudo), "segmento 3\n").unwrap();
        std::fs::write(rot.pieza(4, Pieza::Crudo), "segmento 4\n").unwrap();
        std::fs::write(rot.pieza(5, Pieza::Temporal), "gzip cortado").unwrap();
        let mut s = Sumidero::abrir(rot.clone(), false);
        assert_eq!(numeros(&rot, Pieza::Temporal), Vec::<u64>::new());
        assert_eq!(numeros(&rot, Pieza::Crudo), Vec::<u64>::new());
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![2, 3, 4]);
        assert_eq!(leer_gz(&rot.pieza(4, Pieza::Comprimido)), "segmento 4\n");
        assert_eq!(leer_gz(&rot.pieza(3, Pieza::Comprimido)), "segmento 3\n");
        let q = ColaDeLog::new(64);
        lote(&q, 0, 40);
        s.escribir_lote(&q, 0);
        let gz = numeros(&rot, Pieza::Comprimido);
        assert_eq!(gz.len(), 3);
        assert!(gz.iter().all(|&n| n == 4 || n >= 6), "los nuevos siguen tras el 5: {gz:?}");
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: si el activo no se puede abrir desde el arranque, las líneas
    /// van a stdout sin contar para el tope, no se rota nada y la apertura se
    /// reintenta pasada la espera.
    #[test]
    fn cl144b_sin_activo_se_sigue_en_stdout_y_se_reintenta() {
        let d = dir_temporal("sin_activo");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 1024, conservar: 2 };
        std::fs::create_dir_all(&rot.ruta).unwrap();
        let mut s = Sumidero::abrir(rot.clone(), false);
        assert!(s.archivo.is_none());
        let q = ColaDeLog::new(64);
        for (k, ahora) in [1_000u64, 10_000, 29_999].into_iter().enumerate() {
            lote(&q, k * 2, 2);
            assert_eq!(s.escribir_lote(&q, ahora), 2);
            assert!(s.archivo.is_none(), "{ahora} ms: todavía no toca reintentar");
            assert_eq!(s.escritos, 0, "lo que va a stdout no cuenta para el tope");
        }
        assert_eq!(rot.piezas(), Vec::new(), "sin activo no se rota");
        std::fs::remove_dir(&rot.ruta).unwrap();
        lote(&q, 150, 3);
        s.escribir_lote(&q, REINTENTO_APERTURA_MS);
        assert!(s.archivo.is_some(), "pasada la espera, se reabre");
        let activo = std::fs::read_to_string(&rot.ruta).unwrap();
        assert_eq!(activo.lines().count(), 3);
        assert_eq!(numero(activo.lines().next().unwrap()), 150);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: si el activo se renombra bien pero no se puede
    /// reabrir, queda UN segmento archivado, el resto va a stdout sin rotar
    /// otra vez, y pasada la espera se reabre. Antes de CL-144b, `escritos`
    /// quedaba sobre el tope: cada línea volvía a rotar, recomprimía y
    /// borraba el historial.
    #[test]
    fn cl144c_un_fallo_al_reabrir_tras_rotar_no_rota_en_bucle() {
        let d = dir_temporal("reabrir");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 4 };
        let mut s = Sumidero::abrir(rot.clone(), false);
        let q = ColaDeLog::new(64);
        lote(&q, 0, 15);
        s.escribir_lote(&q, 0);
        s.abrir = |_, _| Err(std::io::Error::other("disco de prueba"));
        lote(&q, 15, 45);
        assert_eq!(s.escribir_lote(&q, 1_000), 45);
        assert!(s.archivo.is_none());
        assert_eq!((s.ultimo, s.escritos), (1, 0));
        for k in 0..5 {
            lote(&q, 60 + k * 40, 40);
            s.escribir_lote(&q, 2_000 + k as u64);
        }
        assert_eq!(s.ultimo, 1, "sin activo no se vuelve a rotar");
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![1]);
        let archivado = leer_gz(&rot.pieza(1, Pieza::Comprimido));
        assert_eq!(numero(archivado.lines().next().unwrap()), 0);
        assert_eq!(archivado.lines().count(), 19, "lo escrito antes del tope");
        s.abrir = abrir_archivo;
        lote(&q, 500, 3);
        s.escribir_lote(&q, 1_000 + REINTENTO_APERTURA_MS);
        assert!(s.archivo.is_some(), "pasada la espera, se reabre");
        let activo = std::fs::read_to_string(&rot.ruta).unwrap();
        assert_eq!(activo.lines().count(), 3);
        assert_eq!(numero(activo.lines().next().unwrap()), 500);
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: con el activo abierto por otro proceso que no comparte el
    /// borrado (no se puede renombrar), la rotación archiva una copia
    /// comprimida antes de vaciarlo y no pierde nada.
    #[cfg(windows)]
    #[test]
    fn cl144b_un_activo_que_no_se_deja_renombrar_se_archiva_igual() {
        use std::os::windows::fs::OpenOptionsExt as _;
        let d = dir_temporal("bloqueado");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 4 };
        let mut s = Sumidero::abrir(rot.clone(), false);
        let q = ColaDeLog::new(64);
        lote(&q, 0, 15);
        s.escribir_lote(&q, 0);
        // FILE_SHARE_READ | FILE_SHARE_WRITE, sin FILE_SHARE_DELETE.
        let lector = std::fs::OpenOptions::new().read(true).share_mode(0x1 | 0x2).open(&rot.ruta).unwrap();
        assert!(std::fs::rename(&rot.ruta, d.join("otro.log")).is_err(), "el lector impide renombrar");
        lote(&q, 15, 15);
        s.escribir_lote(&q, 0);
        let gz = numeros(&rot, Pieza::Comprimido);
        assert_eq!(gz, vec![1], "archivado una vez");
        let archivado = leer_gz(&rot.pieza(1, Pieza::Comprimido));
        let activo = std::fs::read_to_string(&rot.ruta).unwrap();
        let ultima_gz = numero(archivado.lines().last().unwrap());
        assert_eq!(numero(activo.lines().next().unwrap()), ultima_gz + 1, "sin huecos");
        assert!(activo.lines().last().unwrap().starts_with("linea 000029"));
        drop(lector);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: si tras archivar la copia el activo no se deja
    /// vaciar, se sigue añadiendo en él: lo archivado se repite, nada se
    /// pierde.
    #[cfg(windows)]
    #[test]
    fn cl144c_si_no_se_puede_vaciar_se_sigue_sin_perder_nada() {
        use std::os::windows::fs::OpenOptionsExt as _;
        let d = dir_temporal("sin_vaciar");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 4 };
        let mut s = Sumidero::abrir(rot.clone(), false);
        let q = ColaDeLog::new(64);
        lote(&q, 0, 15);
        s.escribir_lote(&q, 0);
        let lector = std::fs::OpenOptions::new().read(true).share_mode(0x1 | 0x2).open(&rot.ruta).unwrap();
        s.abrir = |ruta, modo| match modo {
            Apertura::Vaciar => Err(std::io::Error::other("sin vaciar")),
            Apertura::Anadir => abrir_archivo(ruta, modo),
        };
        lote(&q, 15, 15);
        s.escribir_lote(&q, 0);
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![1]);
        let archivado = leer_gz(&rot.pieza(1, Pieza::Comprimido));
        assert_eq!(numero(archivado.lines().next().unwrap()), 0);
        let activo = std::fs::read_to_string(&rot.ruta).unwrap();
        assert_eq!(numero(activo.lines().next().unwrap()), 0, "no se vació");
        assert_eq!(activo.lines().count(), 30, "todo sigue en el activo");
        drop(lector);
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: con el hilo de compresión, el hilo deja `conservar`
    /// segmentos comprimidos.
    #[test]
    fn cl144b_el_hilo_de_compresion_archiva_fuera_del_hilo_del_log() {
        let d = dir_temporal("hilo");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 4 * 1024, conservar: 2 };
        let mut s = Sumidero::abrir(rot.clone(), true);
        assert!(s.compresor.is_some());
        let q = ColaDeLog::new(64);
        for k in 0..10 {
            lote(&q, k * 40, 40);
            s.escribir_lote(&q, 0);
        }
        let mut listo = false;
        for _ in 0..1_000 {
            if numeros(&rot, Pieza::Crudo).is_empty() && numeros(&rot, Pieza::Comprimido).len() == 2 {
                listo = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(listo, "piezas: {:?}", rot.piezas());
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![s.ultimo - 1, s.ultimo]);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: con un compresor que no consume, la rotación sólo
    /// renombra y avisa: los segmentos quedan crudos hasta que ese hilo pase.
    /// Si el hilo del log comprimiera, aparecería un `.gz`.
    #[test]
    fn cl144c_la_rotacion_no_comprime_en_el_hilo_del_log() {
        let d = dir_temporal("sin_gzip_en_linea");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 4 };
        let mut s = Sumidero::abrir(rot.clone(), false);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        s.compresor = Some(tx);
        let q = ColaDeLog::new(64);
        lote(&q, 0, 40);
        s.escribir_lote(&q, 0);
        assert_eq!(numeros(&rot, Pieza::Crudo), vec![1, 2]);
        assert_eq!(numeros(&rot, Pieza::Comprimido), Vec::<u64>::new());
        assert!(rx.try_recv().is_ok(), "se pidió una pasada");
        mantener_segmentos(&rot);
        assert_eq!(numeros(&rot, Pieza::Crudo), Vec::<u64>::new());
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![1, 2]);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: un `.gz` que un corte de luz dejó a ceros o
    /// truncado no cuenta como hecho; el crudo se vuelve a comprimir. Antes
    /// se borraba el crudo y se perdía el segmento.
    #[test]
    fn cl144c_un_gz_roto_no_cuenta_como_hecho() {
        use std::io::Write as _;
        let d = dir_temporal("gz_roto");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 1 << 20, conservar: 4 };
        std::fs::write(rot.pieza(1, Pieza::Crudo), "segmento 1\n").unwrap();
        std::fs::write(rot.pieza(1, Pieza::Comprimido), [0u8; 64]).unwrap();
        let mut bueno = Vec::new();
        {
            let mut e = flate2::write::GzEncoder::new(&mut bueno, flate2::Compression::fast());
            e.write_all(b"segmento 2\n").unwrap();
            e.finish().unwrap();
        }
        std::fs::write(rot.pieza(2, Pieza::Crudo), "segmento 2\n").unwrap();
        std::fs::write(rot.pieza(2, Pieza::Comprimido), &bueno[..bueno.len() - 6]).unwrap();
        let s = Sumidero::abrir(rot.clone(), false);
        assert_eq!(numeros(&rot, Pieza::Crudo), Vec::<u64>::new());
        assert_eq!(leer_gz(&rot.pieza(1, Pieza::Comprimido)), "segmento 1\n");
        assert_eq!(leer_gz(&rot.pieza(2, Pieza::Comprimido)), "segmento 2\n");
        drop(s);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: un segundo escritor con la misma ruta no la toca
    /// mientras viva el primero (ni sus temporales ni su numeración) y la
    /// toma cuando el primero termina. Antes los dos rotaban con su propia
    /// cuenta y uno borraba los segmentos del otro.
    #[test]
    fn cl144c_dos_procesos_no_comparten_la_ruta() {
        let d = dir_temporal("dos");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 4 };
        let a = Sumidero::abrir(rot.clone(), false);
        assert!(matches!(a.posesion, Some(Posesion::Exclusiva { .. })));
        std::fs::write(rot.pieza(9, Pieza::Temporal), "compresión de A en curso").unwrap();
        let mut b = Sumidero::abrir(rot.clone(), false);
        assert!(b.posesion.is_none() && b.archivo.is_none());
        assert!(rot.pieza(9, Pieza::Temporal).exists(), "B no toca lo de A");
        let q = ColaDeLog::new(64);
        lote(&q, 0, 40);
        b.escribir_lote(&q, 1_000);
        assert_eq!(numeros(&rot, Pieza::Crudo), Vec::<u64>::new(), "B no rota");
        assert_eq!(std::fs::metadata(&rot.ruta).unwrap().len(), 0, "B no escribe en el activo de A");
        drop(a);
        lote(&q, 40, 2);
        b.escribir_lote(&q, REINTENTO_APERTURA_MS);
        assert!(matches!(b.posesion, Some(Posesion::Exclusiva { .. })), "B toma la ruta");
        assert!(!rot.pieza(9, Pieza::Temporal).exists(), "el temporal huérfano se borra");
        assert_eq!(std::fs::read_to_string(&rot.ruta).unwrap().lines().count(), 2);
        drop(b);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: si stdout ya está redirigido al mismo archivo
    /// (otro handle que escribe), el sumidero no lo abre y sigue en stdout.
    /// Antes los dos escribían con su propio desplazamiento y se pisaban.
    #[cfg(windows)]
    #[test]
    fn cl144c_un_archivo_en_el_que_ya_escribe_otro_handle_no_se_abre() {
        let d = dir_temporal("stdout");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 4 };
        let redirigido = std::fs::OpenOptions::new().create(true).append(true).open(&rot.ruta).unwrap();
        let s = Sumidero::abrir(rot.clone(), false);
        assert!(s.archivo.is_none());
        drop(s);
        drop(redirigido);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: un archivo con nombre parecido que no es un
    /// segmento (`motor.log.7`, `motor.log.1.gz`) no cuenta para la
    /// numeración ni se comprime o borra. Antes `.7` fijaba `ultimo` = 7, se
    /// intentaba comprimir `.000007` en cada pasada y nunca se borraba.
    #[test]
    fn cl144c_un_nombre_que_no_es_segmento_se_ignora() {
        let d = dir_temporal("ajenos");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 2 * 1024, conservar: 2 };
        std::fs::write(d.join("motor.log.7"), "copia a mano\n").unwrap();
        std::fs::write(d.join("motor.log.1.gz"), "no es mío").unwrap();
        assert_eq!(rot.piezas(), Vec::new());
        let mut s = Sumidero::abrir(rot.clone(), false);
        assert_eq!(s.ultimo, 0);
        let q = ColaDeLog::new(64);
        for k in 0..2 {
            lote(&q, k * 40, 40);
            s.escribir_lote(&q, 0);
        }
        assert_eq!(numeros(&rot, Pieza::Comprimido), vec![3, 4]);
        assert_eq!(std::fs::read_to_string(d.join("motor.log.7")).unwrap(), "copia a mano\n");
        assert_eq!(std::fs::read_to_string(d.join("motor.log.1.gz")).unwrap(), "no es mío");
        let _ = std::fs::remove_dir_all(d);
    }

    /// Esperado: si la cola no se puede publicar porque su consumidor no se
    /// puede crear, la cola no se publica y las macros siguen escribiendo de
    /// forma síncrona. Antes la cola quedaba publicada sin nadie que la
    /// vaciara y el motor corría sin log.
    #[test]
    fn cl144b_la_cola_solo_se_publica_con_su_consumidor() {
        let celda = OnceLock::new();
        let sin_hilo = |_| Err(std::io::Error::other("sin hilos"));
        assert!(!instalar_en(&celda, 8, sin_hilo));
        assert!(celda.get().is_none());
        assert!(instalar_en(&celda, 8, |_| Ok(())));
        assert_eq!(celda.get().map(|q| q.capacity()), Some(8));
        assert!(!instalar_en(&celda, 16, |_| Ok(())), "idempotente");
    }

    /// Esperado: vaciar al apagar reintenta el búfer que un flush fallido
    /// dejó lleno aunque la cola ya esté vacía.
    #[test]
    fn cl144b_vaciar_al_apagar_escribe_el_bufer_pendiente() {
        use std::io::Write as _;
        let d = dir_temporal("apagado");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 1 << 20, conservar: 2 };
        let mut s = Sumidero::abrir(rot.clone(), false);
        s.archivo.as_mut().unwrap().write_all(b"ultima linea del cierre\n").unwrap();
        let m = std::sync::Mutex::new(s);
        let q = ColaDeLog::new(8);
        assert_eq!(vaciar_telemetria_en(Some(&q), Some(&m)), 0);
        assert_eq!(std::fs::read_to_string(&rot.ruta).unwrap(), "ultima linea del cierre\n");
        assert_eq!(vaciar_telemetria_en(None, Some(&m)), 0, "sin cola no hay nada que vaciar");
        drop(m);
        let _ = std::fs::remove_dir_all(d);
    }

    /// CL-144c — Esperado: el apagado espera al lote que GhostLogger tiene a
    /// medias (cola ya vacía, búfer sin volcar, cerrojo tomado) y vuelve con
    /// ese lote en disco. Con el atajo de la cola vacía volvía antes y
    /// `process::exit` lo perdía.
    #[test]
    fn cl144c_el_apagado_espera_el_lote_a_medias() {
        use std::io::Write as _;
        let d = dir_temporal("apagado_espera");
        let rot = Rotacion { ruta: d.join("motor.log"), tope_bytes: 1 << 20, conservar: 2 };
        let m = std::sync::Arc::new(std::sync::Mutex::new(Sumidero::abrir(rot.clone(), false)));
        let q = ColaDeLog::new(8);
        let (listo_tx, listo_rx) = std::sync::mpsc::channel();
        let del_logger = std::sync::Arc::clone(&m);
        let logger = std::thread::spawn(move || {
            let mut s = del_logger.lock().unwrap();
            s.archivo.as_mut().unwrap().write_all(b"lote a medias\n").unwrap();
            listo_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(200));
            s.archivo.as_mut().unwrap().flush().unwrap();
        });
        listo_rx.recv().unwrap();
        assert_eq!(vaciar_telemetria_en(Some(&q), Some(&*m)), 0);
        assert_eq!(std::fs::read_to_string(&rot.ruta).unwrap(), "lote a medias\n");
        logger.join().unwrap();
        drop(m);
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
