//! SALIDAFALLO v2: salida fatal nativa del colector con cierre y sellado posibles.
//!
//! Piezas (todas fuera de Tokio salvo el supervisor de tareas, que solo observa joins):
//! - `Control`: primer fatal persistente (clase + detalle en un bufer fijo, sin heap ni mutex),
//!   errores secundarios, banderas de cierre y adaptadores de reloj y salida.
//! - `vigilar`: el vigilante `std::thread`. Duerme con `park`; al primer fatal aplica los cortes
//!   nativos (admision, escritor, store), espera con reloj monotono a que el finalizador avise y
//!   termina el proceso con codigo no cero. Si vence el plazo, termina sin esperar a nadie.
//! - `instalar_hook`: hook de panic global (solo en el recorrido `capture`, nunca en el harness).
//! - `Admision`: corte observable de la cola de archivo. Un solo `AtomicU64` empaqueta la
//!   bandera de corte, las ofertas en vuelo y las confirmadas: el corte devuelve una instantanea
//!   coherente y ninguna oferta confirma despues de el (la confirmacion es un CAS que exige la
//!   bandera abierta).
//! - `CompuertaStore`: parada nativa del hilo capture-store. Mismo patron: abrir operacion y
//!   cerrar la compuerta operan sobre la misma palabra atomica, asi que o la operacion entra antes
//!   del corte (y el corte la registra como iniciada) o ve la compuerta cerrada y no empieza.
//! - `Supervisor`: observa el JoinSet superior con la identidad de cada tarea.

use crate::CaptureResult;
use std::{
    cell::UnsafeCell,
    collections::HashMap,
    io,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    thread::{self, Thread},
    time::{Duration, Instant},
};

// ---------------------------------------------------------------------------------------------
// Codigos de salida (identificadores, no cifras operativas). 0, 1 y 3 ya existian.
// ---------------------------------------------------------------------------------------------
/// Fatal con el cierre posible completado (END escrito y contabilidad coherente).
pub const SALIDA_FATAL_CERRADO: u32 = 1;
/// Fatal con cierre incompleto: plazo vencido, contabilidad sin resolver, store sin confirmar
/// su parada o END no persistido.
pub const SALIDA_FATAL_INCOMPLETO: u32 = 4;
/// El vigilante fallo (panic o no pudo vigilar): terminacion nativa inmediata.
pub const SALIDA_VIGILANTE: u32 = 5;

// ---------------------------------------------------------------------------------------------
// Plazo nativo desde el primer fatal (P5). Procedencia:
//   SELLADO_MAX_MEDIDO_NS: el sellado mas lento observado de un segmento, suma de sus fases
//   (sin handoff_to_seal_ns), en las dos capturas v2 cerradas con SEAL_TIMING:
//   logs/captura-l2-40-v2-20261118/mainnet/colector_20261119-031637.stdout.log:17
//   (SHA256 75086EF74DEC34E4ADD251B8459C32B5B7B9760152490B5211A0B0770035E750), segmento
//   hour-1795100400000-0008: 85500+1904600+3290400+24053800+66682075400+7294300+61394800 ns.
//   Maximo de 15 segmentos (6 de colector_20261118-183001.stdout.log, SHA256 8EA38781...,
//   y 9 de colector_20261119-031637.stdout.log).
//   SEGMENTOS_EN_CIERRE: los que pueden quedar por sellar al cortar = 2 abiertos por la regla de
//   rotacion (hora actual y la anterior dentro de ROTATION_GRACE_MS) + SEAL_QUEUE_CAPACITY en
//   cola + 1 en el sellador. Una tercera hora abierta solo aparece con marcas fuera de orden;
//   entonces el plazo puede vencer y el cierre queda incompleto (salida de fallo).
//   El drenaje de la cola de filas no tiene medicion propia: N/D, no se suma nada por el.
// ---------------------------------------------------------------------------------------------
pub const SELLADO_MAX_MEDIDO_NS: u64 = 85_500 + 1_904_600 + 3_290_400 + 24_053_800
    + 66_682_075_400 + 7_294_300 + 61_394_800;
pub const SEGMENTOS_EN_CIERRE: u64 = 2 + crate::SEAL_QUEUE_CAPACITY as u64 + 1;
pub const PLAZO_FATAL: Duration = Duration::from_nanos(SEGMENTOS_EN_CIERRE * SELLADO_MAX_MEDIDO_NS);

// ---------------------------------------------------------------------------------------------
// Adaptadores: reloj monotono y salida del proceso.
// ---------------------------------------------------------------------------------------------
pub trait Reloj: Send + Sync {
    /// Tiempo monotono transcurrido desde un origen fijo.
    fn ahora(&self) -> Duration;
}

pub struct RelojMonotono(Instant);

impl RelojMonotono {
    pub fn nuevo() -> Self {
        Self(Instant::now())
    }
}

impl Reloj for RelojMonotono {
    fn ahora(&self) -> Duration {
        self.0.elapsed()
    }
}

pub trait Salida: Send + Sync {
    /// Termina el proceso. `ordenada`: el cierre posible termino y se puede volcar stdout.
    /// SALIDAFALLO-4 (P5): la salida ordenada NO la ejecuta el vigilante; se pide a otro hilo y el
    /// vigilante sigue haciendo cumplir el plazo hasta la salida real. La no ordenada no vuelve
    /// en produccion; el adaptador de pruebas registra el codigo y vuelve.
    fn terminar(&self, codigo: u32, ordenada: bool);
    /// Verdad si el proceso ya no sigue. En produccion, mientras este codigo corre, el proceso
    /// esta vivo (falso); solo el adaptador de pruebas simula el fin.
    fn terminado(&self) -> bool {
        false
    }
}

/// Peticion de salida ordenada para el hilo `capture-salida` (codigo + 1; 0 = sin peticion).
pub struct SalidaPedida {
    codigo: AtomicU32,
    hilo: OnceLock<Thread>,
}

/// Salida real. Ordenada: el hilo `capture-salida` (arrancado antes del vigilante) vuelca
/// stdout/stderr y termina con `TerminateProcess` (sin `ExitProcess`: no hay desconexion de DLL
/// ni candado del cargador que pueda bloquear); si ese volcado se bloquea, el vigilante, que
/// sigue armado, termina con cierre incompleto al vencer el plazo. No ordenada (plazo vencido o
/// fallo del vigilante): `TerminateProcess` inmediato (sin destructores, atexit ni candados).
pub struct SalidaProceso {
    pedida: Arc<SalidaPedida>,
}

impl SalidaProceso {
    pub fn arrancar() -> io::Result<Self> {
        let pedida = Arc::new(SalidaPedida { codigo: AtomicU32::new(0), hilo: OnceLock::new() });
        let p2 = Arc::clone(&pedida);
        let h = thread::Builder::new().name("capture-salida".into()).spawn(move || salida_ordenada(&p2))?;
        let _ = pedida.hilo.set(h.thread().clone());
        Ok(Self { pedida })
    }
}

fn salida_ordenada(p: &SalidaPedida) {
    let codigo = loop {
        let c = p.codigo.load(Ordering::SeqCst);
        if c != 0 {
            break c - 1;
        }
        thread::park();
    };
    {
        use std::io::Write as _;
        let _ = io::stdout().flush();
        let _ = io::stderr().flush();
    }
    terminar_nativo(codigo)
}

impl Salida for SalidaProceso {
    fn terminar(&self, codigo: u32, ordenada: bool) {
        if ordenada {
            let _ = self.pedida.codigo.compare_exchange(0, codigo + 1, Ordering::SeqCst, Ordering::SeqCst);
            if let Some(h) = self.pedida.hilo.get() {
                h.unpark();
            }
            return;
        }
        terminar_nativo(codigo)
    }
}

#[cfg(windows)]
fn terminar_nativo(codigo: u32) -> ! {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    unsafe {
        TerminateProcess(GetCurrentProcess(), codigo);
    }
    std::process::abort()
}

#[cfg(not(windows))]
fn terminar_nativo(_codigo: u32) -> ! {
    std::process::abort()
}

// ---------------------------------------------------------------------------------------------
// Bufer fijo escrito una sola vez (por quien gana el CAS del primer fatal): sin heap ni mutex.
// ---------------------------------------------------------------------------------------------
const TEXTO_FIJO: usize = 1024;

struct TextoFijo {
    bytes: UnsafeCell<[u8; TEXTO_FIJO]>,
    largo: AtomicUsize,
    listo: AtomicBool,
}

// Seguro: solo escribe el ganador unico del CAS de `Control::clase`, antes de publicar `listo`
// con Release; los lectores leen despues de observar `listo` con Acquire.
unsafe impl Sync for TextoFijo {}

impl TextoFijo {
    const fn nuevo() -> Self {
        Self { bytes: UnsafeCell::new([0; TEXTO_FIJO]), largo: AtomicUsize::new(0), listo: AtomicBool::new(false) }
    }

    fn escribir_una_vez(&self, partes: &[&[u8]]) {
        let mut n = 0;
        let buf = unsafe { &mut *self.bytes.get() };
        for parte in partes {
            for &b in *parte {
                if n == TEXTO_FIJO {
                    break;
                }
                buf[n] = b;
                n += 1;
            }
        }
        self.largo.store(n, Ordering::Relaxed);
        self.listo.store(true, Ordering::Release);
    }

    fn leer(&self) -> Option<String> {
        if !self.listo.load(Ordering::Acquire) {
            return None;
        }
        let n = self.largo.load(Ordering::Relaxed);
        let buf = unsafe { &*self.bytes.get() };
        Some(String::from_utf8_lossy(&buf[..n]).into_owned())
    }
}

/// Entero decimal sin heap (para la linea del panic en el hook).
fn decimal(mut v: u32, buf: &mut [u8; 10]) -> &[u8] {
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            return &buf[i..];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Clase del primer fatal.
// ---------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Clase {
    Ninguna = 0,
    /// Panic observado por el hook (cualquier hilo: worker, driver, tarea, store...).
    Panic = 1,
    /// JoinError de una tarea superior.
    TareaPanic = 2,
    /// Final inesperado (Ok antes de la orden de parada, o cancelacion) de tarea persistente.
    TareaFin = 3,
    /// Result::Err de una tarea o conexion (antes `reader_error`).
    Lector = 4,
    /// El hilo capture-store termino con panic.
    StorePanic = 5,
    /// El hilo capture-store volvio sin orden de parada.
    StoreFin = 6,
    /// El aviso de store se cancelo sin causa conocida.
    StoreCanal = 7,
    /// Fallo del escritor gzip (append, apertura, handoff o fin prematuro).
    Escritor = 8,
    /// Fallo o fin prematuro del sellador.
    Sellador = 9,
    /// Fatal pedido por el finalizador o el cierre (p. ej. END no persistido).
    Cierre = 10,
}

impl Clase {
    fn desde(v: u8) -> Self {
        match v {
            1 => Self::Panic,
            2 => Self::TareaPanic,
            3 => Self::TareaFin,
            4 => Self::Lector,
            5 => Self::StorePanic,
            6 => Self::StoreFin,
            7 => Self::StoreCanal,
            8 => Self::Escritor,
            9 => Self::Sellador,
            10 => Self::Cierre,
            _ => Self::Ninguna,
        }
    }

    pub fn nombre(self) -> &'static str {
        match self {
            Self::Ninguna => "ninguna",
            Self::Panic => "panic",
            Self::TareaPanic => "tarea_panic",
            Self::TareaFin => "tarea_fin_inesperado",
            Self::Lector => "lector_error",
            Self::StorePanic => "store_panic",
            Self::StoreFin => "store_fin_inesperado",
            Self::StoreCanal => "store_canal_cancelado",
            Self::Escritor => "escritor_error",
            Self::Sellador => "sellador_error",
            Self::Cierre => "cierre_error",
        }
    }
}

// Puntos de pausa de prueba (SALIDAFALLO-5): ventanas entre un cambio de estado atomico y la
// publicacion de lo que lo describe. Solo existen bajo cfg(test); produccion no los tiene.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Punto {
    /// Tras ganar el CAS del primer fatal y antes de copiar su detalle.
    TrasCasFatal,
    /// Tras el `fetch_or` del corte de admision y antes de publicar su instantanea.
    TrasCorteAdmision,
    /// Tras el `fetch_or` del cierre de store y antes de publicar su instantanea.
    TrasCierreStore,
}

#[cfg(test)]
pub(crate) type Pausa = Box<dyn Fn(Punto) + Send + Sync>;

#[cfg(test)]
fn pausar(p: &OnceLock<Pausa>, punto: Punto) {
    if let Some(f) = p.get() {
        f(punto);
    }
}

// Banderas de `Control::banderas`.
const FATAL: u32 = 1;
const DESARMADO: u32 = 1 << 1;
const FIN_LISTO: u32 = 1 << 2;
const INCOMPLETO: u32 = 1 << 3;
const VIGILANTE_VIVO: u32 = 1 << 4;
const ASYNC_LISTO: u32 = 1 << 5;
const CORTES_HECHOS: u32 = 1 << 6;

/// Datos que el recorrido async entrega al finalizador al terminar (parada sana o fatal con
/// runtime vivo).
#[derive(Default)]
pub struct FinAsync {
    pub stop_reason: &'static str,
    pub exit: u8,
    pub errors: Vec<String>,
}

pub struct Control {
    banderas: AtomicU32,
    clase: AtomicU8,
    detalle: TextoFijo,
    t_fatal_ns: AtomicU64,
    secundarios: Mutex<Vec<String>>,
    secundarios_perdidos: AtomicUsize,
    codigo_fin: AtomicU8,
    vigilante: OnceLock<Thread>,
    finalizador: OnceLock<Thread>,
    principal: OnceLock<Thread>,
    aviso_async: futures_util::task::AtomicWaker,
    fin_async: Mutex<Option<FinAsync>>,
    pub admision: Arc<Admision>,
    pub store: Arc<CompuertaStore>,
    plazo: Duration,
    reloj: Box<dyn Reloj>,
    salida: Box<dyn Salida>,
    #[cfg(test)]
    pausa: OnceLock<Pausa>,
}

impl Control {
    pub fn nuevo(plazo: Duration, reloj: Box<dyn Reloj>, salida: Box<dyn Salida>, store: Arc<CompuertaStore>) -> Self {
        Self {
            banderas: AtomicU32::new(0),
            clase: AtomicU8::new(Clase::Ninguna as u8),
            detalle: TextoFijo::nuevo(),
            t_fatal_ns: AtomicU64::new(0),
            secundarios: Mutex::new(Vec::new()),
            secundarios_perdidos: AtomicUsize::new(0),
            codigo_fin: AtomicU8::new(0),
            vigilante: OnceLock::new(),
            finalizador: OnceLock::new(),
            principal: OnceLock::new(),
            aviso_async: futures_util::task::AtomicWaker::new(),
            fin_async: Mutex::new(None),
            admision: Arc::new(Admision::nueva()),
            store,
            plazo,
            reloj,
            salida,
            #[cfg(test)]
            pausa: OnceLock::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn fijar_pausa(&self, p: Pausa) {
        let _ = self.pausa.set(p);
    }

    #[cfg(test)]
    pub(crate) fn pausar(&self, punto: Punto) {
        pausar(&self.pausa, punto);
    }

    pub fn plazo(&self) -> Duration {
        self.plazo
    }

    fn banderas(&self) -> u32 {
        self.banderas.load(Ordering::SeqCst)
    }

    pub fn es_fatal(&self) -> bool {
        self.banderas() & FATAL != 0
    }

    pub fn registrar_vigilante(&self, hilo: Thread) {
        let _ = self.vigilante.set(hilo);
    }

    pub fn registrar_finalizador(&self, hilo: Thread) {
        let _ = self.finalizador.set(hilo);
    }

    pub fn registrar_principal(&self, hilo: Thread) {
        let _ = self.principal.set(hilo);
    }

    fn despertar(hilo: &OnceLock<Thread>) {
        if let Some(h) = hilo.get() {
            h.unpark();
        }
    }

    pub fn despertar_vigilante(&self) {
        Self::despertar(&self.vigilante);
    }

    pub fn despertar_finalizador(&self) {
        Self::despertar(&self.finalizador);
    }

    /// Fija el primer fatal (CAS) o agrega un secundario. Devuelve true si fue el primero.
    /// Camino comun: sin await ni E/S; el detalle se copia al bufer fijo sin heap.
    fn fijar(&self, clase: Clase, partes: &[&[u8]], desde_hook: bool) -> bool {
        let primero = self
            .clase
            .compare_exchange(Clase::Ninguna as u8, clase as u8, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if primero {
            #[cfg(test)]
            self.pausar(Punto::TrasCasFatal);
            self.detalle.escribir_una_vez(partes);
            let t = u64::try_from(self.reloj.ahora().as_nanos()).unwrap_or(u64::MAX);
            self.t_fatal_ns.store(t, Ordering::SeqCst);
            self.banderas.fetch_or(FATAL, Ordering::SeqCst);
        } else if desde_hook {
            // El hook no espera ningun mutex: si esta ocupado, solo se cuenta.
            match self.secundarios.try_lock() {
                Ok(mut v) => v.push(texto(clase, partes)),
                Err(_) => {
                    self.secundarios_perdidos.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            match self.secundarios.lock() {
                Ok(mut v) => v.push(texto(clase, partes)),
                Err(p) => p.into_inner().push(texto(clase, partes)),
            }
        }
        if desde_hook && self.banderas() & VIGILANTE_VIVO == 0 {
            // Camino seguro: sin vigilante no hay quien haga cumplir el plazo.
            self.salida.terminar(SALIDA_VIGILANTE, false);
        }
        self.despertar_vigilante();
        primero
    }

    pub fn fatal(&self, clase: Clase, detalle: &str) -> bool {
        self.fijar(clase, &[detalle.as_bytes()], false)
    }

    /// Lo que hace el hook: clase Panic, hilo, archivo:linea y mensaje si es texto estatico o
    /// String, sin formatear ni reservar memoria (solo copia al bufer fijo).
    pub fn fatal_por_panic(&self, info: &std::panic::PanicHookInfo<'_>) {
        let hilo = thread::current();
        let nombre = hilo.name().unwrap_or("sin-nombre");
        let mut linea = [0u8; 10];
        let (archivo, num): (&str, &[u8]) = match info.location() {
            Some(l) => (l.file(), decimal(l.line(), &mut linea)),
            None => ("?", b"?"),
        };
        let carga = info.payload();
        let mensaje: &str = if let Some(s) = carga.downcast_ref::<&str>() {
            s
        } else if let Some(s) = carga.downcast_ref::<String>() {
            s.as_str()
        } else {
            "(carga no textual)"
        };
        self.fijar(
            Clase::Panic,
            &[b"panic en hilo ", nombre.as_bytes(), b" en ", archivo.as_bytes(), b":", num, b": ", mensaje.as_bytes()],
            true,
        );
    }

    /// SALIDAFALLO-5 (N1 hallazgo 2): el primer fatal solo se ve completo. El ganador del CAS
    /// copia el detalle (listo con Release) y despues publica FATAL; aqui se exige FATAL antes de
    /// leer clase y detalle, asi que nunca se devuelve la clase con un detalle vacio. Entre el CAS
    /// y la publicacion, `fatal_pendiente` dice la clase sin inventar el texto.
    pub fn primer_fatal(&self) -> Option<(Clase, String)> {
        if self.banderas() & FATAL == 0 {
            return None;
        }
        let clase = Clase::desde(self.clase.load(Ordering::SeqCst));
        Some((clase, self.detalle.leer()?))
    }

    /// Clase de un primer fatal ya decidido (CAS ganado) cuyo detalle aun no esta publicado.
    pub fn fatal_pendiente(&self) -> Option<Clase> {
        if self.primer_fatal().is_some() {
            return None;
        }
        match Clase::desde(self.clase.load(Ordering::SeqCst)) {
            Clase::Ninguna => None,
            c => Some(c),
        }
    }

    pub fn secundarios(&self) -> (Vec<String>, usize) {
        let v = match self.secundarios.lock() {
            Ok(v) => v.clone(),
            Err(p) => p.into_inner().clone(),
        };
        (v, self.secundarios_perdidos.load(Ordering::Relaxed))
    }

    /// Cortes nativos del primer fatal: admision, escritor y store. Solo atomicos, `unpark` y un
    /// envio no bloqueante; no espera mutex de captura, permisos, capacidad de cola ni scheduler.
    pub fn cortes_nativos(&self) {
        if self.banderas.fetch_or(CORTES_HECHOS, Ordering::SeqCst) & CORTES_HECHOS != 0 {
            return;
        }
        self.admision.cortar(Motivo::Fatal);
        self.store.cerrar();
        self.despertar_finalizador();
    }

    /// Lo llama el finalizador (otro hilo): despierta el `select!` async. Puede tocar el
    /// scheduler de Tokio; por eso no lo hace el vigilante.
    pub fn avisar_async(&self) {
        self.aviso_async.wake();
    }

    /// Future que resuelve cuando hay fatal (despertado por `avisar_async`).
    pub async fn esperar_fatal(&self) {
        std::future::poll_fn(|cx| {
            self.aviso_async.register(cx.waker());
            if self.es_fatal() { std::task::Poll::Ready(()) } else { std::task::Poll::Pending }
        })
        .await
    }

    pub fn entregar_fin_async(&self, fin: FinAsync) {
        match self.fin_async.lock() {
            Ok(mut s) => *s = Some(fin),
            Err(p) => *p.into_inner() = Some(fin),
        }
        self.banderas.fetch_or(ASYNC_LISTO, Ordering::SeqCst);
        self.despertar_finalizador();
    }

    pub fn tomar_fin_async(&self) -> Option<FinAsync> {
        match self.fin_async.lock() {
            Ok(mut s) => s.take(),
            Err(p) => p.into_inner().take(),
        }
    }

    pub fn async_listo(&self) -> bool {
        self.banderas() & ASYNC_LISTO != 0
    }

    /// El finalizador termino (con o sin END persistido).
    pub fn fin_listo(&self, codigo: u8, incompleto: bool) {
        self.codigo_fin.store(codigo, Ordering::SeqCst);
        let extra = if incompleto { INCOMPLETO } else { 0 };
        self.banderas.fetch_or(FIN_LISTO | extra, Ordering::SeqCst);
        self.despertar_vigilante();
        Self::despertar(&self.principal);
    }

    pub fn fin_hecho(&self) -> bool {
        self.banderas() & FIN_LISTO != 0
    }

    pub fn codigo_fin(&self) -> u8 {
        self.codigo_fin.load(Ordering::SeqCst)
    }

    /// Hilo principal: espera al finalizador (park; lo despierta `fin_listo`).
    pub fn esperar_fin(&self) {
        while !self.fin_hecho() {
            thread::park();
        }
    }

    /// Desarma la vigilancia de un cierre sano. Solo se llama con los participantes terminados y
    /// el runtime destruido. Falla si existe fatal: un fatal tardio nunca se vuelve exito.
    pub fn desarmar(&self) -> Result<(), ()> {
        let mut actual = self.banderas();
        loop {
            if actual & FATAL != 0 {
                return Err(());
            }
            match self.banderas.compare_exchange(actual, actual | DESARMADO, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    self.despertar_vigilante();
                    return Ok(());
                }
                Err(v) => actual = v,
            }
        }
    }

    pub fn vigilante_vivo(&self) -> bool {
        self.banderas() & VIGILANTE_VIVO != 0
    }
}

fn texto(clase: Clase, partes: &[&[u8]]) -> String {
    let mut s = String::from(clase.nombre());
    s.push_str(": ");
    for p in partes {
        s.push_str(&String::from_utf8_lossy(p));
    }
    s
}

/// Guarda del vigilante: marca vivo al entrar; si se suelta por panic, termina en nativo.
struct VigilanteVivo<'a>(&'a Control);

impl Drop for VigilanteVivo<'_> {
    fn drop(&mut self) {
        self.0.banderas.fetch_and(!VIGILANTE_VIVO, Ordering::SeqCst);
        if thread::panicking() {
            self.0.salida.terminar(SALIDA_VIGILANTE, false);
        }
    }
}

/// Cuerpo del vigilante. Reloj monotono (adaptador), esperas `park`/`park_timeout`, sin joins,
/// sellado, END ni otra E/S de cierre. Permanece armado hasta `desarmar`.
pub fn vigilar(ctl: &Control) {
    ctl.banderas.fetch_or(VIGILANTE_VIVO, Ordering::SeqCst);
    let _vivo = VigilanteVivo(ctl);
    #[cfg(test)]
    pruebas_vigilante::punto(ctl);
    loop {
        let b = ctl.banderas();
        if b & FATAL != 0 {
            break;
        }
        if b & DESARMADO != 0 {
            return;
        }
        thread::park();
    }
    ctl.cortes_nativos();
    let t0 = Duration::from_nanos(ctl.t_fatal_ns.load(Ordering::SeqCst));
    let mut salida_pedida = false;
    loop {
        let b = ctl.banderas();
        if b & FIN_LISTO != 0 && !salida_pedida {
            salida_pedida = true;
            let codigo = if b & INCOMPLETO != 0 { SALIDA_FATAL_INCOMPLETO } else { SALIDA_FATAL_CERRADO };
            // SALIDAFALLO-4: la salida ordenada la hace otro hilo; el plazo sigue vigilado
            // hasta que el proceso termine de verdad.
            ctl.salida.terminar(codigo, true);
        }
        if salida_pedida && ctl.salida.terminado() {
            return;
        }
        let pasado = ctl.reloj.ahora().saturating_sub(t0);
        if pasado >= ctl.plazo {
            ctl.banderas.fetch_or(INCOMPLETO, Ordering::SeqCst);
            ctl.salida.terminar(SALIDA_FATAL_INCOMPLETO, false);
            return;
        }
        thread::park_timeout(ctl.plazo - pasado);
    }
}

#[cfg(test)]
pub(crate) mod pruebas_vigilante {
    use super::*;
    std::thread_local! {
        pub static FALLAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    /// Inyeccion de prueba: el vigilante falla al arrancar (solo en el hilo que lo pide).
    pub fn punto(_ctl: &Control) {
        if FALLAR.with(|f| f.get()) {
            panic!("fallo inyectado del vigilante");
        }
    }
}

/// Instala el hook global. Solo en el recorrido `capture`; nunca en el proceso del harness.
pub fn instalar_hook(ctl: Arc<Control>) {
    let previo = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ctl.fatal_por_panic(info);
        previo(info);
    }));
}

// ---------------------------------------------------------------------------------------------
// Admision de la cola de archivo (P4a).
// Palabra: bit 63 = cortada; bits 0..21 = ofertas en vuelo; bits 21..63 = ofertas confirmadas.
// Protocolo de cada offer: entrar (en vuelo +1; si ya esta cortada, sale sin reservar) ->
// reservar presupuesto y plaza -> confirmar (CAS que exige no cortada; suma una confirmada) ->
// enviar con el permiso ya reservado -> salir (en vuelo -1, despierta al escritor). Una reserva
// cuya confirmacion falla por el corte suelta permiso y presupuesto al salir.
// Punto observable del corte: el `fetch_or` de `cortar`. Su valor previo da, en la misma lectura,
// las confirmadas (todas con obligacion de drenaje) y las ofertas en vuelo (confirmadas sin enviar
// o reservas pendientes). Despues del corte no hay confirmaciones nuevas. `resuelta` = cortada y
// sin ofertas en vuelo: todas las confirmadas ya estan en el canal.
// ---------------------------------------------------------------------------------------------
const CORTADA: u64 = 1 << 63;
const BITS_VUELO: u32 = 21;
const MASCARA_VUELO: u64 = (1 << BITS_VUELO) - 1;
const UNA_CONFIRMADA: u64 = 1 << BITS_VUELO;
const MASCARA_CONFIRMADAS: u64 = !CORTADA & !MASCARA_VUELO;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motivo {
    Fatal,
    Sana,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Corte {
    pub motivo: Motivo,
    pub confirmadas: u64,
    pub en_vuelo: u64,
}

pub struct Admision {
    estado: AtomicU64,
    corte: OnceLock<Corte>,
    reservas_canceladas: AtomicU64,
    rechazos_tras_corte: AtomicU64,
    escritor: OnceLock<Thread>,
    #[cfg(test)]
    pausa: OnceLock<Pausa>,
}

impl Admision {
    pub fn nueva() -> Self {
        Self {
            estado: AtomicU64::new(0),
            corte: OnceLock::new(),
            reservas_canceladas: AtomicU64::new(0),
            rechazos_tras_corte: AtomicU64::new(0),
            escritor: OnceLock::new(),
            #[cfg(test)]
            pausa: OnceLock::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn fijar_pausa(&self, p: Pausa) {
        let _ = self.pausa.set(p);
    }

    pub fn registrar_escritor(&self, hilo: Thread) {
        let _ = self.escritor.set(hilo);
    }

    pub fn despertar_escritor(&self) {
        if let Some(h) = self.escritor.get() {
            h.unpark();
        }
    }

    pub fn entrar(&self) -> Option<Vuelo<'_>> {
        let previo = self.estado.fetch_add(1, Ordering::SeqCst);
        if previo & CORTADA != 0 {
            self.estado.fetch_sub(1, Ordering::SeqCst);
            self.rechazos_tras_corte.fetch_add(1, Ordering::Relaxed);
            self.despertar_escritor();
            return None;
        }
        Some(Vuelo { adm: self, confirmada: false, cancelada: false })
    }

    /// Corte monotono. Devuelve la instantanea del primer corte (el de este llamador o uno previo).
    pub fn cortar(&self, motivo: Motivo) -> Option<Corte> {
        let previo = self.estado.fetch_or(CORTADA, Ordering::SeqCst);
        if previo & CORTADA == 0 {
            #[cfg(test)]
            pausar(&self.pausa, Punto::TrasCorteAdmision);
            let corte = Corte {
                motivo,
                confirmadas: (previo & MASCARA_CONFIRMADAS) >> BITS_VUELO,
                en_vuelo: previo & MASCARA_VUELO,
            };
            let _ = self.corte.set(corte);
        }
        self.despertar_escritor();
        self.corte.get().copied()
    }

    pub fn corte(&self) -> Option<Corte> {
        self.corte.get().copied()
    }

    /// Verdad solo con la instantanea del corte ya publicada (quien gana el `fetch_or` la fija
    /// justo despues): quien vea el corte puede leer siempre `corte()`.
    pub fn cortada(&self) -> bool {
        self.estado.load(Ordering::SeqCst) & CORTADA != 0 && self.corte.get().is_some()
    }

    pub fn resuelta(&self) -> bool {
        let s = self.estado.load(Ordering::SeqCst);
        s & CORTADA != 0 && s & MASCARA_VUELO == 0 && self.corte.get().is_some()
    }

    pub fn en_vuelo(&self) -> u64 {
        self.estado.load(Ordering::SeqCst) & MASCARA_VUELO
    }

    pub fn confirmadas(&self) -> u64 {
        (self.estado.load(Ordering::SeqCst) & MASCARA_CONFIRMADAS) >> BITS_VUELO
    }

    pub fn reservas_canceladas(&self) -> u64 {
        self.reservas_canceladas.load(Ordering::Relaxed)
    }

    pub fn rechazos_tras_corte(&self) -> u64 {
        self.rechazos_tras_corte.load(Ordering::Relaxed)
    }
}

pub struct Vuelo<'a> {
    adm: &'a Admision,
    confirmada: bool,
    cancelada: bool,
}

impl Vuelo<'_> {
    /// Confirma la oferta si la admision sigue abierta. Tras el corte devuelve false.
    pub fn confirmar(&mut self) -> bool {
        let mut actual = self.adm.estado.load(Ordering::SeqCst);
        loop {
            if actual & CORTADA != 0 {
                self.cancelada = true;
                return false;
            }
            match self.adm.estado.compare_exchange(actual, actual + UNA_CONFIRMADA, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    self.confirmada = true;
                    return true;
                }
                Err(v) => actual = v,
            }
        }
    }
}

impl Drop for Vuelo<'_> {
    fn drop(&mut self) {
        if self.cancelada {
            self.adm.reservas_canceladas.fetch_add(1, Ordering::Relaxed);
        }
        self.adm.estado.fetch_sub(1, Ordering::SeqCst);
        self.adm.despertar_escritor();
    }
}

// ---------------------------------------------------------------------------------------------
// Compuerta nativa de store (P4c). Bit 63 = cerrada; bits bajos = operaciones en curso.
// ---------------------------------------------------------------------------------------------
const CERRADA: u64 = 1 << 63;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OpStore {
    Ninguna = 0,
    Recode = 1,
    RecodeBorraParcialViejo = 2,
    RecodeBorraParcial = 3,
    RecodeBorraReciboParcial = 4,
    RecodeBorraGzip = 5,
    Retencion = 6,
    RetencionBorra = 7,
    Recuperacion = 8,
    RecuperacionBorraGzip = 9,
}

impl OpStore {
    fn desde(v: u8) -> Self {
        match v {
            1 => Self::Recode,
            2 => Self::RecodeBorraParcialViejo,
            3 => Self::RecodeBorraParcial,
            4 => Self::RecodeBorraReciboParcial,
            5 => Self::RecodeBorraGzip,
            6 => Self::Retencion,
            7 => Self::RetencionBorra,
            8 => Self::Recuperacion,
            9 => Self::RecuperacionBorraGzip,
            _ => Self::Ninguna,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorteStore {
    /// Operaciones que ya habian entrado al llegar el corte (no se esperan).
    pub en_curso: u64,
    /// Ultima operacion admitida antes del corte.
    pub ultima: OpStore,
}

type Borrador = Box<dyn Fn(&Path) -> io::Result<()> + Send + Sync>;
type Despertador = Box<dyn Fn() + Send + Sync>;

pub struct CompuertaStore {
    estado: AtomicU64,
    ultima: AtomicU8,
    pub cancel: Arc<AtomicBool>,
    borrador: Borrador,
    despertador: OnceLock<Despertador>,
    corte: OnceLock<CorteStore>,
    #[cfg(test)]
    pausa: OnceLock<Pausa>,
}

impl CompuertaStore {
    pub fn nueva(cancel: Arc<AtomicBool>, borrador: Borrador) -> Self {
        Self {
            estado: AtomicU64::new(0),
            ultima: AtomicU8::new(OpStore::Ninguna as u8),
            cancel,
            borrador,
            despertador: OnceLock::new(),
            corte: OnceLock::new(),
            #[cfg(test)]
            pausa: OnceLock::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn fijar_pausa(&self, p: Pausa) {
        let _ = self.pausa.set(p);
    }

    /// Produccion: el borrado real es `fs::remove_file`.
    pub fn real(cancel: Arc<AtomicBool>) -> Self {
        Self::nueva(cancel, Box::new(|p| std::fs::remove_file(p)))
    }

    pub fn fijar_despertador(&self, d: Despertador) {
        let _ = self.despertador.set(d);
    }

    pub fn admitir(&self, op: OpStore) -> Option<GuardaStore<'_>> {
        let previo = self.estado.fetch_add(1, Ordering::SeqCst);
        if previo & CERRADA != 0 {
            self.estado.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        self.ultima.store(op as u8, Ordering::SeqCst);
        Some(GuardaStore(self))
    }

    /// Borrado ordenado por la compuerta: si esta cerrada, no se borra.
    pub fn borrar(&self, op: OpStore, ruta: &Path) -> io::Result<()> {
        let Some(_g) = self.admitir(op) else {
            return Err(io::Error::other(format!("compuerta de store cerrada; no se borra {}", ruta.display())));
        };
        (self.borrador)(ruta)
    }

    /// Cierre monotono, sin esperar operaciones en curso: las registra y despierta al hilo.
    pub fn cerrar(&self) -> Option<CorteStore> {
        self.cancel.store(true, Ordering::Release);
        let previo = self.estado.fetch_or(CERRADA, Ordering::SeqCst);
        if previo & CERRADA == 0 {
            #[cfg(test)]
            pausar(&self.pausa, Punto::TrasCierreStore);
            let _ = self.corte.set(CorteStore {
                en_curso: previo & !CERRADA,
                ultima: OpStore::desde(self.ultima.load(Ordering::SeqCst)),
            });
        }
        if let Some(d) = self.despertador.get() {
            d();
        }
        self.corte.get().copied()
    }

    /// Como `Admision::cortada`: verdad solo con la instantanea del corte publicada.
    pub fn cerrada(&self) -> bool {
        self.estado.load(Ordering::SeqCst) & CERRADA != 0 && self.corte.get().is_some()
    }

    pub fn en_curso(&self) -> u64 {
        self.estado.load(Ordering::SeqCst) & !CERRADA
    }

    pub fn corte(&self) -> Option<CorteStore> {
        self.corte.get().copied()
    }
}

pub struct GuardaStore<'a>(&'a CompuertaStore);

impl Drop for GuardaStore<'_> {
    fn drop(&mut self) {
        self.0.estado.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Como termino el hilo capture-store (lo fija su envoltorio antes de soltar el aviso oneshot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EstadoStore {
    Corriendo = 0,
    Volvio = 1,
    Panic = 2,
}

pub struct SalidaStore(AtomicU8);

impl SalidaStore {
    pub fn nueva() -> Self {
        Self(AtomicU8::new(EstadoStore::Corriendo as u8))
    }
    pub fn fijar(&self, e: EstadoStore) {
        self.0.store(e as u8, Ordering::SeqCst);
    }
    pub fn leer(&self) -> EstadoStore {
        match self.0.load(Ordering::SeqCst) {
            1 => EstadoStore::Volvio,
            2 => EstadoStore::Panic,
            _ => EstadoStore::Corriendo,
        }
    }
}

/// Clasificacion del aviso de store en el `select!` de captura (P2). La medicion real de disco
/// bajo y el error de almacen conservan su parada ordinaria; panic, fin sin orden y canal
/// cancelado son fatales. Nunca se convierte la cancelacion en una medicion.
#[derive(Debug, PartialEq)]
pub enum EventoStore {
    DiscoBajo { free: u64, minimum: u64 },
    ErrorAlmacen(String),
    FinEsperado,
    FinInesperado,
    Panic,
    CanalCancelado,
}

pub fn clasificar_store(
    aviso: Result<crate::store::StoreStop, tokio::sync::oneshot::error::RecvError>,
    estado: EstadoStore,
    cierre_pedido: bool,
) -> EventoStore {
    match aviso {
        Ok(crate::store::StoreStop::LowDisk { free, minimum }) => EventoStore::DiscoBajo { free, minimum },
        Ok(crate::store::StoreStop::StorageError(e)) => EventoStore::ErrorAlmacen(e),
        Err(_) => match estado {
            EstadoStore::Panic => EventoStore::Panic,
            EstadoStore::Volvio if cierre_pedido => EventoStore::FinEsperado,
            EstadoStore::Volvio => EventoStore::FinInesperado,
            EstadoStore::Corriendo => EventoStore::CanalCancelado,
        },
    }
}

impl EventoStore {
    /// Aplica el evento: los fatales pasan al control; devuelve (exit, error, stop_reason).
    pub fn aplicar(self, ctl: &Control) -> (u8, String, &'static str) {
        match self {
            Self::DiscoBajo { free, minimum } => {
                let s = crate::store::StoreStop::LowDisk { free, minimum };
                (s.code(), s.message(), s.reason())
            }
            Self::ErrorAlmacen(e) => {
                let s = crate::store::StoreStop::StorageError(e);
                (s.code(), s.message(), s.reason())
            }
            Self::FinEsperado => (0, "capture-store termino tras el cierre".into(), "store_ended"),
            Self::FinInesperado => {
                let m = "capture-store termino sin orden de parada";
                ctl.fatal(Clase::StoreFin, m);
                (1, m.into(), "store_ended")
            }
            Self::Panic => {
                let m = "capture-store termino con panic";
                ctl.fatal(Clase::StorePanic, m);
                (1, m.into(), "store_panic")
            }
            Self::CanalCancelado => {
                let m = "aviso de capture-store cancelado con el hilo sin terminar";
                ctl.fatal(Clase::StoreCanal, m);
                (1, m.into(), "store_error")
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Supervisor del JoinSet superior (P2): identidad por tokio::task::Id.
// ---------------------------------------------------------------------------------------------
pub struct Supervisor {
    tareas: tokio::task::JoinSet<CaptureResult<()>>,
    nombres: HashMap<tokio::task::Id, String>,
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::nuevo()
    }
}

impl Supervisor {
    pub fn nuevo() -> Self {
        Self { tareas: tokio::task::JoinSet::new(), nombres: HashMap::new() }
    }

    pub fn lanzar<F>(&mut self, nombre: String, tarea: F)
    where
        F: std::future::Future<Output = CaptureResult<()>> + Send + 'static,
    {
        let h = self.tareas.spawn(tarea);
        self.nombres.insert(h.id(), nombre);
    }

    pub fn vacio(&self) -> bool {
        self.tareas.is_empty()
    }

    /// Siguiente tarea terminada con su nombre (pendiente para siempre si no quedan).
    pub async fn siguiente(&mut self) -> (String, Result<CaptureResult<()>, tokio::task::JoinError>) {
        match self.tareas.join_next_with_id().await {
            Some(Ok((id, r))) => (self.nombres.remove(&id).unwrap_or_else(|| format!("{id}")), Ok(r)),
            Some(Err(e)) => (self.nombres.remove(&e.id()).unwrap_or_else(|| format!("{}", e.id())), Err(e)),
            None => std::future::pending().await,
        }
    }

    /// Clasifica un final. `parada`: ya se pidio stop. Devuelve el error a registrar, si hay.
    pub fn evaluar(
        ctl: &Control,
        nombre: &str,
        r: Result<CaptureResult<()>, tokio::task::JoinError>,
        parada: bool,
    ) -> Option<String> {
        match r {
            Err(e) if e.is_panic() => {
                let m = format!("tarea {nombre}: panic ({e})");
                ctl.fatal(Clase::TareaPanic, &m);
                Some(m)
            }
            Err(e) => {
                let m = format!("tarea {nombre}: cancelada ({e})");
                ctl.fatal(Clase::TareaFin, &m);
                Some(m)
            }
            Ok(Err(e)) => {
                let m = format!("tarea {nombre}: {e}");
                ctl.fatal(Clase::Lector, &m);
                Some(m)
            }
            Ok(Ok(())) if !parada => {
                let m = format!("tarea {nombre}: termino sin orden de parada");
                ctl.fatal(Clase::TareaFin, &m);
                Some(m)
            }
            Ok(Ok(())) => None,
        }
    }

    /// Cierre: recoge TODOS los resultados (no los descarta) y los clasifica con parada=true.
    pub async fn cerrar(&mut self, ctl: &Control) -> Vec<String> {
        let mut errores = Vec::new();
        while !self.tareas.is_empty() {
            let (nombre, r) = self.siguiente().await;
            if let Some(e) = Self::evaluar(ctl, &nombre, r, true) {
                errores.push(e);
            }
        }
        errores
    }
}

/// Final de una conexion efimera (market): un join exitoso es normal (su `Life::Ended` gobierna la
/// reconexion); un JoinError es fatal aunque `Life::Ended` nunca llegue.
pub fn conexion_terminada(r: Result<(), tokio::task::JoinError>) -> CaptureResult<()> {
    match r {
        Ok(()) => Ok(()),
        Err(e) if e.is_panic() => Err(io::Error::other(format!("conexion market: panic ({e})")).into()),
        Err(e) => Err(io::Error::other(format!("conexion market: cancelada ({e})")).into()),
    }
}

#[cfg(test)]
pub(crate) mod tests;
