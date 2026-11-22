//! SALIDAFALLO-9 (P8): inyecciones versionadas del ensayo en subproceso.
//!
//! Solo se activan con el venue `local` (127.0.0.1) y el argumento `inyectar=<caso>`. Con mainnet
//! o testnet el argumento se rechaza al arrancar, `CASO` queda sin fijar y cada punto de abajo es
//! la lectura de un `OnceLock` vacio que vuelve sin hacer nada. Cada caso actua una sola vez.
//! Los eventos se marcan en stdout (`P8_INYECCION <caso> <evento> control_ns=<n>`) para que el
//! banco del padre los feche con su propio reloj monotono.

use crate::salidafallo::{Admision, Clase, CompuertaStore, Control, OpStore};
use std::{
    io,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caso {
    PanicTarea,
    FinTarea,
    CancelarConexionMarket,
    PanicWorkerPark,
    PausaReserva,
    PausaConfirmada,
    RetenerConfirmada,
    ColaLlena,
    FalloAppend,
    FalloManifiesto,
    SelladorBloqueado,
    StoreOcupado,
    FalloEnd,
    BloqueoEnd,
    FatalDuranteEnd,
    FatalDropRuntime,
    FalloVigilante,
    SalidaBloqueada,
}

const CASOS: [(&str, Caso); 18] = [
    ("panic_tarea", Caso::PanicTarea),
    ("fin_tarea", Caso::FinTarea),
    ("cancelar_conexion_market", Caso::CancelarConexionMarket),
    ("panic_worker_park", Caso::PanicWorkerPark),
    ("pausa_reserva", Caso::PausaReserva),
    ("pausa_confirmada", Caso::PausaConfirmada),
    ("retener_confirmada", Caso::RetenerConfirmada),
    ("cola_llena", Caso::ColaLlena),
    ("fallo_append", Caso::FalloAppend),
    ("fallo_manifiesto", Caso::FalloManifiesto),
    ("sellador_bloqueado", Caso::SelladorBloqueado),
    ("store_ocupado", Caso::StoreOcupado),
    ("fallo_end", Caso::FalloEnd),
    ("bloqueo_end", Caso::BloqueoEnd),
    ("fatal_durante_end", Caso::FatalDuranteEnd),
    ("fatal_drop_runtime", Caso::FatalDropRuntime),
    ("fallo_vigilante", Caso::FalloVigilante),
    ("salida_bloqueada", Caso::SalidaBloqueada),
];

impl Caso {
    pub fn nombre(self) -> &'static str {
        CASOS.iter().find(|(_, c)| *c == self).map_or("?", |(n, _)| n)
    }

    /// Casos que necesitan el disco de ensayo (`DiscoP8`); los demas usan `RealDisk`.
    pub fn usa_disco(self) -> bool {
        matches!(
            self,
            Self::FalloAppend | Self::FalloManifiesto | Self::SelladorBloqueado | Self::FalloEnd | Self::BloqueoEnd | Self::FatalDuranteEnd
        )
    }
}

static CASO: OnceLock<Caso> = OnceLock::new();
static CONTROL: OnceLock<Arc<Control>> = OnceLock::new();
/// El inyector ya armo los puntos (el archivo tiene trabajo conocido).
static ARMADO: AtomicBool = AtomicBool::new(false);
/// El punto de un solo uso ya actuo.
static TOMADO: AtomicBool = AtomicBool::new(false);
/// capture-store esta bloqueado dentro de una operacion admitida.
static STORE_OCUPADO: AtomicBool = AtomicBool::new(false);

/// Lee `inyectar=` de los argumentos de `capture`. Solo se admite con el venue `local`.
pub fn desde_args(args: &[String]) -> Result<Option<Caso>, String> {
    let pedido = args.iter().find_map(|a| a.strip_prefix("inyectar="));
    let Some(nombre) = pedido else {
        return Ok(None);
    };
    if args.first().map(String::as_str) != Some("local") {
        return Err("inyectar= solo se admite con el venue local".into());
    }
    CASOS
        .iter()
        .find(|(n, _)| *n == nombre)
        .map(|(_, c)| Some(*c))
        .ok_or_else(|| format!("caso de inyeccion desconocido: {nombre}"))
}

pub fn fijar(caso: Caso, ctl: Arc<Control>) {
    let _ = CONTROL.set(ctl);
    let _ = CASO.set(caso);
}

pub fn caso() -> Option<Caso> {
    CASO.get().copied()
}

pub fn es(c: Caso) -> bool {
    caso() == Some(c)
}

fn armado(c: Caso) -> bool {
    es(c) && ARMADO.load(Ordering::SeqCst)
}

fn tomar(c: Caso) -> bool {
    armado(c) && !TOMADO.swap(true, Ordering::SeqCst)
}

pub fn armar() {
    ARMADO.store(true, Ordering::SeqCst);
}

pub fn marcar(evento: &str) {
    let ns = CONTROL.get().map_or(0, |c| u64::try_from(c.ahora().as_nanos()).unwrap_or(u64::MAX));
    println!("P8_INYECCION {} {evento} control_ns={ns}", caso().map_or("ninguno", Caso::nombre));
}

pub fn fatal(detalle: &str) {
    marcar("fatal");
    if let Some(c) = CONTROL.get() {
        c.fatal(Clase::Inyectado, detalle);
    }
}

fn bloquear_para_siempre() -> ! {
    loop {
        thread::park();
    }
}

/// Espera (bloqueando este hilo) a que el corte de admision fatal este publicado.
fn esperar_corte(adm: &Admision) {
    while !adm.cortada() {
        thread::sleep(Duration::from_millis(1));
    }
}

fn admision() -> Option<&'static Arc<Admision>> {
    CONTROL.get().map(|c| &c.admision)
}

// --- Puntos en el codigo de produccion -------------------------------------------------------

/// ArchiveQueue::offer, tras reservar presupuesto y plaza y antes de confirmar.
pub fn tras_reserva() {
    if tomar(Caso::PausaReserva) {
        fatal("fatal con una reserva sin confirmar (inyectado)");
        if let Some(a) = admision() {
            esperar_corte(a);
        }
        marcar("reserva_reanudada");
    }
}

/// ArchiveQueue::offer, tras confirmar y antes de enviar.
pub fn tras_confirmar() {
    if tomar(Caso::PausaConfirmada) {
        fatal("fatal con una confirmada sin enviar (inyectado)");
        if let Some(a) = admision() {
            esperar_corte(a);
        }
        marcar("confirmada_reanudada");
    } else if tomar(Caso::RetenerConfirmada) {
        fatal("fatal con una confirmada retenida para siempre (inyectado)");
        bloquear_para_siempre();
    }
}

/// archive, antes de drenar: con cola_llena el escritor no saca filas hasta el corte fatal.
pub fn escritor_antes_de_drenar(adm: &Admision) {
    if es(Caso::ColaLlena) {
        marcar("escritor_detenido");
        esperar_corte(adm);
        marcar("escritor_reanudado");
    }
}

/// store::run, al empezar: con store_ocupado el hilo admite una operacion y no vuelve.
pub fn store_al_empezar(gate: &CompuertaStore) {
    if es(Caso::StoreOcupado) {
        let _job = gate.admitir(OpStore::Retencion);
        STORE_OCUPADO.store(true, Ordering::SeqCst);
        marcar("store_ocupado");
        bloquear_para_siempre();
    }
}

pub fn store_ocupado() -> bool {
    STORE_OCUPADO.load(Ordering::SeqCst)
}

/// Builder::on_thread_park de un worker real de Tokio (fuera de Future::poll).
pub fn al_estacionar_worker() {
    if tomar(Caso::PanicWorkerPark) {
        marcar("panic_worker_park");
        panic!("P8 inyectado: panic en on_thread_park de un worker de Tokio");
    }
}

/// run_market: cancelar (abort, sin panic) la conexion activa antes de su Life::Ended.
pub fn cancelar_conexion_market() -> bool {
    let si = tomar(Caso::CancelarConexionMarket);
    if si {
        marcar("cancelar_conexion_market");
    }
    si
}

/// Escritura en el archivo de un segmento (DiscoP8).
pub fn fallo_escritura() -> io::Result<()> {
    if armado(Caso::FalloAppend) {
        if !TOMADO.swap(true, Ordering::SeqCst) {
            marcar("fallo_append");
        }
        return Err(io::Error::other("P8 inyectado: fallo de escritura del segmento"));
    }
    Ok(())
}

/// sync_all de un segmento (DiscoP8): sellador bloqueado.
pub fn al_sincronizar() {
    if es(Caso::SelladorBloqueado) {
        marcar("sellador_bloqueado");
        bloquear_para_siempre();
    }
}

/// write_json (DiscoP8): manifiesto fallido, END fallido o bloqueado, fatal durante END.
pub fn al_escribir_json(path: &std::path::Path) -> io::Result<()> {
    let nombre = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if nombre.ends_with(".manifest.json") && es(Caso::FalloManifiesto) && !TOMADO.swap(true, Ordering::SeqCst) {
        marcar("fallo_manifiesto");
        return Err(io::Error::other("P8 inyectado: fallo del manifiesto tras el rename"));
    }
    if nombre == "END.json" {
        if es(Caso::FalloEnd) {
            marcar("fallo_end");
            return Err(io::Error::other("P8 inyectado: END.json no se puede escribir"));
        }
        if es(Caso::BloqueoEnd) {
            marcar("bloqueo_end");
            bloquear_para_siempre();
        }
        if es(Caso::FatalDuranteEnd) && !TOMADO.swap(true, Ordering::SeqCst) {
            fatal("fatal durante la escritura de END (inyectado)");
        }
    }
    Ok(())
}

/// Vigilante, al observar el primer fatal.
pub fn vigilante_vio_fatal() {
    if es(Caso::FalloVigilante) {
        marcar("fallo_vigilante");
        panic!("P8 inyectado: fallo del vigilante tras el primer fatal");
    }
}

/// Hilo capture-salida, antes de volcar stdout: otro hilo retiene el candado de stdout para
/// siempre, asi que el volcado de la salida ordenada se bloquea de verdad.
pub fn antes_de_volcar_salida() {
    if es(Caso::SalidaBloqueada) {
        marcar("salida_bloqueada");
        let tomado = Arc::new(AtomicBool::new(false));
        let t2 = Arc::clone(&tomado);
        let _ = thread::Builder::new().name("p8-retiene-stdout".into()).spawn(move || {
            let _candado = io::stdout().lock();
            t2.store(true, Ordering::SeqCst);
            bloquear_para_siempre();
        });
        while !tomado.load(Ordering::SeqCst) {
            thread::yield_now();
        }
    }
}

/// Guarda que pide un fatal al soltarse durante la destruccion del runtime.
pub struct FatalAlSoltar;

impl Drop for FatalAlSoltar {
    fn drop(&mut self) {
        if es(Caso::FatalDropRuntime) {
            fatal("fatal durante la destruccion del runtime (inyectado)");
        }
    }
}

/// Tarea "inyector p8" del Supervisor. Espera a que el archivo tenga trabajo conocido (todas las
/// conexiones iniciales abiertas y al menos una fila aceptada despues), actua una vez y despues
/// espera la orden de parada (no termina antes: eso seria un fin inesperado).
pub async fn inyector(
    caso: Caso,
    mut stop: tokio::sync::watch::Receiver<bool>,
    conexiones: impl Fn() -> u64,
    iniciales: u64,
    aceptadas: impl Fn() -> usize,
) -> crate::CaptureResult<()> {
    if caso == Caso::FatalDropRuntime {
        tokio::spawn(async {
            let _guarda = FatalAlSoltar;
            std::future::pending::<()>().await;
        });
    }
    let paso = Duration::from_millis(10);
    let listo = async {
        while conexiones() < iniciales {
            tokio::time::sleep(paso).await;
        }
        let base = aceptadas();
        while aceptadas() <= base {
            tokio::time::sleep(paso).await;
        }
    };
    tokio::select! {
        _ = stop.changed() => return Ok(()),
        _ = listo => {}
    }
    marcar("trabajo_conocido");
    match caso {
        Caso::PanicTarea => panic!("P8 inyectado: panic de tarea"),
        Caso::FinTarea => {
            marcar("fin_tarea");
            return Ok(());
        }
        Caso::CancelarConexionMarket
        | Caso::PanicWorkerPark
        | Caso::PausaReserva
        | Caso::PausaConfirmada
        | Caso::RetenerConfirmada
        | Caso::FalloAppend => armar(),
        Caso::SelladorBloqueado => {
            armar();
            fatal("fatal con el sellador por bloquearse (inyectado)");
        }
        Caso::BloqueoEnd | Caso::FalloVigilante | Caso::SalidaBloqueada => fatal("fatal del ensayo (inyectado)"),
        Caso::StoreOcupado => {
            while !store_ocupado() {
                tokio::time::sleep(paso).await;
            }
            fatal("fatal con capture-store ocupado (inyectado)");
        }
        Caso::ColaLlena | Caso::FalloManifiesto | Caso::FalloEnd | Caso::FatalDuranteEnd | Caso::FatalDropRuntime => {}
    }
    while !*stop.borrow() {
        if stop.changed().await.is_err() {
            break;
        }
    }
    Ok(())
}
