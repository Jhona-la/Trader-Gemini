//! SALIDAFALLO v2, Carril E (PURAS): adaptadores en memoria para disco, reloj, salida y borrado.
//! Sin archivos, sockets, red, hook global ni process::exit. Cada prueba recorre el codigo de
//! produccion (Control, vigilar, Admision, ArchiveQueue::offer, ArchiveReceiver::next_row,
//! archive, seal_segments, CompuertaStore, clasificar_store, Supervisor, conexion_terminada).

use super::*;
use crate::{ArchiveQueue, Disk, Emitter, Life, OfferPoint, QueueEnd, ROW_CHARGE_OVERHEAD, archive, seal_segments};
use flate2::read::GzDecoder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    sync::{Barrier, mpsc as std_mpsc},
};

// ---------------------------------------------------------------------------------------------
// Adaptadores.
// ---------------------------------------------------------------------------------------------
#[derive(Default)]
pub(crate) struct SalidaPrueba {
    codigos: Mutex<Vec<(u32, bool)>>,
    /// Simula una salida ordenada que se bloquea (p. ej. el volcado de stdout): el proceso sigue
    /// vivo tras pedirla, hasta una terminacion no ordenada.
    ordenada_bloqueada: AtomicBool,
}

impl SalidaPrueba {
    pub(crate) fn codigos(&self) -> Vec<(u32, bool)> {
        self.codigos.lock().map(|v| v.clone()).unwrap_or_default()
    }
}

struct SalidaCompartida(Arc<SalidaPrueba>);

impl Salida for SalidaCompartida {
    fn terminar(&self, codigo: u32, ordenada: bool) {
        if let Ok(mut v) = self.0.codigos.lock() {
            v.push((codigo, ordenada));
        }
    }
    fn terminado(&self) -> bool {
        let bloqueada = self.0.ordenada_bloqueada.load(Ordering::SeqCst);
        self.0.codigos().iter().any(|&(_, ordenada)| !ordenada || !bloqueada)
    }
}

#[derive(Default)]
pub(crate) struct RelojPrueba {
    ns: AtomicU64,
}

impl RelojPrueba {
    fn avanzar(&self, d: Duration) {
        self.ns.fetch_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX), Ordering::SeqCst);
    }
}

struct RelojCompartido(Arc<RelojPrueba>);

impl Reloj for RelojCompartido {
    fn ahora(&self) -> Duration {
        Duration::from_nanos(self.0.ns.load(Ordering::SeqCst))
    }
}

type Borrados = Arc<Mutex<Vec<PathBuf>>>;

fn compuerta_memoria() -> (Arc<CompuertaStore>, Borrados) {
    let borrados: Borrados = Arc::default();
    let registro = Arc::clone(&borrados);
    let compuerta = CompuertaStore::nueva(
        Arc::new(AtomicBool::new(false)),
        Box::new(move |p| {
            registro.lock().map_err(|_| io::Error::other("envenenado"))?.push(p.to_path_buf());
            Ok(())
        }),
    );
    (Arc::new(compuerta), borrados)
}

pub(crate) fn control_de_prueba(plazo: Duration) -> (Arc<Control>, Arc<SalidaPrueba>, Arc<RelojPrueba>) {
    let (compuerta, _) = compuerta_memoria();
    control_con_compuerta(plazo, compuerta)
}

fn control_con_compuerta(plazo: Duration, compuerta: Arc<CompuertaStore>) -> (Arc<Control>, Arc<SalidaPrueba>, Arc<RelojPrueba>) {
    let salida = Arc::new(SalidaPrueba::default());
    let reloj = Arc::new(RelojPrueba::default());
    let ctl = Control::nuevo(plazo, Box::new(RelojCompartido(Arc::clone(&reloj))), Box::new(SalidaCompartida(Arc::clone(&salida))), compuerta);
    (Arc::new(ctl), salida, reloj)
}

/// Espera acotada de prueba (no es una cifra operativa del colector).
fn esperar(que: &str, cond: impl Fn() -> bool) {
    let limite = Instant::now() + Duration::from_secs(20);
    while !cond() {
        assert!(Instant::now() < limite, "no se cumplio a tiempo: {que}");
        thread::sleep(Duration::from_millis(1));
    }
}

fn lanzar_vigilante(ctl: &Arc<Control>) -> thread::JoinHandle<()> {
    let c = Arc::clone(ctl);
    let h = thread::spawn(move || vigilar(&c));
    ctl.registrar_vigilante(h.thread().clone());
    esperar("vigilante vivo", || ctl.vigilante_vivo());
    h
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime de prueba")
}

fn fila(seq: u64, ts: u64, payload: &str) -> Value {
    json!({"kind":"frame","route":"public","stream":"xrpusdt@trade","symbol":"XRPUSDT",
        "connection":3,"seq":seq,"recv_utc_ms":ts,"recv_mono_ns":seq,"payload":payload})
}

/// Disco en memoria con fallos inyectables por (prefijo de segmento, fase).
#[derive(Default)]
struct Memoria {
    archivos: Mutex<HashMap<PathBuf, Vec<u8>>>,
    fallos: Mutex<HashSet<(String, &'static str)>>,
}

impl Memoria {
    fn falla(&self, ruta: &Path, fase: &str) -> bool {
        let nombre = ruta.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.fallos.lock().map(|f| f.iter().any(|(s, ph)| *ph == fase && nombre.starts_with(s.as_str()))).unwrap_or(false)
    }
    fn fallar(&self, segmento: &str, fase: &'static str) {
        self.fallos.lock().expect("fallos").insert((segmento.to_owned(), fase));
    }
    fn leer(&self, ruta: &Path) -> Option<Vec<u8>> {
        self.archivos.lock().expect("archivos").get(ruta).cloned()
    }
    fn existe(&self, ruta: &Path) -> bool {
        self.archivos.lock().expect("archivos").contains_key(ruta)
    }
}

fn inyectado(fase: &str) -> io::Error {
    io::Error::other(format!("fallo inyectado: {fase}"))
}

#[derive(Clone)]
struct DiscoMemoria(Arc<Memoria>);

struct ArchivoMemoria {
    ruta: PathBuf,
    mem: Arc<Memoria>,
}

impl Write for ArchivoMemoria {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.mem.falla(&self.ruta, "write") {
            return Err(inyectado("write"));
        }
        self.mem.archivos.lock().map_err(|_| inyectado("lock"))?.entry(self.ruta.clone()).or_default().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.mem.falla(&self.ruta, "flush") {
            return Err(inyectado("flush"));
        }
        Ok(())
    }
}

impl Disk for DiscoMemoria {
    type File = ArchivoMemoria;
    fn create(&self, path: &Path) -> io::Result<ArchivoMemoria> {
        if self.0.falla(path, "create") {
            return Err(inyectado("create"));
        }
        let mut a = self.0.archivos.lock().map_err(|_| inyectado("lock"))?;
        if a.contains_key(path) {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "existe"));
        }
        a.insert(path.to_path_buf(), Vec::new());
        Ok(ArchivoMemoria { ruta: path.to_path_buf(), mem: Arc::clone(&self.0) })
    }
    fn sync(&self, f: &ArchivoMemoria) -> io::Result<()> {
        if self.0.falla(&f.ruta, "sync") { Err(inyectado("sync")) } else { Ok(()) }
    }
    fn digest(&self, path: &Path) -> CaptureResult<(u64, String)> {
        if self.0.falla(path, "sha256") {
            return Err(inyectado("sha256").into());
        }
        let b = self.0.leer(path).ok_or_else(|| inyectado("no existe"))?;
        Ok((b.len() as u64, format!("{:x}", Sha256::digest(&b))))
    }
    fn publish(&self, from: &Path, to: &Path) -> io::Result<()> {
        if self.0.falla(from, "publish") {
            return Err(inyectado("publish"));
        }
        let mut a = self.0.archivos.lock().map_err(|_| inyectado("lock"))?;
        let b = a.remove(from).ok_or_else(|| inyectado("no existe"))?;
        a.insert(to.to_path_buf(), b);
        Ok(())
    }
    fn write_json(&self, path: &Path, value: &Value) -> CaptureResult<()> {
        if self.0.falla(path, "json") {
            return Err(inyectado("json").into());
        }
        let mut b = serde_json::to_vec_pretty(value)?;
        b.push(b'\n');
        self.0.archivos.lock().map_err(|_| inyectado("lock"))?.insert(path.to_path_buf(), b);
        Ok(())
    }
}

fn dir() -> PathBuf {
    PathBuf::from("memoria-run")
}

/// Escritor + sellador de produccion sobre el disco en memoria. Devuelve (outcome, report).
fn archivar(disco: &DiscoMemoria, ctl: &Arc<Control>, rx: crate::ArchiveReceiver) -> (crate::ArchiveOutcome, crate::SealReport) {
    let (seal_tx, seal_rx) = std_mpsc::sync_channel(crate::SEAL_QUEUE_CAPACITY);
    let d2 = disco.clone();
    let c2 = Arc::clone(ctl);
    let sellador = thread::spawn(move || seal_segments(&d2, &dir(), seal_rx, None, Some(&c2)));
    let outcome = archive(disco, &dir(), rx, seal_tx, ctl);
    (outcome, sellador.join().expect("sellador"))
}

fn gz_seqs(bytes: &[u8]) -> Vec<u64> {
    BufReader::new(GzDecoder::new(bytes))
        .lines()
        .map(|l| serde_json::from_str::<Value>(&l.expect("linea")).expect("json")["ingest_seq"].as_u64().expect("seq"))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// T1
// ---------------------------------------------------------------------------------------------
#[test]
fn panic_superior_es_fatal() {
    let (ctl, salida, _) = control_de_prueba(PLAZO_FATAL);
    let vig = lanzar_vigilante(&ctl);
    let rt = rt();
    let _entrada = rt.enter();
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let mut sup = Supervisor::nuevo();
    sup.lanzar("rest gate".into(), async { panic!("driver averiado (inyectado)") });
    for nombre in ["shard 0 public", "shard 4 market"] {
        let mut stop = stop_rx.clone();
        let falla_en_cierre = nombre.ends_with("market");
        sup.lanzar(nombre.into(), async move {
            while !*stop.borrow() {
                let _ = stop.changed().await;
            }
            if falla_en_cierre {
                panic!("panic durante el cierre (inyectado)");
            }
            Ok(())
        });
    }
    // Captura: la tarea superior cae y el supervisor la identifica sin ninguna otra senal de stop.
    let (nombre, r) = rt.block_on(sup.siguiente());
    assert_eq!(nombre, "rest gate");
    assert!(!*stop_rx.borrow(), "el fatal no necesita la orden de parada");
    let error = Supervisor::evaluar(&ctl, &nombre, r, false).expect("error registrado");
    assert!(error.contains("rest gate") && error.contains("panic"), "{error}");
    let (clase, detalle) = ctl.primer_fatal().expect("primer fatal");
    assert_eq!(clase, Clase::TareaPanic);
    assert!(detalle.contains("rest gate"), "{detalle}");
    // El vigilante activa el cierre nativo: corte de admision y compuerta de store.
    esperar("corte de admision", || ctl.admision.cortada());
    assert_eq!(ctl.admision.corte().map(|c| c.motivo), Some(Motivo::Fatal));
    esperar("compuerta de store", || ctl.store.cerrada());
    // Cierre: se recogen TODOS los resultados; el panic del cierre se agrega como secundario.
    let _ = stop_tx.send(true);
    let errores = rt.block_on(sup.cerrar(&ctl));
    assert_eq!(errores.len(), 1, "{errores:?}");
    assert!(errores[0].contains("shard 4 market") && errores[0].contains("panic"));
    assert_eq!(ctl.primer_fatal().map(|f| f.0), Some(Clase::TareaPanic), "la causa inicial no se sustituye");
    let (secundarios, _) = ctl.secundarios();
    assert!(secundarios.iter().any(|s| s.contains("shard 4 market")), "{secundarios:?}");
    assert!(sup.vacio());
    ctl.fin_listo(1, false);
    vig.join().expect("vigilante");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_CERRADO, true)]);
}

// ---------------------------------------------------------------------------------------------
// T2
// ---------------------------------------------------------------------------------------------
#[test]
fn panic_market_es_fatal() {
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let rt = rt();
    let _entrada = rt.enter();
    let (life_tx, mut life) = tokio::sync::mpsc::unbounded_channel::<Life>();
    let mut tasks = tokio::task::JoinSet::new();
    // Una conexion que entra en panic antes de enviar Life::Ended (como connection() si falla dentro).
    let tx = life_tx.clone();
    tasks.spawn(async move {
        let _ = tx.send(Life::Open(7));
        panic!("panic en la conexion market (inyectado)");
    });
    drop(life_tx);
    let resultado = rt.block_on(async {
        let mut abiertas = 0;
        loop {
            tokio::select! {
                Some(joined) = tasks.join_next(), if !tasks.is_empty() => {
                    break crate::salidafallo::conexion_terminada(joined);
                }
                Some(ev) = life.recv() => {
                    assert!(matches!(ev, Life::Open(7)), "{ev:?}");
                    abiertas += 1;
                }
            }
            assert!(abiertas <= 1);
        }
    });
    let error = resultado.expect_err("JoinError de conexion es fatal");
    assert!(error.to_string().contains("conexion market: panic"), "{error}");
    let mut restantes = Vec::new();
    while let Ok(ev) = life.try_recv() {
        restantes.push(ev);
    }
    assert!(!restantes.iter().any(|ev| matches!(ev, Life::Ended(..))), "nunca llego Life::Ended: {restantes:?}");
    assert!(matches!(life.try_recv(), Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)));
    // El shard devuelve ese Err y el supervisor lo vuelve fatal con su identidad.
    let m = Supervisor::evaluar(&ctl, "shard 4 market", Ok(Err(error)), false).expect("registrado");
    assert!(m.contains("shard 4 market"));
    let (clase, detalle) = ctl.primer_fatal().expect("fatal");
    assert_eq!(clase, Clase::Lector);
    assert!(detalle.contains("conexion market: panic"), "{detalle}");
    // Cierre del JoinSet: se recogen los joins (aqui uno con panic) y se devuelve el primero.
    let mut cierre = tokio::task::JoinSet::new();
    cierre.spawn(async {});
    cierre.spawn(async { panic!("panic al cerrar (inyectado)") });
    let primero = rt.block_on(async {
        let mut first = None;
        while let Some(joined) = cierre.join_next().await {
            if let Err(e) = crate::salidafallo::conexion_terminada(joined) {
                first.get_or_insert(e);
            }
        }
        first
    });
    assert!(primero.is_some_and(|e| e.to_string().contains("panic")));
}

// ---------------------------------------------------------------------------------------------
// T3
// ---------------------------------------------------------------------------------------------
#[test]
fn fin_inesperado_es_fatal() {
    let rt = rt();
    let _entrada = rt.enter();
    // Tarea persistente que vuelve Ok antes de la parada: fatal con su nombre.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let mut sup = Supervisor::nuevo();
    sup.lanzar("shard 2 public".into(), async { Ok(()) });
    let (nombre, r) = rt.block_on(sup.siguiente());
    let e = Supervisor::evaluar(&ctl, &nombre, r, false).expect("fin inesperado");
    assert!(e.contains("shard 2 public") && e.contains("sin orden de parada"), "{e}");
    assert_eq!(ctl.primer_fatal().map(|f| f.0), Some(Clase::TareaFin));
    // La misma salida tras la orden de parada es el fin esperado.
    let (sano, _, _) = control_de_prueba(PLAZO_FATAL);
    let mut sup = Supervisor::nuevo();
    sup.lanzar("shard 2 public".into(), async { Ok(()) });
    assert!(rt.block_on(sup.cerrar(&sano)).is_empty());
    assert!(!sano.es_fatal());
    // Conexion efimera que termina bien: join Ok y Life::Ended en los dos ordenes, sin fatal.
    for join_primero in [true, false] {
        let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
        let (life_tx, mut life) = tokio::sync::mpsc::unbounded_channel::<Life>();
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(async move {
            let _ = life_tx.send(Life::Ended(9, "WebSocket ended".into(), false));
        });
        // Cada orden se fuerza de forma explicita (el select! elige al azar entre ramas listas).
        let (por_join, por_life) = rt.block_on(async {
            let mut orden = Vec::new();
            let mut fin = None;
            for paso in if join_primero { ["join", "life"] } else { ["life", "join"] } {
                if paso == "join" {
                    let r = tasks.join_next().await.expect("join");
                    crate::salidafallo::conexion_terminada(r).expect("join exitoso no es fatal");
                } else {
                    fin = life.recv().await;
                }
                orden.push(paso);
            }
            (orden, fin)
        });
        assert_eq!(por_join.len(), 2);
        assert_eq!(por_join[0] == "join", join_primero, "{por_join:?}");
        assert!(matches!(por_life, Some(Life::Ended(9, ref why, false)) if why == "WebSocket ended"));
        assert!(!ctl.es_fatal(), "un cierre recuperable no es fatal");
    }
}

// ---------------------------------------------------------------------------------------------
// T4
// ---------------------------------------------------------------------------------------------
#[test]
fn fallo_store_no_es_disco_bajo() {
    use crate::store::StoreStop;
    use futures_util::FutureExt;
    let cancelado = || {
        let (tx, rx) = tokio::sync::oneshot::channel::<StoreStop>();
        drop(tx);
        rx.now_or_never().expect("listo")
    };
    // Medicion real de disco bajo y error de almacen: paradas ordinarias, no fatales.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let ev = clasificar_store(Ok(StoreStop::LowDisk { free: 7, minimum: 9 }), EstadoStore::Volvio, false);
    assert_eq!(ev, EventoStore::DiscoBajo { free: 7, minimum: 9 });
    assert_eq!(ev.aplicar(&ctl), (3, "DISCO-BAJO: 7 bytes libres < 9".into(), "low_disk"));
    let ev = clasificar_store(Ok(StoreStop::StorageError("x".into())), EstadoStore::Volvio, false);
    assert_eq!(ev.aplicar(&ctl).2, "store_error");
    assert!(!ctl.es_fatal());
    // Cancelacion del oneshot: nunca se vuelve una medicion; depende de como termino el hilo.
    for (estado, cierre, esperado, clase) in [
        (EstadoStore::Panic, false, EventoStore::Panic, Some(Clase::StorePanic)),
        (EstadoStore::Volvio, false, EventoStore::FinInesperado, Some(Clase::StoreFin)),
        (EstadoStore::Volvio, true, EventoStore::FinEsperado, None),
        (EstadoStore::Corriendo, false, EventoStore::CanalCancelado, Some(Clase::StoreCanal)),
    ] {
        let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
        let ev = clasificar_store(cancelado(), estado, cierre);
        assert_eq!(ev, esperado);
        let (code, _, reason) = ev.aplicar(&ctl);
        assert_ne!(reason, "low_disk");
        assert_ne!(code, 3);
        assert_eq!(ctl.primer_fatal().map(|f| f.0), clase, "{estado:?}");
    }
    // Compuerta: fatal entre escaneo y trabajo -> ni trabajo nuevo ni borrados pendientes.
    let destructivas = [
        OpStore::RecodeBorraParcialViejo, OpStore::RecodeBorraParcial, OpStore::RecodeBorraReciboParcial,
        OpStore::RecodeBorraGzip, OpStore::RetencionBorra, OpStore::RecuperacionBorraGzip,
    ];
    let (compuerta, borrados) = compuerta_memoria();
    let (ctl, _, _) = control_con_compuerta(PLAZO_FATAL, Arc::clone(&compuerta));
    let (wake_tx, wake_rx) = std_mpsc::channel();
    compuerta.fijar_despertador(Box::new(move || {
        let _ = wake_tx.send(());
    }));
    // Modo sano: politica ordinaria intacta (el borrado ocurre).
    compuerta.borrar(OpStore::RetencionBorra, Path::new("vieja.tgll")).expect("sano");
    assert_eq!(borrados.lock().expect("b").len(), 1);
    // Un trabajo ya iniciado (recode) bloqueado no bloquea el corte nativo.
    let barrera = Arc::new(Barrier::new(2));
    let (c2, b2) = (Arc::clone(&compuerta), Arc::clone(&barrera));
    let ocupado = thread::spawn(move || {
        let job = c2.admitir(OpStore::Recode).expect("admitido antes del corte");
        b2.wait();
        b2.wait(); // Recode "en curso" hasta que la prueba lo suelte.
        // Fatal durante recode y antes de su punto destructivo: el borrado se niega.
        let negado = c2.borrar(OpStore::RecodeBorraGzip, Path::new("h.jsonl.gz"));
        drop(job);
        negado
    });
    barrera.wait();
    ctl.fatal(Clase::Panic, "fatal inyectado");
    ctl.cortes_nativos(); // lo que hace el vigilante; vuelve sin esperar al trabajo
    assert!(compuerta.cerrada() && compuerta.cancel.load(Ordering::Acquire));
    assert_eq!(compuerta.corte(), Some(CorteStore { en_curso: 1, ultima: OpStore::Recode }));
    wake_rx.recv_timeout(Duration::from_secs(20)).expect("store despertado de forma nativa");
    barrera.wait();
    assert!(ocupado.join().expect("hilo").is_err());
    for op in destructivas {
        assert!(compuerta.borrar(op, Path::new("x")).is_err(), "{op:?}");
    }
    for op in [OpStore::Recode, OpStore::Retencion, OpStore::Recuperacion] {
        assert!(compuerta.admitir(op).is_none(), "{op:?}");
    }
    assert_eq!(borrados.lock().expect("b").len(), 1, "ningun borrado tras el corte");
    // Carrera de admision: cada borrado que ocurrio con la compuerta ya cerrada estaba en curso
    // en la instantanea del corte.
    for _ in 0..50 {
        let tardios = Arc::new(AtomicU64::new(0));
        let ref_c: Arc<OnceLock<Arc<CompuertaStore>>> = Arc::new(OnceLock::new());
        let (t2, r2) = (Arc::clone(&tardios), Arc::clone(&ref_c));
        let compuerta = Arc::new(CompuertaStore::nueva(Arc::new(AtomicBool::new(false)), Box::new(move |_| {
            if r2.get().is_some_and(|c| c.cerrada()) {
                t2.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })));
        let _ = ref_c.set(Arc::clone(&compuerta));
        let hilos: Vec<_> = (0..4).map(|_| {
            let c = Arc::clone(&compuerta);
            thread::spawn(move || while c.borrar(OpStore::RetencionBorra, Path::new("r")).is_ok() {})
        }).collect();
        let corte = compuerta.cerrar().expect("corte");
        for h in hilos {
            h.join().expect("hilo");
        }
        assert!(tardios.load(Ordering::SeqCst) <= corte.en_curso, "borrado no admitido antes del corte");
        assert_eq!(compuerta.en_curso(), 0);
    }
}

// ---------------------------------------------------------------------------------------------
// T5
// ---------------------------------------------------------------------------------------------
#[test]
fn fatal_nativo_sin_runtime() {
    let (compuerta, _) = compuerta_memoria();
    let (wake_tx, wake_rx) = std_mpsc::channel();
    compuerta.fijar_despertador(Box::new(move || {
        let _ = wake_tx.send(());
    }));
    let (ctl, salida, _) = control_con_compuerta(PLAZO_FATAL, compuerta);
    let vig = lanzar_vigilante(&ctl);
    let (queue, mut rx) = ArchiveQueue::with_admission(8, 32 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
    let adm = Arc::clone(&ctl.admision);
    let escritor = thread::spawn(move || {
        adm.registrar_escritor(thread::current());
        let mut n = 0;
        loop {
            match rx.next_row(&adm) {
                Ok(_) => n += 1,
                Err(end) => return (n, end),
            }
        }
    });
    // Ningun runtime de Tokio existe en esta prueba: ni Futures ni scheduler avanzan.
    for s in 1..=3 {
        queue.offer(fila(s, 100, "x"), None).expect("admitida");
    }
    let c = Arc::clone(&ctl);
    thread::spawn(move || c.fatal(Clase::Escritor, "fatal desde un hilo nativo")).join().expect("hilo");
    // El vigilante corta, despierta al escritor (con un emisor clonado todavia vivo) y a store.
    let (n, end) = escritor.join().expect("escritor");
    assert_eq!((n, end), (3, QueueEnd::CutResolved));
    assert_eq!(ctl.admision.corte(), Some(Corte { motivo: Motivo::Fatal, confirmadas: 3, en_vuelo: 0 }));
    wake_rx.recv_timeout(Duration::from_secs(20)).expect("store despertado");
    assert!(queue.offer(fila(4, 100, "tarde"), None).is_err(), "sin admision tras el corte");
    assert_eq!(ctl.admision.rechazos_tras_corte(), 1);
    drop(queue);
    ctl.fin_listo(1, false);
    vig.join().expect("vigilante");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_CERRADO, true)]);
}

// ---------------------------------------------------------------------------------------------
// T6
// ---------------------------------------------------------------------------------------------
fn escritor_de(ctl: &Arc<Control>, mut rx: crate::ArchiveReceiver) -> (thread::JoinHandle<(u64, QueueEnd)>, Arc<AtomicU64>) {
    let adm = Arc::clone(&ctl.admision);
    let vistos = Arc::new(AtomicU64::new(0));
    let v2 = Arc::clone(&vistos);
    let h = thread::spawn(move || {
        adm.registrar_escritor(thread::current());
        loop {
            match rx.next_row(&adm) {
                Ok(_) => {
                    v2.fetch_add(1, Ordering::SeqCst);
                }
                Err(end) => return (v2.load(Ordering::SeqCst), end),
            }
        }
    });
    (h, vistos)
}

fn cola_pausada(ctl: &Arc<Control>, punto: OfferPoint, barrera: Arc<Barrier>, cap: usize) -> (ArchiveQueue, crate::ArchiveReceiver) {
    let (mut q, rx) = ArchiveQueue::with_admission(cap, 32 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
    q.pause = Some(Arc::new(move |p| {
        if p == punto && thread::current().name() == Some("productor-pausado") {
            barrera.wait();
            barrera.wait();
        }
    }));
    (q, rx)
}

#[test]
fn cola_vacia_y_llena_cierra() {
    // a) Cola vacia con emisor clonado retenido: el corte despierta y termina sin EOF.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let (queue, rx) = ArchiveQueue::with_admission(4, 32 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
    let clon = queue.clone();
    let (escritor, _) = escritor_de(&ctl, rx);
    esperar("escritor dormido", || ctl.admision.escritor.get().is_some());
    assert_eq!(ctl.admision.cortar(Motivo::Fatal), Some(Corte { motivo: Motivo::Fatal, confirmadas: 0, en_vuelo: 0 }));
    assert_eq!(escritor.join().expect("escritor"), (0, QueueEnd::CutResolved));
    drop((queue, clon));

    // b) Cola llena: el rechazo por capacidad conserva su causa; se drenan las aceptadas.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let (queue, rx) = ArchiveQueue::with_admission(2, 32 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
    queue.offer(fila(1, 100, "a"), None).expect("1");
    queue.offer(fila(2, 100, "b"), None).expect("2");
    assert!(queue.offer(fila(3, 100, "c"), None).unwrap_err().to_string().contains("row_capacity"));
    let corte = ctl.admision.cortar(Motivo::Fatal).expect("corte");
    assert_eq!((corte.confirmadas, corte.en_vuelo), (2, 0));
    let (escritor, _) = escritor_de(&ctl, rx);
    assert_eq!(escritor.join().expect("escritor"), (2, QueueEnd::CutResolved));
    assert_eq!(queue.audit.accepted.load(Ordering::SeqCst), 2);
    drop(queue);

    // c) Productor pausado tras reservar presupuesto y plaza, ANTES de confirmar: el corte no lo
    // espera (el productor retiene el mutex de admision), y la reserva se libera sin confirmar.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let barrera = Arc::new(Barrier::new(2));
    let (queue, rx) = cola_pausada(&ctl, OfferPoint::AfterReserve, Arc::clone(&barrera), 4);
    queue.offer(fila(1, 100, "previa"), None).expect("previa");
    let q2 = queue.clone();
    let productor = thread::Builder::new().name("productor-pausado".into())
        .spawn(move || q2.offer(fila(2, 100, "reservada"), None)).expect("productor");
    barrera.wait();
    let corte = ctl.admision.cortar(Motivo::Fatal).expect("corte");
    assert_eq!((corte.confirmadas, corte.en_vuelo), (1, 1));
    assert!(!ctl.admision.resuelta(), "una reserva pendiente no es una fila archivada");
    let (escritor, _) = escritor_de(&ctl, rx);
    barrera.wait();
    let r = productor.join().expect("productor");
    assert!(r.unwrap_err().to_string().contains("cut during reservation"));
    assert_eq!(escritor.join().expect("escritor"), (1, QueueEnd::CutResolved));
    assert_eq!(ctl.admision.reservas_canceladas(), 1);
    assert_eq!(ctl.admision.confirmadas(), 1, "ninguna confirmacion tras el corte");
    assert_eq!(queue.audit.accepted.load(Ordering::SeqCst), 1);
    assert_eq!(queue.budget.available_permits(), queue.byte_limit, "presupuesto liberado");
    assert_eq!(queue.tx.capacity(), queue.tx.max_capacity(), "plaza liberada");
    drop(queue);

    // d) Pausado tras confirmar y antes de enviar: es una fila con obligacion de drenaje.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let barrera = Arc::new(Barrier::new(2));
    let (queue, rx) = cola_pausada(&ctl, OfferPoint::AfterConfirm, Arc::clone(&barrera), 4);
    let q2 = queue.clone();
    let productor = thread::Builder::new().name("productor-pausado".into())
        .spawn(move || q2.offer(fila(1, 100, "confirmada"), None)).expect("productor");
    barrera.wait();
    let corte = ctl.admision.cortar(Motivo::Fatal).expect("corte");
    assert_eq!((corte.confirmadas, corte.en_vuelo), (1, 1));
    let (escritor, vistos) = escritor_de(&ctl, rx);
    assert!(!ctl.admision.resuelta());
    assert!(!escritor.is_finished(), "no hay falsa completitud con una confirmada sin enviar");
    assert_eq!(vistos.load(Ordering::SeqCst), 0);
    barrera.wait();
    productor.join().expect("productor").expect("la confirmada se envia");
    assert_eq!(escritor.join().expect("escritor"), (1, QueueEnd::CutResolved));
    drop(queue);

    // e) Permiso retenido para siempre: el reloj inyectado vence y el vigilante sale con fallo
    // (cierre incompleto) sin esperar EOF ni al escritor.
    let plazo = Duration::from_secs(1);
    let (ctl, salida, reloj) = control_de_prueba(plazo);
    let vig = lanzar_vigilante(&ctl);
    let barrera = Arc::new(Barrier::new(2));
    let (queue, rx) = cola_pausada(&ctl, OfferPoint::AfterConfirm, Arc::clone(&barrera), 4);
    let q2 = queue.clone();
    let productor = thread::Builder::new().name("productor-pausado".into())
        .spawn(move || q2.offer(fila(1, 100, "retenida"), None)).expect("productor");
    barrera.wait();
    let (escritor, _) = escritor_de(&ctl, rx);
    ctl.fatal(Clase::Escritor, "fatal con permiso retenido");
    esperar("corte fatal", || ctl.admision.cortada());
    reloj.avanzar(plazo);
    ctl.despertar_vigilante();
    vig.join().expect("vigilante");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_INCOMPLETO, false)]);
    assert!(!escritor.is_finished() && !ctl.admision.resuelta());
    barrera.wait();
    productor.join().expect("productor").expect("envio");
    assert_eq!(escritor.join().expect("escritor"), (1, QueueEnd::CutResolved));
}

// ---------------------------------------------------------------------------------------------
// T7
// ---------------------------------------------------------------------------------------------
#[test]
fn plazo_nativo_impide_bloqueo() {
    // Sin notificaciones de cierre, store ocupado y escritor/sellador/END que nunca avisan.
    let plazo = Duration::from_secs(5);
    let (compuerta, _) = compuerta_memoria();
    let (ctl, salida, reloj) = control_con_compuerta(plazo, Arc::clone(&compuerta));
    let vig = lanzar_vigilante(&ctl);
    let ocupado = compuerta.admitir(OpStore::Recode).expect("store ocupado");
    ctl.fatal(Clase::Sellador, "sellador bloqueado (inyectado)");
    esperar("cortes", || ctl.store.cerrada());
    reloj.avanzar(plazo - Duration::from_nanos(1));
    ctl.despertar_vigilante();
    thread::sleep(Duration::from_millis(20));
    assert!(salida.codigos().is_empty(), "antes del plazo no termina");
    assert!(!vig.is_finished(), "sigue armado");
    reloj.avanzar(Duration::from_nanos(1));
    ctl.despertar_vigilante();
    vig.join().expect("vigilante");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_INCOMPLETO, false)]);
    assert_eq!(compuerta.en_curso(), 1, "no espero al trabajo de store");
    drop(ocupado);

    // Fallo del vigilante: terminacion nativa sin runtime.
    let (ctl, salida, _) = control_de_prueba(plazo);
    let c = Arc::clone(&ctl);
    let vig = thread::spawn(move || {
        pruebas_vigilante::FALLAR.with(|f| f.set(true));
        vigilar(&c);
    });
    assert!(vig.join().is_err());
    assert_eq!(salida.codigos(), [(SALIDA_VIGILANTE, false)]);
    assert!(!ctl.vigilante_vivo());
    // Con el vigilante caido, el camino del hook termina por su cuenta.
    ctl.fijar(Clase::Panic, &[b"panic tardio"], true);
    assert_eq!(salida.codigos(), [(SALIDA_VIGILANTE, false), (SALIDA_VIGILANTE, false)]);

    // Armado hasta la destruccion del runtime: el finalizador sano ya aviso (codigo 0) y un fatal
    // posterior (p. ej. durante la destruccion) sigue siendo una salida de fallo.
    let (ctl, salida, _) = control_de_prueba(plazo);
    let vig = lanzar_vigilante(&ctl);
    ctl.fin_listo(0, false);
    thread::sleep(Duration::from_millis(20));
    assert!(!vig.is_finished(), "el fin sano no desarma");
    ctl.fatal(Clase::Panic, "panic durante la destruccion del runtime");
    vig.join().expect("vigilante");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_CERRADO, true)]);
    assert!(ctl.desarmar().is_err());
    // Desarme ordenado sin fatal: el vigilante vuelve sin terminar el proceso.
    let (ctl, salida, _) = control_de_prueba(plazo);
    let vig = lanzar_vigilante(&ctl);
    ctl.fin_listo(0, false);
    assert!(ctl.desarmar().is_ok());
    vig.join().expect("vigilante");
    assert!(salida.codigos().is_empty());
}

// ---------------------------------------------------------------------------------------------
// T8
// ---------------------------------------------------------------------------------------------
fn cola_con_horas(ctl: &Arc<Control>) -> crate::ArchiveReceiver {
    let (queue, rx) = ArchiveQueue::with_admission(16, 64 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
    let h = crate::HOUR_MS;
    for (s, ts) in [(1, h - 1), (2, h + 1), (3, h - 2), (4, h + crate::ROTATION_GRACE_MS)] {
        queue.offer(fila(s, ts, "{}"), Some(crate::Cont::Agg { symbol: "XRPUSDT".into(), a: s })).expect("fila");
    }
    rx
}

#[test]
fn error_sellado_no_da_exito() {
    let h = crate::HOUR_MS;
    let (seg1, seg2) = ("hour-0-0001", format!("hour-{h}-0002"));
    let p = |n: &str, ext: &str| dir().join(format!("{n}{ext}"));
    // Recorrido sano completo.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let disco = DiscoMemoria(Arc::default());
    let (out, rep) = archivar(&disco, &ctl, cola_con_horas(&ctl));
    assert_eq!((out.rows_written, out.rows_unwritten, rep.segments), (4, 0, 2));
    assert!(rep.error.is_none() && rep.failures.is_empty() && !ctl.es_fatal());
    for (seg, esperado) in [(seg1, [1, 3]), (seg2.as_str(), [2, 4])] {
        let gz = disco.0.leer(&p(seg, ".jsonl.gz")).expect("gzip final");
        assert_eq!(gz_seqs(&gz), esperado, "hora UTC y gracia de rotacion");
        let m: Value = serde_json::from_slice(&disco.0.leer(&p(seg, ".manifest.json")).expect("manifiesto")).expect("json");
        assert_eq!(m["bytes"], gz.len() as u64);
        assert_eq!(m["sha256"], format!("{:x}", Sha256::digest(&gz)));
        assert_eq!(m["rows"], 2);
        assert_eq!(m["utc_hour_ms"], if seg == seg1 { 0 } else { h });
        assert!(!disco.0.existe(&p(seg, ".jsonl.gz.partial")));
    }
    // Manifiesto fallido DESPUES del rename: gzip final sin manifiesto valido, no admitido; el
    // otro segmento sano se sella igual.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let disco = DiscoMemoria(Arc::default());
    disco.0.fallar(seg1, "json");
    let (out, rep) = archivar(&disco, &ctl, cola_con_horas(&ctl));
    assert_eq!(out.rows_written, 4);
    assert_eq!(rep.segments, 1);
    assert_eq!(rep.failures.len(), 1);
    let f = &rep.failures[0];
    assert_eq!((f["segment"].as_str(), f["phase"].as_str(), f["state"].as_str(), f["admitted"].as_bool()),
        (Some(seg1), Some("publish_manifest"), Some("jsonl.gz_final_sin_manifiesto_valido"), Some(false)));
    assert!(disco.0.existe(&p(seg1, ".jsonl.gz")) && !disco.0.existe(&p(seg1, ".manifest.json")));
    assert!(disco.0.existe(&p(&seg2, ".manifest.json")));
    assert!(rep.error.is_some());
    assert_eq!(ctl.primer_fatal().map(|f| f.0), Some(Clase::Sellador));
    assert_ne!(rep.counts.rows, out.rows_written, "lo sellado no iguala lo escrito: no hay exito");
    // Fallos ANTES del rename: el artefacto queda .jsonl.gz.partial; el otro segmento se sella.
    for (fase, esperadas) in [("write", &["gzip_finish", "flush"][..]), ("flush", &["flush"][..]),
        ("sync", &["sync_all"][..]), ("sha256", &["sha256"][..]), ("publish", &["publish_gzip"][..])] {
        let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
        let disco = DiscoMemoria(Arc::default());
        let (seal_tx, seal_rx) = std_mpsc::sync_channel(crate::SEAL_QUEUE_CAPACITY);
        let out = archive(&disco, &dir(), cola_con_horas(&ctl), seal_tx, &ctl);
        assert_eq!(out.rows_written, 4);
        disco.0.fallar(seg1, fase);
        let rep = seal_segments(&disco, &dir(), seal_rx, None, Some(&ctl));
        assert_eq!(rep.segments, 1, "{fase}");
        let f = &rep.failures[0];
        assert!(esperadas.contains(&f["phase"].as_str().unwrap_or("")), "{fase}: {f}");
        assert_eq!(f["state"], "jsonl.gz.partial", "{fase}");
        assert!(disco.0.existe(&p(seg1, ".jsonl.gz.partial")) && !disco.0.existe(&p(seg1, ".jsonl.gz")), "{fase}");
        assert!(disco.0.existe(&p(&seg2, ".manifest.json")), "{fase}");
        assert!(ctl.es_fatal());
    }
    // Append fallido: la hora afectada no se sella, sus filas cuentan como no escritas y la otra
    // hora sigue.
    // (Una fila grande y poco compresible para que el append llegue al archivo a traves de los
    // buferes del gzip y del BufWriter.)
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let disco = DiscoMemoria(Arc::default());
    disco.0.fallar(seg1, "write");
    let (queue, rx) = ArchiveQueue::with_admission(16, 4 << 20, Arc::clone(&ctl.admision));
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    let grande: String = (0..400_000).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        char::from(b"0123456789abcdef"[(x & 15) as usize])
    }).collect();
    for (s, ts, pl) in [(1, h - 1, grande.as_str()), (2, h + 1, "{}"), (3, h - 2, "{}"), (4, h + crate::ROTATION_GRACE_MS, "{}")] {
        queue.offer(fila(s, ts, pl), None).expect("fila");
    }
    drop(queue);
    let (out, rep) = archivar(&disco, &ctl, rx);
    assert_eq!((out.rows_written, out.rows_unwritten), (2, 2));
    assert_eq!(out.failures[0]["phase"], "append");
    assert_eq!(rep.segments, 1);
    assert!(disco.0.existe(&p(seg1, ".jsonl.gz.partial")));
    assert_eq!(ctl.primer_fatal().map(|f| f.0), Some(Clase::Escritor));
    // Handoff fallido (sellador ausente): registrado por segmento, sin abandonar los demas.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let disco = DiscoMemoria(Arc::default());
    let (seal_tx, seal_rx) = std_mpsc::sync_channel(crate::SEAL_QUEUE_CAPACITY);
    drop(seal_rx);
    let out = archive(&disco, &dir(), cola_con_horas(&ctl), seal_tx, &ctl);
    assert_eq!((out.rows_written, out.rows_unwritten), (0, 4));
    assert_eq!(out.failures.iter().filter(|f| f["phase"] == "handoff").count(), 2);
}

// ---------------------------------------------------------------------------------------------
// T9
// ---------------------------------------------------------------------------------------------
#[test]
fn parada_normal_no_borra_fatal() {
    // Fatal y stop en los dos ordenes: el fatal persiste y nunca se desarma.
    for fatal_primero in [true, false] {
        let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
        let parada = || ctl.admision.cortar(Motivo::Sana);
        if fatal_primero {
            ctl.fatal(Clase::Lector, "fatal antes del stop");
            ctl.cortes_nativos();
            parada();
            assert_eq!(ctl.admision.corte().map(|c| c.motivo), Some(Motivo::Fatal), "el corte fatal prevalece");
        } else {
            parada();
            ctl.fatal(Clase::Lector, "fatal despues del stop");
        }
        assert!(ctl.primer_fatal().is_some());
        assert!(ctl.desarmar().is_err());
    }
    // Fatal en cada fase del cierre sano: drenaje, sellado, END y destruccion del runtime.
    for fase in ["drenaje", "sellado", "END", "destruccion_runtime"] {
        let (ctl, salida, _) = control_de_prueba(PLAZO_FATAL);
        let vig = lanzar_vigilante(&ctl);
        ctl.admision.cortar(Motivo::Sana);
        match fase {
            "drenaje" | "sellado" => {
                ctl.fatal(Clase::Sellador, fase);
                ctl.fin_listo(0, false);
            }
            "END" => {
                ctl.fatal(Clase::Cierre, "END.json no persistido");
                ctl.fin_listo(0, true);
            }
            _ => {
                ctl.fin_listo(0, false);
                ctl.fatal(Clase::Panic, fase);
            }
        }
        vig.join().expect("vigilante");
        let codigos = salida.codigos();
        assert_eq!(codigos.len(), 1, "{fase}");
        assert_ne!(codigos[0].0, 0, "{fase}: un fatal tardio no es exito");
        assert!(ctl.desarmar().is_err(), "{fase}");
        assert_eq!(ctl.primer_fatal().map(|f| f.1), Some(if fase == "END" { "END.json no persistido".into() } else { fase.into() }));
    }
    // Parada sana con el orden de public_l2: las conexiones terminaron, el drenaje L2 final emite
    // a la cola AUN ABIERTA, despues se corta la admision (sana), el escritor drena exactamente las
    // aceptadas (con un emisor clonado vivo) y se sellan los segmentos.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let vig = lanzar_vigilante(&ctl);
    let (queue, rx) = ArchiveQueue::with_admission(64, 128 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
    let disco = DiscoMemoria(Arc::default());
    let (d2, c2) = (disco.clone(), Arc::clone(&ctl));
    let archivador = thread::spawn(move || archivar(&d2, &c2, rx));
    let emisor = Emitter { queue: queue.clone(), clock: crate::Clock(Instant::now()), route: "public", shard: Some(0), seq: Arc::new(AtomicU64::new(0)) };
    let h = crate::HOUR_MS;
    let finales = 5;
    for i in 0..finales {
        let at = crate::Stamp { utc_ms: h + i, mono_ns: i };
        emisor.emit("frame", 11, at, json!({"stream":"btcusdt@depth@100ms","symbol":"BTCUSDT","payload":"{}","depth_generation":1}), None)
            .expect("drenaje L2 admitido antes del corte");
    }
    let corte = ctl.admision.cortar(Motivo::Sana).expect("corte sano");
    assert_eq!(corte, Corte { motivo: Motivo::Sana, confirmadas: finales, en_vuelo: 0 });
    assert!(emisor.emit_now("frame", 11, json!({}), None).is_err(), "tras el corte no se admite");
    let (out, rep) = archivador.join().expect("archivador");
    assert_eq!(out.end, Some(QueueEnd::CutResolved));
    assert_eq!((out.rows_written, out.rows_unwritten, rep.counts.rows), (finales, 0, finales));
    assert_eq!(rep.segments, 1);
    assert!(rep.failures.is_empty() && !ctl.es_fatal());
    drop((emisor, queue));
    ctl.fin_listo(0, false);
    assert!(ctl.desarmar().is_ok(), "sin fatal: resultado sano");
    vig.join().expect("vigilante");
}

// ---------------------------------------------------------------------------------------------
// T10 (SALIDAFALLO-4, hallazgo 1 de la N1): la salida ordenada no retira al vigilante del plazo.
// ---------------------------------------------------------------------------------------------
#[test]
fn salida_ordenada_bloqueada_vence_plazo() {
    let plazo = Duration::from_secs(5);
    let (ctl, salida, reloj) = control_de_prueba(plazo);
    salida.ordenada_bloqueada.store(true, Ordering::SeqCst);
    let vig = lanzar_vigilante(&ctl);
    ctl.fatal(Clase::Lector, "fatal con salida ordenada bloqueada");
    esperar("cortes", || ctl.admision.cortada());
    // El cierre posible termino: el vigilante pide la salida ordenada, que no llega a terminar.
    ctl.fin_listo(1, false);
    esperar("salida ordenada pedida", || !salida.codigos().is_empty());
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_CERRADO, true)]);
    // Antes del plazo el vigilante sigue armado y no termina de otra forma.
    reloj.avanzar(plazo - Duration::from_nanos(1));
    ctl.despertar_vigilante();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_CERRADO, true)], "antes del plazo no hay otra salida");
    // Vencido el plazo desde el primer fatal: terminacion nativa con cierre incompleto.
    reloj.avanzar(Duration::from_nanos(1));
    ctl.despertar_vigilante();
    vig.join().expect("vigilante");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_CERRADO, true), (SALIDA_FATAL_INCOMPLETO, false)]);
}
// ---------------------------------------------------------------------------------------------
// T11 (SALIDAFALLO-5, hallazgo 2 de la N1): pausa determinista entre el CAS del primer fatal y
// la publicacion de su detalle. El lector nunca ve la clase con un detalle vacio.
// ---------------------------------------------------------------------------------------------
#[test]
fn primer_fatal_sin_detalle_vacio() {
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let barrera = Arc::new(Barrier::new(2));
    let b2 = Arc::clone(&barrera);
    ctl.fijar_pausa(Box::new(move |p| {
        if p == Punto::TrasCasFatal && thread::current().name() == Some("fatal-pausado") {
            b2.wait();
            b2.wait();
        }
    }));
    let c = Arc::clone(&ctl);
    let ganador = thread::Builder::new().name("fatal-pausado".into())
        .spawn(move || c.fatal(Clase::Lector, "detalle completo del primer fatal")).expect("hilo");
    barrera.wait(); // CAS ganado; el detalle aun no esta copiado ni FATAL publicado.
    assert_eq!(ctl.primer_fatal(), None, "nunca la clase con un detalle vacio");
    assert_eq!(ctl.fatal_pendiente(), Some(Clase::Lector));
    assert!(!ctl.es_fatal());
    assert!(!ctl.fatal(Clase::Escritor, "segundo"), "el CAS ya tiene dueno: es secundario");
    barrera.wait();
    assert!(ganador.join().expect("hilo"));
    assert_eq!(ctl.primer_fatal(), Some((Clase::Lector, "detalle completo del primer fatal".to_string())));
    assert_eq!(ctl.fatal_pendiente(), None);
    assert_eq!(ctl.secundarios().0, vec!["escritor_error: segundo".to_string()]);
}

// ---------------------------------------------------------------------------------------------
// T12 (SALIDAFALLO-5, sobre SALIDAFALLO-3): pausa determinista entre el cambio de estado del
// corte (admision y store) y la publicacion de su instantanea.
// ---------------------------------------------------------------------------------------------
#[test]
fn instantaneas_de_corte_con_pausa() {
    let adm = Arc::new(Admision::nueva());
    let barrera = Arc::new(Barrier::new(2));
    let b2 = Arc::clone(&barrera);
    adm.fijar_pausa(Box::new(move |p| {
        if p == Punto::TrasCorteAdmision {
            b2.wait();
            b2.wait();
        }
    }));
    let a2 = Arc::clone(&adm);
    let cortador = thread::spawn(move || a2.cortar(Motivo::Fatal));
    barrera.wait(); // bit de corte puesto; instantanea sin publicar.
    assert!(!adm.cortada(), "sin instantanea publicada no se ve cortada");
    assert!(!adm.resuelta(), "sin instantanea publicada no se ve resuelta");
    assert_eq!(adm.corte(), None);
    assert!(adm.entrar().is_none(), "la admision ya esta cerrada aunque la instantanea falte");
    barrera.wait();
    assert_eq!(cortador.join().expect("hilo"), Some(Corte { motivo: Motivo::Fatal, confirmadas: 0, en_vuelo: 0 }));
    assert!(adm.cortada() && adm.resuelta());
    assert_eq!(adm.rechazos_tras_corte(), 1);

    let (compuerta, borrados) = compuerta_memoria();
    let barrera = Arc::new(Barrier::new(2));
    let b2 = Arc::clone(&barrera);
    compuerta.fijar_pausa(Box::new(move |p| {
        if p == Punto::TrasCierreStore {
            b2.wait();
            b2.wait();
        }
    }));
    let c2 = Arc::clone(&compuerta);
    let cerrador = thread::spawn(move || c2.cerrar());
    barrera.wait(); // bit de cierre puesto; instantanea sin publicar.
    assert!(!compuerta.cerrada(), "sin instantanea publicada no se ve cerrada");
    assert_eq!(compuerta.corte(), None);
    assert!(compuerta.admitir(OpStore::Retencion).is_none(), "ya no admite trabajo");
    assert!(compuerta.borrar(OpStore::RetencionBorra, Path::new("x.tgll")).is_err(), "ni borrados");
    barrera.wait();
    assert_eq!(cerrador.join().expect("hilo"), Some(CorteStore { en_curso: 0, ultima: OpStore::Ninguna }));
    assert!(compuerta.cerrada());
    assert!(borrados.lock().expect("b").is_empty());
}
// ---------------------------------------------------------------------------------------------
// T13 (SALIDAFALLO-6, hallazgo 3 de la N1): la operacion de store admitida antes del cierre queda
// identificada en la misma instantanea que su contador (pausa tras admitir).
// ---------------------------------------------------------------------------------------------
#[test]
fn identidad_store_en_la_instantanea() {
    let (compuerta, _) = compuerta_memoria();
    // Una operacion anterior ya termino: no es la que esta en curso al cerrar.
    drop(compuerta.admitir(OpStore::Retencion).expect("previa"));
    let barrera = Arc::new(Barrier::new(2));
    let b2 = Arc::clone(&barrera);
    compuerta.fijar_pausa(Box::new(move |p| {
        if p == Punto::TrasAdmitirStore && thread::current().name() == Some("store-pausado") {
            b2.wait();
            b2.wait();
        }
    }));
    let c2 = Arc::clone(&compuerta);
    let store = thread::Builder::new().name("store-pausado".into()).spawn(move || {
        let guarda = c2.admitir(OpStore::Recuperacion).expect("admitida antes del cierre");
        drop(guarda);
    }).expect("hilo");
    barrera.wait(); // admitida; la guarda aun no se devolvio.
    let corte = compuerta.cerrar().expect("corte");
    assert_eq!(corte, CorteStore { en_curso: 1, ultima: OpStore::Recuperacion }, "la operacion en curso queda identificada");
    barrera.wait();
    store.join().expect("hilo");
    assert_eq!(compuerta.en_curso(), 0);
    assert!(compuerta.admitir(OpStore::Recode).is_none());
    assert_eq!(compuerta.corte(), Some(corte));
}
// ---------------------------------------------------------------------------------------------
// T14 (SALIDAFALLO-7, hallazgo 4 de la N1): END cubre los fatales tardios. Recorre el
// finalizador real (`finish_capture`) sobre el disco en memoria.
// ---------------------------------------------------------------------------------------------
type AlEscribirEnd = Mutex<Option<Box<dyn FnOnce() + Send>>>;

/// Disco en memoria que avisa cada escritura de END.json y puede ejecutar una accion durante la
/// primera (un fatal "durante la escritura de END").
struct DiscoEnd {
    mem: DiscoMemoria,
    avisos: std_mpsc::Sender<&'static str>,
    al_escribir: AlEscribirEnd,
}

impl Disk for DiscoEnd {
    type File = ArchivoMemoria;
    fn create(&self, path: &Path) -> io::Result<ArchivoMemoria> {
        self.mem.create(path)
    }
    fn sync(&self, f: &ArchivoMemoria) -> io::Result<()> {
        self.mem.sync(f)
    }
    fn digest(&self, path: &Path) -> CaptureResult<(u64, String)> {
        self.mem.digest(path)
    }
    fn publish(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.mem.publish(from, to)
    }
    fn write_json(&self, path: &Path, value: &Value) -> CaptureResult<()> {
        if path.file_name().is_some_and(|n| n == "END.json") {
            let accion = self.al_escribir.lock().ok().and_then(|mut a| a.take());
            if let Some(a) = accion {
                a();
            }
            let _ = self.avisos.send("END");
        }
        self.mem.write_json(path, value)
    }
}

fn partes_fin(ctl: &Arc<Control>) -> crate::FinishParts {
    let hecho = || Arc::new(AtomicBool::new(true));
    crate::FinishParts {
        ctl: Arc::clone(ctl),
        dir: dir(),
        clock: crate::Clock(Instant::now()),
        audit: Arc::new(crate::QueueAudit::default()),
        connections: Arc::new(AtomicU64::new(0)),
        writer_thread: thread::spawn(crate::ArchiveOutcome::default),
        sealer_thread: thread::spawn(crate::SealReport::default),
        store_thread: thread::spawn(|| ()),
        writer_finished: hecho(),
        sealer_finished: hecho(),
        store_finished: hecho(),
    }
}

fn lanzar_finalizador(ctl: &Arc<Control>, disco: DiscoEnd) -> thread::JoinHandle<()> {
    let partes = partes_fin(ctl);
    let h = thread::spawn(move || crate::finish_capture(partes, &disco));
    ctl.registrar_finalizador(h.thread().clone());
    h
}

fn end_de(mem: &DiscoMemoria) -> Value {
    serde_json::from_slice(&mem.0.leer(&dir().join("END.json")).expect("END.json")).expect("json")
}

#[test]
fn end_cubre_fatal_tardio() {
    let sano = || FinAsync { stop_reason: "deadline", exit: 0, errors: Vec::new() };
    // a) Parada sana: el finalizador no compone END con el runtime vivo. Un fatal durante la
    // destruccion del runtime entra en END y el codigo no es cero.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let (tx, rx) = std_mpsc::channel();
    let t2 = tx.clone();
    ctl.fijar_pausa(Box::new(move |p| {
        if p == Punto::FinalizadorEsperaRuntime {
            let _ = t2.send("espera_runtime");
        }
    }));
    let mem = DiscoMemoria(Arc::default());
    let fin = lanzar_finalizador(&ctl, DiscoEnd { mem: mem.clone(), avisos: tx, al_escribir: Mutex::new(None) });
    ctl.entregar_fin_async(sano());
    assert_eq!(rx.recv_timeout(Duration::from_secs(20)), Ok("espera_runtime"), "END no se compone con el runtime vivo");
    ctl.fatal(Clase::Panic, "panic durante la destruccion del runtime");
    ctl.runtime_destruido();
    fin.join().expect("finalizador");
    let end = end_de(&mem);
    assert_eq!(end["salidafallo"]["first_fatal"]["class"], "panic");
    assert_eq!(end["salidafallo"]["first_fatal"]["detail"], "panic durante la destruccion del runtime");
    assert_eq!(end["status"], "error");
    assert_eq!(ctl.codigo_fin(), 1);

    // b) Fatal durante la escritura de END: END se reescribe con ese fatal.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let (tx, rx) = std_mpsc::channel();
    let c2 = Arc::clone(&ctl);
    let mem = DiscoMemoria(Arc::default());
    let accion: Box<dyn FnOnce() + Send> = Box::new(move || {
        c2.fatal(Clase::Cierre, "fatal durante la escritura de END");
    });
    let fin = lanzar_finalizador(&ctl, DiscoEnd { mem: mem.clone(), avisos: tx, al_escribir: Mutex::new(Some(accion)) });
    ctl.entregar_fin_async(sano());
    ctl.runtime_destruido();
    fin.join().expect("finalizador");
    assert_eq!(rx.try_iter().count(), 2, "END escrito dos veces");
    let end = end_de(&mem);
    assert_eq!(end["salidafallo"]["first_fatal"]["class"], "cierre_error");
    assert_eq!(end["salidafallo"]["end_rewrites"], 1);
    assert_eq!(end["status"], "error");
    assert_eq!(ctl.codigo_fin(), 1);
}
// ---------------------------------------------------------------------------------------------
// T15 (SALIDAFALLO-8, hallazgo 5 de la N1): los fallos del escritor en append y handoff
// registran ruta, estado de conservacion, validacion y filas, como los del sellador.
// ---------------------------------------------------------------------------------------------
#[test]
fn fallo_escritor_con_ruta_y_estado() {
    let h = crate::HOUR_MS;
    let (seg1, seg2) = ("hour-0-0001", format!("hour-{h}-0002"));
    let partial = |n: &str| json!(dir().join(format!("{n}.jsonl.gz.partial")));
    // Append fallido en la primera hora (fila grande y poco compresible, como en T8).
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let disco = DiscoMemoria(Arc::default());
    disco.0.fallar(seg1, "write");
    let (queue, rx) = ArchiveQueue::with_admission(16, 4 << 20, Arc::clone(&ctl.admision));
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    let grande: String = (0..400_000).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        char::from(b"0123456789abcdef"[(x & 15) as usize])
    }).collect();
    for (s, ts, pl) in [(1, h - 1, grande.as_str()), (2, h + 1, "{}"), (3, h - 2, "{}"), (4, h + crate::ROTATION_GRACE_MS, "{}")] {
        queue.offer(fila(s, ts, pl), None).expect("fila");
    }
    drop(queue);
    let (out, _) = archivar(&disco, &ctl, rx);
    let f = &out.failures[0];
    assert_eq!(f["phase"], "append");
    assert_eq!(f["segment"], seg1);
    assert_eq!(f["path"], partial(seg1));
    assert_eq!(f["state"], "jsonl.gz.partial_conservado_sin_sellar");
    assert_eq!((f["validated"].as_bool(), f["admitted"].as_bool()), (Some(false), Some(false)));
    assert_eq!(f["rows"], 1);
    assert!(disco.0.existe(&dir().join(format!("{seg1}.jsonl.gz.partial"))), "el parcial se conserva");
    // Handoff fallido (sellador ausente): ruta y estado de cada segmento afectado.
    let (ctl, _, _) = control_de_prueba(PLAZO_FATAL);
    let disco = DiscoMemoria(Arc::default());
    let (seal_tx, seal_rx) = std_mpsc::sync_channel(crate::SEAL_QUEUE_CAPACITY);
    drop(seal_rx);
    let out = archive(&disco, &dir(), cola_con_horas(&ctl), seal_tx, &ctl);
    let handoffs: Vec<&Value> = out.failures.iter().filter(|f| f["phase"] == "handoff").collect();
    assert_eq!(handoffs.len(), 2);
    for (f, seg) in handoffs.iter().zip([seg1, seg2.as_str()]) {
        assert_eq!(f["segment"], seg);
        assert_eq!(f["path"], partial(seg));
        assert_eq!(f["state"], "jsonl.gz.partial_conservado_sin_sellar");
        assert_eq!((f["validated"].as_bool(), f["admitted"].as_bool()), (Some(false), Some(false)));
        assert_eq!(f["rows"], 2);
    }
}
// ---------------------------------------------------------------------------------------------
// T16 (SALIDAFALLO-9, hallazgo 6 de la N1): el venue local del ensayo P8 solo admite 127.0.0.1,
// sin TLS, sin valor por defecto, y las inyecciones solo existen con el venue local.
// ---------------------------------------------------------------------------------------------
#[test]
fn venue_local_solo_loopback() {
    let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let v = crate::Venue::local(&a(&["ws_local=127.0.0.1:9001", "rest_local=http://127.0.0.1:9002"])).expect("loopback");
    assert_eq!((v.name, v.ws, v.rest), ("local", "127.0.0.1:9001", "http://127.0.0.1:9002"));
    assert!(v.url("public", &["FSTUSDT".into()], false).starts_with("ws://127.0.0.1:9001/public/stream?"));
    for malos in [
        &["ws_local=10.0.0.1:9001", "rest_local=http://127.0.0.1:9002"][..],
        &["ws_local=127.0.0.1:9001", "rest_local=http://192.168.1.2:9002"][..],
        &["ws_local=127.0.0.1:9001", "rest_local=https://127.0.0.1:9002"][..],
        &["ws_local=fstream.binance.com:443", "rest_local=http://127.0.0.1:9002"][..],
        &["rest_local=http://127.0.0.1:9002"][..],
        &["ws_local=127.0.0.1:9001"][..],
    ] {
        assert!(crate::Venue::local(&a(malos)).is_err(), "{malos:?}");
    }
    assert!(crate::Venue::parse("mainnet").expect("mainnet").url("public", &["FSTUSDT".into()], false).starts_with("wss://fstream.binance.com/"));
    assert!(crate::Venue::parse("local").is_err(), "local no tiene destino por defecto");
    assert!(crate::inyeccion::desde_args(&a(&["mainnet", "out", "r.csv", "inyectar=panic_tarea"])).is_err());
    assert!(crate::inyeccion::desde_args(&a(&["testnet", "out", "r.csv", "inyectar=panic_tarea"])).is_err());
    assert_eq!(crate::inyeccion::desde_args(&a(&["mainnet", "out", "r.csv"])), Ok(None));
    assert_eq!(crate::inyeccion::desde_args(&a(&["local", "out", "r.csv", "inyectar=store_ocupado"])), Ok(Some(crate::inyeccion::Caso::StoreOcupado)));
    assert!(crate::inyeccion::desde_args(&a(&["local", "out", "r.csv", "inyectar=desconocido"])).is_err());
}
// ---------------------------------------------------------------------------------------------
// T17 (SALIDAFALLO-10, C1 de la N1 v3): con FIN_LISTO publicado y el plazo ya vencido en la misma
// vuelta, el vigilante da prioridad al vencimiento: termina sin ordenar y con cierre incompleto, y
// nunca pide una salida ordenada iniciada despues del vencimiento. La pausa determinista pone las
// dos condiciones desde el propio hilo del vigilante, despues de los cortes y antes de su primera
// lectura de reloj y banderas.
// ---------------------------------------------------------------------------------------------
#[test]
fn vencimiento_prioriza_sobre_salida_ordenada() {
    let plazo = Duration::from_secs(5);
    let (ctl, salida, reloj) = control_de_prueba(plazo);
    let debil = Arc::downgrade(&ctl);
    let r2 = Arc::clone(&reloj);
    let ambas = Arc::new(AtomicBool::new(false));
    let a2 = Arc::clone(&ambas);
    ctl.fijar_pausa(Box::new(move |p| {
        if p == Punto::VigilanteTrasCortes {
            if let Some(c) = debil.upgrade() {
                r2.avanzar(plazo);
                c.fin_listo(SALIDA_FATAL_CERRADO as u8, false);
                a2.store(c.fin_hecho() && c.ahora() >= plazo, Ordering::SeqCst);
            }
        }
    }));
    let vig = lanzar_vigilante(&ctl);
    ctl.fatal(Clase::Lector, "fatal con fin listo y plazo vencido a la vez");
    vig.join().expect("vigilante");
    assert!(ambas.load(Ordering::SeqCst), "las dos condiciones estaban presentes al mirar");
    assert_eq!(salida.codigos(), [(SALIDA_FATAL_INCOMPLETO, false)], "el vencimiento gana; ninguna salida ordenada");
}