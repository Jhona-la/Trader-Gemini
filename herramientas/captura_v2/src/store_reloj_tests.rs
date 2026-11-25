//! FALLAS-1 (pedido 20261123-0013, filas A18-A33 y A55): arranque reciente y cadencia de STATUS
//! por el camino real de `store::run`, con el reloj de store inyectado.
//!
//! `I` es `cfg.scan_every` efectivo (`crate::STORE_SCAN_EVERY`). `EPSILON` es el paso minimo
//! representable del reloj de prueba: cuenta nanosegundos en un `u64`, asi que es 1 ns. El guion
//! corre dentro de `RelojStore::esperar` (la espera real de `run`) en el mismo hilo que `run`: el
//! tiempo avanza solo cuando el guion lo decide, sin sleeps de pared. Ningun borrado es fisico: las
//! purgas pasan por una compuerta cuyo borrador solo anota la ruta.
//!
//! Cada caso imprime `TRAZA_RELOJ <json>` con las lecturas del reloj (mono, utc, esperar), las
//! publicaciones de STATUS observadas y el resultado del hilo.

use super::retention_tests::{DiskProbe, Fixture, fijar_disco, llamadas_disco};
use super::*;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

const EPSILON: Duration = Duration::from_nanos(1);
const I: Duration = crate::STORE_SCAN_EVERY;
/// UTC de partida del guion: hora 2 de 1970 + 1 ms. La fixture (hora 1) vence en 1 h + 365 d.
const UTC0: u64 = 2 * crate::HOUR_MS + 1;
/// Origen de las pruebas de conservacion (A21, A31-A33): representable tambien en la base (>= I),
/// para aislar la propiedad conservada del defecto de arranque, que prueban A23-A24.
const ORIGEN_CONSERVACION: Duration = I;

/// Lo que ve el guion en cada espera de `run`.
struct Paso<'a> {
    reloj: &'a RelojGuion,
    vuelta: usize,
}

type Guion = Box<dyn FnMut(&Paso<'_>) + Send>;

struct RelojGuion {
    mono_ns: AtomicU64,
    utc: AtomicU64,
    esperas: AtomicUsize,
    traza: Mutex<Vec<Value>>,
    guion: Mutex<Option<Guion>>,
}

impl RelojGuion {
    fn nuevo(origen: Duration, guion: Guion) -> Arc<Self> {
        Arc::new(Self {
            mono_ns: AtomicU64::new(u64::try_from(origen.as_nanos()).expect("origen en u64 ns")),
            utc: AtomicU64::new(UTC0),
            esperas: AtomicUsize::new(0),
            traza: Mutex::default(),
            guion: Mutex::new(Some(guion)),
        })
    }
    fn anotar(&self, v: Value) {
        self.traza.lock().expect("traza").push(v);
    }
    fn ahora(&self) -> Duration {
        Duration::from_nanos(self.mono_ns.load(Ordering::SeqCst))
    }
    fn fijar(&self, t: Duration) {
        self.mono_ns
            .store(u64::try_from(t.as_nanos()).expect("ns"), Ordering::SeqCst);
        self.anotar(json!({"ev": "guion_fija_mono", "ns": t.as_nanos() as u64}));
    }
    fn fijar_utc(&self, ms: u64) {
        self.utc.store(ms, Ordering::SeqCst);
        self.anotar(json!({"ev": "guion_fija_utc", "ms": ms}));
    }
}

impl RelojStore for RelojGuion {
    fn mono(&self) -> Duration {
        let t = self.ahora();
        self.anotar(json!({"ev": "mono", "ns": t.as_nanos() as u64}));
        t
    }
    fn utc_ms(&self) -> u64 {
        let ms = self.utc.load(Ordering::SeqCst);
        self.anotar(json!({"ev": "utc", "ms": ms}));
        ms
    }
    fn esperar(
        &self,
        sealed: &mpsc::Receiver<(PathBuf, String)>,
        max: Duration,
    ) -> Result<(), mpsc::RecvTimeoutError> {
        let vuelta = self.esperas.fetch_add(1, Ordering::SeqCst) + 1;
        self.anotar(
            json!({"ev": "esperar", "vuelta": vuelta, "max_ns": max.as_nanos() as u64,
            "mono_ns": self.ahora().as_nanos() as u64}),
        );
        let mut guion = self.guion.lock().expect("guion").take();
        if let Some(g) = guion.as_mut() {
            g(&Paso {
                reloj: self,
                vuelta,
            });
        }
        *self.guion.lock().expect("guion") = guion;
        match sealed.try_recv() {
            Ok(_) => Ok(()),
            Err(mpsc::TryRecvError::Empty) => Err(mpsc::RecvTimeoutError::Timeout),
            Err(mpsc::TryRecvError::Disconnected) => Err(mpsc::RecvTimeoutError::Disconnected),
        }
    }
}

/// STATUS publicado: su `utc_ms` (cada vuelta usa un UTC distinto, asi se distingue cada escritura).
fn status_utc(root: &Path) -> Option<u64> {
    let bytes = fs::read(root.join("STATUS.json")).ok()?;
    serde_json::from_slice::<Value>(&bytes).ok()?["utc_ms"].as_u64()
}

fn raiz_vacia() -> CaptureResult<PathBuf> {
    static N: AtomicU64 = AtomicU64::new(0);
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".retention-tests")
        .join(format!(
            "reloj-{}-{stamp}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&root)?;
    Ok(root)
}

type Borrados = Arc<Mutex<Vec<PathBuf>>>;

fn cfg_con(
    root: &Path,
    reloj: Arc<RelojGuion>,
    cancel: Arc<AtomicBool>,
) -> (StoreConfig, Borrados) {
    let borrados: Borrados = Arc::default();
    let registro = Arc::clone(&borrados);
    let gate = CompuertaStore::nueva(
        Arc::clone(&cancel),
        Box::new(move |p: &Path| {
            // Borrado simulado: se anota, el archivo queda en disco.
            registro
                .lock()
                .map_err(|_| io::Error::other("envenenado"))?
                .push(p.to_path_buf());
            Ok(())
        }),
    );
    (
        StoreConfig {
            root: root.to_path_buf(),
            retention_ms: crate::RETENTION_MS,
            cancel,
            min_free_bytes: crate::MIN_FREE_BYTES,
            codec: Fixture::params(),
            scan_every: I,
            gate: Arc::new(gate),
            reloj,
        },
        borrados,
    )
}

/// Resultado de una corrida de `run` en el hilo del test.
struct Corrida {
    panic: Option<String>,
    aviso: Option<StoreStop>,
    aviso_consumido: bool,
    traza: Vec<Value>,
}

fn correr(cfg: StoreConfig, reloj: &Arc<RelojGuion>) -> Corrida {
    let (sealed_tx, sealed_rx) = mpsc::channel();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
    let mut stop_tx = Some(stop_tx);
    let r = catch_unwind(AssertUnwindSafe(|| run(cfg, sealed_rx, &mut stop_tx)));
    drop(sealed_tx);
    let aviso_consumido = stop_tx.is_none();
    drop(stop_tx);
    let aviso = stop_rx.try_recv().ok();
    let panic = r.err().map(|e| {
        e.downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic sin texto".into())
    });
    let traza = reloj.traza.lock().map(|v| v.clone()).unwrap_or_default();
    Corrida {
        panic,
        aviso,
        aviso_consumido,
        traza,
    }
}

fn imprimir(caso: &str, c: &Corrida, extra: Value) {
    println!(
        "TRAZA_RELOJ {}",
        json!({"caso": caso, "I_ns": I.as_nanos() as u64, "epsilon_ns": EPSILON.as_nanos() as u64,
        "panic": c.panic, "aviso": c.aviso.as_ref().map(|s| s.message()), "aviso_consumido": c.aviso_consumido,
        "extra": extra, "traza": c.traza})
    );
}

/// A18, A19, A23-A26: arranque desde `origen`. Primera vuelta: cancelacion, disco, STATUS; en la
/// primera espera el STATUS ya esta publicado con el UTC de esa vuelta y el disco se midio una vez.
fn arranque(caso: &str, origen: Duration) {
    let root = raiz_vacia().expect("raiz");
    let probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    let cancel = Arc::new(AtomicBool::new(false));
    let visto: Arc<Mutex<Option<(Option<u64>, usize)>>> = Arc::default();
    let (v2, c2, r2) = (Arc::clone(&visto), Arc::clone(&cancel), root.clone());
    let reloj = RelojGuion::nuevo(
        origen,
        Box::new(move |p: &Paso<'_>| {
            if p.vuelta == 1 {
                *v2.lock().expect("visto") = Some((status_utc(&r2), llamadas_disco()));
                p.reloj.anotar(json!({"ev": "guion_cancela"}));
                c2.store(true, Ordering::Release);
            }
        }),
    );
    let (cfg, borrados) = cfg_con(&root, Arc::clone(&reloj), Arc::clone(&cancel));
    let c = correr(cfg, &reloj);
    let primera = *visto.lock().expect("visto");
    imprimir(
        caso,
        &c,
        json!({"origen_ns": origen.as_nanos() as u64, "en_primera_espera": primera.map(|(s, d)| json!({"status_utc": s, "lecturas_disco": d}))}),
    );
    assert!(
        c.panic.is_none(),
        "A23: store::run arranca sin panic desde el origen {origen:?}: {:?}",
        c.panic
    );
    assert_eq!(
        primera,
        Some((Some(UTC0), 1)),
        "A23: en la primera vuelta store mide el disco una vez y publica el primer STATUS antes de esperar"
    );
    assert!(
        !c.aviso_consumido && c.aviso.is_none(),
        "A23: el canal de aviso sigue sin usar tras una cancelacion ordinaria"
    );
    assert_eq!(
        probe.calls(),
        1,
        "la cancelacion se ve antes de la segunda medida de disco"
    );
    assert!(borrados.lock().expect("borrados").is_empty());
    let esperas: Vec<_> = c.traza.iter().filter(|e| e["ev"] == "esperar").collect();
    assert_eq!(esperas.len(), 1);
    assert_eq!(
        esperas[0]["max_ns"].as_u64(),
        Some(I.as_nanos() as u64),
        "la espera real usa I"
    );
}

#[test]
fn a23_arranque_en_origen_cero() {
    arranque("a23_origen_cero", Duration::ZERO);
}

#[test]
fn a24_arranque_interior_menor_que_i() {
    // I - epsilon es positivo y representable: I = 60 s, epsilon = 1 ns.
    assert!(I > EPSILON);
    arranque("a24_i_menos_epsilon", I - EPSILON);
}

#[test]
fn a25_arranque_exactamente_i() {
    arranque("a25_i", I);
}

#[test]
fn a26_arranque_i_mas_epsilon() {
    arranque("a26_i_mas_epsilon", I + EPSILON);
}

/// A20, A22, A27, A28: cadencia sin purgas. `pasos` son desplazamientos desde la publicacion de
/// referencia vigente; `esperado` dice si la vuelta siguiente debe publicar. Tras cada publicacion la
/// referencia pasa a ser esa vuelta.
fn cadencia(caso: &str, origen: Duration, pasos: &'static [(Duration, bool)]) {
    let root = raiz_vacia().expect("raiz");
    let _probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    let cancel = Arc::new(AtomicBool::new(false));
    let observado: Arc<Mutex<Vec<Value>>> = Arc::default();
    let (o2, c2, r2) = (Arc::clone(&observado), Arc::clone(&cancel), root.clone());
    let mut referencia = origen;
    let mut ultimo_status = None;
    let reloj = RelojGuion::nuevo(
        origen,
        Box::new(move |p: &Paso<'_>| {
            // Lo publicado en la vuelta que acaba de terminar.
            let s = status_utc(&r2);
            let publico = s != ultimo_status;
            let utc_vuelta = UTC0 + (p.vuelta as u64 - 1) * 1_000;
            if publico {
                referencia = p.reloj.ahora();
                ultimo_status = s;
            }
            o2.lock().expect("obs").push(json!({"vuelta": p.vuelta, "publico": publico, "status_utc": s,
            "utc_vuelta": utc_vuelta, "mono_ns": p.reloj.ahora().as_nanos() as u64, "referencia_ns": referencia.as_nanos() as u64}));
            match pasos.get(p.vuelta - 1) {
                Some((delta, _)) => {
                    p.reloj.fijar(referencia + *delta);
                    // UTC independiente del monotono: avanza 1 s por vuelta, sin relacion con el delta.
                    p.reloj.fijar_utc(UTC0 + p.vuelta as u64 * 1_000);
                }
                None => c2.store(true, Ordering::Release),
            }
        }),
    );
    let (cfg, _) = cfg_con(&root, Arc::clone(&reloj), Arc::clone(&cancel));
    let pared = std::time::Instant::now(); // reloj de pared real, no el del adaptador
    let c = correr(cfg, &reloj);
    let pared_ns = pared.elapsed().as_nanos() as u64;
    let obs = observado.lock().expect("obs").clone();
    imprimir(
        caso,
        &c,
        json!({"origen_ns": origen.as_nanos() as u64, "observado": obs, "pared_ns": pared_ns}),
    );
    assert!(c.panic.is_none(), "cadencia: sin panic: {:?}", c.panic);
    assert_eq!(obs.len(), pasos.len() + 1, "una espera por vuelta");
    assert_eq!(obs[0]["publico"], true, "primera vuelta: STATUS");
    for (i, (delta, esperado)) in pasos.iter().enumerate() {
        assert_eq!(
            obs[i + 1]["publico"],
            *esperado,
            "A27/A28: tras avanzar {delta:?} desde la referencia, la vuelta {} publica = {esperado}",
            i + 2
        );
    }
    assert!(
        pared_ns < I.as_nanos() as u64,
        "sin espera de pared: el reloj lo avanza el guion"
    );
}

const CADENCIA_I: &[(Duration, bool)] = &[
    (Duration::from_nanos(60_000_000_000 - 1), false), // I - epsilon: no publica
    (Duration::from_secs(60), true),                   // I: publica, nueva referencia
    (Duration::from_nanos(60_000_000_000 - 1), false), // I - epsilon desde la nueva referencia
    (Duration::from_secs(60), true),
];

const CADENCIA_I_MAS_EPSILON: &[(Duration, bool)] = &[
    (Duration::from_nanos(60_000_000_001), true), // I + epsilon: publica
    (Duration::from_nanos(60_000_000_000 - 1), false), // desde esa referencia, I - epsilon: no
    (Duration::from_secs(60), true),
];

#[test]
fn a27_a28_cadencia_en_i() {
    assert_eq!(
        I,
        Duration::from_secs(60),
        "las tablas de pasos estan escritas para I = 60 s"
    );
    cadencia("a27_a28_cadencia_en_i", I, CADENCIA_I);
}

#[test]
fn a28_cadencia_en_i_mas_epsilon() {
    assert_eq!(I, Duration::from_secs(60));
    cadencia("a28_cadencia_en_i_mas_epsilon", I, CADENCIA_I_MAS_EPSILON);
}

#[test]
fn a20_cadencia_desde_origen_cero() {
    // Candidato: el mismo contrato al origen del reloj (la base entra en panic antes).
    cadencia("a20_cadencia_origen_cero", Duration::ZERO, CADENCIA_I);
}

/// A29: purga valida antes de vencer I -> STATUS extraordinario y nueva referencia. Borrado simulado.
#[test]
fn a29_status_extraordinario_tras_purga() -> CaptureResult<()> {
    let fixture = Fixture::archived_with_gzip()?;
    // Sin gzip coexistente: el grupo es un archivo cerrado (tgll + recibo + manifiesto). El gzip
    // recreado por la fixture se aparta con rename (sin borrar) para que la purga no lo salte.
    fs::rename(
        fixture.path(".jsonl.gz"),
        fixture.root.join("gzip_apartado_por_el_test.jsonl.gz"),
    )?;
    let _probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    let vence = 2 * crate::HOUR_MS + crate::RETENTION_MS; // fin de la hora 1 + 365 d
    let cancel = Arc::new(AtomicBool::new(false));
    let observado: Arc<Mutex<Vec<Value>>> = Arc::default();
    let (o2, c2, r2) = (
        Arc::clone(&observado),
        Arc::clone(&cancel),
        fixture.root.clone(),
    );
    let mut referencia = I;
    let mut ultimo = None;
    let reloj = RelojGuion::nuevo(
        I,
        Box::new(move |p: &Paso<'_>| {
            let s = status_utc(&r2);
            let publico = s != ultimo;
            if publico {
                referencia = p.reloj.ahora();
                ultimo = s;
            }
            o2.lock().expect("obs").push(
                json!({"vuelta": p.vuelta, "publico": publico, "status_utc": s,
            "mono_ns": p.reloj.ahora().as_nanos() as u64}),
            );
            match p.vuelta {
                // Antes de vencer I: la hora caduca por UTC (retencion) y el monotono avanza I/2.
                1 => {
                    p.reloj.fijar(referencia + I / 2);
                    p.reloj.fijar_utc(vence);
                }
                // Tras la purga el UTC vuelve atras (independiente del monotono) para no repetirla.
                2 => {
                    p.reloj.fijar(referencia + I - EPSILON);
                    p.reloj.fijar_utc(UTC0 + 7_000);
                }
                3 => {
                    p.reloj.fijar(referencia + I);
                    p.reloj.fijar_utc(UTC0 + 9_000);
                }
                _ => c2.store(true, Ordering::Release),
            }
        }),
    );
    let (cfg, borrados) = cfg_con(&fixture.root, Arc::clone(&reloj), Arc::clone(&cancel));
    let c = correr(cfg, &reloj);
    let obs = observado.lock().expect("obs").clone();
    let borr = borrados.lock().expect("b").clone();
    let log = fs::read_to_string(fixture.root.join("RETENTION.log")).unwrap_or_default();
    imprimir(
        "a29_purga",
        &c,
        json!({"observado": obs, "borrados_simulados": borr, "retention_log": log}),
    );
    assert!(c.panic.is_none(), "{:?}", c.panic);
    assert_eq!(
        obs.iter().map(|o| o["publico"] == true).collect::<Vec<_>>(),
        [true, true, false, true],
        "A29: STATUS inicial, extraordinario tras la purga, nada en I - epsilon desde la purga, periodico en I"
    );
    assert_eq!(
        obs[1]["status_utc"].as_u64(),
        Some(vence),
        "STATUS actualizado en la vuelta de la purga"
    );
    assert_eq!(
        borr.len(),
        3,
        "tres borrados simulados (tgll, recibo, manifiesto)"
    );
    assert!(log.contains("expired_365d"));
    for ext in [".tgll", ".recode.json", ".manifest.json"] {
        assert!(
            fixture.path(ext).exists(),
            "borrado simulado: {ext} sigue en disco"
        );
    }
    Ok(())
}

/// A30: arranca ya cancelado. Sale sin medir disco, sin STATUS ni espera, canal sin usar.
fn cancelado(caso: &str, origen: Duration) {
    let root = raiz_vacia().expect("raiz");
    let probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    let cancel = Arc::new(AtomicBool::new(true));
    let reloj = RelojGuion::nuevo(origen, Box::new(|_| {}));
    let (cfg, _) = cfg_con(&root, Arc::clone(&reloj), cancel);
    let c = correr(cfg, &reloj);
    imprimir(
        caso,
        &c,
        json!({"origen_ns": origen.as_nanos() as u64, "lecturas_disco": probe.calls()}),
    );
    assert!(c.panic.is_none(), "A30: sin panic: {:?}", c.panic);
    assert!(
        !c.aviso_consumido && c.aviso.is_none(),
        "A30: cancelacion ordinaria, no aviso fatal"
    );
    assert_eq!(probe.calls(), 0, "sin operaciones tras ver la cancelacion");
    assert!(status_utc(&root).is_none(), "sin STATUS");
    assert!(!c.traza.iter().any(|e| e["ev"] == "esperar"));
}

#[test]
fn a30_cancelado_al_arrancar_origen_i() {
    cancelado("a30_cancelado_origen_i", I);
}

#[test]
fn a30_cancelado_al_arrancar_origen_cero() {
    cancelado("a30_cancelado_origen_cero", Duration::ZERO);
}

/// A31: igualdad con el minimo permitida; el inmediato inferior para sin recode ni purga; y el
/// descenso durante el trabajo (tras una vuelta sana) tambien para.
#[test]
fn a31_disco_minimo_inferior_y_descenso() -> CaptureResult<()> {
    let min = crate::MIN_FREE_BYTES;
    // Inferior desde el arranque con una hora pendiente de recode.
    let fixture = Fixture::sealed()?;
    let _probe = DiskProbe::new(min - 1);
    let cancel = Arc::new(AtomicBool::new(false));
    let reloj = RelojGuion::nuevo(ORIGEN_CONSERVACION, Box::new(|_| {}));
    let (cfg, _) = cfg_con(&fixture.root, Arc::clone(&reloj), cancel);
    let c = correr(cfg, &reloj);
    imprimir("a31_inferior", &c, json!({"min": min}));
    assert!(c.panic.is_none());
    assert!(
        matches!(c.aviso, Some(StoreStop::LowDisk { free, minimum }) if free == min - 1 && minimum == min),
        "A31: disco bajo avisado"
    );
    assert!(c.aviso_consumido);
    assert!(
        status_utc(&fixture.root).is_none()
            && !fixture.path(".tgll").exists()
            && !fixture.path(".tgll.partial").exists(),
        "A31: sin STATUS ni recode tras la guarda fallida"
    );
    fixture.assert_gzip_kept()?;
    drop(_probe);
    // Igualdad y descenso durante el trabajo.
    let root = raiz_vacia()?;
    let probe = DiskProbe::new(min);
    let cancel = Arc::new(AtomicBool::new(false));
    let reloj = RelojGuion::nuevo(
        ORIGEN_CONSERVACION,
        Box::new(move |p: &Paso<'_>| {
            if p.vuelta == 1 {
                fijar_disco(min - 1, false);
            }
        }),
    );
    let (cfg, _) = cfg_con(&root, Arc::clone(&reloj), cancel);
    let c = correr(cfg, &reloj);
    imprimir(
        "a31_igualdad_y_descenso",
        &c,
        json!({"min": min, "lecturas": probe.calls()}),
    );
    assert!(c.panic.is_none());
    assert_eq!(
        status_utc(&root),
        Some(UTC0),
        "A31: con disco igual al minimo store publica"
    );
    assert!(
        matches!(c.aviso, Some(StoreStop::LowDisk { .. })),
        "A31: el descenso durante el trabajo para store"
    );
    assert_eq!(c.aviso.as_ref().map(StoreStop::code), Some(3));
    Ok(())
}

/// A32: el sondeo de disco devuelve error: disco incierto, aviso StorageError, sin STATUS.
#[test]
fn a32_sondeo_de_disco_falla() {
    let root = raiz_vacia().expect("raiz");
    let _probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    fijar_disco(crate::MIN_FREE_BYTES, true);
    let cancel = Arc::new(AtomicBool::new(false));
    let reloj = RelojGuion::nuevo(ORIGEN_CONSERVACION, Box::new(|_| {}));
    let (cfg, _) = cfg_con(&root, Arc::clone(&reloj), cancel);
    let c = correr(cfg, &reloj);
    imprimir("a32_sondeo_falla", &c, json!({}));
    assert!(c.panic.is_none());
    assert!(
        matches!(&c.aviso, Some(StoreStop::StorageError(m)) if m.contains("no se puede medir disco")),
        "A32: disco incierto avisado: {:?}",
        c.aviso.as_ref().map(StoreStop::message)
    );
    assert_eq!(c.aviso.as_ref().map(StoreStop::code), Some(1));
    assert!(status_utc(&root).is_none());
}

/// A33: error autentico de la escritura de STATUS (el temporal `STATUS.json.tmp` es una carpeta).
#[test]
fn a33_error_de_escritura_de_status() {
    let root = raiz_vacia().expect("raiz");
    fs::create_dir_all(root.join("STATUS.json.tmp")).expect("carpeta que bloquea el temporal");
    let _probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    let cancel = Arc::new(AtomicBool::new(false));
    let reloj = RelojGuion::nuevo(ORIGEN_CONSERVACION, Box::new(|_| {}));
    let (cfg, _) = cfg_con(&root, Arc::clone(&reloj), cancel);
    let c = correr(cfg, &reloj);
    imprimir("a33_status_falla", &c, json!({}));
    assert!(c.panic.is_none());
    assert!(
        matches!(&c.aviso, Some(StoreStop::StorageError(m)) if m.starts_with("STATUS:")),
        "A33: fallo de STATUS avisado: {:?}",
        c.aviso.as_ref().map(StoreStop::message)
    );
    assert!(
        !c.traza.iter().any(|e| e["ev"] == "esperar"),
        "store para en esa vuelta"
    );
}

/// A21: el reloj inyectado es el que lee `run` (inicio, condicion de STATUS y espera).
#[test]
fn a21_run_usa_el_reloj_inyectado() {
    let root = raiz_vacia().expect("raiz");
    let _probe = DiskProbe::new(crate::MIN_FREE_BYTES);
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = Arc::clone(&cancel);
    let reloj = RelojGuion::nuevo(
        ORIGEN_CONSERVACION,
        Box::new(move |p: &Paso<'_>| {
            if p.vuelta == 2 {
                c2.store(true, Ordering::Release);
            } else {
                p.reloj.fijar(ORIGEN_CONSERVACION + I);
            }
        }),
    );
    let (cfg, _) = cfg_con(&root, Arc::clone(&reloj), cancel);
    let c = correr(cfg, &reloj);
    imprimir("a21_lecturas", &c, json!({}));
    assert!(c.panic.is_none());
    let ev: Vec<&str> = c
        .traza
        .iter()
        .filter_map(|e| e["ev"].as_str())
        .filter(|e| !e.starts_with("guion"))
        .collect();
    // Por vuelta: utc (retencion), mono (condicion de STATUS), esperar.
    assert_eq!(
        ev,
        ["utc", "mono", "esperar", "utc", "mono", "esperar"],
        "A21: lecturas del reloj inyectado en el camino real"
    );
    assert_eq!(status_utc(&root), Some(UTC0), "STATUS con el UTC inyectado");
}
