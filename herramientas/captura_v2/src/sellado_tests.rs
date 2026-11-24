//! FALLAS-3 (pedido 20261123-0013, filas A46-A47): presion inversa del sellado sobre la cola de
//! archivo con el escritor (`archive`) y el sellador (`seal_segments`) de produccion, sobre archivos
//! reales en `.retention-tests` (sin borrar nada) y la cola de produccion derivada.
//!
//! Criterio declarado antes de ejecutar (A47): el candidato NO desacopla la cola del sellado; se
//! admite presion inversa finita. Se exige: (1) con el primer sello retenido por barrera dentro de
//! `Segment::seal` (fase gzip_finish, con acuse) y SEAL_QUEUE_CAPACITY segmentos ya entregados, el
//! escritor queda bloqueado en el siguiente `seal.send` y la testigo no se desencola antes de liberar;
//! (2) las llegadas que siguen durante la retencion se admiten sin rechazo, dentro de B y de las
//! plazas; (3) tras liberar el sello, el atraso vuelve a cero y cada fila queda archivada una vez, en
//! orden de ingest_seq. Los tiempos medidos se informan (`MEDIDA_SELLADO`); no son cifras operativas.

use super::*;
use std::io::BufRead;
use std::sync::{Condvar, mpsc as std_mpsc};

struct SalidaNula;
impl salidafallo::Salida for SalidaNula {
    fn terminar(&self, _codigo: u32, _ordenada: bool) {}
}

/// Disco real con el primer `gzip_finish` retenido hasta que el test lo libere.
struct DiscoRetenido {
    real: RealDisk,
    estado: Arc<(Mutex<(bool, bool)>, Condvar)>, // (retenido_ahora, liberado)
}

impl Disk for DiscoRetenido {
    type File = File;
    fn create(&self, path: &Path) -> io::Result<File> {
        self.real.create(path)
    }
    fn sync(&self, file: &File) -> io::Result<()> {
        self.real.sync(file)
    }
    fn digest(&self, path: &Path) -> CaptureResult<(u64, String)> {
        self.real.digest(path)
    }
    fn publish(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.real.publish(from, to)
    }
    fn write_json(&self, path: &Path, value: &Value) -> CaptureResult<()> {
        self.real.write_json(path, value)
    }
    fn antes_de_fase(&self, fase: &'static str) -> io::Result<()> {
        if fase != "gzip_finish" {
            return Ok(());
        }
        let (m, cv) = &*self.estado;
        let mut g = m.lock().map_err(|_| io::Error::other("envenenado"))?;
        if !g.1 {
            g.0 = true; // acuse: el sellador esta dentro de Segment::seal
            cv.notify_all();
            while !g.1 {
                g = cv.wait(g).map_err(|_| io::Error::other("envenenado"))?;
            }
        }
        Ok(())
    }
}

fn fila(id: u64, ts: u64) -> Value {
    json!({"kind": "frame", "route": "market", "shard": 0, "connection": 1, "seq": id + 1,
        "recv_utc_ms": ts, "recv_mono_ns": id, "oferta_id": id,
        "stream": "fstusdt@aggTrade", "symbol": "FSTUSDT", "payload": format!("{{\"a\":{id}}}")})
}

fn esperar_hasta(que: &str, limite: Duration, cond: impl Fn() -> bool) -> Duration {
    let t = Instant::now();
    while !cond() {
        assert!(t.elapsed() < limite, "no se cumplio dentro del limite de la prueba: {que}");
        thread::yield_now();
    }
    t.elapsed()
}

#[test]
fn a47_presion_inversa_del_sellado_con_testigo() -> CaptureResult<()> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".retention-tests").join(format!("sellado-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&dir)?;
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ctl = Arc::new(Control::nuevo(salidafallo::PLAZO_FATAL, Box::new(salidafallo::RelojMonotono::nuevo()),
        Box::new(SalidaNula), Arc::new(salidafallo::CompuertaStore::real(cancel))));
    let (queue, rx, cap) = ArchiveQueue::derivada(QUEUE_BYTE_BUDGET, ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision))
        .map_err(io::Error::other)?;
    let audit = Arc::clone(&queue.audit);
    let estado = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let disco = DiscoRetenido { real: RealDisk, estado: Arc::clone(&estado) };
    let (seal_tx, seal_rx) = std_mpsc::sync_channel(SEAL_QUEUE_CAPACITY);
    let (d2, c2) = (Arc::new(disco), Arc::clone(&ctl));
    let d3 = Arc::clone(&d2);
    let dir2 = dir.clone();
    let sellador = thread::spawn(move || seal_segments(&*d3, &dir2, seal_rx, None, Some(&c2)));
    let (c3, dir3) = (Arc::clone(&ctl), dir.clone());
    let escritor = thread::spawn(move || archive(&*d2, &dir3, rx, seal_tx, &c3));
    let limite = Duration::from_secs(60); // limite de la prueba, no cifra del colector
    let deq = || audit.dequeued.load(Ordering::SeqCst);
    let acc = || audit.accepted.load(Ordering::SeqCst);
    let h = |n: u64| (100 + n) * HOUR_MS;
    let mut id = 0u64;
    let mut ofrecer = |ts: u64| -> CaptureResult<u64> {
        let k = id;
        queue.offer(fila(k, ts), None)?;
        id += 1;
        Ok(k)
    };
    // 1) Hora 1 y una fila de la hora 2 pasada la gracia: el escritor entrega el segmento 1 y el
    // sellador queda retenido en gzip_finish (acuse).
    ofrecer(h(1) + 1)?;
    ofrecer(h(2) + ROTATION_GRACE_MS)?;
    {
        let (m, cv) = &*estado;
        let mut g = m.lock().expect("estado");
        while !g.0 {
            g = cv.wait(g).expect("estado");
        }
    }
    // 2) SEAL_QUEUE_CAPACITY entregas mas llenan el canal; la siguiente rotacion deja al escritor en
    // el siguiente seal.send.
    let rotaciones = SEAL_QUEUE_CAPACITY as u64 + 1;
    for n in 0..rotaciones {
        ofrecer(h(3 + n) + ROTATION_GRACE_MS)?;
    }
    let antes = 2 + rotaciones as usize;
    esperar_hasta("el escritor desencola la fila que provoca el envio bloqueado", limite, || deq() == antes);
    // 3) Testigo y llegadas activas durante la retencion.
    let hora_abierta = h(3 + rotaciones - 1) + ROTATION_GRACE_MS + 1;
    let testigo = ofrecer(hora_abierta)?;
    let llegadas = 20_000u64;
    for k in 0..llegadas {
        ofrecer(hora_abierta + 1 + k / 100)?;
    }
    let t_obs = Instant::now();
    while t_obs.elapsed() < Duration::from_millis(300) {
        assert_eq!(deq(), antes, "A47: con el sello retenido y el canal lleno la testigo no se desencola");
        thread::yield_now();
    }
    let (ocupadas, cargados) = (queue.tx.max_capacity() - queue.tx.capacity(), audit.charged_bytes().unwrap_or(0));
    assert!(audit.failure_reason().is_none(), "A47: sin rechazo durante la retencion");
    assert!(cargados <= QUEUE_BYTE_BUDGET && ocupadas <= cap.filas);
    // 4) Liberar el sello: independiente del retorno del envio.
    let t_lib = Instant::now();
    {
        let (m, cv) = &*estado;
        m.lock().expect("estado").1 = true;
        cv.notify_all();
    }
    let a_testigo = esperar_hasta("la testigo se desencola tras liberar", limite, || deq() > antes);
    let total = acc();
    let a_cero = esperar_hasta("el atraso vuelve a cero", limite, || deq() == total);
    drop(ofrecer);
    drop(queue);
    // Parada sana como en produccion: el corte de admision despierta al escritor (park) y este drena y
    // termina; soltar el emisor no lo despierta (el primer intento de esta prueba se colgo ahi).
    ctl.admision.cortar(salidafallo::Motivo::Sana);
    let out = escritor.join().expect("escritor");
    let rep = sellador.join().expect("sellador");
    // 5) Conciliacion: cada fila archivada una vez y en orden de ingest_seq.
    let mut ids = Vec::new();
    let mut seqs = Vec::new();
    let mut nombres: Vec<_> = fs::read_dir(&dir)?.flatten().map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".jsonl.gz")).collect();
    nombres.sort();
    for p in &nombres {
        for l in std::io::BufReader::new(flate2::read::GzDecoder::new(File::open(p)?)).lines() {
            let v: Value = serde_json::from_str(&l?)?;
            ids.push(v["oferta_id"].as_u64().unwrap_or(u64::MAX));
            seqs.push(v["ingest_seq"].as_u64().unwrap_or(0));
        }
    }
    let mut unicos = ids.clone();
    unicos.sort_unstable();
    unicos.dedup();
    seqs.sort_unstable();
    println!("MEDIDA_SELLADO {}", json!({"caso": "a47", "criterio": "presion_inversa_finita",
        "seal_queue_capacity": SEAL_QUEUE_CAPACITY, "segmentos_entregados_antes_del_bloqueo": 1 + SEAL_QUEUE_CAPACITY,
        "testigo": testigo, "desencoladas_antes_de_liberar": antes, "llegadas_durante_retencion": llegadas,
        "ocupadas_durante_retencion": ocupadas, "bytes_cargados_durante_retencion": cargados,
        "liberacion_a_testigo_ns": a_testigo.as_nanos() as u64, "liberacion_a_atraso_cero_ns": (a_testigo + a_cero).as_nanos() as u64,
        "pared_liberacion_total_ns": t_lib.elapsed().as_nanos() as u64,
        "aceptadas": total, "archivadas": ids.len(), "identidades_unicas": unicos.len(),
        "escritor": out.json(), "segmentos_sellados": rep.segments, "fallos_sellado": rep.failures}));
    assert_eq!(ids.len(), total, "A47/A51: todas las aceptadas archivadas");
    assert_eq!(unicos.len(), total, "A40: ninguna identidad repetida");
    assert_eq!(seqs, (1..=total as u64).collect::<Vec<_>>(), "A41: ingest_seq unico y sin huecos");
    assert!(ids.contains(&testigo));
    assert!(rep.failures.is_empty() && out.failures.is_empty());
    assert_eq!(audit.charged_bytes(), Some(0), "A42: reservas devueltas");
    Ok(())
}
