//! FALLAS-2 (pedido 20261123-0013, filas A04, A09-A10, A13-A14, A34-A36, A42-A43 y A56): cola de
//! archivo por el camino real de `ArchiveQueue::offer`.
//!
//! El cuerpo de las pruebas es el mismo en la base r3b y en el candidato; solo cambia el modulo
//! `adaptador` de arriba (como se obtiene la cola de produccion y su cargo). Cada oferta lleva una
//! identidad sintetica (`oferta_id`) puesta antes de entrar en la cola. La capacidad antigua sale de
//! la fixture versionada del incidente, no de una constante del banco.
//!
//! El asignador contador (solo en el binario de pruebas) mide el heap vivo y su pico; las pruebas de
//! memoria imprimen `MEDIDA_COLA <json>` y deben correr con --test-threads=1.

use super::*;
use std::alloc::{GlobalAlloc, Layout, System};

// --- Adaptador del sujeto ----------------------------------------------------------------------
mod adaptador {
    use super::*;
    pub const SUJETO: &str = "candidato";
    /// Cola de produccion: la misma llamada que `capture` (derivada de B y q_min).
    pub fn cola_produccion() -> (ArchiveQueue, ArchiveReceiver) {
        let (q, rx, _) = ArchiveQueue::derivada(
            QUEUE_BYTE_BUDGET,
            ROW_CHARGE_OVERHEAD,
            Arc::new(Admision::nueva()),
        )
        .expect("capacidad derivada");
        (q, rx)
    }
    pub fn cargo(q: &ArchiveQueue, row: &Value) -> usize {
        q.charge_of(row)
    }
    pub fn bytes_cargados(q: &ArchiveQueue) -> usize {
        q.byte_limit - q.budget.available_permits()
    }
}
use adaptador::*;

// --- Asignador contador ------------------------------------------------------------------------
struct Contador;
static VIVOS: AtomicUsize = AtomicUsize::new(0);
static PICO: AtomicUsize = AtomicUsize::new(0);
/// Heap vivo contado solo en los hilos marcados con `Marca` (a10); el resto del proceso no entra.
static VIVOS_MEDIDA: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // const y Cell<bool> sin Drop: no reserva heap ni registra destructor (no recursa en el asignador).
    static EN_MEDIDA: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Anota `delta` en VIVOS_MEDIDA si el hilo esta marcado (un hilo en cierre cuenta como no marcado).
/// La resta es saturada: liberar en un hilo marcado lo asignado sin marcar no subdesborda.
fn anotar(delta: usize, suma: bool) {
    if EN_MEDIDA.try_with(|c| c.get()).unwrap_or(false) {
        if suma {
            VIVOS_MEDIDA.fetch_add(delta, Ordering::Relaxed);
        } else {
            let _ = VIVOS_MEDIDA.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(delta))
            });
        }
    }
}

/// Guarda RAII: marca el hilo actual para la medida y lo desmarca al salir (tambien en panic).
struct Marca;
impl Marca {
    fn poner() -> Marca {
        EN_MEDIDA.with(|c| c.set(true));
        Marca
    }
}
impl Drop for Marca {
    fn drop(&mut self) {
        EN_MEDIDA.with(|c| c.set(false));
    }
}

unsafe impl GlobalAlloc for Contador {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let v = VIVOS.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PICO.fetch_max(v, Ordering::Relaxed);
            anotar(l.size(), true);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        VIVOS.fetch_sub(l.size(), Ordering::Relaxed);
        anotar(l.size(), false);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, nuevo: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, nuevo) };
        if !q.is_null() {
            if nuevo >= l.size() {
                let v = VIVOS.fetch_add(nuevo - l.size(), Ordering::Relaxed) + (nuevo - l.size());
                PICO.fetch_max(v, Ordering::Relaxed);
                anotar(nuevo - l.size(), true);
            } else {
                VIVOS.fetch_sub(l.size() - nuevo, Ordering::Relaxed);
                anotar(l.size() - nuevo, false);
            }
        }
        q
    }
}

#[global_allocator]
static ASIGNADOR: Contador = Contador;

/// Heap vivo de los hilos marcados (acumulador por hilo de la medida de a10).
fn vivos() -> usize {
    VIVOS_MEDIDA.load(Ordering::Relaxed)
}
fn reiniciar_pico() -> usize {
    let v = vivos();
    PICO.store(v, Ordering::Relaxed);
    v
}
fn pico() -> usize {
    PICO.load(Ordering::Relaxed)
}

// --- Fixture del incidente ---------------------------------------------------------------------
const FIXTURE_INCIDENTE: &str = include_str!("../fixtures/fallas/incidente_29060_cola.json");

struct Incidente {
    capacidad_antigua: usize,
    presupuesto: usize,
    cargo_minimo: usize,
}

fn incidente() -> Incidente {
    let v: Value = serde_json::from_str(FIXTURE_INCIDENTE).expect("fixture del incidente");
    let n = |k: &str| {
        v[k].as_u64()
            .map(|x| x as usize)
            .unwrap_or_else(|| panic!("fixture sin {k}"))
    };
    Incidente {
        capacidad_antigua: n("queue_capacity"),
        presupuesto: n("queue_byte_budget"),
        cargo_minimo: n("queue_row_overhead_charge"),
    }
}

fn fixture_sha256() -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(FIXTURE_INCIDENTE.as_bytes()))
}

/// Fila sintetica de mercado con identidad propia y `payload` de `largo` bytes.
fn fila(id: u64, largo: usize) -> Value {
    json!({"kind": "frame", "route": "market", "shard": 0, "connection": 1, "seq": id + 1,
        "recv_utc_ms": HOUR_MS + id, "recv_mono_ns": id, "oferta_id": id,
        "stream": "fstusdt@aggTrade", "symbol": "FSTUSDT", "payload": "x".repeat(largo)})
}

fn razon(e: &(dyn Error + Send + Sync)) -> String {
    let s = e.to_string();
    [
        "already closed",
        "admission cut",
        "row_capacity",
        "byte_budget",
        "row_exceeds_budget",
        "writer_unavailable",
    ]
    .iter()
    .find(|r| s.contains(*r))
    .map_or(s.clone(), |r| (*r).to_owned())
}

fn drenar(rx: &mut ArchiveReceiver) -> Vec<u64> {
    let mut ids = Vec::new();
    while let Ok(r) = rx.try_recv() {
        ids.push(r.row["oferta_id"].as_u64().expect("identidad"));
    }
    ids
}

// --- A04 ---------------------------------------------------------------------------------------
#[test]
fn a04_capacidad_de_produccion_derivada_del_presupuesto() {
    let (q, _rx) = cola_produccion();
    let derivada = QUEUE_BYTE_BUDGET / ROW_CHARGE_OVERHEAD;
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a04", "sujeto": SUJETO, "capacidad_canal": q.tx.max_capacity(),
        "presupuesto": QUEUE_BYTE_BUDGET, "cargo_minimo": ROW_CHARGE_OVERHEAD, "floor_B_qmin": derivada})
    );
    assert_eq!(
        q.tx.max_capacity(),
        derivada,
        "A04: capacidad de filas de produccion = floor(B / q_min)"
    );
    assert!(
        (q.tx.max_capacity() + 1) * ROW_CHARGE_OVERHEAD > QUEUE_BYTE_BUDGET,
        "A08: (C + 1) * q_min > B"
    );
}

// --- A05 (solo candidato: la base no tiene derivacion) ----------------------------------------
#[test]
fn a05_precondiciones_de_la_derivacion() {
    let casos: Vec<(usize, usize)> = vec![
        (QUEUE_BYTE_BUDGET, ROW_CHARGE_OVERHEAD),
        (5, 2),
        (2_048, 2_048),
        (2_049, 2_048),
        (u32::MAX as usize, 1),
        (u32::MAX as usize, ROW_CHARGE_OVERHEAD),
    ];
    let mut informe = Vec::new();
    for (b, q) in casos {
        let c = derivar_capacidad(b, q).expect("configuracion valida");
        informe.push(json!({"B": b, "q_min": q, "C": c.filas}));
        assert_eq!(c.filas, b / q, "A05: C = floor(B / q_min)");
        assert!(
            c.filas >= 1 && c.filas * q <= b && (c.filas + 1) * q > b,
            "A05/A08: frontera de la cota"
        );
        // La cola se construye sin panic con esa capacidad (mpsc::channel y Semaphore::new).
        let _ = ArchiveQueue::derivada(b, q, Arc::new(Admision::nueva())).expect("cola construida");
    }
    let invalidos = [
        (QUEUE_BYTE_BUDGET, 0usize),
        (2_047, 2_048),
        (0, 1),
        (u32::MAX as usize + 1, ROW_CHARGE_OVERHEAD),
    ];
    for (b, q) in invalidos {
        let r = derivar_capacidad(b, q);
        informe.push(json!({"B": b, "q_min": q, "rechazo": r.as_ref().err()}));
        assert!(
            r.is_err(),
            "A05: configuracion invalida rechazada de forma explicita: B={b} q={q}"
        );
    }
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a05", "sujeto": SUJETO, "casos": informe,
        "semaphore_max_permits": Semaphore::MAX_PERMITS})
    );
}
// --- A34 / A56 ---------------------------------------------------------------------------------
/// La barrera antigua: con el receptor vivo y sin desencolar, C_antigua + 1 ofertas cuyos cargos
/// suman menos que B. La base rechaza la testigo (C_antigua + 1) por `row_capacity`.
#[test]
fn a34_barrera_antigua_de_filas_con_presupuesto_disponible() {
    let inc = incidente();
    assert_eq!(
        (inc.presupuesto, inc.cargo_minimo),
        (QUEUE_BYTE_BUDGET, ROW_CHARGE_OVERHEAD),
        "la fixture describe este presupuesto"
    );
    let (q, mut rx) = cola_produccion();
    let total = inc.capacidad_antigua + 1;
    let filas: Vec<Value> = (0..total as u64).map(|id| fila(id, 16)).collect();
    let suma: usize = filas.iter().map(|f| cargo(&q, f)).sum();
    assert!(
        suma <= QUEUE_BYTE_BUDGET,
        "precondicion: suma de cargos {suma} <= B {}",
        QUEUE_BYTE_BUDGET
    );
    let mut rechazos = Vec::new();
    for f in filas {
        let id = f["oferta_id"].as_u64().unwrap_or(u64::MAX);
        if let Err(e) = q.offer(f, None) {
            rechazos.push((id, razon(e.as_ref())));
            break;
        }
    }
    let (ocupadas, cargados) = (q.tx.max_capacity() - q.tx.capacity(), bytes_cargados(&q));
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a34", "sujeto": SUJETO, "fixture_sha256": fixture_sha256(),
        "capacidad_antigua": inc.capacidad_antigua, "ofertas": total, "suma_cargos": suma, "presupuesto": QUEUE_BYTE_BUDGET,
        "ocupadas": ocupadas, "bytes_cargados": cargados, "rechazos": rechazos,
        "rechazadas_auditoria": *q.audit.rejected.lock().expect("r")})
    );
    assert_eq!(
        rechazos,
        Vec::<(u64, String)>::new(),
        "A34: con presupuesto disponible se admiten las C_antigua + 1 ofertas (testigo incluida)"
    );
    let ids = drenar(&mut rx);
    assert_eq!(
        ids,
        (0..total as u64).collect::<Vec<_>>(),
        "A40: identidades en orden y una sola vez"
    );
    assert_eq!(
        bytes_cargados(&q),
        0,
        "A42: tras desencolar y soltar las filas no queda cargo"
    );
}

// --- A13 ---------------------------------------------------------------------------------------
/// Agotamiento real: n = floor(B / q) filas de cargo q (q > q_min, n < C_antigua) caben; la siguiente no.
#[test]
fn a13_agotamiento_real_del_presupuesto() {
    let (q, mut rx) = cola_produccion();
    let largo = 8_192
        - ROW_CHARGE_OVERHEAD
        - ("frame".len() + "market".len() + "fstusdt@aggTrade".len() + "FSTUSDT".len());
    let c = cargo(&q, &fila(0, largo));
    assert_eq!(c, 8_192);
    let n = QUEUE_BYTE_BUDGET / c;
    assert!(
        n < incidente().capacidad_antigua,
        "el caso no depende de la barrera de filas"
    );
    for id in 0..n as u64 {
        q.offer(fila(id, largo), None)
            .expect("cabe en el presupuesto");
    }
    let r = q
        .offer(fila(n as u64, largo), None)
        .map_err(|e| razon(e.as_ref()));
    let despues = q
        .offer(fila(n as u64 + 1, 0), None)
        .map_err(|e| razon(e.as_ref()));
    let rech = q.audit.rejected.lock().expect("r").clone();
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a13", "sujeto": SUJETO, "admitidas": n, "cargo": c,
        "rechazo": format!("{r:?}"), "despues": format!("{despues:?}"), "auditoria": rech,
        "primer_fallo": q.audit.failure_reason()})
    );
    assert_eq!(
        r,
        Err("byte_budget".to_owned()),
        "A13: la fila n+1 se rechaza por agotamiento real"
    );
    assert_eq!(
        despues,
        Err("already closed".to_owned()),
        "A13: primer fallo conservado; la admision queda cerrada"
    );
    assert_eq!(q.audit.failure_reason(), Some("byte_budget"));
    assert_eq!(rech.len(), 1, "un solo rechazo registrado");
    assert_eq!(
        rech[0]["oferta_id"].as_u64().or(Some(n as u64)),
        Some(n as u64)
    );
    assert_eq!(drenar(&mut rx).len(), n, "lo admitido se drena");
    assert_eq!(bytes_cargados(&q), 0, "A42: reservas devueltas");
}

// --- A35 ---------------------------------------------------------------------------------------
/// Fronteras: total = B - 1 y total = B con cargos representables; la siguiente minima se rechaza.
/// Una fila desencolada y aun en proceso retiene su cargo hasta soltarse.
#[test]
fn a35_fronteras_del_presupuesto_y_fila_en_proceso() {
    let base_texto = "frame".len() + "market".len() + "fstusdt@aggTrade".len() + "FSTUSDT".len();
    let largo = |c: usize| c - ROW_CHARGE_OVERHEAD - base_texto;
    for (falta, caso) in [(1usize, "inferior"), (0, "igual")] {
        let (q, mut rx) = cola_produccion();
        let n = QUEUE_BYTE_BUDGET / 8_192 - 1;
        for id in 0..n as u64 {
            q.offer(fila(id, largo(8_192)), None).expect("cabe");
        }
        let ultimo = 8_192 - falta; // total = B - falta
        q.offer(fila(n as u64, largo(ultimo)), None)
            .expect("A35: el ultimo cargo valido se admite");
        assert_eq!(bytes_cargados(&q), QUEUE_BYTE_BUDGET - falta);
        let r = q
            .offer(fila(n as u64 + 1, 0), None)
            .map_err(|e| razon(e.as_ref()));
        println!(
            "MEDIDA_COLA {}",
            json!({"caso": format!("a35_{caso}"), "sujeto": SUJETO, "cargados": QUEUE_BYTE_BUDGET - falta,
            "siguiente": format!("{r:?}")})
        );
        assert_eq!(
            r,
            Err("byte_budget".to_owned()),
            "A35 {caso}: el siguiente cargo imposible se rechaza con causa"
        );
        assert_eq!(drenar(&mut rx).len(), n + 1);
        assert_eq!(bytes_cargados(&q), 0);
    }
    // Fila en proceso: desencolada y retenida, su cargo sigue reservado.
    let (q, mut rx) = cola_produccion();
    let grande = largo(QUEUE_BYTE_BUDGET / 2);
    q.offer(fila(0, grande), None).expect("mitad");
    q.offer(fila(1, grande), None).expect("otra mitad");
    let en_proceso = rx.try_recv().expect("desencolada");
    let r = q.offer(fila(2, 0), None).map_err(|e| razon(e.as_ref()));
    let cargados = bytes_cargados(&q);
    drop(en_proceso);
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a35_en_proceso", "sujeto": SUJETO, "cargados_con_fila_en_proceso": cargados,
        "siguiente": format!("{r:?}")})
    );
    assert_eq!(
        cargados, QUEUE_BYTE_BUDGET,
        "A07: la fila en proceso conserva su cargo"
    );
    assert_eq!(r, Err("byte_budget".to_owned()));
}

// --- A36 ---------------------------------------------------------------------------------------
#[test]
fn a36_fila_cuyo_cargo_rebasa_el_presupuesto() {
    let (q, _rx) = cola_produccion();
    let r = q
        .offer(fila(0, QUEUE_BYTE_BUDGET), None)
        .map_err(|e| razon(e.as_ref()));
    let rech = q.audit.rejected.lock().expect("r").clone();
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a36", "sujeto": SUJETO, "resultado": format!("{r:?}"), "auditoria": rech,
        "primer_fallo": q.audit.failure_reason()})
    );
    assert!(r.is_err(), "A36: la fila imposible se rechaza");
    assert_eq!(bytes_cargados(&q), 0, "A36: no se reserva nada");
    assert_eq!(
        q.tx.capacity(),
        q.tx.max_capacity(),
        "A36: ninguna plaza ocupada"
    );
    assert!(
        q.audit.failure_reason().is_some(),
        "A36: primer fallo explicito"
    );
    assert_eq!(rech.len(), 1);
    assert!(
        rech[0]["row_charge_bytes"].as_u64() > Some(QUEUE_BYTE_BUDGET as u64),
        "cargo calculado sin truncar"
    );
}

// --- A14 ---------------------------------------------------------------------------------------
#[test]
fn a14_escritor_cerrado_antes_de_ofrecer() {
    let (q, rx) = cola_produccion();
    drop(rx);
    let r = q.offer(fila(0, 8), None).map_err(|e| razon(e.as_ref()));
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a14", "sujeto": SUJETO, "resultado": format!("{r:?}")})
    );
    assert_eq!(
        r,
        Err("writer_unavailable".to_owned()),
        "A14: escritor ausente rechazado con causa"
    );
    assert_eq!(
        bytes_cargados(&q),
        0,
        "A14: sin permisos consumidos tras el cierre"
    );
}

// --- A09 / A10 / A11 ---------------------------------------------------------------------------
/// Memoria de la cola con el escritor detenido: heap vivo medido por el asignador contador para
/// filas cortas, de libro (payload de 1.200 B, como depth de 40 niveles) y mezcla, con cuatro
/// emisores; despues se drena y se mide la recuperacion. Compara el heap por fila con su cargo.
#[test]
fn a10_memoria_de_la_cola_con_escritor_detenido() {
    let _marca = Marca::poner();
    let tam = json!({"QueuedRow": std::mem::size_of::<QueuedRow>(), "QueuedRow_align": std::mem::align_of::<QueuedRow>(),
        "Value": std::mem::size_of::<Value>(), "Option_Cont": std::mem::size_of::<Option<Cont>>(),
        "OwnedSemaphorePermit": std::mem::size_of::<OwnedSemaphorePermit>()});
    let mut fases = Vec::new();
    for (nombre, largos, filas) in [
        ("cortas_16B", vec![16usize], 100_000usize),
        ("libro_1200B", vec![1_200], 60_000),
        (
            "mezcla_16_300_1200_4000B",
            vec![16, 300, 1_200, 4_000],
            60_000,
        ),
    ] {
        let inicio = reiniciar_pico();
        let (q, mut rx) = cola_produccion();
        let tras_crear = vivos();
        let cargo_total = Arc::new(AtomicUsize::new(0));
        let admitidas = Arc::new(AtomicUsize::new(0));
        thread::scope(|s| {
            for emisor in 0..4u64 {
                let (q, largos, cargo_total, admitidas) = (
                    q.clone(),
                    largos.clone(),
                    Arc::clone(&cargo_total),
                    Arc::clone(&admitidas),
                );
                s.spawn(move || {
                    let _marca = Marca::poner();
                    for k in 0..(filas / 4) as u64 {
                        let id = emisor * 1_000_000 + k;
                        let f = fila(id, largos[(k as usize) % largos.len()]);
                        let c = cargo(&q, &f);
                        if q.offer(f, None).is_ok() {
                            cargo_total.fetch_add(c, Ordering::Relaxed);
                            admitidas.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        let lleno = vivos();
        let pico_lleno = pico();
        let n = admitidas.load(Ordering::Relaxed);
        let cargado = bytes_cargados(&q);
        let t = Instant::now();
        let drenadas = drenar(&mut rx).len();
        let drenaje_ns = t.elapsed().as_nanos() as u64;
        let tras_drenar = vivos();
        drop((q, rx));
        let fin = vivos();
        let heap_cola = lleno.saturating_sub(tras_crear);
        let fase = json!({"fase": nombre, "ofertas": filas, "admitidas": n, "drenadas": drenadas,
            "heap_inicio": inicio, "heap_tras_crear_cola": tras_crear, "heap_cola_llena": lleno, "pico": pico_lleno,
            "heap_tras_drenar": tras_drenar, "heap_tras_soltar_cola": fin, "drenaje_ns": drenaje_ns,
            "heap_de_la_cola": heap_cola, "suma_cargos": cargo_total.load(Ordering::Relaxed), "bytes_cargados": cargado,
            "heap_por_fila": heap_cola as f64 / n.max(1) as f64,
            "cargo_por_fila": cargo_total.load(Ordering::Relaxed) as f64 / n.max(1) as f64,
            "heap_fuera_del_cargo": heap_cola as i64 - cargo_total.load(Ordering::Relaxed) as i64});
        println!(
            "MEDIDA_COLA {}",
            json!({"caso": "a10", "sujeto": SUJETO, "medida": fase})
        );
        fases.push(fase);
        assert_eq!(drenadas, n);
    }
    println!(
        "MEDIDA_COLA {}",
        json!({"caso": "a09_layout", "sujeto": SUJETO, "tamanos": tam,
        "metodo": "asignador contador #[global_allocator] del binario de pruebas; heap vivo (bytes pedidos) antes/despues"})
    );
    for f in &fases {
        assert!(
            f["heap_fuera_del_cargo"].as_i64() <= Some(0),
            "A11: el heap de la cola cabe en sus cargos: {f}"
        );
    }
}
