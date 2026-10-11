//! CL-60 — Medidor del camino caliente (plan de Línea C, §7 paso 5).
//!
//! Pasa una cinta sintética fija por `GodEngineCore::process_event` y da, por
//! evento, los ns (mediana, p99, p99,9 y máximo) y las asignaciones de heap.
//! Es la medida que pide la puerta de rendimiento: «asignaciones por evento y
//! ns por evento, mediana y p99, de la base y de la rama, en release, en la
//! misma máquina y con la misma cinta».
//!
//! Por qué no sirven los medidores que ya había:
//! - `nano_benchmark` mide `process_tick_dual`, no `process_event`, y pasa
//!   ciclos a ns suponiendo 3,5 GHz.
//! - `latency_flow_audit` (bin del crate raíz) sí llama a `process_event`,
//!   pero con el precio fijo en 60 000, sólo eventos de libro y una moneda: el
//!   espectro no madura, no hay intenciones y `evaluate_quantum_order` no
//!   corre. Tampoco cuenta asignaciones.
//!
//! Aquí la cinta mezcla trades, libro y cierres de vela de varias monedas,
//! con el mismo reparto de campos que el host vivo (`god_engine.rs`: el trade
//! llega con bid/ask a 0 y las cantidades del último libro; la vela con su
//! cierre; el libro con su mejor nivel). Los ciclos del TSC se pasan a ns con
//! la frecuencia medida sobre el mismo bucle, no supuesta.
//!
//! Uso (release; el test está ignorado para que la CI no lo corra en debug):
//!
//! ```text
//! cargo test --release -p god-engine-core --test medidor_camino_caliente \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Variables: `MEDIDOR_EVENTOS` (medidos, 200 000), `MEDIDOR_CALENTAMIENTO`
//! (sin medir, 50 000), `MEDIDOR_REPETICIONES` (5), `MEDIDOR_MONEDAS` (4),
//! `MEDIDOR_SEMILLA`. Cada repetición construye un núcleo nuevo y repite la
//! misma trayectoria de estado, de modo que lo que difiere entre repeticiones
//! es ruido de la máquina. Cada repetición imprime una línea `MEDIDOR\t…`;
//! `scripts/plan_linea_c.py rendimiento` compara las de la base y la rama.
//!
//! El fichero es autónomo a propósito: `scripts/medir_camino_caliente.sh` lo
//! copia tal cual a un worktree de la base para medirla con la misma cinta.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Asignador que cuenta
// ---------------------------------------------------------------------------

/// Cuenta toda petición de memoria nueva (`alloc`, `alloc_zeroed`,
/// `realloc`) de cualquier hilo. No usa estado por hilo a propósito: en
/// windows-gnu un `thread_local!` puede reservar memoria la primera vez que
/// se toca, y hacerlo desde el asignador sería reentrante. El ruido de otros
/// hilos se mide aparte (`asignaciones_de_fondo_por_s`).
struct ContadorDeAsignaciones;

static ASIGNACIONES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for ContadorDeAsignaciones {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ASIGNACIONES.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ASIGNACIONES.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ASIGNACIONES.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ASIGNADOR: ContadorDeAsignaciones = ContadorDeAsignaciones;

#[inline(always)]
fn asignaciones() -> u64 {
    ASIGNACIONES.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Reloj
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn ciclos() -> u64 {
    // SAFETY: RDTSC existe en todo x86_64.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Fuera de x86_64 no hay TSC: el «ciclo» es un ns de `Instant` desde un
/// origen fijo, y la calibración da 1 ns por ciclo.
#[cfg(not(target_arch = "x86_64"))]
fn ciclos() -> u64 {
    use std::sync::OnceLock;
    static ORIGEN: OnceLock<Instant> = OnceLock::new();
    ORIGEN.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Cinta sintética
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tipo {
    Trade,
    Libro,
    Vela,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Evento {
    moneda: usize,
    tipo: Tipo,
    cierra_vela: bool,
    precio: f64,
    cantidad: f64,
    bid: f64,
    ask: f64,
    bid_qty: f64,
    ask_qty: f64,
    obi: f64,
    t_ms: u64,
    comprador_maker: bool,
}

/// Generador congruencial de 64 bits (constantes de Knuth, MMIX). Basta para
/// una cinta reproducible; no se le pide calidad estadística.
struct Lcg(u64);

impl Lcg {
    fn siguiente(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniforme en [0, 1) con los 53 bits altos.
    fn uniforme(&mut self) -> f64 {
        (self.siguiente() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Normal aproximada: suma de 4 uniformes centrada y con varianza 1
    /// (Irwin–Hall). Las colas están truncadas en ±2√3, que basta aquí.
    fn normal(&mut self) -> f64 {
        let s: f64 = (0..4).map(|_| self.uniforme()).sum();
        (s - 2.0) * 3.0f64.sqrt()
    }
}

/// Parámetros de la cinta. Están fijados para que la cinta ejercite el camino
/// completo y no para imitar un mercado concreto:
/// - 25 eventos/s en total (40 ms de media entre eventos, exponencial), es
///   decir, ~6 por moneda y segundo con 4 monedas;
/// - 55 % trades, 45 % libro, y una vela cerrada por moneda y minuto;
/// - volatilidad de ~15 pb por minuto y un ciclo de 3 h con deriva de ±30 pb
///   por hora, para que el espectro vea tendencia y el núcleo emita
///   intenciones (sin intenciones `evaluate_quantum_order` no corre y la
///   medida se queda corta; con el genoma y los bosques de `nucleo` y
///   `preparar_universo` salen unas 13 por cada 50 000 eventos);
/// - medio spread de 0,5 pb y libro de ~20 000 USD por lado.
const MEDIA_ENTRE_EVENTOS_MS: f64 = 40.0;
const PROB_LIBRO: f64 = 0.45;
const VOL_POR_MINUTO: f64 = 0.0015;
const DERIVA_POR_HORA: f64 = 0.0030;
const PERIODO_CICLO_MS: f64 = 3.0 * 3_600_000.0;
const MEDIO_SPREAD: f64 = 0.000_05;
const LIBRO_USD: f64 = 20_000.0;
const TRADE_USD: f64 = 2_000.0;
const T0_MS: u64 = 60_000;

fn precio_inicial(moneda: usize) -> f64 {
    // Escalas de precio distintas por moneda (BTC, ETH, SOL, BNB…), para que
    // ningún cálculo dependa por accidente de un nivel de precio.
    const BASES: [f64; 6] = [60_000.0, 3_000.0, 150.0, 550.0, 0.6, 25.0];
    BASES[moneda % BASES.len()]
}

fn cinta(n: usize, monedas: usize, semilla: u64) -> Vec<Evento> {
    assert!(monedas >= 1, "la cinta necesita al menos una moneda");
    let mut azar = Lcg(semilla);
    let mut t = T0_MS;
    let mut ln_p: Vec<f64> = (0..monedas).map(|m| precio_inicial(m).ln()).collect();
    let mut t_moneda = vec![T0_MS; monedas];
    let mut minuto = vec![T0_MS / 60_000; monedas];
    let mut libro: Vec<(f64, f64)> = (0..monedas)
        .map(|m| {
            let q = LIBRO_USD / precio_inicial(m);
            (q, q)
        })
        .collect();
    let eventos_por_minuto_y_moneda = 60_000.0 / MEDIA_ENTRE_EVENTOS_MS / monedas as f64;
    let sigma_evento = VOL_POR_MINUTO / eventos_por_minuto_y_moneda.sqrt();
    let mut salida = Vec::with_capacity(n);

    while salida.len() < n {
        let espera = (-(1.0 - azar.uniforme()).ln() * MEDIA_ENTRE_EVENTOS_MS).round();
        t += (espera as u64).max(1);
        let m = (azar.siguiente() >> 33) as usize % monedas;

        // Una vela por moneda y minuto: si el reloj de la moneda cruzó un
        // minuto, el evento es el cierre de la vela anterior.
        if t / 60_000 > minuto[m] {
            minuto[m] = t / 60_000;
            let p = ln_p[m].exp();
            let (bq, aq) = libro[m];
            salida.push(Evento {
                moneda: m,
                tipo: Tipo::Vela,
                cierra_vela: true,
                precio: p,
                cantidad: TRADE_USD * 30.0 / p,
                bid: 0.0,
                ask: 0.0,
                bid_qty: bq,
                ask_qty: aq,
                obi: 0.0,
                t_ms: t,
                comprador_maker: false,
            });
            continue;
        }

        let dt = (t - t_moneda[m]) as f64;
        t_moneda[m] = t;
        let fase = 2.0 * std::f64::consts::PI * t as f64 / PERIODO_CICLO_MS;
        let deriva = DERIVA_POR_HORA * fase.sin() * dt / 3_600_000.0;
        let paso = deriva + sigma_evento * azar.normal();
        ln_p[m] += paso;
        let p = ln_p[m].exp();

        if azar.uniforme() < PROB_LIBRO {
            let base = LIBRO_USD / p;
            let bq = base * (0.5 + azar.uniforme());
            let aq = base * (0.5 + azar.uniforme());
            libro[m] = (bq, aq);
            salida.push(Evento {
                moneda: m,
                tipo: Tipo::Libro,
                cierra_vela: false,
                precio: p,
                cantidad: 0.0,
                bid: p * (1.0 - MEDIO_SPREAD),
                ask: p * (1.0 + MEDIO_SPREAD),
                bid_qty: bq,
                ask_qty: aq,
                obi: (bq - aq) / (bq + aq),
                t_ms: t,
                comprador_maker: false,
            });
        } else {
            let (bq, aq) = libro[m];
            let q = -(1.0 - azar.uniforme()).ln() * TRADE_USD / p;
            salida.push(Evento {
                moneda: m,
                tipo: Tipo::Trade,
                cierra_vela: false,
                precio: p,
                cantidad: q,
                bid: 0.0,
                ask: 0.0,
                bid_qty: bq,
                ask_qty: aq,
                obi: 0.0,
                t_ms: t,
                // Una bajada la empuja el vendedor agresor: el comprador era
                // el que tenía la orden en el libro.
                comprador_maker: paso < 0.0,
            });
        }
    }
    salida
}

// ---------------------------------------------------------------------------
// Estadística
// ---------------------------------------------------------------------------

/// Percentil por rango más cercano sobre una muestra ORDENADA: el menor valor
/// con al menos una fracción `q` de la muestra en o por debajo de él.
fn percentil(ordenados: &[u64], q: f64) -> u64 {
    assert!(!ordenados.is_empty(), "percentil de una muestra vacía");
    assert!((0.0..=1.0).contains(&q), "q fuera de [0, 1]: {q}");
    let n = ordenados.len();
    let rango = (q * n as f64).ceil() as usize;
    ordenados[rango.clamp(1, n) - 1]
}

#[derive(Debug, Clone, PartialEq)]
struct Resumen {
    eventos: usize,
    mediana_ns: f64,
    p99_ns: f64,
    p999_ns: f64,
    max_ns: f64,
    media_ns: f64,
    asignaciones_por_evento: f64,
    fraccion_con_asignacion: f64,
    max_asignaciones_evento: u32,
}

/// Resume una repetición. `ciclos_evento` y `asign_evento` van en paralelo;
/// `ns_por_ciclo` es la calibración medida sobre el mismo bucle.
fn resumir(ciclos_evento: &[u64], asign_evento: &[u32], ns_por_ciclo: f64) -> Resumen {
    assert_eq!(ciclos_evento.len(), asign_evento.len());
    assert!(!ciclos_evento.is_empty());
    assert!(ns_por_ciclo.is_finite() && ns_por_ciclo > 0.0);
    let n = ciclos_evento.len();
    let mut ordenados = ciclos_evento.to_vec();
    ordenados.sort_unstable();
    let a_ns = |c: u64| c as f64 * ns_por_ciclo;
    let total_asign: u64 = asign_evento.iter().map(|&a| a as u64).sum();
    Resumen {
        eventos: n,
        mediana_ns: a_ns(percentil(&ordenados, 0.5)),
        p99_ns: a_ns(percentil(&ordenados, 0.99)),
        p999_ns: a_ns(percentil(&ordenados, 0.999)),
        max_ns: a_ns(*ordenados.last().unwrap()),
        media_ns: ordenados.iter().map(|&c| a_ns(c)).sum::<f64>() / n as f64,
        asignaciones_por_evento: total_asign as f64 / n as f64,
        fraccion_con_asignacion: asign_evento.iter().filter(|&&a| a > 0).count() as f64
            / n as f64,
        max_asignaciones_evento: asign_evento.iter().copied().max().unwrap_or(0),
    }
}

// ---------------------------------------------------------------------------
// Banco
// ---------------------------------------------------------------------------

const SIMBOLOS: [&str; 6] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ATOMUSDT"];

/// Registra el universo, los specs y un bosque por moneda. Sin spec,
/// `evaluate_quantum_order` rechaza toda orden (B3.20); sin bosque del roster
/// el núcleo sólo abre la sonda de arranque frío. En los dos casos el camino
/// medido se queda corto. Los specs son los de `get_official_binance_spec`
/// con las comisiones del spec nativo del backtest. El bosque es el
/// predictor sintético direccional del T-1 (un corte en la dimensión 0,
/// ±3 según el signo, base 0,5): pasa por la inferencia que corre en vivo.
fn preparar_universo(monedas: usize) {
    assert!(monedas <= SIMBOLOS.len(), "como mucho {} monedas", SIMBOLOS.len());
    let simbolos: Vec<String> = SIMBOLOS[..monedas].iter().map(|s| s.to_string()).collect();
    quantum_arena::symbols::update_dynamic_universe(simbolos.clone());
    let specs = simbolos
        .iter()
        .map(|s| {
            let mut spec = quantum_arena::symbol_registry::get_official_binance_spec(s);
            spec.maker_fee = 0.0002;
            spec.taker_fee = 0.0005;
            spec
        })
        .collect();
    quantum_arena::symbol_registry::update_registry(specs);
    for s in &simbolos {
        let datos = god_engine_core::ml_inference::NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.0, 0.0, 0.0],
            value: vec![0.0, -3.0, 3.0],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let bosque = god_engine_core::ml_inference::NanoForest::from_data(datos)
            .expect("bosque sintético fuera de contrato");
        god_engine_core::ml_inference::NanoForest::store_global(&format!("{s}_MOTOR"), bosque);
    }
}

/// Núcleo con el genoma base del T-1. Con la configuración por defecto del
/// arena, sin genoma aplicado, ninguna intención llega al riesgo (medido:
/// 0 intenciones en 250 000 eventos).
fn nucleo() -> GodEngineCore {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    quantum_arena::genome::SuperGenotype::new_baseline(0.0002, 0.0005).apply_to_arena(&arena);
    // Lo que el host escribe antes de cada evento y no cambia en la cinta.
    arena.registry.set("macro_staleness_ms", 0.0);
    GodEngineCore::new(arena)
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct Salidas {
    intenciones: usize,
    cierres: usize,
}

#[inline(always)]
fn alimentar(core: &mut GodEngineCore, e: &Evento, omni: &[f64; 54], s: &mut Salidas) {
    let (nueva, cerrada) = core.process_event(
        e.moneda,
        e.tipo == Tipo::Trade,
        e.cierra_vela,
        e.tipo == Tipo::Libro,
        e.precio,
        e.cantidad,
        e.bid,
        e.ask,
        e.bid_qty,
        e.ask_qty,
        e.obi,
        0.0,
        e.t_ms,
        false,
        omni,
        e.comprador_maker,
    );
    s.intenciones += nueva.is_some() as usize;
    s.cierres += cerrada.is_some() as usize;
    black_box((nueva, cerrada));
}

fn entero_de_entorno(nombre: &str, defecto: u64) -> u64 {
    match std::env::var(nombre) {
        Ok(v) => v
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{nombre}={v:?} no es un entero")),
        Err(_) => defecto,
    }
}

// ---------------------------------------------------------------------------
// Tests que corre la CI (debug): el instrumento y la cinta
// ---------------------------------------------------------------------------

#[test]
fn percentil_por_rango_mas_cercano() {
    let v: Vec<u64> = (1..=100).collect();
    assert_eq!(percentil(&v, 0.5), 50);
    assert_eq!(percentil(&v, 0.99), 99);
    assert_eq!(percentil(&v, 1.0), 100);
    assert_eq!(percentil(&v, 0.0), 1);
    assert_eq!(percentil(&[7], 0.99), 7);
    // Con 10 elementos el p99 es el máximo: no se interpola.
    let w: Vec<u64> = (1..=10).collect();
    assert_eq!(percentil(&w, 0.99), 10);
    assert_eq!(percentil(&w, 0.5), 5);
}

#[test]
fn resumen_convierte_con_la_calibracion_y_cuenta_asignaciones() {
    let ciclos = [30, 10, 20, 40];
    let asign = [0, 2, 0, 1];
    let r = resumir(&ciclos, &asign, 0.5);
    assert_eq!(r.eventos, 4);
    assert_eq!(r.mediana_ns, 10.0); // rango 2 de [10, 20, 30, 40] = 20 ciclos
    assert_eq!(r.max_ns, 20.0);
    assert_eq!(r.media_ns, 12.5);
    assert_eq!(r.asignaciones_por_evento, 0.75);
    assert_eq!(r.fraccion_con_asignacion, 0.5);
    assert_eq!(r.max_asignaciones_evento, 2);
}

#[test]
fn el_asignador_cuenta_las_reservas() {
    let antes = asignaciones();
    let v: Vec<u64> = black_box(Vec::with_capacity(16));
    let b = black_box(Box::new(5u64));
    let despues = asignaciones();
    drop((v, b));
    assert!(despues >= antes + 2, "{antes} -> {despues}");
}

#[test]
fn la_cinta_es_reproducible_y_coherente() {
    let a = cinta(20_000, 4, 7);
    assert_eq!(a, cinta(20_000, 4, 7), "misma semilla, misma cinta");
    assert_ne!(a, cinta(20_000, 4, 8), "otra semilla, otra cinta");
    assert_eq!(a.len(), 20_000);

    let mut t_previo = 0;
    let mut velas = [0usize; 4];
    let mut tipos = [0usize; 3];
    for e in &a {
        assert!(e.t_ms > t_previo, "el tiempo avanza estrictamente");
        t_previo = e.t_ms;
        assert!(e.moneda < 4);
        assert!(e.precio.is_finite() && e.precio > 0.0);
        assert!(e.bid_qty > 0.0 && e.ask_qty > 0.0);
        match e.tipo {
            Tipo::Libro => {
                assert!(e.bid > 0.0 && e.bid < e.ask, "libro sin cruzar");
                assert!(e.obi.abs() < 1.0);
                tipos[1] += 1;
            }
            Tipo::Trade => {
                assert!(e.cantidad > 0.0 && e.bid == 0.0 && e.ask == 0.0);
                tipos[0] += 1;
            }
            Tipo::Vela => {
                assert!(e.cierra_vela);
                velas[e.moneda] += 1;
                tipos[2] += 1;
            }
        }
    }
    // ~800 s de cinta: unas 13 velas por moneda.
    let minutos = (t_previo - T0_MS) / 60_000;
    for v in velas {
        assert!(v as u64 + 1 >= minutos && v as u64 <= minutos + 1, "{velas:?} en {minutos} min");
    }
    let libro = tipos[1] as f64 / (tipos[0] + tipos[1]) as f64;
    assert!((libro - PROB_LIBRO).abs() < 0.02, "fracción de libro {libro}");
}

#[test]
fn la_cinta_pasa_por_el_nucleo_sin_romperlo() {
    // Humo en debug: el banco puede alimentar la cinta al núcleo. La medida
    // de verdad es `medir_camino_caliente`, en release.
    preparar_universo(4);
    let mut core = nucleo();
    let omni = [0.0f64; 54];
    let mut s = Salidas::default();
    for e in cinta(3_000, 4, 11).iter() {
        alimentar(&mut core, e, &omni, &mut s);
    }
    assert!(s.intenciones <= 3_000 && s.cierres <= 3_000);
}

// ---------------------------------------------------------------------------
// La medida (release, ignorada en la CI)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "banco: cargo test --release ... -- --ignored --nocapture"]
fn medir_camino_caliente() {
    let medidos = entero_de_entorno("MEDIDOR_EVENTOS", 200_000) as usize;
    let calentamiento = entero_de_entorno("MEDIDOR_CALENTAMIENTO", 50_000) as usize;
    let repeticiones = entero_de_entorno("MEDIDOR_REPETICIONES", 5) as usize;
    let monedas = entero_de_entorno("MEDIDOR_MONEDAS", 4) as usize;
    let semilla = entero_de_entorno("MEDIDOR_SEMILLA", 0x5DEE_CE66D);
    assert!(medidos > 0 && repeticiones > 0);
    if cfg!(debug_assertions) {
        eprintln!("[MEDIDOR] AVISO: build debug; la puerta exige --release");
    }

    preparar_universo(monedas);
    let eventos = cinta(calentamiento + medidos, monedas, semilla);
    let omni = [0.0f64; 54];
    let mut ciclos_evento = vec![0u64; medidos];
    let mut asign_evento = vec![0u32; medidos];

    // Asignaciones de otros hilos sin trabajo en este: si no es 0, las
    // asignaciones por evento llevan ruido de fondo.
    let a0 = asignaciones();
    std::thread::sleep(Duration::from_millis(300));
    let fondo_por_s = (asignaciones() - a0) as f64 / 0.3;

    println!(
        "MEDIDOR_CAB\trep\teventos\tmediana_ns\tp99_ns\tp999_ns\tmax_ns\tmedia_ns\t\
         asign_por_evento\tfraccion_con_asign\tmax_asign_evento\tintenciones\tcierres\t\
         ghz\tfondo_asign_por_s\tmonedas\tsemilla\tdebug"
    );
    for rep in 0..repeticiones {
        let mut core = nucleo();
        let mut s = Salidas::default();
        for e in &eventos[..calentamiento] {
            alimentar(&mut core, e, &omni, &mut s);
        }
        let s_calentamiento = s;

        let t_ini = Instant::now();
        let c_ini = ciclos();
        for (i, e) in eventos[calentamiento..].iter().enumerate() {
            let a = asignaciones();
            let c = ciclos();
            alimentar(&mut core, e, &omni, &mut s);
            ciclos_evento[i] = ciclos() - c;
            asign_evento[i] = (asignaciones() - a).min(u32::MAX as u64) as u32;
        }
        let c_total = ciclos() - c_ini;
        let ns_total = t_ini.elapsed().as_nanos() as f64;
        let ns_por_ciclo = ns_total / c_total as f64;
        let r = resumir(&ciclos_evento, &asign_evento, ns_por_ciclo);
        println!(
            "MEDIDOR\t{rep}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.3}\t{:.4}\t{}\t{}\t{}\t{:.3}\t{:.1}\t{monedas}\t{semilla}\t{}",
            r.eventos,
            r.mediana_ns,
            r.p99_ns,
            r.p999_ns,
            r.max_ns,
            r.media_ns,
            r.asignaciones_por_evento,
            r.fraccion_con_asignacion,
            r.max_asignaciones_evento,
            s.intenciones - s_calentamiento.intenciones,
            s.cierres - s_calentamiento.cierres,
            1.0 / ns_por_ciclo,
            fondo_por_s,
            cfg!(debug_assertions) as u8,
        );

        // Desglose por tipo de evento (sólo informativo).
        for tipo in [Tipo::Trade, Tipo::Libro, Tipo::Vela] {
            let mut c: Vec<u64> = Vec::new();
            let mut a: Vec<u32> = Vec::new();
            for (i, e) in eventos[calentamiento..].iter().enumerate() {
                if e.tipo == tipo {
                    c.push(ciclos_evento[i]);
                    a.push(asign_evento[i]);
                }
            }
            if !c.is_empty() {
                let r = resumir(&c, &a, ns_por_ciclo);
                println!(
                    "MEDIDOR_TIPO\t{rep}\t{tipo:?}\t{}\t{:.1}\t{:.1}\t{:.3}",
                    r.eventos, r.mediana_ns, r.p99_ns, r.asignaciones_por_evento
                );
            }
        }
    }

    // Coste unitario de operaciones que el camino caliente repite muchas
    // veces por evento (sólo informativo): leer el reloj de pared, que el
    // registro hace en cada escritura desde #720, y escribir y leer una clave
    // por moneda en un registro ya poblado por el calentamiento.
    const VECES: u32 = 1_000_000;
    let por_llamada = |t: Instant| t.elapsed().as_nanos() as f64 / VECES as f64;
    let t = Instant::now();
    for _ in 0..VECES {
        black_box(chrono::Utc::now());
    }
    println!("MEDIDOR_MICRO\tchrono::Utc::now\t{:.1}", por_llamada(t));
    let mut core = nucleo();
    let mut s = Salidas::default();
    for e in &eventos[..calentamiento] {
        alimentar(&mut core, e, &omni, &mut s);
    }
    let registro = &core.arena.registry;
    println!("MEDIDOR_MICRO\tclaves_del_registro\t{}", registro.scan_all().len());
    registro.set_for_coin(0, "medidor_clave", 0.0);
    let t = Instant::now();
    for i in 0..VECES {
        registro.set_for_coin(0, "medidor_clave", black_box(i as f64));
    }
    println!("MEDIDOR_MICRO\tregistry.set_for_coin\t{:.1}", por_llamada(t));
    let t = Instant::now();
    for _ in 0..VECES {
        black_box(registro.get_for_coin_or(0, black_box("medidor_clave"), 0.0));
    }
    println!("MEDIDOR_MICRO\tregistry.get_for_coin_or\t{:.1}", por_llamada(t));
}
