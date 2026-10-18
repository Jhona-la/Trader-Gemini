//! C-F5 · T1-04 y T1-05 de la revisión de la Línea G (Línea C, CL-115):
//! qué cierres alimentan el freno de la estrategia.
//!
//! El freno contrasta H0 «la estrategia tiene +0,30 R por operación» con la
//! R de cada cierre. Dos clases de cierre no son evidencia de esa hipótesis:
//! - T1-04: una posición ADOPTADA del exchange (sin TP ni SL propios). El
//!   núcleo la gestiona con un stop de respaldo inventado (`sl_at_tau` y el
//!   ATR), y su R saldría de ese stop, no del real;
//! - T1-05: una SONDA (D4, ADR-0016): se abre para medir, por presupuesto y
//!   sin la puerta del valor esperado, no porque la estrategia vea ventaja.
//!
//! Fixture. Un núcleo con aprendizaje local (simulación aislada) y una
//! posición larga a 100 con cierre por stop: a 98,5 con stop propio a 99, o
//! a 90 sin stop propio (el de respaldo está como mucho al 3 %).

use god_engine_core::{outcome_context::OutcomeContext, GodEngineCore};
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use std::{
    path::PathBuf,
    sync::{atomic::Ordering, Arc, Mutex},
};

static ENTORNO: Mutex<()> = Mutex::new(());

/// Directorio temporal propio como directorio de trabajo (el cierre puede
/// escribir archivos relativos).
struct Directorio {
    raiz: PathBuf,
    previo: PathBuf,
}
impl Directorio {
    fn nuevo(caso: &str) -> Self {
        let raiz = std::env::temp_dir().join(format!("cl115-{caso}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&raiz);
        std::fs::create_dir_all(raiz.join("data")).unwrap();
        let previo = std::env::current_dir().unwrap();
        std::env::set_current_dir(&raiz).unwrap();
        Self { raiz, previo }
    }
}
impl Drop for Directorio {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previo).unwrap();
        let _ = std::fs::remove_dir_all(&self.raiz);
    }
}

/// Cierra por stop una posición larga a 100 y devuelve cuántas observaciones
/// recibió el freno. `cierres_previos` es el recuento de la moneda (< 5: fase
/// de sonda); `stop_propio` decide si la posición trae TP y SL.
fn observaciones_del_freno(cierres_previos: usize, stop_propio: bool) -> usize {
    quantum_arena::symbols::update_dynamic_universe(vec!["CLAAAUSDT".into(), "CLBBBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("CLAAAUSDT"),
        symbol_registry::get_official_binance_spec("CLBBBUSDT"),
    ]);
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new_with_outcome_context(arena.clone(), OutcomeContext::IsolatedSimulation);
    core.swing_nn = None;
    core.scalp_forest = None;
    arena.coins[0].metrics.trade_count.store(cierres_previos, Ordering::Relaxed);
    let (tp, sl) = if stop_propio { (101.0, 99.0) } else { (0.0, 0.0) };
    let pos = arena.coins[0].positions.slots()[0];
    assert!(pos.open_with_horizon(true, 100.0, 1.0, 10.0, 1_000, tp, sl, PositionHorizon::Continuous));
    arena.used_margin.store(10.0, Ordering::Relaxed);
    let antes = core.risk_engine.freno_estrategia.ville.count;
    let bid = if stop_propio { 98.5 } else { 90.0 };
    let (_, cierre, _) = core.process_tick_dual(0, bid, bid + 0.01, 5.0, 5.0, 2_000, &[0.0; 54], true, false);
    assert!(cierre.is_some(), "precondición: la posición cierra");
    assert_eq!(
        arena.coins[0].metrics.trade_count.load(Ordering::Relaxed),
        cierres_previos + 1,
        "precondición: el cierre se aprende"
    );
    core.risk_engine.freno_estrategia.ville.count - antes
}

/// Esperado: un cierre con stop propio fuera de la fase de sonda alimenta el
/// freno; una sonda y una posición adoptada sin stop propio, no.
#[test]
fn cl115_solo_alimentan_el_freno_los_cierres_de_la_estrategia() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let _d = Directorio::nuevo("estrategia");
    let estrategia = observaciones_del_freno(10, true);
    let sonda = observaciones_del_freno(0, true);
    let adoptada = observaciones_del_freno(10, false);
    println!("observado: observaciones del freno — estrategia {estrategia}, sonda {sonda}, adoptada sin stop {adoptada}");
    assert_eq!(estrategia, 1, "un cierre de la estrategia cuenta");
    assert_eq!(sonda, 0, "una sonda no es evidencia de la estrategia");
    assert_eq!(adoptada, 0, "sin stop propio la R no es medible");
}

/// Firma del estado tras cerrar una posición larga a 100 por stop, con la
/// moneda fuera de sonda. `confirmada` decide si la entrada existió en el
/// exchange (con el contexto del host, uno de papel no enseña nada).
fn firma_tras_cierre(contexto: OutcomeContext, confirmada: bool) -> (u64, u64) {
    quantum_arena::symbols::update_dynamic_universe(vec!["CLAAAUSDT".into(), "CLBBBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("CLAAAUSDT"),
        symbol_registry::get_official_binance_spec("CLBBBUSDT"),
    ]);
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new_with_outcome_context(arena.clone(), contexto);
    core.swing_nn = None;
    core.scalp_forest = None;
    arena.coins[0].metrics.trade_count.store(10, Ordering::Relaxed);
    let pos = arena.coins[0].positions.slots()[0];
    assert!(pos.open_with_horizon(true, 100.0, 1.0, 10.0, 1_000, 101.0, 99.0, PositionHorizon::Continuous));
    pos.exchange_confirmed.store(confirmada, Ordering::Relaxed);
    arena.used_margin.store(10.0, Ordering::Relaxed);
    let antes = core.firma_estado_aprendido();
    let _ = core.process_tick_dual(0, 98.5, 98.51, 5.0, 5.0, 2_000, &[0.0; 54], true, false);
    (antes, core.firma_estado_aprendido())
}

/// CL-120 (T2-4 y T2-6 de la revisión de la Línea G). Esperado: la firma del
/// estado cambia con un cierre que se aprende y no con uno de papel (el host
/// guardaba también en esos, sin que hubiera nada nuevo), y cambia cuando la
/// reconciliación ajusta un recuento (que antes no disparaba ningún
/// guardado).
#[test]
fn cl120_la_firma_del_estado_cambia_solo_cuando_hay_algo_que_guardar() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let _d = Directorio::nuevo("firma");
    let (a, b) = firma_tras_cierre(OutcomeContext::IsolatedSimulation, false);
    let (c, d) = firma_tras_cierre(OutcomeContext::ExchangeLocalEstimate, false);
    println!("observado: firma con un cierre aprendido {a} → {b}; con un cierre de papel {c} → {d}");
    assert_ne!(a, b, "un cierre aprendido cambia la firma");
    assert_eq!(c, d, "un cierre de papel no");
    // Un ajuste de la reconciliación (recuento de una ranura fantasma).
    let arena = GlobalArena::build_in_own_stack(100.0);
    let core = GodEngineCore::new(arena.clone());
    let antes = core.firma_estado_aprendido();
    arena.coins[1].metrics.trade_count.fetch_add(1, Ordering::Relaxed);
    assert_ne!(core.firma_estado_aprendido(), antes, "un ajuste del recuento cambia la firma");
}

/// CL-120: el host no escribe el estado en el hilo del bucle de eventos.
/// Contrato de fuente (sin comentarios): el único `guardar` está en el hilo
/// escritor, que recibe las fotos por un canal; al apagar se envía la última
/// y se espera a que el escritor termine.
#[test]
fn cl120_el_host_guarda_el_estado_en_un_hilo_aparte() {
    let host: String = include_str!("../../../src/bin/god_engine.rs")
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<String>()
        .split_whitespace()
        .collect();
    assert!(host.contains(
        "let(tx_estado,rx_estado)=std::sync::mpsc::channel::<god_engine_core::estado_aprendido::EstadoAprendido>();"
    ));
    assert_eq!(host.matches("estado_aprendido::guardar(").count(), 1, "un solo sitio escribe el estado");
    assert!(host.contains("escribir_estado_aprendido(&ruta,&foto);"), "y lo llama el hilo escritor");
    assert!(!host.contains(".expect(\"hilo escritor"), "sin hilo escritor no se cae: se escribe en el bucle");
    assert!(
        host.contains("drop(tx_estado);ifletOk(escritor)=escritor_estado{let_=escritor.join();}"),
        "al apagar se espera al escritor"
    );
}
