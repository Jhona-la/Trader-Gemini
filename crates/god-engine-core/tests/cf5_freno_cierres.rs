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
    let pos = &arena.coins[0].positions.position;
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
