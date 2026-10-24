//! C-F3 · F3.2 (Línea C): la τ con la que aprende el cierre.
//!
//! Teoría que se prueba:
//! - El Kelly que el riesgo lee para la orden SIGUIENTE es un escalar por
//!   moneda (`metrics.kelly_fraction`), sin τ. Si al cerrar se evalúa la
//!   curva genómica de Kelly a la τ de la operación cerrada, la orden
//!   siguiente hereda el horizonte de la anterior (LC-F1-36; F3.2-04 quedó
//!   retirado como duplicado, este test es su testigo de integración en el
//!   núcleo). Sólo muerde en modo supervivencia (capital/base <
//!   kelly_survival_cap_ratio, risk-engine/src/kelly.rs), que la prueba fija
//!   y comprueba.
//! - Una posición sin τ de decisión (`entry_tau_ms` = 0, p. ej. adoptada
//!   tras un reinicio sin diario) no sabe a qué escala se decidió: su cierre
//!   no puede premiar ni castigar las 32 escalas del espectro (F3.2-06).
//!   Hoy cae al nodo crudo `dominant_tau_ms` o al gen `temporal_scale`.
//!
//! Arnés: el de close_outcome_contract.rs (directorio temporal propio,
//! IsolatedSimulation, posición abierta a mano y cierre defensivo por
//! `process_tick_dual` con entradas vetadas).
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN …"]` con el assert intacto.

use god_engine_core::{outcome_context::OutcomeContext, GodEngineCore};
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use std::{
    path::PathBuf,
    sync::{atomic::Ordering::Relaxed, Arc, Mutex},
};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

/// Directorio de trabajo temporal y único: el cierre puede escribir en
/// `data/`. Sólo se borra el directorio creado aquí (prefijo comprobado).
struct FixtureDirectory {
    root: PathBuf,
    previous: PathBuf,
}
impl FixtureDirectory {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tg-cf3-close-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("data")).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();
        Self { root, previous }
    }
}
impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous).unwrap();
        let exact = self.root.canonicalize().unwrap();
        assert!(exact.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(exact
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("tg-cf3-close-"));
        std::fs::remove_dir_all(exact).unwrap();
    }
}

/// Largo abierto a 100 (TP 101, SL 99), 1 unidad, margen 10, capital 100.
fn core_con_posicion() -> (Arc<GlobalArena>, GodEngineCore) {
    quantum_arena::symbols::update_dynamic_universe(vec!["XXIXUSDT".into(), "XXIYUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("XXIXUSDT"),
        symbol_registry::get_official_binance_spec("XXIYUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core =
        GodEngineCore::new_with_outcome_context(arena.clone(), OutcomeContext::IsolatedSimulation);
    core.swing_nn = None;
    let pos = &arena.coins[0].positions.slot_2;
    assert!(pos.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        1000,
        101.0,
        99.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(10.0, Relaxed);
    (arena, core)
}

/// Cierre defensivo: entradas vetadas, la propuesta de cierre llega.
fn close_at(core: &mut GodEngineCore, coin_id: usize, bid: f64) -> f64 {
    let (entry, proposal, _) = core.process_tick_dual(
        coin_id,
        bid,
        bid + 0.01,
        5.0,
        5.0,
        2000,
        &[0.0; 54],
        true,
        false,
    );
    assert!(entry.is_none());
    let (_, pnl, _) = proposal.expect("el cierre defensivo llega al llamador");
    pnl
}

/// Cierra el largo a 101,1 (ganador, sobre el TP) con `entry_tau_ms` dado y
/// devuelve `metrics.kelly_fraction` tras el cierre. Curva genómica de
/// Kelly creciente: b = ln 2,4/ln 240, a = ln 0,25 − b·ln 60 000, es decir
/// Kelly 0,25 a 60 s y 0,6 a 4 h. Supervivencia: ratio 10 con capital 100 y
/// base 100 (ratio 1 < 10). Historial: 200 operaciones, WR 0,6, PF 3.
fn kelly_tras_cierre(entry_tau_ms: u64) -> f64 {
    let (arena, mut core) = core_con_posicion();
    let cfg = &arena.config;
    cfg.kelly_survival_cap_ratio.store(10.0, Relaxed);
    cfg.kelly_clamp_min.store(0.01, Relaxed);
    cfg.kelly_clamp_max.store(0.5, Relaxed);
    let b = 2.4f64.ln() / 240.0f64.ln();
    cfg.kelly_curve_b.store(b, Relaxed);
    cfg.kelly_curve_a
        .store(0.25f64.ln() - b * 60_000f64.ln(), Relaxed);
    assert!((cfg.kelly_at_tau(60_000.0) - 0.25).abs() < 1e-9);
    assert!((cfg.kelly_at_tau(14_400_000.0) - 0.6).abs() < 1e-9);
    // Precondición (escéptico de LC-F1-36): modo supervivencia.
    let ratio = arena.unified_capital.load(Relaxed) / cfg.base_capital.load(Relaxed).max(1.0);
    assert!(
        ratio < cfg.kelly_survival_cap_ratio.load(Relaxed),
        "precondición: modo supervivencia (capital/base = {ratio})"
    );
    let m = &arena.coins[0].metrics;
    m.trade_count.store(200, Relaxed);
    m.win_rate.store(0.6, Relaxed);
    m.gross_wins.store(300.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    arena.coins[0]
        .positions
        .slot_2
        .entry_tau_ms
        .store(entry_tau_ms, Relaxed);
    close_at(&mut core, 0, 101.1);
    assert_eq!(
        m.trade_count.load(Relaxed),
        201,
        "precondición: el cierre aprende"
    );
    m.kelly_fraction.load(Relaxed)
}

/// Verdad: el Kelly escalar que dimensiona la orden siguiente no puede
/// depender del horizonte de la operación cerrada: mismo valor (|Δ| <
/// 1e-12, sólo redondeo) tras cerrar a 60 s y a 4 h el mismo resultado.
#[test]
#[ignore = "OPEN LC-F1-36 (Línea C, C-F4; F3.2-04 retirado como duplicado): el cierre guarda kelly_at_tau(τ de la operación cerrada) como escalar por moneda y el riesgo lo usa para la orden siguiente, sea cual sea su τ"]
fn cf3_kelly_de_la_moneda_no_depende_de_la_tau_de_la_operacion_cerrada() {
    let _g = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _d = FixtureDirectory::new();
    let corta = kelly_tras_cierre(60_000);
    let larga = kelly_tras_cierre(14_400_000);
    println!("observado: kelly_fraction tras cerrar a 60 s = {corta:.6}; a 4 h = {larga:.6}");
    assert!(
        (corta - larga).abs() < 1e-12,
        "kelly_fraction tras cerrar a 60 s = {corta}; a 4 h = {larga}: la orden siguiente hereda la τ anterior"
    );
}

/// Verdad: una posición sin τ de decisión (entry_tau_ms = 0) cerrada con
/// pérdida (97, bajo el SL de 99) no mueve ninguna de las 32
/// `epigenetic_gain` del espectro (igualdad exacta). Precondición: el cierre
/// aprende (trade_count = 1).
#[test]
fn cf3_posicion_sin_tau_de_decision_no_entrena_el_espectro() {
    let _g = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _d = FixtureDirectory::new();
    let (arena, mut core) = core_con_posicion();
    arena.coins[0]
        .positions
        .slot_2
        .entry_tau_ms
        .store(0, Relaxed);
    let antes: Vec<f64> = core.temporal_spectrum[0]
        .scales
        .iter()
        .map(|s| s.epigenetic_gain)
        .collect();
    close_at(&mut core, 0, 97.0);
    assert_eq!(
        arena.coins[0].metrics.trade_count.load(Relaxed),
        1,
        "precondición: el cierre aprende"
    );
    let despues: Vec<f64> = core.temporal_spectrum[0]
        .scales
        .iter()
        .map(|s| s.epigenetic_gain)
        .collect();
    let movidas: Vec<(usize, f64, f64)> = antes
        .iter()
        .zip(&despues)
        .enumerate()
        .filter(|(_, (a, d))| a != d)
        .map(|(k, (a, d))| (k, *a, *d))
        .collect();
    println!("observado: escalas cuya ganancia cambió (k, antes, después): {movidas:?}");
    assert_eq!(
        antes, despues,
        "una posición sin τ de decisión no debe premiar ni castigar escalas espectrales"
    );
}
