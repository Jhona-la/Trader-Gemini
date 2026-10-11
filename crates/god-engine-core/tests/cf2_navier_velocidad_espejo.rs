//! C-F2.6 (Línea C): simetría espejo (G-TEO §30.1.3) del voto escalar vivo de
//! `NavierStokesManifoldEngine` y del signo de `price_velocity`.
//!
//! Teoría: un mercado espejado debe producir el voto opuesto. En la rama
//! turbulenta aislada (laminar = 0, OFI = 0) el voto es tanh(100·velocidad),
//! así que la velocidad de un precio que cae debe ser negativa y el voto de
//! un mercado que cae sin desequilibrio de libro no puede ser largo. Los
//! caminos son deterministas y monótonos: el signo verdadero de la velocidad
//! es conocido. Hoy `price_velocity` (stateful_engine.rs, `dir_velocity`)
//! es la EWMA(0,15) de |Δmid|, sin signo: ≈ +0,05 en los dos caminos, y el
//! voto ≈ +0,9999 en los dos.
//!
//! Siguiendo al escéptico de refutar, el productor y el consumidor van en
//! pruebas separadas: un arreglo correcto del consumidor (AGY) no depende de
//! que se arregle el productor (FMT-001, dueño de stateful_engine), y al
//! revés. El camino bajista es el espejo LOGARÍTMICO del alcista
//! (bid_baja = p0²/ask_sube, ask_baja = p0²/bid_sube), para que un voto con
//! retorno firmado normalizado por σ sea antisimétrico.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use signal_engine::navier_stokes_manifold::NavierStokesManifoldEngine;
use std::sync::Mutex;
use strategy_core::QuantumStrategy;

/// Los tramos cambian el universo y el registro de símbolos globales.
static ENTORNO_GLOBAL: Mutex<()> = Mutex::new(());

/// Mid inicial del tramo alcista: p0 del espejo logarítmico.
const P0: f64 = 100.005;

/// Libro del evento i: alcista bid = 100 + 0,05·i, ask = bid + 0,01; el
/// bajista es su espejo logarítmico respecto a p0.
fn libro(i: u64, sube: bool) -> (f64, f64) {
    let bid = 100.0 + 0.05 * i as f64;
    let ask = bid + 0.01;
    if sube {
        (bid, ask)
    } else {
        (P0 * P0 / ask, P0 * P0 / bid)
    }
}

/// 50 eventos de depth separados 1 s en una arena nueva; devuelve
/// (`price_velocity` publicada, voto escalar de Navier con la rama
/// turbulenta aislada: laminar = 0 y OFI = 0 en el ámbito del símbolo).
fn tramo(sym: &str, sube: bool) -> (f64, f64) {
    symbols::update_dynamic_universe(vec![sym.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(sym)]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    for i in 0..50u64 {
        let (bid, ask) = libro(i, sube);
        let _ = core.process_tick_dual(
            0,
            bid,
            ask,
            5.0,
            5.0,
            10_000 + 1_000 * i,
            &[0.0; 54],
            true,
            false,
        );
    }
    let velocidad = arena
        .registry
        .get_for_coin_or(0, "price_velocity", f64::NAN);
    arena.registry.set_scoped(sym, "navier_laminar_share", 0.0);
    arena.registry.set_scoped(sym, "order_flow_imbalance", 0.0);
    let mut ns = NavierStokesManifoldEngine::new();
    ns.init(arena.registry.clone()).expect("init");
    (velocidad, ns.evaluate_for_coin(0, sym))
}

/// Consumidor (pedido a AGY). Verdad: voto_sube > 0, voto_baja < 0 y
/// |voto_sube + voto_baja| ≤ 0,05·|voto_sube|. La tolerancia relativa del 5 %
/// cubre la diferencia de magnitud entre los dos tramos (el espejo es exacto
/// en log-precio, no en dólares) y la del pequeño cambio de spread; con un
/// voto saturado en tanh la diferencia real es mucho menor. Hoy: +0,9999 en
/// los dos.
#[test]
#[ignore = "OPEN F2.6-01 (AGY, consumidor NavierStokesManifoldEngine): el voto escalar vivo lee una velocidad sin signo y vota largo en cualquier mercado turbulento"]
fn cf2_6_navier_voto_escalar_es_antisimetrico_con_el_mercado_espejado() {
    let _guarda = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    let (v_sube, voto_sube) = tramo("CF26SUBEUSDT", true);
    let (v_baja, voto_baja) = tramo("CF26BAJAUSDT", false);
    assert!(
        voto_sube > 0.0
            && voto_baja < 0.0
            && (voto_sube + voto_baja).abs() <= 0.05 * voto_sube.abs(),
        "espejo roto: voto sube {voto_sube:.6}, baja {voto_baja:.6} (price_velocity sube {v_sube:.6}, baja {v_baja:.6})"
    );
}

/// Productor (FMT-001, pedido al dueño de stateful_engine; aviso de paridad
/// con el bosque del daemon, que también consume la velocidad). Verdad: la
/// velocidad de un precio que sube 5 céntimos por evento es positiva y la
/// de su espejo, negativa. Hoy: 0,05·(1 − 0,85^49) ≈ +0,04998 en los dos.
#[test]
#[ignore = "OPEN F2.6-01 (núcleo, productor FMT-001 en stateful_engine.rs): price_velocity es la EWMA de |Δmid| y no cambia de signo con el mercado"]
fn cf2_6_price_velocity_cambia_de_signo_con_el_mercado() {
    let _guarda = ENTORNO_GLOBAL.lock().unwrap_or_else(|p| p.into_inner());
    let (v_sube, _) = tramo("CF26VSUBEUSDT", true);
    let (v_baja, _) = tramo("CF26VBAJAUSDT", false);
    assert!(
        v_sube > 0.0 && v_baja < 0.0,
        "price_velocity con el precio subiendo {v_sube:.6} y bajando {v_baja:.6} (debe cambiar de signo)"
    );
}
