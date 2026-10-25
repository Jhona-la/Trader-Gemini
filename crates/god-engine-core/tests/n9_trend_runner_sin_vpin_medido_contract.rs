//! N9 (censo C-F7b; CL-174): sin VPIN MEDIDO, la sombra espectral de
//! TrendRunner se abstiene (espectro nulo), con la regla de K12 (CL-158).
//!
//! Antes la sombra leía `cvpin` con un 0,50 por defecto: una toxicidad media
//! que nadie midió. En `voto_espectral_con_sigma` el VPIN sólo contrae la
//! convicción (penalización max(1 − tanh(1,5·v), 0,2) ∈ [0,2; 1]), nunca la
//! anula ni le cambia el signo, así que ningún valor expresa «desconocido».
//! El motor devuelve un VPIN no finito a 0,50, por eso la guarda vive en el
//! núcleo. En los caminos de producción el 0,50 sólo se leía en el primer
//! evento de un núcleo nuevo, con H aún en 0,5 (peso de persistencia nulo) y
//! el espectro recién sembrado (lo fija la tercera prueba). Para que el
//! contrato sea observable, la primera prueba siembra el espectro y H
//! directamente, sin pasar por el dual (que escribiría `cvpin`).
//! Eventos sintéticos locales; sin exchange ni red.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use signal_engine::trend_runner::HighPayoffTrendRunner;
use std::sync::{Arc, Mutex};

const SIM: &str = "N9TRENDUSDT";
const H: f64 = 0.75;
const SE: f64 = 0.05;

/// El universo y el registro de símbolos son globales del proceso.
static ENTORNO: Mutex<()> = Mutex::new(());

fn nucleo() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    // Las sombras se publican en todo evento, no sólo bajo el muestreo F1-043.
    arena.registry.set("telemetria_sombras_forzada", 1.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    (arena, core)
}

/// Diez pasos alcistas de 5 pb, uno por segundo, directos al espectro de la
/// moneda 0: el registro no recibe `cvpin` ni ninguna otra clave.
fn sembrar_espectro(core: &mut GodEngineCore) {
    for i in 0..10u64 {
        core.temporal_spectrum[0].update(100.0 + 0.05 * i as f64, 10_000 + 1_000 * i);
    }
}

/// Persistencia medida por moneda: el dual de cada evento la reescribe con
/// la del DFA, así que se fija justo antes del evento, que es cuando la lee
/// la sombra.
fn fijar_hurst(arena: &GlobalArena) {
    arena.registry.set_for_coin(0, "hurst_exponent", H);
    arena.registry.set_for_coin(0, "hurst_std_error", SE);
}

/// Evento de profundidad puro (sin trade): compone las sombras espectrales.
fn profundidad(core: &mut GodEngineCore, mid: f64, t_ms: u64) {
    let _ = core.process_event(
        0,
        false,
        false,
        true,
        mid,
        0.0,
        mid,
        mid + 0.01,
        5.0,
        5.0,
        0.1,
        0.0,
        t_ms,
        false,
        &[0.0; 54],
        false,
    );
}

/// Los desplazamientos con los que votó la sombra: momentum_z de cada escala
/// con el recorte del núcleo (bloque #609 de lib.rs). El dual no vuelve a
/// actualizar el espectro dentro del mismo evento (F1-076).
fn desplazamientos(core: &GodEngineCore) -> [f64; 32] {
    let mut x = [0.0f64; 32];
    for (k, s) in core.temporal_spectrum[0].scales.iter().enumerate() {
        x[k] = if s.momentum_z.is_finite() {
            s.momentum_z.clamp(-10.0, 10.0)
        } else {
            0.0
        };
    }
    x
}

fn media_trend(x: &[f64; 32], vpin: f64) -> f64 {
    HighPayoffTrendRunner::voto_espectral_con_sigma(x, H, SE, vpin)
        .media_banda(0, 31)
        .unwrap_or(f64::NAN)
}

/// RED antes de CL-174: la sombra publicaba el voto con VPIN = 0,50 (no nulo
/// en esta malla, con H = 0,75) y su escala dominante. El mismo voto entra en
/// votos_espectrales[MOTOR_TREND_RUNNER].
#[test]
fn n9_sin_vpin_medido_la_sombra_trend_se_abstiene() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    sembrar_espectro(&mut core);
    fijar_hurst(&arena);
    assert_eq!(
        arena.registry.get_for_coin_strict(0, "cvpin"),
        None,
        "cvpin por moneda"
    );
    assert_eq!(arena.registry.get_value_fast("cvpin"), None, "cvpin global");

    profundidad(&mut core, 100.55, 20_000);

    let x = desplazamientos(&core);
    assert!(
        x.iter().any(|v| v.abs() >= 1e-6),
        "la malla sembrada tiene desplazamientos: {x:?}"
    );
    let con_literal = media_trend(&x, 0.50);
    assert!(
        con_literal.is_finite() && con_literal != 0.0,
        "el 0,50 retirado votaba: {con_literal}"
    );

    assert_eq!(
        arena
            .registry
            .get_for_coin_strict(0, "sombra_trend_consenso"),
        Some(0.0),
        "sin VPIN medido la sombra TrendRunner se abstiene (con el 0,50: {con_literal})"
    );
    assert_eq!(
        arena
            .registry
            .get_for_coin_strict(0, "sombra_trend_tau_max"),
        None
    );
    assert_eq!(
        arena.registry.get_for_coin_strict(0, "sombra_trend_v_max"),
        None
    );
}

/// Verde antes y después: con `cvpin` medido, la sombra es bit a bit el voto
/// del motor con esa toxicidad, y la toxicidad discrimina (0,2 frente a 0,8
/// en la misma malla). Si la sombra dejara de leer `cvpin`, cae.
#[test]
fn n9_con_vpin_medido_la_sombra_trend_es_el_motor_con_esa_toxicidad() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    sembrar_espectro(&mut core);
    for (i, cvpin) in [0.2f64, 0.8].into_iter().enumerate() {
        // El dual reescribe `cvpin` (0 sin trades), H y su error típico: se
        // fijan justo antes del evento.
        arena.registry.set_for_coin(0, "cvpin", cvpin);
        fijar_hurst(&arena);
        profundidad(
            &mut core,
            100.55 + 0.05 * i as f64,
            20_000 + 1_000 * i as u64,
        );
        let x = desplazamientos(&core);
        assert!(
            media_trend(&x, 0.2).abs() > media_trend(&x, 0.8).abs() + 0.01,
            "la toxicidad discrimina en esta malla: {} frente a {}",
            media_trend(&x, 0.2),
            media_trend(&x, 0.8)
        );
        let esperado = media_trend(&x, cvpin);
        let publicado = arena
            .registry
            .get_for_coin_strict(0, "sombra_trend_consenso");
        assert_eq!(
            publicado.map(f64::to_bits),
            Some(esperado.to_bits()),
            "cvpin {cvpin}: publicado {publicado:?}, esperado {esperado}"
        );
    }
}

/// El argumento de conducta de CL-174, comprobado. En el primer evento de un
/// núcleo nuevo (el único en que falta `cvpin`) H tampoco está publicado y el
/// espectro recién sembrado tiene x ≡ 0, así que el voto ya era nulo; y al
/// terminar ese evento el dual ya publicó `cvpin` y H por moneda y global.
/// Verde antes y después. Si cae, la elección de N9 pasa a ser observable y
/// hay que medirla con golden y T-1.
#[test]
fn n9_el_unico_evento_sin_cvpin_tiene_hurst_neutro_y_espectro_recien_sembrado() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    assert_eq!(arena.registry.get_value_fast("cvpin"), None);
    assert_eq!(
        arena.registry.get_for_coin_strict(0, "hurst_exponent"),
        None
    );

    profundidad(&mut core, 100.0, 10_000);

    assert!(
        desplazamientos(&core).iter().all(|v| *v == 0.0),
        "x ≡ 0 en el primer evento"
    );
    assert_eq!(
        arena
            .registry
            .get_for_coin_strict(0, "sombra_trend_consenso"),
        Some(0.0)
    );
    assert_eq!(
        arena
            .registry
            .get_for_coin_strict(0, "sombra_trend_tau_max"),
        None
    );
    for clave in ["cvpin", "hurst_exponent"] {
        let por_moneda = arena.registry.get_for_coin_strict(0, clave);
        let global = arena.registry.get_value_fast(clave);
        assert!(
            por_moneda.is_some_and(f64::is_finite),
            "{clave} por moneda: {por_moneda:?}"
        );
        assert!(
            global.is_some_and(f64::is_finite),
            "{clave} global: {global:?}"
        );
    }
}
