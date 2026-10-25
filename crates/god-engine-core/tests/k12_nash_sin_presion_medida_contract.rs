//! K12 (censo C-F7b, lote 27; CL-158): sin presión adversarial MEDIDA, la
//! sombra espectral de Nash se abstiene (espectro nulo) en vez de jugar
//! contra un adversario inventado.
//!
//! Antes la cadena acababa en `nash_equilibrium_drift`, una clave sin
//! escritor, con un 0,50 literal. El voto por escala es
//! tanh(x_k·(1 − p·k/31)) con p ∈ [0; 0,9]: ninguna p expresa «adversario
//! desconocido», porque la escala 0 vota tanh(x₀) con cualquier p. En los
//! caminos de producción el 0,50 sólo se leía en el primer evento de un
//! núcleo nuevo, con el espectro recién sembrado (x ≡ 0: voto nulo con
//! cualquier p; lo fija la tercera prueba). Para que el contrato sea
//! observable, la primera prueba siembra el espectro directamente, sin
//! pasar por el núcleo (que escribiría `cvpin`).
//! Eventos sintéticos locales; sin exchange ni red.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use signal_engine::game_theoretic_nash::GameTheoreticNashEngine;
use std::sync::{Arc, Mutex};

const SIM: &str = "K12NASHUSDT";

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
/// moneda 0: el registro no recibe `cvpin` ni ninguna otra presión.
fn sembrar_espectro(core: &mut GodEngineCore) {
    for i in 0..10u64 {
        core.temporal_spectrum[0].update(100.0 + 0.05 * i as f64, 10_000 + 1_000 * i);
    }
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

fn media_nash(x: &[f64; 32], presion: f64) -> f64 {
    GameTheoreticNashEngine::voto_espectral(x, presion)
        .media_banda(0, 31)
        .unwrap_or(f64::NAN)
}

/// RED antes de CL-158: la sombra publicaba el minimax contra p = 0,50
/// (media ≈ 0,77 en esta malla) y su escala dominante. El mismo voto entra
/// en votos_espectrales[MOTOR_NASH].
#[test]
fn k12_sin_presion_medida_la_sombra_nash_se_abstiene() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    sembrar_espectro(&mut core);
    for clave in ["game_theory_adversarial_pressure", "cvpin", "nash_equilibrium_drift"] {
        assert_eq!(arena.registry.get_for_coin_strict(0, clave), None, "{clave} por moneda");
        assert_eq!(arena.registry.get_value_fast(clave), None, "{clave} global");
    }

    profundidad(&mut core, 100.55, 20_000);

    let x = desplazamientos(&core);
    assert!(x.iter().any(|v| *v != 0.0), "la malla sembrada tiene desplazamientos: {x:?}");
    let con_literal = media_nash(&x, 0.50);
    assert!(con_literal > 0.5, "el 0,50 retirado votaba con convicción: {con_literal}");

    assert_eq!(
        arena.registry.get_for_coin_strict(0, "sombra_nash_consenso"),
        Some(0.0),
        "sin presión medida la sombra Nash se abstiene (con el 0,50: {con_literal})"
    );
    assert_eq!(arena.registry.get_for_coin_strict(0, "sombra_nash_tau_max"), None);
    assert_eq!(arena.registry.get_for_coin_strict(0, "sombra_nash_v_max"), None);
}

/// Verde antes y después: con `cvpin` medido, la sombra es bit a bit el
/// minimax del motor con esa presión, y la presión discrimina (0,2 frente a
/// 0,8 en la misma malla). Si la sombra dejara de leer `cvpin`, cae (lo que
/// R5-A4 decía probar; linea_g/g7_hallazgos.tsv:76).
#[test]
fn k12_con_cvpin_medido_la_sombra_nash_es_el_minimax_con_esa_presion() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    sembrar_espectro(&mut core);
    for (i, cvpin) in [0.2f64, 0.8].into_iter().enumerate() {
        // El dual de cada evento reescribe `cvpin` con el VPIN medido (0 sin
        // trades): se fija justo antes del evento, que es cuando lo lee la sombra.
        arena.registry.set_for_coin(0, "cvpin", cvpin);
        profundidad(&mut core, 100.55 + 0.05 * i as f64, 20_000 + 1_000 * i as u64);
        let x = desplazamientos(&core);
        assert!(
            (media_nash(&x, 0.2) - media_nash(&x, 0.8)).abs() > 0.05,
            "la presión discrimina en esta malla"
        );
        let esperado = media_nash(&x, cvpin);
        let publicado = arena.registry.get_for_coin_strict(0, "sombra_nash_consenso");
        assert_eq!(
            publicado.map(f64::to_bits),
            Some(esperado.to_bits()),
            "cvpin {cvpin}: publicado {publicado:?}, esperado {esperado}"
        );
    }
}

/// El argumento de conducta de CL-158, comprobado. En el primer evento de un
/// núcleo nuevo (el único en que faltaba `cvpin`) el espectro recién
/// sembrado tiene x ≡ 0, así que el voto Nash es nulo con cualquier presión;
/// y al terminar ese evento el dual ya publicó `cvpin` por moneda y global.
/// Verde antes y después. Si cae, la elección de K12 pasa a ser observable
/// y hay que medirla con golden y T-1.
#[test]
fn k12_el_unico_evento_sin_cvpin_tiene_el_espectro_recien_sembrado() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    assert_eq!(arena.registry.get_value_fast("cvpin"), None);

    profundidad(&mut core, 100.0, 10_000);

    assert!(
        desplazamientos(&core).iter().all(|v| *v == 0.0),
        "x ≡ 0 en el primer evento"
    );
    assert_eq!(arena.registry.get_for_coin_strict(0, "sombra_nash_consenso"), Some(0.0));
    let por_moneda = arena.registry.get_for_coin_strict(0, "cvpin");
    let global = arena.registry.get_value_fast("cvpin");
    assert!(por_moneda.is_some_and(f64::is_finite), "cvpin por moneda: {por_moneda:?}");
    assert!(global.is_some_and(f64::is_finite), "cvpin global: {global:?}");
}

/// N10 / CL-175: la sombra Nash lee el cvpin medido aunque alguien escriba
/// `game_theory_adversarial_pressure`, la clave sin escritor que encabezaba
/// la cadena. Roja antes: publicaba el minimax con la presión de la clave
/// huérfana (0,8) en vez de la medida (0,2).
#[test]
fn n10_la_clave_sin_escritor_no_pisa_el_cvpin_medido() {
    let _guarda = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, mut core) = nucleo();
    sembrar_espectro(&mut core);
    arena.registry.set_for_coin(0, "cvpin", 0.2);
    arena
        .registry
        .set_for_coin(0, "game_theory_adversarial_pressure", 0.8);

    profundidad(&mut core, 100.55, 20_000);

    let x = desplazamientos(&core);
    assert!(
        (media_nash(&x, 0.2) - media_nash(&x, 0.8)).abs() > 0.05,
        "la presión discrimina en esta malla"
    );
    let publicado = arena.registry.get_for_coin_strict(0, "sombra_nash_consenso");
    assert_eq!(
        publicado.map(f64::to_bits),
        Some(media_nash(&x, 0.2).to_bits()),
        "publicado {publicado:?}: con cvpin 0,2 {}, con la clave huérfana 0,8 {}",
        media_nash(&x, 0.2),
        media_nash(&x, 0.8)
    );
}
