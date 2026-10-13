//! CONTRATO — StatArb con física viva y paridad lector/escritor en el núcleo.
//!
//! OLA 73 (R6-A3/B1/B2/A5) cableó una SDE OU por evento sobre la base
//! futuro-spot. CL-97 (C-F3, F3.7-02 v2) la sustituye por una prueba de
//! reversión sobre pares síncronos (`strategy_core::prueba_reversion`): la
//! SDE pasaba la guarda con caminatas independientes. Contrato vigente:
//!
//! 1. El núcleo publica en cada evento, por moneda, `statarb_ou_zscore` (z
//!    de la base si la reversión está demostrada, 0 si no o con el spot
//!    caducado), `statarb_half_life_ms` (la cota 2·τ_k demostrada en el nodo
//!    τ_k ≤ τ*, ∞ sin demostración) y `statarb_beta` (1: la base es
//!    ln F − ln S, sin estimar β).
//! 2. El lector sólo vota con reversión demostrada (t½ publicada ≤ 2τ*). Sin
//!    prueba propia se abstiene: ya no cae a `vecm_zscore`.
//! 3. TTL: con el spot caducado (> 30 s) la z publicada es 0, también con el
//!    spot presente pero sin refrescar (CL-113).

mod comun_base_spot;

use comun_base_spot::*;
use quantum_arena::GlobalArena;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use strategy_core::stat_arb::StatArbEngine;
use strategy_core::QuantumStrategy;

/// El mismo lector que el orquestador registra.
fn lector(arena: &GlobalArena) -> StatArbEngine {
    let mut engine = StatArbEngine::new(30, 1.5);
    assert!(engine.init(arena.registry.clone()).is_ok());
    engine
}

/// Verdad: sin spot no hay prueba; el núcleo publica la abstención (z 0,
/// t½ ∞) y el lector no vota aunque `vecm_zscore` valga 2,5.
#[test]
fn cl97_sin_spot_publica_abstencion_y_el_lector_no_vota() {
    let _g = bloqueo();
    let (arena, mut core) = nucleo();
    evento(&arena, &mut core, 1_000, 50_000.0, None);
    assert_eq!(leer(&arena, "statarb_ou_zscore"), Some(0.0));
    assert_eq!(leer(&arena, "statarb_half_life_ms"), Some(f64::INFINITY));
    arena.registry.set("vecm_zscore", 2.5);
    arena.registry.set_scoped("BTCUSDT", "vecm_zscore", 2.5);
    assert_eq!(lector(&arena).evaluate_for_coin(0, "BTCUSDT"), 0.0);
    assert_eq!(lector(&arena).evaluate(), 0.0);
}

/// Verdad: con una base que revierte rápido (t½ = 14 s) y pares síncronos,
/// la prueba demuestra la reversión: la t½ publicada es 2·τ_k de un nodo de
/// la malla con τ_k ≤ τ*, β = 1 y la z es finita.
#[test]
fn cl97_con_base_que_revierte_publica_la_cota_de_un_nodo() {
    let _g = bloqueo();
    let (arena, _core, _m, _ts) = nucleo_demostrado(1);
    let th = leer(&arena, "statarb_half_life_ms").expect("t½ publicada");
    let tau = leer(&arena, "dominant_tau_ms").expect("τ* publicada");
    let nodo = (17..=22).find(|&k| (2.0 * SPECTRUM_SCALES_MS[k]).to_bits() == th.to_bits());
    println!("observado: t½ publicada = {th} ms, τ* = {tau} ms, nodo = {nodo:?}");
    assert!(nodo.is_some(), "la t½ publicada es una cota 2·τ_k: {th}");
    assert!(SPECTRUM_SCALES_MS[nodo.unwrap()] <= tau);
    assert_eq!(leer(&arena, "statarb_beta"), Some(1.0));
    let z = leer(&arena, "statarb_ou_zscore").expect("z publicada");
    assert!(z.is_finite() && z != 0.0, "z = {z}");
}

/// Verdad: si el feed de spot desaparece (spot a 0), la z publicada vuelve a
/// 0 y el lector calla. (Prueba el spot AUSENTE; el TTL lo prueba la
/// siguiente, CL-113.)
#[test]
fn cl97_spot_ausente_publica_cero() {
    let _g = bloqueo();
    let (arena, mut core, _m, ts) = nucleo_demostrado(2);
    arena.update_spot_data(0, 0.0, 0.0, 0.0, 0.0);
    evento(&arena, &mut core, ts + 31_000, 50_100.0, None);
    assert_eq!(leer(&arena, "statarb_ou_zscore"), Some(0.0));
    assert_eq!(lector(&arena).evaluate_for_coin(0, "BTCUSDT"), 0.0);
}

/// CL-113 (W-4 de la revisión del PR #60). Verdad: con el spot PRESENTE
/// pero sin refrescar, la z se publica mientras su edad no pasa del TTL
/// (30 s) y vuelve a 0 después. La prueba anterior ponía el spot a 0: probaba
/// un spot ausente y habría pasado con un TTL infinito.
#[test]
fn cl113_spot_presente_pero_caducado_publica_cero() {
    let _g = bloqueo();
    let (arena, mut core, mut m, mut ts) = nucleo_demostrado(2);
    // El futuro sigue (un evento por segundo) y el spot deja de llegar. El
    // último llegó justo antes del evento ts − 1 000; su marca es el reloj de
    // eventos del núcleo en ese momento.
    let mut z_a = |segundos: u64| {
        let mut z = f64::NAN;
        for _ in 0..segundos {
            let (f, _) = m.paso();
            evento(&arena, &mut core, ts, f, None);
            ts += 1_000;
            z = leer(&arena, "statarb_ou_zscore").expect("z publicada");
        }
        z
    };
    let dentro = z_a(20);
    let fuera = z_a(20);
    println!("observado: z con el spot de ~20 s = {dentro}; de ~40 s = {fuera}");
    assert!(dentro.is_finite() && dentro != 0.0, "dentro del TTL la z sigue: {dentro}");
    assert_eq!(fuera, 0.0, "pasado el TTL la z es 0");
    assert_eq!(lector(&arena).evaluate_for_coin(0, "BTCUSDT"), 0.0);
}

/// Verdad (C-02): por el escritor de producción (`apply_spot_tick`), la
/// prueba recibe sus pares, demuestra la reversión y el lector evalúa.
#[test]
fn ola57_c02_feed_spot_madura_la_prueba_y_publica() {
    let _g = bloqueo();
    let (arena, mut core) = nucleo();
    let omni = data_pipeline::omni_multiplexer::OmniState::new();
    let mut m = Mercado::new(4, 14.0);
    let mut ts = 1_000u64;
    let mut demostrada = false;
    for i in 0..3_600u64 {
        let (f, sp) = m.paso();
        data_pipeline::spot_feed::apply_spot_tick(&arena, &omni, 0, "BTCUSDT", sp - 0.1, sp + 0.1, 2.0, 2.0);
        evento(&arena, &mut core, ts, f, None);
        ts += 1_000;
        if i >= 600 && guarda_pasa(&arena) {
            demostrada = true;
            break;
        }
    }
    assert!(demostrada, "con el feed de producción la prueba demuestra la reversión");
    assert_eq!(leer(&arena, "statarb_beta"), Some(1.0));
    assert!(lector(&arena).evaluate_for_coin(0, "BTCUSDT").is_finite());
}
