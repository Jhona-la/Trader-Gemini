//! C-F3.8 (Línea C): K-26 ejecutando el núcleo con base honesta, espejo del
//! sesgo spot (F3.8-01) y el fixture de #714 (F3.8-03).
//!
//! Teoría que se prueba:
//! - K-26 (`p_win_directional`): el largo gana con p y el corto con la
//!   reflexión normalizada de p alrededor de la base b; en neutralidad
//!   (p = b) los dos lados tienen p_win = b, y el calibrador conformal
//!   (`ConformalCalibrator::p_value`, `accepts`) les da el mismo p-valor.
//! - El núcleo calcula `ml_model_base = sigmoid(init)` del bosque del
//!   símbolo (0,5 sin bosque) y la PUBLICA en el registro en cada tick, antes
//!   del bloque conformal (lib.rs, `set_reg("ml_model_base", …)`).
//! - `ml_prob = compose_ml_prob(ensamble, spot_bias)` suma el sesgo spot en
//!   unidades de probabilidad (s = clamp(basis_pb/10; ±0,05)). Con b ≠ ½ la
//!   divergencia normalizada lo mide como +s/(1 − b) en el largo y −s/b en
//!   el corto (F3.8-01, misma raíz que F3.1-07, F3.3-04 y F3.5-04).
//!
//! Solución conocida (las tres pruebas):
//! - Bosque de sólo sesgo (`tree_offsets = [0, 0]`, la representación
//!   legada que aceptan los oráculos) con init = logit(0,25): p = 0,25 salvo
//!   el redondeo f32 (≲ 3·10⁻⁸), de ahí la tolerancia 1e-6.
//! - Calibrador caliente (mínimo 30) con 40 cierres en la rejilla
//!   p_i = (i + 0,5)/41, gana si i es par. La no conformidad de «gana» con
//!   p ≈ 0,25 es ≈ 0,75; ninguna puntuación de la rejilla cae a menos de
//!   0,25/41 de 0,75, así que el redondeo f32 no mueve el rango y el
//!   p-valor es exactamente 11/41 = 0,2683 en los dos lados.
//! - Un tick con bid 100 y ask 100,01; con spot, basis ±5 pb (satura s).
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlos:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::{
    calibration::normalized_directional_divergence,
    ml_inference::{NanoForest, NanoForestData},
    GodEngineCore,
};
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex};

/// Serializa el estado global (universo, registro de specs, bosques) entre
/// las pruebas de este binario.
static CORE_ENV: Mutex<()> = Mutex::new(());
const SIM: &str = "CFTRESUSDT";
const BASE: f64 = 0.25;

fn bosque_constante(b: f64) -> NanoForest {
    NanoForest::from_data(NanoForestData {
        children_left: vec![],
        children_right: vec![],
        feature: vec![],
        threshold: vec![],
        value: vec![],
        tree_offsets: vec![0, 0],
        init_score: (b / (1.0 - b)).ln() as f32,
    })
    .expect("bosque de sólo sesgo")
}

fn nucleo_caliente() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    NanoForest::store_global(&format!("{SIM}_MOTOR"), bosque_constante(BASE));
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    arena.config.conformal_alpha.store(0.05, Relaxed);
    // 40 cierres en rejilla: calibrador caliente (mínimo 30), sin empates en 0,25 ni 0,75.
    for i in 0..40 {
        core.conformal_by_coin[0].update((i as f64 + 0.5) / 41.0, i % 2 == 0);
    }
    (arena, core)
}

fn leer(a: &GlobalArena, k: &str) -> f64 {
    a.registry.get_for_coin_strict_or(0, k, f64::NAN)
}

fn tick(arena: &GlobalArena, core: &mut GodEngineCore, basis_bps: Option<f64>) {
    let (bid, ask) = (100.0, 100.01);
    if let Some(bps) = basis_bps {
        let s = 0.5 * (bid + ask) * (1.0 + bps * 1e-4);
        arena.update_spot_data(0, s - 0.0005, s + 0.0005, 1.0, 1.0);
    }
    let _ = core.process_tick_dual(0, bid, ask, 5.0, 5.0, 10_000, &[0.0; 54], true, false);
}

/// Control positivo de K-26 con el núcleo: base honesta 0,25 alcanzada,
/// calibrador caliente (p < 1) y los dos lados con el mismo p-valor y la
/// misma aceptación. Es la versión que el test de #714 declara y no ejecuta
/// (F3.4-08, F3.8-03).
#[test]
fn cf3_k26_neutral_con_base_honesta_ejecutando_el_nucleo() {
    let _g = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo_caliente();
    tick(&arena, &mut core, None);
    let base = leer(&arena, "ml_model_base");
    let ml = leer(&arena, "ml_prob");
    let (pl, ps) = (
        leer(&arena, "conformal_p_value"),
        leer(&arena, "conformal_p_value_short"),
    );
    let (al, ash) = (
        leer(&arena, "conformal_accept_long"),
        leer(&arena, "conformal_accept_short"),
    );
    println!(
        "observado: base = {base:.8}, ml_prob = {ml:.8}, p-valor largo = {pl:.4}, corto = {ps:.4}, \
         acepta largo = {al}, corto = {ash}"
    );
    assert!(
        (base - BASE).abs() < 1e-6,
        "el núcleo no llegó a la base honesta: {base}"
    );
    assert!((ml - BASE).abs() < 1e-6, "ml_prob no es la base: {ml}");
    assert!(
        pl < 1.0,
        "calibrador en calentamiento: el test no discriminaría"
    );
    assert_eq!(
        pl, ps,
        "K-26: en neutralidad los dos lados comparten p-valor"
    );
    assert_eq!(
        al, ash,
        "K-26: en neutralidad los dos lados comparten la aceptación"
    );
}

/// Verdad: con la misma base del modelo (0,25) y basis en espejo (+5 pb en
/// A, −5 pb en B), el apoyo que ve la puerta es opuesto, d_A + d_B = 0
/// (± 1e-6, redondeo f32 del bosque), y el p-valor del largo en A es el del
/// corto en B (y al revés). Hoy: ml_prob 0,30 y 0,20, d_A = +0,0667 y
/// d_B = −0,2000 (suma −0,133); p_win del largo en A 0,30 (p-valor 0,3171)
/// frente al del corto en B 0,40 (p-valor 0,4146).
///
/// Nota de los escépticos: d_A + d_B = 0 supone que el espejo correcto es la
/// divergencia normalizada. F3.8-02 muestra que esa reflexión tampoco es el
/// espejo de primer toque; las dos pruebas deben compartir una definición de
/// espejo antes de cerrar cualquiera. Un arreglo en log-odds dejaría esta
/// prueba roja.
#[test]
#[ignore = "OPEN F3.8-01 (Línea C, lib.rs fuera de las ramas 1-10 y calibration.rs): el sesgo spot aditivo en p se mide con la divergencia normalizada por semiintervalos; con b = 0,25 el corto espejo recibe 3 veces el apoyo del largo"]
fn cf3_espejo_del_sesgo_spot_con_base_honesta() {
    let _g = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let lectura = |bps: f64| {
        let (arena, mut core) = nucleo_caliente();
        tick(&arena, &mut core, Some(bps));
        (
            leer(&arena, "ml_prob"),
            leer(&arena, "ml_model_base"),
            leer(&arena, "conformal_p_value"),
            leer(&arena, "conformal_p_value_short"),
        )
    };
    let (ml_a, b_a, pl_a, ps_a) = lectura(5.0);
    let (ml_b, b_b, pl_b, ps_b) = lectura(-5.0);
    let d_a = normalized_directional_divergence(ml_a, b_a);
    let d_b = normalized_directional_divergence(ml_b, b_b);
    println!(
        "observado: A (+5 pb): ml_prob = {ml_a:.4}, d = {d_a:+.4}, p-valor largo {pl_a:.4}, corto {ps_a:.4}"
    );
    println!(
        "observado: B (-5 pb): ml_prob = {ml_b:.4}, d = {d_b:+.4}, p-valor largo {pl_b:.4}, corto {ps_b:.4}"
    );
    assert!(
        (b_a - BASE).abs() < 1e-6 && (b_b - BASE).abs() < 1e-6,
        "precondición: base honesta en los dos núcleos ({b_a}, {b_b})"
    );
    assert!(
        (d_a + d_b).abs() < 1e-6,
        "mismo basis en espejo, apoyo distinto: d_A={d_a} d_B={d_b}"
    );
    assert_eq!(
        pl_a, ps_b,
        "p-valor del largo con +5 pb frente al del corto con -5 pb"
    );
    assert_eq!(
        ps_a, pl_b,
        "p-valor del corto con +5 pb frente al del largo con -5 pb"
    );
}

/// Testigo del fixture literal de #714 (`conformal_direction_symmetry_contract.rs`):
/// universo BTCSYM, spec BTCUSDT, sin bosque ni red, base 0,25 escrita en el
/// registro, α = 0,05 y un tick. Verdad: si ese test evaluara lo que declara,
/// el registro conservaría la base 0,25 y el p-valor sería < 1. Hoy el núcleo
/// pisa la base con la de «sin bosque» (0,5) antes del bloque conformal, y el
/// calibrador sin cierres devuelve p = 1 (calentamiento, acepta siempre).
///
/// El arreglo es del test de #714 (inyectar un bosque de base 0,25 y calentar
/// el calibrador, como `cf3_k26_neutral_con_base_honesta_ejecutando_el_nucleo`),
/// no del núcleo. Este testigo reproduce el fixture viejo: no se «activa»;
/// se retira cuando el test de #714 adopte el fixture correcto.
#[test]
#[ignore = "OPEN F3.8-03 (AGY, autor de #714; la fila del ledger es de Línea E): el fixture de #714 escribe ml_model_base = 0,25 pero el núcleo sin bosque la pisa con 0,5 y el calibrador frío da p = 1; el test certifica K-26 sin ejecutarlo. Testigo: se retira cuando #714 use un bosque de base 0,25 y el calibrador caliente"]
fn cf3_fixture_714_no_alcanza_la_base_que_declara() {
    let _g = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    symbols::update_dynamic_universe(vec!["BTCSYM".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    arena.config.conformal_alpha.store(0.05, Relaxed);
    arena.registry.set_for_coin(0, "ml_model_base", 0.25);
    let _ = core.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false);
    let base = leer(&arena, "ml_model_base");
    let p = leer(&arena, "conformal_p_value");
    println!("observado: ml_model_base = {base}, conformal_p_value = {p}");
    assert!(
        (base - 0.25).abs() < 1e-9,
        "#714 declara base 0,25; el núcleo evaluó con {base}"
    );
    assert!(
        p < 1.0,
        "#714 evalúa en calentamiento (p = 1, accept = true)"
    );
}
