//! C-F3.5 (Línea C), hallazgo F3.5-05: el ensamble bosque⊕DarkAlpha promedia
//! probabilidades de eventos distintos y todos los consumidores lo miden
//! contra la base del bosque.
//!
//! Teoría que se prueba:
//! - El bosque sirve P(TP del largo antes que el SL), con base
//!   b_f = sigmoid(init_score); DarkAlpha sirve P(retorno neto > 5 pb), sin
//!   lado y con otra base b_n.
//! - `ModelEnsemble::combined` es la media ponderada de las opiniones
//!   presentes; con pesos iniciales iguales (log-pesos 0, sin habilidad
//!   medida) vale (p_f + p_n)/2.
//! - El núcleo publica `ml_model_base` = b_f (sólo la del bosque) y
//!   `ml_prob_motor` = combined (sin spot). Las puertas, `nn_score`,
//!   FlowExcitation y el consejo leen el lift p − b contra esa base.
//!
//! Solución conocida: un bosque de hoja única con base 0,20 (p_f = 0,20 en
//! todo estado: sin información) y una red constante en 0,45 (pesos y sesgos
//! a 0, sesgo final logit(0,45), Scaler con media 0 y desviación 1: p_n =
//! 0,45 en todo estado). Ninguno aporta información sobre su propia base.
//! Hoy: combined = (0,20 + 0,45)/2 = 0,325 frente a base 0,20, un lift de
//! +0,125 fabricado (+0,156 en la escala normalizada de las puertas).
//!
//! Propiedad: dos modelos sin información no fabrican lift sobre la base
//! publicada: |ml_prob_motor − ml_model_base| < 1e-6. El borrador pedía
//! 1e-9, pero el bosque devuelve su probabilidad en f32 y la base se calcula
//! en f64 desde el mismo init: aun con un arreglo correcto que sirviera sólo
//! el bosque, la diferencia es el redondeo f32 de σ(init), hasta 2⁻²⁵ ≈
//! 3·10⁻⁸ (con b = 0,20: 0,2f32 − 0,2 = 3,0·10⁻⁹ > 1e-9). 1e-6 deja 30
//! veces ese redondeo y está 10⁵ veces por debajo del defecto (0,125).
//! Precondición: opinan los dos modelos (snapshot [Some, Some]); el NN sólo
//! opina en BTCUSDT (C-06), de ahí el símbolo.
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlo:
//! `cargo test ... -- --include-ignored`.

use dark_alpha_engine::{DarkAlphaEngine, Scaler};
use god_engine_core::ml_inference::{NanoForest, NanoForestData};
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};

fn bosque_con_base(p: f32) -> NanoForest {
    NanoForest::from_data(NanoForestData {
        children_left: vec![-1],
        children_right: vec![-1],
        feature: vec![-1],
        threshold: vec![0.0],
        value: vec![0.0],
        tree_offsets: vec![0, 1],
        init_score: (p / (1.0 - p)).ln(),
    })
    .expect("bosque sintético fuera de contrato")
}

fn red_constante(p: f64) -> DarkAlphaEngine {
    let mut red = DarkAlphaEngine::new(54, 64, 32);
    for capa in [&mut red.layer1, &mut red.layer2, &mut red.layer3] {
        capa.weights.iter_mut().for_each(|w| *w = 0.0);
        capa.biases.iter_mut().for_each(|b| *b = 0.0);
    }
    red.layer3.biases[0] = (p / (1.0 - p)).ln();
    red.scaler = Some(Scaler::new(vec![0.0; 54], vec![1.0; 54]));
    red
}

#[test]
#[ignore = "OPEN F3.5-05 (Línea C, ensamble): combined() promedia P(TP largo) del bosque con P(retorno > 5 pb) de DarkAlpha y ml_model_base publica sólo la base del bosque; dos modelos sin información fabrican lift (0,325 frente a 0,20)"]
fn cf3_dos_modelos_sin_informacion_dejan_neutro_el_canal_ml() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    NanoForest::store_global("BTCUSDT_MOTOR", bosque_con_base(0.20));

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.scalp_forest = None;
    core.swing_nn = Some(red_constante(0.45));
    let _ = core.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false);

    let snap = core.ensembles[0].prediction_snapshot();
    let p = arena
        .registry
        .get_for_coin_strict_or(0, "ml_prob_motor", f64::NAN);
    let b = arena
        .registry
        .get_for_coin_strict_or(0, "ml_model_base", f64::NAN);
    println!("observado: snapshot = {snap:?}, ml_prob_motor = {p:.6}, ml_model_base = {b:.6}");
    assert!(
        snap[0].is_some() && snap[1].is_some(),
        "precondición: opinan los dos modelos {snap:?}"
    );
    assert!(
        (p - b).abs() < 1e-6,
        "dos modelos sin información fabrican lift: ml_prob={p} base={b}"
    );
}
