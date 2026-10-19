//! LF-1 (Línea F, revisión de composición de Ω84/#722) — cierra el respaldo a
//! BTC de LC-F1-32.
//!
//! Ω84 unificó las tres identidades del slot dentro de `process_tick_dual` en
//! una sola, pero con respaldo `"BTCUSDT"`: un slot que no está en el universo
//! ni en el registro de specs se convertía en BTC. Heredaba el bosque
//! `BTCUSDT_MOTOR` y su base, escribía las claves `BTCUSDT_*` del registro (la
//! guarda `!sym.is_empty()` de `set_reg` dejaba de saltar) y alimentaba el
//! líder del lead-lag con su propio flujo. Contradice ADR-0011 (un solo mapa de
//! identidad) y el comentario B3.18b («SIN FALLBACK a BTC»).
//!
//! Es latente en vivo (el host registra universo y specs al arrancar), pero
//! alcanzable en `run_booktick_replay`, que a propósito no registra nada, y en
//! `votes_export`. Binario propio: el universo y el registro de specs son
//! globales del proceso. Eventos sintéticos locales; sin exchange ni red.
use god_engine_core::GodEngineCore;
use god_engine_core::ml_inference::{NanoForest, NanoForestData};
use quantum_arena::{GlobalArena, symbol_registry, symbols};

fn bosque_con_base(p: f32) -> NanoForest {
    // Un árbol de hoja única en 0: la predicción es la base sigmoid(init).
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

#[test]
fn lf1_slot_sin_identidad_no_se_convierte_en_btc() {
    // Universo de un solo símbolo: el slot 1 no tiene identidad.
    symbols::update_dynamic_universe(vec!["LFIDAUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "LFIDAUSDT",
    )]);
    assert!(symbol_registry::try_symbol(1).is_none(), "el fixture exige un slot sin identidad");
    NanoForest::store_global("BTCUSDT_MOTOR", bosque_con_base(0.30));

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    let _ = core.process_tick_dual(1, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false);

    // 1) No hereda el bosque de BTC: sin modelo propio, la base es la neutra.
    let base = arena.registry.get_for_coin_strict_or(1, "ml_model_base", f64::NAN);
    assert!(
        (base - 0.5).abs() < 1e-12,
        "un slot sin identidad no debe heredar la base 0,30 de BTCUSDT_MOTOR; leída {base}"
    );

    // 2) No escribe las claves del símbolo BTCUSDT.
    assert!(
        arena.registry.get_scoped_strict("BTCUSDT", "mid_price").is_none(),
        "un slot sin identidad no debe escribir BTCUSDT_mid_price"
    );
    assert!(
        arena.registry.get_scoped_strict("BTCUSDT", "ml_model_base").is_none(),
        "un slot sin identidad no debe escribir BTCUSDT_ml_model_base"
    );

    // 3) Sí publica su propio estado por moneda (el arreglo no aborta el evento:
    //    run_booktick_replay no registra identidades a propósito).
    let mid = arena.registry.get_for_coin_strict_or(1, "mid_price", f64::NAN);
    assert!(mid.is_finite(), "el slot sin identidad sigue publicando por moneda; leído {mid}");
}
