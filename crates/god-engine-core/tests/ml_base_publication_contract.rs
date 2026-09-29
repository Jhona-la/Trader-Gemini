//! CL-21 (FMT-159) — la base del modelo del roster llega al registro.
//! Eventos sintéticos locales; sin exchange.
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
fn cl21_la_base_del_modelo_se_publica_por_simbolo_y_moneda() {
    symbols::update_dynamic_universe(vec!["MBPAUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "MBPAUSDT",
    )]);
    NanoForest::store_global("MBPAUSDT_MOTOR", bosque_con_base(0.30));

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let _ = core.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false);

    let por_simbolo =
        arena
            .registry
            .get_scoped_val_or(Some("MBPAUSDT"), None, "ml_model_base", f64::NAN);
    let por_moneda = arena.registry.get_for_coin_or(0, "ml_model_base", f64::NAN);
    assert!(
        (por_simbolo - 0.30).abs() < 1e-6,
        "la base del bosque del roster debe publicarse (FMT-159); leída {por_simbolo}"
    );
    assert!((por_moneda - 0.30).abs() < 1e-6, "por moneda: {por_moneda}");
}
