//! CL-37 — con el universo del bootloader, el núcleo encuentra el bosque del
//! roster. El host publicaba la lista del bootloader en minúsculas y carga los
//! bosques por el stem del archivo (`models/ATOMUSDT_MOTOR.json`): la clave
//! `"atomusdt_MOTOR"` nunca coincidía, `has_roster_model` quedaba en falso en
//! todos los slots y la base del modelo caía al 0,5 neutro. Eventos
//! sintéticos locales; sin exchange. Binario propio: el universo es global.
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
fn cl37_el_nucleo_encuentra_el_bosque_del_roster_con_el_universo_del_bootloader() {
    // Mismo orden de operaciones que el host: universo del bootloader,
    // registro en MAYÚSCULAS (B3.38) y bosque por el stem del archivo.
    let universo: Vec<String> = vec!["idnmusdt".into()];
    symbols::update_dynamic_universe(universo.clone());
    symbol_registry::update_registry(
        universo
            .iter()
            .map(|s| symbol_registry::get_official_binance_spec(&s.to_uppercase()))
            .collect(),
    );
    NanoForest::store_global("IDNMUSDT_MOTOR", bosque_con_base(0.30));

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let _ = core.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false);

    let por_moneda = arena.registry.get_for_coin_or(0, "ml_model_base", f64::NAN);
    assert!(
        (por_moneda - 0.30).abs() < 1e-6,
        "el slot 0 es IDNMUSDT y su bosque tiene base 0,30; leída {por_moneda}"
    );
}

/// R7-R4-A-2 — el desacople de ámbito en spoof_score y whale_burst_z queda resuelto:
/// la lectura unificada bidireccional (por coin_id y por símbolo scoped)
/// garantiza que el Consejo de Seniors recibe las señales reales tanto si
/// el escritor usa set_scoped como si usa set_for_coin.
#[test]
fn r7_r4_a2_spoof_y_whale_burst_se_resuelven_en_ambitos_simbolo_y_coin_id() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let sym = "BTCUSDT";
    let coin_id = 0;

    // Caso 1: Escritor usa set_scoped (como el host clásico)
    arena.registry.set_scoped(sym, "spoof_score", 0.85);
    arena.registry.set_scoped(sym, "whale_burst_z", 4.2);

    let spoof_read_1 = arena.registry
        .get_for_coin_or(coin_id, "spoof_score", 0.0)
        .max(arena.registry.get_scoped_value_or(sym, "spoof_score", 0.0));
    let whale_read_1 = arena.registry
        .get_for_coin_or(coin_id, "whale_burst_z", 0.0)
        .max(arena.registry.get_scoped_value_or(sym, "whale_burst_z", 0.0));

    assert_eq!(spoof_read_1, 0.85, "spoof_score debe resolverse desde set_scoped");
    assert_eq!(whale_read_1, 4.2, "whale_burst_z debe resolverse desde set_scoped");

    // Caso 2: Escritor usa set_for_coin
    arena.registry.set_for_coin(coin_id, "spoof_score", 0.92);
    arena.registry.set_for_coin(coin_id, "whale_burst_z", 5.5);

    let spoof_read_2 = arena.registry
        .get_for_coin_or(coin_id, "spoof_score", 0.0)
        .max(arena.registry.get_scoped_value_or(sym, "spoof_score", 0.0));
    let whale_read_2 = arena.registry
        .get_for_coin_or(coin_id, "whale_burst_z", 0.0)
        .max(arena.registry.get_scoped_value_or(sym, "whale_burst_z", 0.0));

    assert_eq!(spoof_read_2, 0.92, "spoof_score debe resolverse desde set_for_coin");
    assert_eq!(whale_read_2, 5.5, "whale_burst_z debe resolverse desde set_for_coin");
}

