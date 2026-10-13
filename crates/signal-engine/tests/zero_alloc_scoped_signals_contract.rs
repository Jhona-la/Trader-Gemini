use omniscient_registry::OmniscientRegistry;
use signal_engine::flow_impulse::FlowImpulseEngine;
use signal_engine::RenyiTsallisEntropyEngine;
use strategy_core::QuantumStrategy;
use std::sync::Arc;

#[test]
fn test_flow_impulse_scoped_evaluation_strict_isolation() {
    let registry = Arc::new(OmniscientRegistry::new());

    // Publicamos parámetros para ETHUSDT (coin_id = 1)
    // Hawkes > 1.60 (STEADY_STATE_RATIO) representa excitación por cascada de liquidez
    registry.set_scoped("ETHUSDT", "order_book_imbalance", 0.75);
    registry.set_scoped("ETHUSDT", "order_flow_imbalance", 0.60);
    registry.set_scoped("ETHUSDT", "hawkes_intensity", 2.50);

    let mut engine = FlowImpulseEngine::default();
    let _ = engine.init(registry.clone());

    // ETHUSDT (coin_id = 1) debe evaluar exitosamente con sus datos propios
    let voto_eth = engine.evaluate_for_coin(1, "ETHUSDT");
    assert!(
        voto_eth > 0.0,
        "ETHUSDT con OBI/OFI y Hawkes excitado (>1.6) debe emitir voto alcista positivo: {}",
        voto_eth
    );

    // SOLUSDT (coin_id = 2) sin parámetros NO debe contaminarse de ETHUSDT ni del global
    let voto_sol = engine.evaluate_for_coin(2, "SOLUSDT");
    assert_eq!(
        voto_sol, 0.0,
        "SOLUSDT sin parámetros propios debe neutralizarse estrictamente a 0.0 sin fuga de ETHUSDT"
    );
}

#[test]
fn test_renyi_tsallis_scoped_evaluation_strict_isolation() {
    let registry = Arc::new(OmniscientRegistry::new());

    // Publicamos parámetros para SOLUSDT (coin_id = 2)
    // Entropía de Tsallis baja (0.30 < 0.60) representa alta certeza de orden; OBI > 0.15 indica desequilibrio
    registry.set_scoped("SOLUSDT", "tsallis_q_entropy", 0.30);
    registry.set_scoped("SOLUSDT", "order_book_imbalance", 0.50);

    let mut engine = RenyiTsallisEntropyEngine::new(1.5, 2.0);
    let _ = engine.init(registry.clone());

    // SOLUSDT debe evaluar con sus datos propios
    let voto_sol = engine.evaluate_for_coin(2, "SOLUSDT");
    assert!(
        voto_sol > 0.0,
        "SOLUSDT con baja entropía Tsallis (<0.60) y OBI alto debe emitir voto positivo: {}",
        voto_sol
    );

    // ADAUSDT (coin_id = 3) sin parámetros debe neutralizarse estrictamente a 0.0
    let voto_ada = engine.evaluate_for_coin(3, "ADAUSDT");
    assert_eq!(
        voto_ada, 0.0,
        "ADAUSDT sin parámetros debe neutralizarse a 0.0 sin heredar de SOLUSDT"
    );
}

#[test]
fn test_cero_heap_en_consultas_estrictas_de_registro() {
    let registry = OmniscientRegistry::new();

    registry.set_scoped("BTCUSDT", "vol_forecast_pct", 0.025);
    registry.set_for_coin(0, "local_spread", 0.0001);

    // 1. Consulta por símbolo estricta
    let val_sym = registry.get_scoped_val_strict(Some("BTCUSDT"), Some(0), "vol_forecast_pct");
    assert_eq!(val_sym, Some(0.025));

    // 2. Consulta por coin_id estricta
    let val_cid = registry.get_scoped_val_strict(None, Some(0), "local_spread");
    assert_eq!(val_cid, Some(0.0001));

    // 3. Consulta de parámetro ausente retorna None sin caer a global ni alojar
    let val_absent = registry.get_scoped_val_strict(Some("BTCUSDT"), Some(0), "parametro_inexistente");
    assert_eq!(val_absent, None);

    // 4. Parámetro global no es devuelto por la API estricta
    registry.set("parametro_global_solo", 999.0);
    let val_global = registry.get_scoped_val_strict(Some("ETHUSDT"), Some(1), "parametro_global_solo");
    assert_eq!(val_global, None, "API estricta jamás debe caer al parámetro global");
}
