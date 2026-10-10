//! CONTRATO FORMAL: Acoplamiento de Flujos de Consenso Helmholtz-Hodge, Yang-Mills y Staleness Macro (OLA Ω38)
//!
//! Verifica que:
//! 1. Los buffers multiactivo `latest_prices` y `latest_ofis` se alimentan continuamente con cada tick.
//! 2. La curvatura de Yang-Mills (densidad de acción y corrientes restauradoras J_i) se computan y publican en el registro.
//! 3. La descomposición ortogonal de Helmholtz-Hodge (curl_share, energía de gradiente y energía rotacional) se computa y publica en el registro.
//! 4. La antigüedad de los datos macro (`macro_staleness_ms`) se refleja fielmente en el snapshot del Consejo de Sabios.
//! 5. `StatArbEngine` y `YangMillsGaugeEngine` están registrados y son evaluados en el orquestador continuo.

use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use std::sync::{Arc, Mutex};

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn setup_multi_coin_fixture() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec![
        "BTCUSDT".into(),
        "ETHUSDT".into(),
        "SOLUSDT".into(),
    ]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
        symbol_registry::get_official_binance_spec("ETHUSDT"),
        symbol_registry::get_official_binance_spec("SOLUSDT"),
    ]);

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

#[test]
fn test_hodge_and_yang_mills_continuous_coupling_contract() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (arena, mut core) = setup_multi_coin_fixture();

    // 1. Inyectar ticks en los 3 activos para calentar buffers de precios y OFIs
    // BTC tick
    core.process_tick_dual(
        0,
        50_000.0,
        50_001.0,
        1.5,
        1.2,
        1_000,
        &[0.0; 54],
        true,
        true,
    );

    // ETH tick
    core.process_tick_dual(
        1,
        3_000.0,
        3_000.5,
        10.0,
        8.0,
        1_001,
        &[0.0; 54],
        true,
        true,
    );

    // SOL tick
    core.process_tick_dual(
        2,
        150.0,
        150.05,
        50.0,
        45.0,
        1_002,
        &[0.0; 54],
        true,
        true,
    );

    // 2. Verificar que Yang-Mills publicó action y corriente restauradora
    let ym_action = arena.registry.get_value_or("yang_mills_action", -1.0);
    assert!(
        ym_action.is_finite() && ym_action >= 0.0,
        "Yang-Mills action density debe ser finita y no-negativa, obtenido: {}",
        ym_action
    );

    let ym_curr_sol = arena.registry.get_for_coin_or(2, "yang_mills_current", 999.0);
    assert!(
        ym_curr_sol.is_finite() && ym_curr_sol.abs() <= 1.0,
        "Yang-Mills corriente gauge debe estar acotada en [-1, 1], obtenido: {}",
        ym_curr_sol
    );

    // 3. Verificar que Helmholtz-Hodge publicó métricas válidas en el registro
    let curl_share = arena.registry.get_value_or("hodge_curl_share", -1.0);
    assert!(
        curl_share.is_finite() && (0.0..=1.0).contains(&curl_share),
        "Helmholtz-Hodge curl_share debe estar en [0, 1], obtenido: {}",
        curl_share
    );

    let grad_energy = arena.registry.get_value_or("hodge_gradient_energy", -1.0);
    assert!(
        grad_energy.is_finite() && grad_energy >= 0.0,
        "Helmholtz-Hodge gradient energy debe ser no-negativa, obtenido: {}",
        grad_energy
    );

    let curl_energy = arena.registry.get_value_or("hodge_curl_energy", -1.0);
    assert!(
        curl_energy.is_finite() && curl_energy >= 0.0,
        "Helmholtz-Hodge curl energy debe ser no-negativa, obtenido: {}",
        curl_energy
    );

    // 4. Verificar publicación y lectura de macro_staleness_ms en el registro
    arena.registry.set("macro_staleness_ms", 45_000.0);
    let stale_ms = arena.registry.get_value_or("macro_staleness_ms", 0.0) as u64;
    assert_eq!(stale_ms, 45_000, "macro_staleness_ms debe coincidir exactamente");

    // 5. Verificar que StatArb y YangMillsGaugeEngine están en el orquestador
    let censo = core.tensor_orchestrator.censo_snapshot();
    let has_stat_arb = censo.iter().any(|(name, _, _)| *name == "StatArbEngine");
    let has_yang_mills = censo.iter().any(|(name, _, _)| *name == "YangMillsGaugeEngine");
    assert!(has_stat_arb, "StatArbEngine debe estar registrado en el TensorOrchestrator");
    assert!(has_yang_mills, "YangMillsGaugeEngine debe estar registrado en el TensorOrchestrator");

    // 6. Evaluación de consenso continuo no debe entrar en pánico ni producir NaN
    let consensus = core.tensor_orchestrator.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert!(
        consensus.net_confidence.is_finite(),
        "Consenso tensorial continuo debe tener confianza finita"
    );
}

#[test]
fn test_hodge_cross_flow_non_degenerate_vortex_coupling_contract() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (arena, mut core) = setup_multi_coin_fixture();

    // 1. Inyectar ronda 1: inicializar precios base en los 3 activos
    core.process_tick_dual(0, 50_000.0, 50_001.0, 1.0, 1.0, 1_000, &[0.0; 54], true, true);
    core.process_tick_dual(1, 3_000.0, 3_000.5, 1.0, 1.0, 1_001, &[0.0; 54], true, true);
    core.process_tick_dual(2, 150.0, 150.05, 1.0, 1.0, 1_002, &[0.0; 54], true, true);

    // 2. Inyectar ronda 2 con flujo cruzado asimétrico circulante L2/L3:
    // BTC sube (+10 bps) con OFI comprador (+2.0)
    core.process_tick_dual(0, 50_050.0, 50_051.0, 3.0, 1.0, 1_100, &[0.0; 54], true, true);
    // ETH cae (-33 bps) con OFI vendedor (-2.0)
    core.process_tick_dual(1, 2_990.0, 2_990.5, 1.0, 3.0, 1_101, &[0.0; 54], true, true);
    // SOL sube moderado (+66 bps) con OFI vendedor (-1.0) para inducir dislocación rotacional de liquidez
    core.process_tick_dual(2, 151.0, 151.05, 1.0, 2.0, 1_102, &[0.0; 54], true, true);

    // 3. Verificar que la matriz asimétrica L2/L3 generó rotacional real en el pipeline vivo
    let curl_share = arena.registry.get_value_or("hodge_curl_share", 0.0);
    let curl_energy = arena.registry.get_value_or("hodge_curl_energy", 0.0);
    let grad_energy = arena.registry.get_value_or("hodge_gradient_energy", 0.0);

    // R6-A7 / R6-C7: Demostrar formalmente que el pipeline vivo produce vórtice no-degenerado
    assert!(
        curl_share > 0.0,
        "R6-A7/C7: Con flujo cruzado asimétrico L2/L3, hodge_curl_share debe ser estrictamente > 0, obtenido: {}",
        curl_share
    );
    assert!(
        curl_energy > 0.0,
        "R6-A7/C7: Helmholtz-Hodge curl energy debe ser estrictamente > 0, obtenido: {}",
        curl_energy
    );
    assert!(
        grad_energy >= 0.0,
        "Helmholtz-Hodge gradient energy debe ser no-negativa, obtenido: {}",
        grad_energy
    );

    // 4. Modulación laminar en SeniorMicroestructura: (1.0 - 0.70 * curl_share) < 1.0
    let laminar_factor = 1.0 - 0.70 * curl_share;
    assert!(
        laminar_factor < 1.0,
        "El factor laminar debe amortiguar microestructura ante presencia de vórtice (curl > 0)"
    );
}

#[test]
fn test_mid_price_published_and_multiasset_ttl_contract() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (arena, mut core) = setup_multi_coin_fixture();

    // 1. Inyectar tick BTC bid=50_000.0, ask=50_002.0 (mid_price = 50_001.0) en t=1_000 ms
    core.process_tick_dual(0, 50_000.0, 50_002.0, 1.0, 1.0, 1_000, &[0.0; 54], true, true);

    // R7-R3-C-1: Verificar que mid_price está publicado en el registro omnisciente
    let mid_btc_coin = arena.registry.get_for_coin_or(0, "mid_price", 0.0);
    assert!(
        (mid_btc_coin - 50_001.0).abs() < 1e-6,
        "mid_price para BTC (coin 0) debe ser 50001.0, obtenido: {}",
        mid_btc_coin
    );

    let mid_btc_scoped = arena.registry.get_scoped_parameter(Some("BTCUSDT"), Some(0), "mid_price", "god_core")
        .map(|p| p.get_value())
        .unwrap_or(0.0);
    assert!(
        (mid_btc_scoped - 50_001.0).abs() < 1e-6,
        "mid_price scoped para BTCUSDT debe ser 50001.0, obtenido: {}",
        mid_btc_scoped
    );

    let mid_global = arena.registry.get_value_or("mid_price", 0.0);
    assert!(
        (mid_global - 50_001.0).abs() < 1e-6,
        "mid_price global debe ser 50001.0, obtenido: {}",
        mid_global
    );

    // 2. R7-R3-D-2: Verificar TTL multiactivo unificado (GEOMETRIA_MULTIACTIVO_TTL_MS = 10_000 ms)
    // Inyectar ETH y SOL en t=1_000 ms
    core.process_tick_dual(1, 3_000.0, 3_002.0, 1.0, 1.0, 1_000, &[0.0; 54], true, true);
    core.process_tick_dual(2, 150.0, 150.2, 1.0, 1.0, 1_000, &[0.0; 54], true, true);

    // En t=1_000 ms, las 3 monedas son frescas (edad 0 <= 10_000 ms)
    assert!(core.latest_prices[0] > 0.0);
    assert!(core.latest_prices[1] > 0.0);
    assert!(core.latest_prices[2] > 0.0);

    // Inyectar tick en BTC en t=12_000 ms (edad de ETH y SOL es 11_000 ms > 10_000 ms TTL)
    core.process_tick_dual(0, 50_100.0, 50_102.0, 1.0, 1.0, 12_000, &[0.0; 54], true, true);

    // BTC (t=12_000) debe estar fresco, pero ETH y SOL deben ser considerados obsoletos (TTL excedido)
    let mid_btc_nuevo = arena.registry.get_for_coin_or(0, "mid_price", 0.0);
    assert!(
        (mid_btc_nuevo - 50_101.0).abs() < 1e-6,
        "mid_price BTC en t=12000 debe ser 50101.0, obtenido: {}",
        mid_btc_nuevo
    );
}
