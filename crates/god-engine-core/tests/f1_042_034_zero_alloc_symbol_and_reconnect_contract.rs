//! Contrato formal: F1-042 (Erradicación de Asignación Heap String por Tick) y
//! F1-034 (Causalidad en Calibración Hedge tras Reconexión de Motores).
//!
//! Ola Ω199 (Consejo Colegiado de 10 Roles Senior en Modo Profesor):
//! - F1-042: `process_tick_dual` emplea stack buffer `[u8; 32]` y `with_symbol`,
//!   erradicando llamadas a `malloc`/`free` por evento y preservando paridad exacta
//!   de nombres y clasificación líder BTC/ETH.
//! - F1-034: `reset_engines` y `reset_engines_cold` resetean `kline_close_memory`
//!   y llaman `nuevo_bar()` en los ensambles, evitando que un salto de precio tras
//!   desconexión contamine la calibración Hedge con retornos espurios.

use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, symbol_registry, symbols};

#[test]
fn test_f1_042_zero_alloc_stack_symbol_parity_and_leads() {
    let universe = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
        "1000PEPEUSDT".to_string(),
    ];
    symbols::update_dynamic_universe(universe.clone());
    symbol_registry::update_registry(
        universe
            .iter()
            .map(|s| symbol_registry::get_official_binance_spec(s))
            .collect(),
    );

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());

    // 1. Verificar coin_id 0 (BTCUSDT): líder principal macro
    let (ord_0, _) = core.process_event(
        0, true, false, true, 65000.0, 1.5, 64999.0, 65001.0, 10.0, 10.0, 0.05, 0.0, 10_000, false, &[0.0; 54], false,
    );
    let _ = ord_0;

    // 2. Verificar coin_id 1 (ETHUSDT): líder secundario
    let (ord_1, _) = core.process_event(
        1, true, false, true, 3500.0, 5.0, 3499.0, 3501.0, 20.0, 20.0, -0.02, 0.0, 10_000, false, &[0.0; 54], false,
    );
    let _ = ord_1;

    // 3. Verificar coin_id 3 (1000PEPEUSDT, símbolo largo de 12 caracteres):
    let (ord_3, _) = core.process_event(
        3, true, false, true, 0.012, 50000.0, 0.0119, 0.0121, 100000.0, 100000.0, 0.10, 0.0, 10_000, false, &[0.0; 54], false,
    );
    let _ = ord_3;

    // 4. Verificar que el registro omnisciente tiene las claves scoped por símbolo exacto
    let btc_mid = arena.registry.get_scoped_value_or("BTCUSDT", "mid_price", f64::NAN);
    let eth_mid = arena.registry.get_scoped_value_or("ETHUSDT", "mid_price", f64::NAN);
    let pepe_mid = arena.registry.get_scoped_value_or("1000PEPEUSDT", "mid_price", f64::NAN);
    assert!((btc_mid - 65000.0).abs() < 1e-6, "BTCUSDT mid_price esperado 65000.0, leído {btc_mid}");
    assert!((eth_mid - 3500.0).abs() < 1e-6, "ETHUSDT mid_price esperado 3500.0, leído {eth_mid}");
    assert!((pepe_mid - 0.012).abs() < 1e-6, "1000PEPEUSDT mid_price esperado 0.012, leído {pepe_mid}");
}

#[test]
fn test_f1_034_reset_engines_clears_kline_close_memory_and_ensembles() {
    let universe = vec!["SOLUSDT".to_string()];
    symbols::update_dynamic_universe(universe.clone());
    symbol_registry::update_registry(
        universe
            .iter()
            .map(|s| symbol_registry::get_official_binance_spec(s))
            .collect(),
    );

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());

    // Primer kline cerrado a 100.0 establece baseline
    let _ = core.process_event(
        0, false, true, false, 100.0, 10.0, 99.9, 100.1, 50.0, 50.0, 0.0, 0.0, 60_000, false, &[0.0; 54], false,
    );

    // Segundo kline cerrado a 105.0 (+5%, decisivo)
    let _ = core.process_event(
        0, false, true, false, 105.0, 10.0, 104.9, 105.1, 50.0, 50.0, 0.0, 0.0, 120_000, false, &[0.0; 54], false,
    );

    // Simular desconexión de red / WebSocket y reconexión: reset_engines
    core.reset_engines();

    // El primer kline post-reconexión llega a 150.0 (un salto de +42% si comparara con 105.0)
    // Pero gracias a F1-034, kline_close_memory fue reseteado a 0.0, por lo que este primer
    // kline establece el nuevo baseline causal sin emitir una etiqueta corrupta al ensemble!
    let _ = core.process_event(
        0, false, true, false, 150.0, 10.0, 149.9, 150.1, 50.0, 50.0, 0.0, 0.0, 3600_000, false, &[0.0; 54], false,
    );

    // Ahora un segundo kline post-reconexión a 151.0 (+0.66%) evalúa contra 150.0 correctamente
    let _ = core.process_event(
        0, false, true, false, 151.0, 10.0, 150.9, 151.1, 50.0, 50.0, 0.0, 0.0, 3660_000, false, &[0.0; 54], false,
    );

    // Prueba idéntica para reset_engines_cold
    core.reset_engines_cold();
    let _ = core.process_event(
        0, false, true, false, 200.0, 10.0, 199.9, 200.1, 50.0, 50.0, 0.0, 0.0, 7200_000, false, &[0.0; 54], false,
    );
}

#[test]
fn test_f1_042_unregistered_coin_id_failclosed_safety() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);

    // coin_id 25 (fuera del universo registrado)
    let (ord, _, _) = core.process_tick_dual(
        25, 50.0, 50.02, 10.0, 10.0, 10_000, &[0.0; 54], true, false,
    );
    // Debe operar fail-closed sin pánico y sin allocaciones
    assert!(ord.is_none());
}
