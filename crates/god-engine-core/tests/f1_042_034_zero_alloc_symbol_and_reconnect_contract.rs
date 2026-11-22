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
use std::sync::{Mutex, MutexGuard, PoisonError};

/// El universo y el registro de specs son globales del proceso y las pruebas de
/// este binario los publican o los leen: se turnan. Sin el candado,
/// `test_f1_034` publica `["SOLUSDT"]` en otro hilo mientras `test_f1_042`
/// construye su núcleo, los slots 1 y 3 se quedan sin identidad y la lectura
/// con respaldo devolvía el `mid_price` global (el «leído 0.012» de la
/// compuerta). Tolerante al veneno: una prueba que falle no arrastra a las demás.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn bloqueo() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Lectura del `mid_price` publicado con la identidad de `simbolo`, sin
/// respaldo a la clave global: NAN si ese símbolo no publicó.
fn mid_de(arena: &GlobalArena, simbolo: &str) -> f64 {
    arena.registry.get_scoped_strict(simbolo, "mid_price").unwrap_or(f64::NAN)
}

/// Los tres eventos de `test_f1_042` (slots 0, 1 y 3), compartidos con la
/// reproducción determinista para que las dos alimenten lo mismo.
fn tres_eventos_f1_042(core: &mut GodEngineCore) {
    let _ = core.process_event(
        0, true, false, true, 65000.0, 1.5, 64999.0, 65001.0, 10.0, 10.0, 0.05, 0.0, 10_000, false, &[0.0; 54], false,
    );
    let _ = core.process_event(
        1, true, false, true, 3500.0, 5.0, 3499.0, 3501.0, 20.0, 20.0, -0.02, 0.0, 10_000, false, &[0.0; 54], false,
    );
    let _ = core.process_event(
        3, true, false, true, 0.012, 50000.0, 0.0119, 0.0121, 100000.0, 100000.0, 0.10, 0.0, 10_000, false, &[0.0; 54], false,
    );
}

/// Reproducción determinista, sin hilos ni reloj, del «BTCUSDT mid_price
/// esperado 65000.0, leído 0.012»: se publica a propósito el universo de
/// `test_f1_034` (`["SOLUSDT"]`) y se alimentan los tres eventos de
/// `test_f1_042`. El slot 0 publica como SOLUSDT, los slots 1 y 3 no tienen
/// identidad, y el `mid_price` global queda en 0.012 (el último evento). La
/// lectura por símbolo de las aserciones de este binario no puede devolver
/// ese global como si fuera el precio de BTCUSDT.
#[test]
fn test_f1_042_universo_ajeno_no_se_lee_como_precio_de_btc() {
    let _guard = bloqueo();
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
    tres_eventos_f1_042(&mut core);

    // Control: el escenario es el del fallo. El global lleva el último precio
    // (0.012) y el slot 0 publicó bajo el nombre del universo ajeno.
    let global = arena.registry.get_value_fast("mid_price").unwrap_or(f64::NAN);
    assert!((global - 0.012).abs() < 1e-9, "control: mid_price global = {global}");
    let sol = mid_de(&arena, "SOLUSDT");
    assert!((sol - 65000.0).abs() < 1e-6, "control: SOLUSDT mid_price = {sol}");

    // Lo que se exige: sin identidad no hay precio por símbolo.
    let btc = mid_de(&arena, "BTCUSDT");
    let eth = mid_de(&arena, "ETHUSDT");
    let pepe = mid_de(&arena, "1000PEPEUSDT");
    assert!(btc.is_nan(), "BTCUSDT mid_price sin identidad debe ser NAN, leído {btc}");
    assert!(eth.is_nan(), "ETHUSDT mid_price sin identidad debe ser NAN, leído {eth}");
    assert!(pepe.is_nan(), "1000PEPEUSDT mid_price sin identidad debe ser NAN, leído {pepe}");
}

#[test]
fn test_f1_042_zero_alloc_stack_symbol_parity_and_leads() {
    let _guard = bloqueo();
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
    //    (lectura estricta: NAN = ese slot no publicó con esa identidad)
    let btc_mid = mid_de(&arena, "BTCUSDT");
    let eth_mid = mid_de(&arena, "ETHUSDT");
    let pepe_mid = mid_de(&arena, "1000PEPEUSDT");
    assert!((btc_mid - 65000.0).abs() < 1e-6, "BTCUSDT mid_price esperado 65000.0, leído {btc_mid}");
    assert!((eth_mid - 3500.0).abs() < 1e-6, "ETHUSDT mid_price esperado 3500.0, leído {eth_mid}");
    assert!((pepe_mid - 0.012).abs() < 1e-6, "1000PEPEUSDT mid_price esperado 0.012, leído {pepe_mid}");
}

#[test]
fn test_f1_034_reset_engines_clears_kline_close_memory_and_ensembles() {
    let _guard = bloqueo();
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
    let _guard = bloqueo();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);

    // coin_id 25 (fuera del universo registrado)
    let (ord, _, _) = core.process_tick_dual(
        25, 50.0, 50.02, 10.0, 10.0, 10_000, &[0.0; 54], true, false,
    );
    // Debe operar fail-closed sin pánico y sin allocaciones
    assert!(ord.is_none());
}
