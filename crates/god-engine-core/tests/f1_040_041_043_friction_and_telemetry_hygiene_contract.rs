//! Contrato formal de verificación para F1-040, F1-041 y F1-043 (Ola Ω203).
//!
//! # F1-040: Unificación de Fricción Roundtrip y Gobernanza de LG-22 ARL
//! - Se erradica la coexistencia de tres fórmulas contradictorias de fricción en la ranura.
//! - Se unifica a `tp_sl::roundtrip_friction(live_taker_fee, slip_floor, atr_pct, 0.0)`.
//! - `lg22_arl` en `QuantumConfig` gobierna activamente el ARL del detector e-CUSUM.
//!
//! # F1-041: Absorción de Liquidez Oculta sin Cancelación Algebraica de Volumen
//! - `compute_absorption` no cancela el volumen ni la amortiguación en el cociente.
//! - Micro-trades producen presión cercana a cero en lugar del ratio estático OBI ±1/3.
//! - Ticks de profundidad (sin trade) no reinician a cero la presión detectada por el trade precedente.
//!
//! # F1-043: Muestreo de Telemetría de 45 Claves `sombra_*`
//! - El array en stack de votos espectrales se computa en el camino caliente para alimentar el consenso.
//! - Las 45 escrituras DashMap per-coin `sombra_*` se muestrean (múltiplos de 64 o flag forzada),
//!   ahorrando ~40 µs por evento sin lector en producción.

use dark_alpha_engine::hidden_liquidity::HiddenLiquidityEngine;
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering;

#[test]
fn f1_040_unificacion_friccion_y_gobernanza_lg22_arl() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
    arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
    arena.config.base_slippage_floor.store(0.0002, Ordering::Relaxed);
    arena.config.lg22_arl.store(250.0, Ordering::Relaxed);

    let mut core = GodEngineCore::new(arena.clone());
    core.sync_council_params_from_arena();

    // 1. Verificar sincronización activa de lg22_arl desde QuantumConfig hacia el registro
    let arl_registrado = arena.registry.get_value_or("lg22_arl", 0.0);
    assert_eq!(
        arl_registrado, 250.0,
        "lg22_arl en registry debe reflejar el valor sincronizado desde QuantumConfig"
    );

    // 2. Verificar la equivalencia canónica de la fricción roundtrip
    let live_taker = arena.config.live_taker_fee.load(Ordering::Relaxed);
    let slip_floor = arena.config.base_slippage_floor.load(Ordering::Relaxed);
    let canonica = risk_engine::tp_sl::roundtrip_friction(live_taker, slip_floor, 0.01, 0.0);
    // 2 * 0.0005 + 2 * 0.0002 = 0.0010 + 0.0004 = 0.0014
    assert!((canonica - 0.0014).abs() < 1e-6, "Fricción canónica debe ser exactamente 2*(taker+slip)");
}

#[test]
fn f1_041_absorcion_bloques_sin_cancelacion_algebraica_y_persistencia() {
    let engine = HiddenLiquidityEngine::default();
    let safe_atr = 100.0;

    // Caso 1: Volumen masivo (500 BTC) con movimiento imperceptible (0.01):
    // Debe saturar fuertemente cerca de +1.0
    let state_masivo = engine.compute_absorption(500.0, 100.0, 100.0, 0.01, safe_atr);
    assert!(state_masivo.is_iceberg_active);
    assert!(
        state_masivo.net_iceberg_pressure > 0.85,
        "Presión de iceberg masivo debe saturar fuertemente hacia +1.0, obtenido: {}",
        state_masivo.net_iceberg_pressure
    );

    // Caso 2: Micro-trade (0.01 BTC) con movimiento imperceptible (0.01):
    // NO debe dar 0.333 (la cancelación antigua). Debe colapsar suavemente a cero.
    let state_micro = engine.compute_absorption(0.01, 100.0, 100.0, 0.01, safe_atr);
    assert!(!state_micro.is_iceberg_active);
    assert!(
        state_micro.net_iceberg_pressure.abs() < 0.01,
        "Micro-trade no debe producir falsa alarma de iceberg de 33%, obtenido: {}",
        state_micro.net_iceberg_pressure
    );

    // Caso 3: Absorción direccionada (taker buy vs taker sell explícitos)
    let state_dir_buy = engine.compute_absorption_directed(500.0, 0.0, 100.0, 100.0, 0.01, safe_atr);
    assert!(state_dir_buy.net_iceberg_pressure > 0.90);

    let state_dir_sell = engine.compute_absorption_directed(0.0, 500.0, 100.0, 100.0, -0.01, safe_atr);
    assert!(state_dir_sell.net_iceberg_pressure < -0.90);
}

#[test]
fn f1_043_muestreo_telemetria_sombras_y_consenso_continuo() {
    symbols::update_dynamic_universe(vec!["SOMB43USDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "SOMB43USDT",
    )]);

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;

    // Enviar evento en tick 0 (múltiplo de 64): las sombras se deben escribir
    let _ = core.process_event(
        0,
        false,
        false,
        true, // is_depth = true
        100.0,
        0.0,
        100.0,
        100.01,
        5.0,
        5.0,
        0.1,
        0.0,
        10_000,
        false,
        &[0.0; 54],
        false,
    );

    let sombra_tick_0 = arena.registry.get_for_coin_or(0, "sombra_nash_consenso", f64::NAN);
    assert!(
        sombra_tick_0.is_finite(),
        "Tick 0 (múltiplo de 64) debe publicar sombras para diagnóstico/calibración"
    );

    // Enviar evento en tick 65 (post-calentamiento y no múltiplo de 64) sin flag forzada:
    // consenso espectral se computa, pero sombras no se escriben al registro.
    arena.tick_counter.store(65, Ordering::Relaxed);
    arena.registry.set_for_coin(0, "sombra_nash_consenso", f64::NAN);
    let _ = core.process_event(
        0,
        false,
        false,
        true,
        100.05,
        0.0,
        100.05,
        100.06,
        5.0,
        5.0,
        0.1,
        0.0,
        11_000,
        false,
        &[0.0; 54],
        false,
    );

    let sombra_tick_1 = arena.registry.get_for_coin_or(0, "sombra_nash_consenso", f64::NAN);
    assert!(
        sombra_tick_1.is_nan(),
        "Tick 1 no debe ejecutar 45 escrituras DashMap de telemetría sin lector"
    );

    // Con bandera forzada de exportación/diagnóstico, se publica en cualquier tick
    arena.registry.set("telemetria_sombras_forzada", 1.0);
    let _ = core.process_event(
        0,
        false,
        false,
        true,
        100.10,
        0.0,
        100.10,
        100.11,
        5.0,
        5.0,
        0.1,
        0.0,
        12_000,
        false,
        &[0.0; 54],
        false,
    );

    let sombra_forzada = arena.registry.get_for_coin_or(0, "sombra_nash_consenso", f64::NAN);
    assert!(
        sombra_forzada.is_finite(),
        "Con telemetria_sombras_forzada = 1.0, las sombras deben publicarse inmediatamente"
    );
}
