//! C-F3.7 (Línea C), F3.7-07: el núcleo no debe calcular una cotización
//! maker que nadie consume.
//!
//! Teoría que se prueba: `process_tick_dual` devuelve como tercer valor una
//! `MakerQuote` que calcula cuando la latencia del WS es ≤ 25 ms (bloque
//! «MARKET MAKING» de lib.rs: recorre los slots para el inventario y lee
//! cuatro genes). Ningún llamador vivo la lee: el host la descarta (`_maker`)
//! y el backtest de evolución también (`_`). Es CPU por evento no bloqueado
//! sin ninguna decisión, y los genes maker_spread_pct y maker_obi_threshold
//! no gobiernan nada en el núcleo vivo. Contrato: sin consumidor, ninguna
//! cotización.
//!
//! Notas de los escépticos: la cotización sale en cada evento no bloqueado
//! (trades y velas), no en cada tick; maker_spread_pct sigue en el simulador
//! de backtest-engine. Retirarla cambia la firma de `process_tick` y exige el
//! oráculo T-1.
//!
//! Convención: `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert
//! intacto mientras el hallazgo siga abierto.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering;

/// 50 ticks de 1 s con mid 50 000 + (i mod 3), libro válido (spread 1 USD,
/// cantidades 1/1), entradas permitidas y la latencia WS por defecto del
/// arena (0 ms ≤ 25). Esperado: 0 cotizaciones de 50.
#[test]
#[ignore = "OPEN F3.7-07 (Línea C): process_tick_dual calcula una MakerQuote (latencia ≤ 25 ms) en cada evento no bloqueado y los dos llamadores la descartan"]
fn cf3_el_nucleo_no_calcula_cotizacion_sin_consumidor() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    let latencia = arena.last_ws_latency_ms.load(Ordering::Relaxed);
    assert!(
        latencia <= 25,
        "premisa: latencia WS inicial {latencia} ms (el bloque maker exige ≤ 25)"
    );
    let mut cotizaciones = 0;
    let mut ts = 1_000u64;
    for i in 0..50u64 {
        let mid = 50_000.0 + (i % 3) as f64;
        let (_, _, q) = core.process_tick_dual(
            0,
            mid - 0.5,
            mid + 0.5,
            1.0,
            1.0,
            ts,
            &[0.0; 54],
            true,
            true,
        );
        if q.is_some() {
            cotizaciones += 1;
        }
        ts += 1_000;
    }
    println!("observado: {cotizaciones} cotizaciones maker en 50 ticks, ningún consumidor");
    assert_eq!(
        cotizaciones, 0,
        "cotización maker calculada sin consumidor en {cotizaciones} de 50 ticks"
    );
}
