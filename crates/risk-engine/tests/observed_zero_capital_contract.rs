//! CONTRATO FORMAL DE CAPITAL OBSERVADO CERO (Fase 2 / Ola Omega220)
//!
//! QUÉ: Certificación formal de la cadena de gobernanza de capital en `risk-engine`:
//!      - `capital_del_freno` distingue ausencia de saldo (NaN) de un saldo real observado.
//!      - Un saldo observado de 0.0 USDT en el exchange gobierna con `capital_del_freno == 0.0`.
//!      - `freno_duro` se dispara de inmediato ante capital observado cero o negativo.
//!
//! POR QUÉ: Si un saldo de 0.0 se interpretaba como "sin exchange" (backtest), el sistema
//!          seguía operando con el capital simulado del arena aunque la cuenta real estuviera
//!          liquidada o vacía.

use quantum_arena::GlobalArena;
use risk_engine::drawdown::{capital_del_freno, freno_duro};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn unobserved_exchange_capital_defaults_to_arena_simulation() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.unified_capital.store(150.0, Relaxed);
    // En el arranque o en backtest, capital_exchange es f64::NAN por diseño
    assert!(arena.capital_exchange.load(Relaxed).is_nan());
    assert_eq!(
        capital_del_freno(&arena),
        150.0,
        "Sin exchange observado (NaN), gobierna el arena"
    );
}

#[test]
fn observed_positive_exchange_capital_takes_conservative_minimum() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.unified_capital.store(120.0, Relaxed);
    arena.capital_exchange.store(95.0, Relaxed);
    assert_eq!(
        capital_del_freno(&arena),
        95.0,
        "Plano conservador: min(120, 95) = 95"
    );

    arena.unified_capital.store(80.0, Relaxed);
    assert_eq!(
        capital_del_freno(&arena),
        80.0,
        "Plano conservador: min(80, 95) = 80"
    );
}

#[test]
fn observed_zero_exchange_capital_governs_and_triggers_freno_duro() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.unified_capital.store(100.0, Relaxed);
    arena.pico_capital.store(100.0, Relaxed);

    // Binance envía ACCOUNT_UPDATE con wallet_balance = 0.0
    arena.capital_exchange.store(0.0, Relaxed);
    let cap = capital_del_freno(&arena);
    assert_eq!(cap, 0.0, "Cero observado en exchange DEBE gobernar (0.0)");

    let freno = freno_duro(&arena, 100.0, cap);
    assert!(
        freno.echado(),
        "Freno duro debe dispararse inmediatamente cuando cap <= 0.0"
    );
    assert!(
        freno.por_capital(),
        "Freno duro debe activarse por capital cuando cap <= 0.0"
    );
}

#[test]
fn observed_negative_exchange_capital_governs_and_triggers_freno_duro() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.unified_capital.store(50.0, Relaxed);

    // Saldo negativo por liquidación / deuda de margen
    arena.capital_exchange.store(-2.5, Relaxed);
    let cap = capital_del_freno(&arena);
    assert_eq!(cap, -2.5, "Saldo negativo observado en exchange debe gobernar");

    let freno = freno_duro(&arena, 100.0, cap);
    assert!(
        freno.echado(),
        "Freno duro debe dispararse inmediatamente cuando cap < 0.0"
    );
    assert!(
        freno.por_capital(),
        "Freno duro debe activarse por capital cuando cap < 0.0"
    );
}
