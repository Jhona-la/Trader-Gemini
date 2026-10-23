//! Constructor, sincronización EXPLÍCITA y cálculo directo de comisiones.
//! Estos casos no procesan ticks ni demuestran un débito o cierre del ledger.
//! El contrato por eventos reales está en qra_fee_tick_accounting_contract.rs.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

#[test]
fn god_engine_core_initializes_with_arena_fees() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .live_maker_fee
        .store(0.00015, Ordering::Relaxed);
    arena
        .config
        .live_taker_fee
        .store(0.00045, Ordering::Relaxed);

    let core = GodEngineCore::new(arena);
    let (maker, taker) = core.reality_fees();
    assert!(
        (maker - 0.00015).abs() < 1e-12,
        "maker fee inicial incorrecto: {maker}"
    );
    assert!(
        (taker - 0.00045).abs() < 1e-12,
        "taker fee inicial incorrecto: {taker}"
    );
}

#[test]
fn explicit_sync_preserves_signed_finite_rates_and_retains_known_close_estimates() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    // Tasas base VIP0 estándar
    arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
    arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);

    let mut core = GodEngineCore::new(arena);
    let (m0, t0) = core.reality_fees();
    assert!((m0 - 0.0002).abs() < 1e-12);
    assert!((t0 - 0.0005).abs() < 1e-12);

    // Cambio en caliente de comisiones del exchange (ej. actualización de nivel VIP o promo)
    core.arena
        .config
        .live_maker_fee
        .store(0.0001, Ordering::Relaxed);
    core.arena
        .config
        .live_taker_fee
        .store(0.00035, Ordering::Relaxed);

    // Sincronización explícita
    core.sync_reality_fees_from_arena();
    let (m1, t1) = core.reality_fees();
    assert!((m1 - 0.0001).abs() < 1e-12, "maker no se sincronizó: {m1}");
    assert!((t1 - 0.00035).abs() < 1e-12, "taker no se sincronizó: {t1}");

    // Cálculo directo sobre el nocional suministrado; no debita el ledger.
    let notional = 1000.0;
    let (_, fee_taker) = core
        .reality
        .calculate_market_entry(50000.0, true, notional, 0.001, 0.0005, 10.0);
    assert!((fee_taker - notional * 0.00035).abs() < 1e-10);

    let (_, fee_maker) = core
        .reality
        .calculate_exit(50000.0, true, notional, true, 0.001, 0.0005, 10.0);
    assert!((fee_maker - notional * 0.0001).abs() < 1e-10);

    // Cero y tasas negativas finitas conservan el dominio de la evidencia REST.
    core.arena
        .config
        .live_maker_fee
        .store(-0.01, Ordering::Relaxed);
    core.arena
        .config
        .live_taker_fee
        .store(0.0, Ordering::Relaxed);
    core.sync_reality_fees_from_arena();
    assert_eq!(core.reality_fees(), (-0.01, 0.0));

    // Retener una estimación de cierre no vuelve finita ni actual la observación.
    // Este caso directo no certifica que la admisión del tick quede bloqueada.
    core.arena
        .config
        .live_maker_fee
        .store(f64::NAN, Ordering::Relaxed);
    core.arena
        .config
        .live_taker_fee
        .store(f64::INFINITY, Ordering::Relaxed);
    core.sync_reality_fees_from_arena();

    let (m2, t2) = core.reality_fees();
    assert_eq!(m2, -0.01, "último maker finito de la estimación de cierre");
    assert_eq!(t2, 0.0, "último taker finito de la estimación de cierre");
}
