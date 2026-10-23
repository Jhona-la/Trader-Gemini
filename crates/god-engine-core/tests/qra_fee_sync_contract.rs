//! Contrato de paridad física de comisiones en admisión y ejecución interna (QRA-FEE-01).
//! Verifica que GodEngineCore inicialice y sincronice dinámicamente sus tasas con el arena
//! en cada tick de mercado, eliminando desalineaciones entre la física interna y el ledger.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

#[test]
fn god_engine_core_initializes_with_arena_fees() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena.config.live_maker_fee.store(0.00015, Ordering::Relaxed);
    arena.config.live_taker_fee.store(0.00045, Ordering::Relaxed);

    let core = GodEngineCore::new(arena);
    let (maker, taker) = core.reality_fees();
    assert!((maker - 0.00015).abs() < 1e-12, "maker fee inicial incorrecto: {maker}");
    assert!((taker - 0.00045).abs() < 1e-12, "taker fee inicial incorrecto: {taker}");
}

#[test]
fn god_engine_core_syncs_arena_fees_dynamically() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    // Tasas base VIP0 estándar
    arena.config.live_maker_fee.store(0.0002, Ordering::Relaxed);
    arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);

    let mut core = GodEngineCore::new(arena);
    let (m0, t0) = core.reality_fees();
    assert!((m0 - 0.0002).abs() < 1e-12);
    assert!((t0 - 0.0005).abs() < 1e-12);

    // Cambio en caliente de comisiones del exchange (ej. actualización de nivel VIP o promo)
    core.arena.config.live_maker_fee.store(0.0001, Ordering::Relaxed);
    core.arena.config.live_taker_fee.store(0.00035, Ordering::Relaxed);

    // Sincronización explícita
    core.sync_reality_fees_from_arena();
    let (m1, t1) = core.reality_fees();
    assert!((m1 - 0.0001).abs() < 1e-12, "maker no se sincronizó: {m1}");
    assert!((t1 - 0.00035).abs() < 1e-12, "taker no se sincronizó: {t1}");

    // Entradas y salidas deben reflejar la nueva física en dólares exactos
    let notional = 1000.0;
    let (_, fee_taker) = core
        .reality
        .calculate_market_entry(50000.0, true, notional, 0.001, 0.0005, 10.0);
    assert!((fee_taker - notional * 0.00035).abs() < 1e-10);

    let (_, fee_maker) = core
        .reality
        .calculate_exit(50000.0, true, notional, true, 0.001, 0.0005, 10.0);
    assert!((fee_maker - notional * 0.0001).abs() < 1e-10);

    // Tasas inválidas (negativas, NaN) deben ser rechazadas por fail-closed sin alterar las sanas
    core.arena.config.live_maker_fee.store(-0.01, Ordering::Relaxed);
    core.arena.config.live_taker_fee.store(f64::NAN, Ordering::Relaxed);
    core.sync_reality_fees_from_arena();

    let (m2, t2) = core.reality_fees();
    assert_eq!(m2, 0.0001, "maker corrupto tras tasa negativa");
    assert_eq!(t2, 0.00035, "taker corrupto tras tasa NaN");
}
