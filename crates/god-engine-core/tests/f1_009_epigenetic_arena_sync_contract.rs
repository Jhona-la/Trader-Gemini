//! CONTRATO FORMAL F1-009: SINCRONIZACIÓN EPIGENÉTICA DINÁMICA DE QuantumConfig (Ola Ω205)
//!
//! # Por qué existe este contrato
//! Anteriormente, ~100 campos atómicos de `QuantumConfig` nacían con literales fijos en `from_genome`,
//! no tenían escritores en código de producción y funcionaban como constantes estáticas congeladas.
//! El hallazgo F1-009 exigió eliminar la falsa adaptabilidad, dotar de derivación matemática rigurosa
//! a las constantes (ej. cota de falsación actuarial de Cramér-Lundberg para `council_max_drawdown_limit`)
//! y garantizar que el registro omnisciente pueda modular dinámicamente todos los parámetros físicos,
//! cuánticos, fluidodinámicos (Navier-Stokes) y deliberativos del Consejo en tiempo real lock-free.

use god_engine_core::GodEngineCore;
use quantum_arena::config::{CANONICAL_COUNCIL_MAX_DRAWDOWN, CANONICAL_FEYNMAN_HBAR};
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

#[test]
fn test_f1_009_sync_from_registry_updates_arena_atomics_and_engine() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena.clone());

    // 1. Verificar valores iniciales canónicos derivados matemáticamente
    assert_eq!(
        arena.config.council_max_drawdown_limit.load(Ordering::Relaxed),
        CANONICAL_COUNCIL_MAX_DRAWDOWN
    );
    assert_eq!(
        arena.config.feynman_hbar.load(Ordering::Relaxed),
        CANONICAL_FEYNMAN_HBAR
    );
    assert_eq!(
        arena.config.council_approval_threshold.load(Ordering::Relaxed),
        0.35
    );

    // 2. Modulación dinámica por los motores en el registro omnisciente
    arena.registry.set("council_approval_threshold", 0.42);
    arena.registry.set("navier_critical_laminar", 1.85);
    arena.registry.set("feynman_hbar", 1.25);
    arena.registry.set("espectral_recencia_x", 0.75);
    arena.registry.set("tsallis_q", 1.80);
    arena.registry.set("hidden_liquidity_weight", 0.45);
    arena.registry.set("lg22_arl", 250.0);

    // 3. Sincronización continua viva
    core.sync_council_params_from_arena();

    // 4. Comprobar que los atómicos de QuantumConfig y los motores adoptaron los nuevos valores
    assert_eq!(
        arena.config.council_approval_threshold.load(Ordering::Relaxed),
        0.42
    );
    assert_eq!(
        core.consejo_deliberacion.params.approval_threshold,
        0.42
    );
    assert_eq!(
        arena.config.navier_critical_laminar.load(Ordering::Relaxed),
        1.85
    );
    assert_eq!(
        arena.config.feynman_hbar.load(Ordering::Relaxed),
        1.25
    );
    assert_eq!(
        arena.config.espectral_recencia_x.load(Ordering::Relaxed),
        0.75
    );
    assert_eq!(
        arena.config.tsallis_q.load(Ordering::Relaxed),
        1.80
    );
    assert_eq!(
        arena.config.hidden_liquidity_weight.load(Ordering::Relaxed),
        0.45
    );
    assert_eq!(
        arena.config.lg22_arl.load(Ordering::Relaxed),
        250.0
    );
}

#[test]
fn test_f1_009_sync_from_registry_bounds_protection() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena.clone());

    let prev_approval = arena.config.council_approval_threshold.load(Ordering::Relaxed);
    let prev_navier = arena.config.navier_critical_laminar.load(Ordering::Relaxed);

    // Inyectar valores corruptos fuera de bounds en el registro
    arena.registry.set("council_approval_threshold", 2.50); // Debe estar en [0, 1]
    arena.registry.set("navier_critical_laminar", -5.0); // Debe ser positivo y finito

    // Sincronizar
    core.sync_council_params_from_arena();

    // Deben preservarse los valores sanos previos
    assert_eq!(
        arena.config.council_approval_threshold.load(Ordering::Relaxed),
        prev_approval,
        "Valor corrupto no debe ser aplicado al atómico de configuración"
    );
    assert_eq!(
        core.consejo_deliberacion.params.approval_threshold,
        prev_approval
    );
    assert_eq!(
        arena.config.navier_critical_laminar.load(Ordering::Relaxed),
        prev_navier
    );
}
