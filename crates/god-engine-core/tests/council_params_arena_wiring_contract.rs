//! CONTRATO FORMAL: CABLEADO DINÁMICO DE PARÁMETROS DEL CONSEJO A QuantumConfig (Ola Ω95).
//!
//! # Por qué existe
//! Anteriormente, `CouncilParams` vivía con 6 constantes estáticas congeladas en `Default::default()`,
//! aisladas de `QuantumConfig` / `GlobalArena`. Toda mutación en caliente del genoma o ajuste
//! adaptativo de umbrales no afectaba la deliberación del consejo.
//!
//! # Solución (Ola Ω95)
//! 1. `QuantumConfig` expone los 6 parámetros del consejo como atómicos lock-free (`AtomicF64`):
//!    - `council_approval_threshold`
//!    - `council_supermajority_override`
//!    - `council_override_signal_floor`
//!    - `council_override_penalty`
//!    - `council_shrinkage_k`
//!    - `council_cascade_severity_breaker`
//! 2. `GodEngineCore::sync_council_params_from_arena` sincroniza de forma lock-free y segura
//!    (validando cotas) los parámetros de deliberación en cada ciclo de decisión.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

#[test]
fn council_params_sincronizan_dinamicamente_desde_arena_config() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena.clone());

    // 1. Valores canónicos iniciales en arena y consejo
    assert_eq!(core.consejo_deliberacion.params.approval_threshold, 0.35);
    assert_eq!(core.consejo_deliberacion.params.supermajority_override, 0.80);
    assert_eq!(core.consejo_deliberacion.params.override_signal_floor, 0.28);
    assert_eq!(core.consejo_deliberacion.params.override_penalty, 0.75);
    assert_eq!(core.consejo_deliberacion.params.shrinkage_k, 8.0);
    assert_eq!(core.consejo_deliberacion.params.cascade_severity_breaker, 0.85);

    // 2. Modificación lock-free desde arena.config (ej. Evolución adaptativa o feedback cuántico)
    arena.config.council_approval_threshold.store(0.45, Ordering::Relaxed);
    arena.config.council_supermajority_override.store(0.88, Ordering::Relaxed);
    arena.config.council_override_signal_floor.store(0.32, Ordering::Relaxed);
    arena.config.council_override_penalty.store(0.60, Ordering::Relaxed);
    arena.config.council_shrinkage_k.store(12.0, Ordering::Relaxed);
    arena.config.council_cascade_severity_breaker.store(0.92, Ordering::Relaxed);

    // 3. Ejecutar sincronización
    core.sync_council_params_from_arena();

    // 4. Verificar que el consejo adoptó los nuevos valores exactos
    assert_eq!(core.consejo_deliberacion.params.approval_threshold, 0.45);
    assert_eq!(core.consejo_deliberacion.params.supermajority_override, 0.88);
    assert_eq!(core.consejo_deliberacion.params.override_signal_floor, 0.32);
    assert_eq!(core.consejo_deliberacion.params.override_penalty, 0.60);
    assert_eq!(core.consejo_deliberacion.params.shrinkage_k, 12.0);
    assert_eq!(core.consejo_deliberacion.params.cascade_severity_breaker, 0.92);
}

#[test]
fn council_params_protege_contra_valores_corruptos_o_fuera_de_cota() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena.clone());

    // Estado base conocido
    let prev_approval = core.consejo_deliberacion.params.approval_threshold;
    let prev_k = core.consejo_deliberacion.params.shrinkage_k;

    // Intentar inyectar valor corrupto/fuera de cota (approval > 1.0, k negativo o NaN)
    arena.config.council_approval_threshold.store(2.50, Ordering::Relaxed);
    arena.config.council_shrinkage_k.store(-5.0, Ordering::Relaxed);

    core.sync_council_params_from_arena();

    // Deben preservarse los valores sanos anteriores
    assert_eq!(
        core.consejo_deliberacion.params.approval_threshold, prev_approval,
        "Approval corrupto > 1.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.shrinkage_k, prev_k,
        "Shrinkage k corrupto <= 0 no debe aplicarse al consejo"
    );
}
