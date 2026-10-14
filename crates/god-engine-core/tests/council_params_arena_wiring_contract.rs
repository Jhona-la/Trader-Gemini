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
    assert_eq!(core.consejo_deliberacion.params.utility_veto_floor, 0.02);
    assert_eq!(core.consejo_deliberacion.params.utility_veto_wr, 0.35);
    assert_eq!(core.consejo_deliberacion.params.fast_slippage_floor, 35.0);
    assert_eq!(core.consejo_deliberacion.params.slow_slippage_span, 65.0);
    assert_eq!(core.consejo_deliberacion.params.max_drawdown_limit, 0.6315968501359614);
    assert_eq!(core.consejo_deliberacion.params.streak_decay_rate, 0.50);
    assert_eq!(core.consejo_deliberacion.params.causal_veto_threshold, 0.75);
    assert_eq!(core.consejo_deliberacion.params.causal_min_book_imbalance, 0.25);
    assert_eq!(core.consejo_deliberacion.params.causal_aligned_max_risk, 0.88);
    assert_eq!(core.consejo_deliberacion.params.hodge_curl_attenuation, 0.70);
    assert_eq!(core.consejo_deliberacion.params.hodge_laminar_floor, 0.30);
    assert_eq!(core.consejo_deliberacion.params.ym_ref_current, 0.05);
    assert_eq!(core.consejo_deliberacion.params.ym_max_weight, 0.20);

    // 2. Modificación lock-free desde arena.config (ej. Evolución adaptativa o feedback cuántico)
    arena.config.council_approval_threshold.store(0.45, Ordering::Relaxed);
    arena.config.council_supermajority_override.store(0.88, Ordering::Relaxed);
    arena.config.council_override_signal_floor.store(0.32, Ordering::Relaxed);
    arena.config.council_override_penalty.store(0.60, Ordering::Relaxed);
    arena.config.council_shrinkage_k.store(12.0, Ordering::Relaxed);
    arena.config.council_cascade_severity_breaker.store(0.92, Ordering::Relaxed);
    arena.config.council_utility_veto_floor.store(0.012, Ordering::Relaxed);
    arena.config.council_utility_veto_wr.store(0.30, Ordering::Relaxed);
    arena.config.council_fast_slippage_floor.store(45.0, Ordering::Relaxed);
    arena.config.council_slow_slippage_span.store(80.0, Ordering::Relaxed);
    arena.config.council_max_drawdown_limit.store(0.40, Ordering::Relaxed);
    arena.config.council_streak_decay_rate.store(0.75, Ordering::Relaxed);
    arena.config.council_causal_veto_threshold.store(0.65, Ordering::Relaxed);
    arena.config.council_causal_min_book_imbalance.store(0.35, Ordering::Relaxed);
    arena.config.council_causal_aligned_max_risk.store(0.92, Ordering::Relaxed);
    arena.config.council_hodge_curl_attenuation.store(0.85, Ordering::Relaxed);
    arena.config.council_hodge_laminar_floor.store(0.15, Ordering::Relaxed);
    arena.config.council_ym_ref_current.store(0.08, Ordering::Relaxed);
    arena.config.council_ym_max_weight.store(0.30, Ordering::Relaxed);

    // 3. Ejecutar sincronización
    core.sync_council_params_from_arena();

    // 4. Verificar que el consejo adoptó los nuevos valores exactos
    assert_eq!(core.consejo_deliberacion.params.approval_threshold, 0.45);
    assert_eq!(core.consejo_deliberacion.params.supermajority_override, 0.88);
    assert_eq!(core.consejo_deliberacion.params.override_signal_floor, 0.32);
    assert_eq!(core.consejo_deliberacion.params.override_penalty, 0.60);
    assert_eq!(core.consejo_deliberacion.params.shrinkage_k, 12.0);
    assert_eq!(core.consejo_deliberacion.params.cascade_severity_breaker, 0.92);
    assert_eq!(core.consejo_deliberacion.params.utility_veto_floor, 0.012);
    assert_eq!(core.consejo_deliberacion.params.utility_veto_wr, 0.30);
    assert_eq!(core.consejo_deliberacion.params.fast_slippage_floor, 45.0);
    assert_eq!(core.consejo_deliberacion.params.slow_slippage_span, 80.0);
    assert_eq!(core.consejo_deliberacion.params.max_drawdown_limit, 0.40);
    assert_eq!(core.consejo_deliberacion.params.streak_decay_rate, 0.75);
    assert_eq!(core.consejo_deliberacion.params.causal_veto_threshold, 0.65);
    assert_eq!(core.consejo_deliberacion.params.causal_min_book_imbalance, 0.35);
    assert_eq!(core.consejo_deliberacion.params.causal_aligned_max_risk, 0.92);
    assert_eq!(core.consejo_deliberacion.params.hodge_curl_attenuation, 0.85);
    assert_eq!(core.consejo_deliberacion.params.hodge_laminar_floor, 0.15);
    assert_eq!(core.consejo_deliberacion.params.ym_ref_current, 0.08);
    assert_eq!(core.consejo_deliberacion.params.ym_max_weight, 0.30);
}

#[test]
fn council_params_protege_contra_valores_corruptos_o_fuera_de_cota() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena.clone());

    // Estado base conocido
    let prev_approval = core.consejo_deliberacion.params.approval_threshold;
    let prev_k = core.consejo_deliberacion.params.shrinkage_k;
    let prev_max_dd = core.consejo_deliberacion.params.max_drawdown_limit;
    let prev_decay = core.consejo_deliberacion.params.streak_decay_rate;
    let prev_causal_veto = core.consejo_deliberacion.params.causal_veto_threshold;
    let prev_causal_imb = core.consejo_deliberacion.params.causal_min_book_imbalance;
    let prev_causal_risk = core.consejo_deliberacion.params.causal_aligned_max_risk;
    let prev_curl_att = core.consejo_deliberacion.params.hodge_curl_attenuation;
    let prev_lam_floor = core.consejo_deliberacion.params.hodge_laminar_floor;
    let prev_ym_ref = core.consejo_deliberacion.params.ym_ref_current;
    let prev_ym_max_w = core.consejo_deliberacion.params.ym_max_weight;

    // Intentar inyectar valor corrupto/fuera de cota
    arena.config.council_approval_threshold.store(2.50, Ordering::Relaxed);
    arena.config.council_shrinkage_k.store(-5.0, Ordering::Relaxed);
    arena.config.council_max_drawdown_limit.store(1.50, Ordering::Relaxed);
    arena.config.council_streak_decay_rate.store(-2.0, Ordering::Relaxed);
    arena.config.council_causal_veto_threshold.store(1.50, Ordering::Relaxed);
    arena.config.council_causal_min_book_imbalance.store(-0.50, Ordering::Relaxed);
    arena.config.council_causal_aligned_max_risk.store(1.20, Ordering::Relaxed);
    arena.config.council_hodge_curl_attenuation.store(1.50, Ordering::Relaxed);
    arena.config.council_hodge_laminar_floor.store(-0.10, Ordering::Relaxed);
    arena.config.council_ym_ref_current.store(-0.05, Ordering::Relaxed);
    arena.config.council_ym_max_weight.store(2.0, Ordering::Relaxed);

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
    assert_eq!(
        core.consejo_deliberacion.params.max_drawdown_limit, prev_max_dd,
        "Max drawdown corrupto > 1.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.streak_decay_rate, prev_decay,
        "Streak decay corrupto < 0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.causal_veto_threshold, prev_causal_veto,
        "Causal veto threshold corrupto > 1.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.causal_min_book_imbalance, prev_causal_imb,
        "Causal min book imbalance corrupto < 0.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.causal_aligned_max_risk, prev_causal_risk,
        "Causal aligned max risk corrupto > 1.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.hodge_curl_attenuation, prev_curl_att,
        "Hodge curl attenuation corrupto > 1.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.hodge_laminar_floor, prev_lam_floor,
        "Hodge laminar floor corrupto < 0.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.ym_ref_current, prev_ym_ref,
        "YM ref current corrupto <= 0.0 no debe aplicarse al consejo"
    );
    assert_eq!(
        core.consejo_deliberacion.params.ym_max_weight, prev_ym_max_w,
        "YM max weight corrupto > 1.0 no debe aplicarse al consejo"
    );
}
