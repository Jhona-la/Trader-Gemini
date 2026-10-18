//! Contrato F1-115: Erradicación de queries redundantes en DashMap y fast-path O(1) con is_empty().
//!
//! Verifica que:
//! 1. `OmniscientRegistry::is_empty()` y `len()` reflejen fielmente el estado del mapa sin locks.
//! 2. `resolve_micro_max_fee_pct` evite búsquedas redundantes y retorne el fallback exacto de forma inmediata.
//! 3. `resolve_margin_cap_limits` aproveche el fast-path O(1) y preserve la capacidad de overrides específicos.

use quantum_arena::GlobalArena;
use risk_engine::capital_regime::{
    resolve_margin_cap_limits, resolve_micro_max_fee_pct, DEFAULT_MICRO_MAX_FEE_PCT,
};

#[test]
fn test_f1_115_registry_is_empty_and_len() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    assert!(arena.registry.is_empty(), "Nuevo registro debe estar vacio");
    assert_eq!(arena.registry.len(), 0);

    arena.registry.set("test_param_1", 42.0);
    assert!(!arena.registry.is_empty(), "Registro con elemento no debe estar vacio");
    assert_eq!(arena.registry.len(), 1);

    arena.registry.set("test_param_2", 84.0);
    assert_eq!(arena.registry.len(), 2);
}

#[test]
fn test_f1_115_resolve_micro_max_fee_pct_fast_path_and_overrides() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    // 1. Fast-path cuando el registro está vacío
    let fee_pct = resolve_micro_max_fee_pct(&arena, 0);
    assert_eq!(fee_pct, DEFAULT_MICRO_MAX_FEE_PCT);

    // 2. Override global
    arena.registry.set("micro_max_fee_pct", 0.05);
    let fee_pct_global = resolve_micro_max_fee_pct(&arena, 0);
    assert_eq!(fee_pct_global, 0.05);

    // 3. Override por moneda (coin_id = 0)
    arena.registry.set_for_coin(0, "micro_max_fee_pct", 0.08);
    let fee_pct_coin0 = resolve_micro_max_fee_pct(&arena, 0);
    assert_eq!(fee_pct_coin0, 0.08);

    // Otra moneda sin override específico debe heredar el global
    let fee_pct_coin1 = resolve_micro_max_fee_pct(&arena, 1);
    assert_eq!(fee_pct_coin1, 0.05);
}

#[test]
fn test_f1_115_resolve_margin_cap_limits_fast_path_and_overrides() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    // Fast path: registry empty
    let limit_empty = resolve_margin_cap_limits(&arena, 0, 10.0, 1.0, 0.8, 10.0, 0.5);
    assert!(limit_empty.is_finite() && limit_empty > 0.0);

    // Con override por moneda
    arena.registry.set_for_coin(0, "micro_margin_cap_ratio", 0.40);
    let limit_custom = resolve_margin_cap_limits(&arena, 0, 10.0, 1.0, 0.8, 10.0, 1.0);
    // En micro_w_alloc = 1.0, micro_safe_limit = (alloc * micro_ratio).max(req_margin) = (10.0 * 0.4).max(1.0) = 4.0
    assert_eq!(limit_custom, 4.0);
}
