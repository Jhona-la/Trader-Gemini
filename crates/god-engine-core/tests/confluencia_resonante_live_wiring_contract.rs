//! Contrato formal: Cableado Dinámico de Confluencia Resonante en el Camino Caliente (Ola Ω105).
//!
//! F3.1-06: Verifica que la ruta de resonancia cuántica espectral continua en `god-engine-core`
//! lea los umbrales de confluencia dinámicamente desde el registro omnisciente por moneda,
//! preservando bit a bit el contrato canónico h2_11 cuando no se proveen overrides.

use god_engine_core::{
    confluencia_resonante, confluencia_resonante_con_umbrales, COHERENCIA_MINIMA,
    FUSED_UMBAL_MODERADO, FUSED_UMBAL_PLENO, HURST_CONTINUACION, MAREA_MACRO_TOLERANCIA,
};
use quantum_arena::GlobalArena;

#[test]
fn test_confluencia_resonante_defaults_preserve_h2_11() {
    // Paridad exacta con el contrato h2_11
    let canonical = confluencia_resonante(0.40, 0.50, 0.20, 0.10, 0.0);
    let dynamic_default = confluencia_resonante_con_umbrales(
        0.40,
        0.50,
        0.20,
        0.10,
        0.0,
        FUSED_UMBAL_PLENO,
        FUSED_UMBAL_MODERADO,
        HURST_CONTINUACION,
        COHERENCIA_MINIMA,
        MAREA_MACRO_TOLERANCIA,
    );
    assert_eq!(canonical, dynamic_default);
    assert!(canonical.is_some());
    let (is_long, conf) = canonical.unwrap();
    assert!(is_long);
    assert!((conf - (0.50 + 0.20 * 0.35)).abs() < 1e-12);
}

#[test]
fn test_confluencia_resonante_custom_registry_thresholds() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin_id = 1; // Altcoin específica

    // Un fused score de 0.30 con hurst neutral (0.50) no pasa con umbral pleno estándar (0.38)
    let res_standard = confluencia_resonante_con_umbrales(
        0.30,
        0.50,
        0.20,
        0.10,
        0.0,
        FUSED_UMBAL_PLENO,
        FUSED_UMBAL_MODERADO,
        HURST_CONTINUACION,
        COHERENCIA_MINIMA,
        MAREA_MACRO_TOLERANCIA,
    );
    assert_eq!(res_standard, None, "0.30 está por debajo de 0.38 con Hurst neutral");

    // Configuramos en el registro omnisciente para coin_id = 1 un umbral adaptado más sensible (0.28)
    arena.registry.set_for_coin(coin_id, "fused_umbral_pleno", 0.28);

    let fused_pleno = arena.registry.get_for_coin_or(
        coin_id,
        "fused_umbral_pleno",
        FUSED_UMBAL_PLENO,
    );
    let fused_moderado = arena.registry.get_for_coin_or(
        coin_id,
        "fused_umbral_moderado",
        FUSED_UMBAL_MODERADO,
    );
    let hurst_cont = arena.registry.get_for_coin_or(
        coin_id,
        "hurst_continuacion",
        HURST_CONTINUACION,
    );
    let coh_min = arena.registry.get_for_coin_or(
        coin_id,
        "coherencia_minima",
        COHERENCIA_MINIMA,
    );
    let marea_tol = arena.registry.get_for_coin_or(
        coin_id,
        "marea_macro_tolerancia",
        MAREA_MACRO_TOLERANCIA,
    );

    let res_adapted = confluencia_resonante_con_umbrales(
        0.30,
        0.50,
        0.20,
        0.10,
        0.0,
        fused_pleno,
        fused_moderado,
        hurst_cont,
        coh_min,
        marea_tol,
    );
    assert!(res_adapted.is_some(), "Con umbral adaptado a 0.28, 0.30 debe producir señal");
    let (is_long, _) = res_adapted.unwrap();
    assert!(is_long);
}
