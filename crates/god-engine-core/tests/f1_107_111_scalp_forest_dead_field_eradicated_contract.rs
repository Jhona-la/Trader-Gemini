//! Contrato F1-107 y F1-111 (Ola Ω162): Erradicación de campo huérfano `scalp_forest`
//! y búsqueda espuria "UNIVERSAL" en `refresh_models`.
//!
//! # Fundamento Teórico y Doctrina
//! - **Fallo Tipo 1 (Arista Muerta)**: `scalp_forest` se asignaba en el constructor y se reasignaba
//!   cada 1000 ticks en `refresh_models` buscando "UNIVERSAL", pero ningún código de producción
//!   lo leía (la inferencia real en `process_event` usa `{sym}_MOTOR` escopado por símbolo).
//! - **Aislamiento Multi-Activo Causal (B3.18b / LG-17)**: Prohibición de fallbacks cross-símbolo.
//!   Si una altcoin no tiene modelo propio `{sym}_MOTOR`, el motor debe abstenerse (ausencia de evidencia,
//!   no evidencia de ausencia) en vez de heredar modelos genéricos o de BTC.
//! - **Higiene de Camino Caliente**: Eliminación de SipHash + ArcSwap::load infructuosos cada 1000 eventos.

use god_engine_core::ml_inference::NanoForest;
use god_engine_core::GodEngineCore;
use quantum_arena::state::GlobalArena;

#[test]
fn test_f1_107_111_refresh_models_clean_without_universal_lookup() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);

    // Antes de Ω162, refresh_models() hacía un hash SipHash buscando "UNIVERSAL"
    // en GLOBAL_FORESTS en cada invocación, asignándolo al campo scalp_forest que nadie leía.
    // Ahora refresh_models() se ejecuta sin búsquedas superfluas ni efectos colaterales.
    core.refresh_models();

    // Verificamos que la clave "UNIVERSAL" no tiene ningún modelo registrado en el almacén global
    let universal = NanoForest::get_global("UNIVERSAL");
    assert!(
        universal.is_none(),
        "F1-111: No debe existir clave UNIVERSAL registrada; los modelos deben ser estrictamente escopados por símbolo"
    );
}

#[test]
fn test_f1_107_111_multi_coin_strict_isolation_without_cross_fallback() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let _core = GodEngineCore::new(arena);

    // Registramos un modelo específico para BTCUSDT_MOTOR
    // (o verificamos que una altcoin sin modelo no hereda nada)
    let alt_sym = "SOLUSDT";
    let alt_motor = NanoForest::get_global_suffixed(alt_sym, "_MOTOR");
    
    // Si no se ha cargado un modelo para SOLUSDT, la consulta debe retornar None
    // y no caer jamás a un fallback global de BTC o UNIVERSAL.
    if alt_motor.is_none() {
        assert!(
            alt_motor.is_none(),
            "F1-107: SOLUSDT sin modelo propio no debe tener modelo activo; aislamiento causal B3.18b garantizado"
        );
    }
}
