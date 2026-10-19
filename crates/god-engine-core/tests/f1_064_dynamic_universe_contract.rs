//! Contratos de Certificación Formal para Ola Ω172:
//! - F1-064: Carga, validación y saneamiento de universo dinámico en bootloader con disciplina de modelos B3.37 y fallback a ROSTER.

use god_engine_core::bootloader::{SystemDiagnostics, ROSTER};

#[test]
fn f1_064_dynamic_universe_validation_and_roster_discipline() {
    let temp_dir = std::env::temp_dir().join(format!("f1_064_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let models_dir = temp_dir.join("models");
    let _ = std::fs::create_dir_all(&models_dir);

    // Simular 2 modelos activos en models_dir
    let _ = std::fs::write(models_dir.join("ALGOUSDT_MOTOR.json"), "{}");
    let _ = std::fs::write(models_dir.join("NEARUSDT_MOTOR.json"), "{}");
    // Un candidato no promovido (no debe contar)
    let _ = std::fs::write(models_dir.join("FAKEUSDT_SCALP_CANDIDATE.json"), "{}");

    let config_path = temp_dir.join("dynamic_config.json");

    // 1. Probar con archivo inexistente -> None
    assert!(SystemDiagnostics::load_validated_dynamic_universe(&config_path, &models_dir).is_none());

    // 2. Probar con JSON inválido -> None
    let _ = std::fs::write(&config_path, "{ broken_json ");
    assert!(SystemDiagnostics::load_validated_dynamic_universe(&config_path, &models_dir).is_none());

    // 3. Probar con símbolos que incluyen:
    // - "ALGOUSDT" (tiene modelo en models_dir)
    // - "BTCUSDT" (en ROSTER canónico)
    // - "FAKEUSDT" (sin modelo promovido, no en ROSTER -> DEBE SER DESCARTADO)
    // - "garbage_symbol_123" -> sin modelo ni roster -> descartado
    // - Duplicado "BTCUSDT" -> debe ser deduplicado
    let test_json = r#"{
        "symbols": [
            "ALGOUSDT",
            "btcusdt",
            "FAKEUSDT",
            "BTCUSDT",
            "NEARUSDT"
        ]
    }"#;
    let _ = std::fs::write(&config_path, test_json);

    let res = SystemDiagnostics::load_validated_dynamic_universe(&config_path, &models_dir);
    assert!(res.is_some(), "Debe cargar configuración válida");
    let syms = res.unwrap();

    // Debe contener ALGOUSDT (modelo), BTCUSDT (roster), NEARUSDT (modelo y roster)
    assert!(syms.contains(&"ALGOUSDT".to_string()));
    assert!(syms.contains(&"BTCUSDT".to_string()));
    assert!(syms.contains(&"NEARUSDT".to_string()));
    // FAKEUSDT no debe estar porque no tiene modelo ni está en ROSTER
    assert!(!syms.contains(&"FAKEUSDT".to_string()));

    // No debe haber duplicados
    let mut dedup = syms.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(syms.len(), dedup.len(), "No debe contener símbolos duplicados");

    // El backfill debe completar plazas hasta al menos ROSTER.len()
    assert!(syms.len() >= ROSTER.len(), "Backfill debe completar hasta al menos ROSTER.len()");
    assert!(syms.len() <= quantum_arena::state::MAX_COINS, "Nunca debe exceder MAX_COINS");

    // Limpieza
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn f1_064_execute_phase_1_and_2_isolation_and_fallback() {
    // Modo demo/test siempre garantiza el ROSTER canónico intacto
    let (_, syms_demo) = SystemDiagnostics::execute_phase_1_and_2(true)
        .await
        .expect("phase 1 and 2 demo");
    assert_eq!(syms_demo, ROSTER.iter().map(|s| s.to_string()).collect::<Vec<_>>());

    // Modo live con data/dynamic_config.json ausente o fallback garantiza ROSTER
    let (_, syms_live) = SystemDiagnostics::execute_phase_1_and_2(false)
        .await
        .expect("phase 1 and 2 live");
    assert!(!syms_live.is_empty());
    assert!(syms_live.len() <= quantum_arena::state::MAX_COINS);
    assert!(syms_live.iter().all(|s| s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())));
}
