use metacortex_engine::{AdnBackupCatalog, GenerationMetadata, ReminiscenceModule};

#[test]
fn test_adn_backup_and_reminiscence_resurrection() {
    let temp_dir = std::env::temp_dir().join("trader_gemini_test_adn");
    let catalog = AdnBackupCatalog::new(&temp_dir);

    // 1. Archive Generation 1 (BTC Bull Regime)
    let gen1 = GenerationMetadata {
        gen_id: 1,
        timestamp_ns: 1000000,
        teleonomia_score: 2.45,
        sharpe_ratio: 3.12,
        max_drawdown: 0.015,
        win_rate: 0.88,
        active_regime: "BTC_Bull".to_string(),
        code_hash: 0xAABBCC,
    };
    catalog
        .archive_generation(&gen1, "// Cortex Gen 1 Rust Code")
        .expect("Failed to archive Gen 1");

    // 2. Archive Generation 2 (ETH Volatile Regime)
    let gen2 = GenerationMetadata {
        gen_id: 2,
        timestamp_ns: 2000000,
        teleonomia_score: 1.95,
        sharpe_ratio: 2.05,
        max_drawdown: 0.035,
        win_rate: 0.72,
        active_regime: "ETH_Volatile".to_string(),
        code_hash: 0xDDEEFF,
    };
    catalog
        .archive_generation(&gen2, "// Cortex Gen 2 Rust Code")
        .expect("Failed to archive Gen 2");

    // 3. Verify listing archived generations
    let archived = catalog.list_archived_generations();
    assert_eq!(archived.len(), 2, "Must list 2 archived generations");
    assert_eq!(archived[0].gen_id, 1);
    assert_eq!(archived[1].gen_id, 2);

    // 4. Test Reminiscence Module when market shifts back to BTC Bull Regime
    let reminiscence = ReminiscenceModule::new(catalog);
    let ancestral_seed = reminiscence.find_ancestral_seed("BTC_Bull", 2.5);
    assert!(
        ancestral_seed.is_some(),
        "Reminiscence MUST find Gen 1 seed for BTC_Bull regime"
    );

    let seed = ancestral_seed.unwrap();
    assert_eq!(seed.gen_id, 1);
    assert_eq!(seed.sharpe_ratio, 3.12);

    let _ = std::fs::remove_dir_all(temp_dir);
}
