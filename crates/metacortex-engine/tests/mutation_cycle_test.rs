use chrono::Utc;
use metacortex_engine::{
    EpigenomaSymbolParams, MetacortexEngine, TraumaRecord, WaveletFeatureParams, WaveletType,
};
use tempfile::tempdir;

#[test]
fn test_template_generation_and_immune_system() {
    let dir = tempdir().unwrap();
    let root = dir.path();

    let engine = MetacortexEngine::new(root);

    // 1. Test Wavelet Feature Generation
    let params = WaveletFeatureParams {
        id: 101,
        wavelet_type: WaveletType::Haar,
        window_size: 128,
        threshold: 2.0,
        activation_factor: 1.0,
    };

    let generated_file = engine
        .template_engine
        .generate_wavelet_feature(&params)
        .unwrap();
    assert!(generated_file.exists());
    let content = std::fs::read_to_string(&generated_file).unwrap();
    assert!(content.contains("FeatureWaveletRegimen_v101"));
    assert!(content.contains("compute_wavelet_v101"));

    // 2. Test Living Immune System Test Generation
    let trauma = TraumaRecord {
        id: "trauma-test-1".to_string(),
        timestamp: Utc::now(),
        symbol: "BTCUSDT".to_string(),
        regime: "Trending".to_string(),
        expected_pnl_pct: 0.05,
        actual_pnl_pct: -0.04,
        predictor_name: "WaveletPredictor".to_string(),
        inputs_snapshot: vec![100.0, 102.5, 98.0, 105.0],
    };

    let _ = engine.immune_system.record_trauma(&trauma).unwrap();
    let immune_tests = engine.immune_system.generate_immune_tests().unwrap();
    assert_eq!(immune_tests.len(), 1);
    assert!(immune_tests[0].exists());
    let test_content = std::fs::read_to_string(&immune_tests[0]).unwrap();
    assert!(test_content.contains("test_immune_antibody_trauma_test_1"));

    // 3. Test Epigenoma State Persistence
    let symbol_params = EpigenomaSymbolParams {
        symbol: "BTCUSDT".to_string(),
        scalp_tp: 0.007,
        scalp_sl: 0.005,
        swing_tp: 0.02,
        swing_sl: 0.01,
        min_confidence: 0.65,
        max_leverage: 20.0,
    };

    let saved_path = engine
        .hot_swap
        .save_symbol_epigenoma(&symbol_params)
        .unwrap();
    assert!(saved_path.exists());
    let epigenoma_content = std::fs::read_to_string(&saved_path).unwrap();
    assert!(epigenoma_content.contains("BTCUSDT"));
}
