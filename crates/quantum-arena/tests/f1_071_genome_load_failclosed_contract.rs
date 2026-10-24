//! Contract test for F1-071 (Ola Ω175):
//!
//! F1-071: Verification and contract enforcement that `load_or_baseline` and fallback loaders
//! enforce strict sanitization through `SuperGenotype::from_vector()`, and that the host
//! operates under a strict fail-closed policy (no unvalidated raw fallbacks in production).

use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use quantum_arena::paths::{config_join, config_root};
use std::sync::Mutex;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn test_f1_071_config_paths_resolution() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let root = config_root();
    assert!(root.ends_with("config_dir"));

    let champion = config_join("genotypes/quantum_champion.json");
    assert!(champion.ends_with("config_dir/genotypes/quantum_champion.json"));
    assert!(std::path::Path::new(&champion).exists(), "{champion} must exist physically");
}

#[test]
fn test_f1_071_raw_champion_fallback_is_sanitized() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let path = config_join("genotypes/quantum_champion.json");
    let alt_path = "../../config_dir/genotypes/quantum_champion.json";

    let content = std::fs::read_to_string(&path)
        .or_else(|_| std::fs::read_to_string(alt_path))
        .expect("quantum_champion.json must exist in repo");

    let raw: SuperGenotype = serde_json::from_str(&content)
        .expect("raw deserialization of quantum_champion.json");

    // In raw json, global_max_drawdown might be 0.35 (below 0.50 lower bound)
    let sanitized = SuperGenotype::from_vector(&raw.to_vector());

    // Sanitized genome must have global_max_drawdown clamped to at least 0.50
    assert!(
        sanitized.global_max_drawdown >= 0.50,
        "sanitized global_max_drawdown must be >= 0.50, got {}",
        sanitized.global_max_drawdown
    );

    // Sanitized genome must satisfy GenomeEnvelope::validate()
    let val_res = GenomeEnvelope::validate(&sanitized);
    assert!(
        val_res.is_ok(),
        "sanitized fallback genome must pass GenomeEnvelope::validate(): {:?}",
        val_res.err()
    );
}

#[test]
fn test_f1_071_load_active_prod_envelope_is_valid() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        std::env::set_var("TG_GENOME_ENV", "prod");
    }

    let envelope_opt = GenomeEnvelope::load_active();
    assert!(envelope_opt.is_some(), "prod active envelope must exist");

    let envelope = envelope_opt.unwrap();
    assert!(envelope.generation >= 1);
    assert!(!envelope.source.is_empty());

    let val_res = GenomeEnvelope::validate(&envelope.genome);
    assert!(
        val_res.is_ok(),
        "prod active genome must pass GenomeEnvelope::validate(): {:?}",
        val_res.err()
    );
}

#[test]
fn test_f1_071_load_active_unknown_env_fails_closed() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        std::env::set_var("TG_GENOME_ENV", "unknown_invalid_env");
    }

    let envelope_opt = GenomeEnvelope::load_active();
    assert!(
        envelope_opt.is_none(),
        "invalid env must return None fail-closed without cross-env fallback"
    );
}

#[test]
fn test_f1_071_arbitrary_unbounded_fallback_is_sanitized() {
    let mut bad_genome = SuperGenotype::default();
    bad_genome.global_max_drawdown = 0.05; // Below 0.50 floor
    bad_genome.scalp_obi_threshold = 0.001; // Below 0.10 floor
    bad_genome.scalp_trail_act_atr = 0.01; // Below 1.5 floor

    let sanitized = SuperGenotype::from_vector(&bad_genome.to_vector());

    assert!(sanitized.global_max_drawdown >= 0.50);
    assert!(sanitized.scalp_obi_threshold >= 0.10);
    assert!(sanitized.scalp_trail_act_atr >= 1.5);

    let val_res = GenomeEnvelope::validate(&sanitized);
    assert!(
        val_res.is_ok(),
        "sanitized genome must satisfy all validation gates: {:?}",
        val_res.err()
    );
}
