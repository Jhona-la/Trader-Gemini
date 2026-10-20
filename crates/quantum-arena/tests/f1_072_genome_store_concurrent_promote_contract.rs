//! Contract test for F1-072 (Ola Ω176):
//!
//! F1-072: Concurrencia segura y atomizado en `promote()`, reclamación O_EXCL de
//! número de generación en `gen_{N}.json`, tmp único (`{path}.{pid}.{nonce}.tmp`),
//! eliminación de ventana `remove_file()` en `atomic_write()`, y auto-reparación
//! de historial faltante en `load_active()`.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use std::sync::Mutex;
use std::thread;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[test]
fn test_f1_072_concurrent_promotions_produce_unique_generations_and_history() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let temp_root = std::env::temp_dir().join(format!("tg_test_promote_concurrent_{}_{}", std::process::id(), now_ms()));
    let _ = std::fs::create_dir_all(&temp_root);

    let temp_root_str = temp_root.to_str().expect("valid utf8 temp path").to_string();
    unsafe {
        std::env::set_var("TG_CONFIG_ROOT", &temp_root_str);
        std::env::set_var("TG_GENOME_ENV", "demo");
    }

    let num_threads = 8;
    let mut handles = Vec::new();

    for i in 0..num_threads {
        let handle = thread::spawn(move || {
            let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
            let source = format!("worker_{}", i);
            let reason = format!("concurrent test thread {}", i);
            GenomeEnvelope::promote(genome, &source, &reason)
        });
        handles.push(handle);
    }

    let mut promoted_generations = Vec::new();
    for handle in handles {
        let res = handle.join().expect("thread did not panic");
        let envelope = res.expect("promote must succeed concurrently without error");
        promoted_generations.push(envelope.generation);
    }

    // Verify all returned generations are unique and in range 1..=num_threads
    promoted_generations.sort();
    let original_len = promoted_generations.len();
    promoted_generations.dedup();
    assert_eq!(
        promoted_generations.len(),
        original_len,
        "Every concurrent promote must claim a strictly unique generation: {:?}",
        promoted_generations
    );
    assert_eq!(promoted_generations.len(), num_threads);
    assert_eq!(promoted_generations[0], 1);
    assert_eq!(*promoted_generations.last().unwrap(), num_threads as u64);

    // Verify history directory contains exactly num_threads files
    let history_dir = temp_root.join("genomes/demo/history");
    let mut history_files: Vec<String> = std::fs::read_dir(&history_dir)
        .expect("read_dir history")
        .filter_map(|e| e.ok().map(|de| de.file_name().to_string_lossy().to_string()))
        .collect();
    history_files.sort();

    assert_eq!(history_files.len(), num_threads);
    for g in 1..=(num_threads as u64) {
        let expected_name = format!("gen_{:06}.json", g);
        assert!(
            history_files.contains(&expected_name),
            "History dir must contain {expected_name}"
        );

        let file_path = history_dir.join(&expected_name);
        let content = std::fs::read_to_string(&file_path).expect("read history file");
        let env: GenomeEnvelope = serde_json::from_str(&content).expect("parse history envelope");
        assert_eq!(env.generation, g);
    }

    // Verify active.json points to the final generation
    let active_path = temp_root.join("genomes/demo/active.json");
    let active_content = std::fs::read_to_string(&active_path).expect("read active.json");
    let active_env: GenomeEnvelope = serde_json::from_str(&active_content).expect("parse active.json");
    assert_eq!(active_env.generation, num_threads as u64);

    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn test_f1_072_load_active_auto_repairs_missing_history_file() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let temp_root = std::env::temp_dir().join(format!("tg_test_repair_{}_{}", std::process::id(), now_ms()));
    let demo_dir = temp_root.join("genomes/prod");
    std::fs::create_dir_all(&demo_dir).expect("create test env dir");

    let temp_root_str = temp_root.to_str().expect("valid utf8 temp path").to_string();
    unsafe {
        std::env::set_var("TG_CONFIG_ROOT", &temp_root_str);
        std::env::set_var("TG_GENOME_ENV", "prod");
    }

    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let envelope = GenomeEnvelope {
        schema_version: 1,
        generation: 42,
        created_ms: now_ms(),
        source: "isolated_repair_test".to_string(),
        parent_generation: 41,
        promotion_reason: "test missing history repair".to_string(),
        genome: base,
    };

    let active_path = demo_dir.join("active.json");
    let active_json = serde_json::to_string_pretty(&envelope).expect("serialize envelope");
    std::fs::write(&active_path, &active_json).expect("write active.json");

    let history_file = demo_dir.join("history/gen_000042.json");
    assert!(!history_file.exists(), "History file must not exist prior to repair");

    // Calling load_active() should detect missing history and reconstruct it
    let loaded = GenomeEnvelope::load_active();
    assert!(loaded.is_some(), "load_active() must return Some(envelope)");
    assert_eq!(loaded.as_ref().unwrap().generation, 42);

    // History file must now exist and be parseable
    assert!(history_file.exists(), "History file must be reconstructed automatically");
    let hist_content = std::fs::read_to_string(&history_file).expect("read repaired history");
    let hist_env: GenomeEnvelope = serde_json::from_str(&hist_content).expect("parse repaired history");
    assert_eq!(hist_env.generation, 42);
    assert_eq!(hist_env.source, "isolated_repair_test");

    // load_generation and rollback must succeed now
    let loaded_gen = GenomeEnvelope::load_generation(42);
    assert!(loaded_gen.is_ok(), "load_generation(42) must succeed after auto-repair");

    let rollback_res = GenomeEnvelope::rollback(42);
    assert!(rollback_res.is_ok(), "rollback(42) must succeed after auto-repair");
    let rolled_back = rollback_res.unwrap();
    assert_eq!(rolled_back.generation, 43);
    assert_eq!(rolled_back.parent_generation, 42);
    assert_eq!(rolled_back.source, "rollback");

    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn test_f1_072_real_prod_history_self_heals() {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    unsafe {
        std::env::remove_var("TG_CONFIG_ROOT");
        std::env::remove_var("TGM_CONFIG_DIR");
        std::env::set_var("TG_GENOME_ENV", "prod");
    }

    // Before this test, prod/history might be missing gen_000001.json
    let loaded = GenomeEnvelope::load_active();
    assert!(loaded.is_some(), "Production active genome must load");

    let gen = loaded.as_ref().unwrap().generation;
    assert_eq!(gen, 1);

    // load_generation(1) must succeed now (auto-healed if it was missing)
    let gen_1 = GenomeEnvelope::load_generation(1);
    assert!(gen_1.is_ok(), "Production gen_000001.json must exist and be loadable: {:?}", gen_1.err());
    assert_eq!(gen_1.unwrap().generation, 1);
}
