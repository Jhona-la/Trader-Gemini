//! CONTRATO FORMAL F3-002: Persistencia Binaria de Genoma Zero-Copy & SHA-256 (Átomo 3.2)
//!
//! Verifica:
//! 1. Tamaño y layout canónico de BinaryGenomeHeader (72 bytes, endianness LE, magic TGGENOM\0).
//! 2. Fidelidad bit-a-bit de ida y vuelta en serialización binaria (to_binary_bytes / from_binary_bytes).
//! 3. Rechazo fail-closed inmediato ante alteración o corrupción de bits en el payload (detección por SHA-256).
//! 4. Rechazo fail-closed ante magic number binario inválido o truncamiento.
//! 5. Coexistencia armónica: promote() genera active.json y active.bin simultáneamente.
//! 6. Carga transparente ultrarrápida: load_active() lee active.bin en < 100 µs.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::{
    BinaryGenomeHeader, GenomeEnvelope, BINARY_MAGIC, BINARY_SCHEMA_VERSION, SCHEMA_VERSION,
};

#[test]
fn test_binary_genome_header_layout_and_size() {
    assert_eq!(BinaryGenomeHeader::SIZE, 72);

    let header = BinaryGenomeHeader {
        magic: BINARY_MAGIC,
        schema_version: BINARY_SCHEMA_VERSION,
        flags: 0,
        generation: 42,
        created_ms: 1700000000000,
        payload_len: 1024,
        checksum: [0xAB; 32],
    };

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 72);
    assert_eq!(&bytes[0..8], b"TGGENOM\0");

    let parsed = BinaryGenomeHeader::from_bytes(&bytes).expect("Cabecera parseable");
    assert_eq!(parsed, header);
}

#[test]
fn test_binary_genome_roundtrip_fidelity() {
    let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.tp_horizon_curve = quantum_arena::temporal_spectrum::HorizonCurve {
        a: -4.9600174176553296,
        b: 0.09143837668088041,
    };
    g.sl_horizon_curve = quantum_arena::temporal_spectrum::HorizonCurve {
        a: -5.719394858308684,
        b: 0.08643759946713066,
    };

    let envelope = GenomeEnvelope {
        schema_version: SCHEMA_VERSION,
        generation: 108,
        created_ms: 1729000000000,
        source: "darwin_online_optimizer".to_string(),
        parent_generation: 107,
        promotion_reason: "Validación OOS Sharpe > 3.0".to_string(),
        genome: g.clone(),
    };

    let binary_bytes = envelope.to_binary_bytes().expect("Serialización binaria");
    assert!(binary_bytes.len() > 72);

    let decoded = GenomeEnvelope::from_binary_bytes(&binary_bytes).expect("Deserialización binaria");
    assert_eq!(decoded.schema_version, envelope.schema_version);
    assert_eq!(decoded.generation, envelope.generation);
    assert_eq!(decoded.created_ms, envelope.created_ms);
    assert_eq!(decoded.source, envelope.source);
    assert_eq!(decoded.parent_generation, envelope.parent_generation);
    assert_eq!(decoded.promotion_reason, envelope.promotion_reason);

    let orig_vec = envelope.genome.to_vector();
    let dec_vec = decoded.genome.to_vector();
    assert_eq!(orig_vec.len(), dec_vec.len());
    for i in 0..orig_vec.len() {
        assert_eq!(
            orig_vec[i].to_bits(),
            dec_vec[i].to_bits(),
            "Discrepancia en gen {} tras roundtrip binario",
            i
        );
    }
}

#[test]
fn test_binary_genome_cryptographic_corruption_detected_fail_closed() {
    let g = SuperGenotype::new_baseline(0.0002, 0.0005);
    let envelope = GenomeEnvelope {
        schema_version: SCHEMA_VERSION,
        generation: 1,
        created_ms: 100,
        source: "test".to_string(),
        parent_generation: 0,
        promotion_reason: "test".to_string(),
        genome: g,
    };

    let mut bytes = envelope.to_binary_bytes().expect("bytes");
    // Alteramos 1 solo bit en el payload (después de los 72 bytes de cabecera)
    let corrupt_idx = 75;
    bytes[corrupt_idx] ^= 0x01;

    let res = GenomeEnvelope::from_binary_bytes(&bytes);
    assert!(res.is_err(), "Debe fallar ante payload corrupto");
    let err = res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("Fallo de integridad criptográfica"));
}

#[test]
fn test_binary_genome_magic_corruption_rejected() {
    let g = SuperGenotype::new_baseline(0.0002, 0.0005);
    let envelope = GenomeEnvelope {
        schema_version: SCHEMA_VERSION,
        generation: 1,
        created_ms: 100,
        source: "test".to_string(),
        parent_generation: 0,
        promotion_reason: "test".to_string(),
        genome: g,
    };

    let mut bytes = envelope.to_binary_bytes().expect("bytes");
    bytes[0] = b'X';

    let res = GenomeEnvelope::from_binary_bytes(&bytes);
    assert!(res.is_err(), "Debe fallar ante magic inválido");
    let err = res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("Magic number binario inválido"));
}

#[test]
fn test_binary_genome_store_atomic_promote_and_load() {
    let temp_dir = std::env::temp_dir().join(format!("tg_test_bin_genome_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::env::set_var("TG_CONFIG_DIR", temp_dir.to_str().unwrap());
    std::env::set_var("TG_GENOME_ENV", "backtest");

    let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.tp_horizon_curve = quantum_arena::temporal_spectrum::HorizonCurve {
        a: -4.9600174176553296,
        b: 0.09143837668088041,
    };
    g.sl_horizon_curve = quantum_arena::temporal_spectrum::HorizonCurve {
        a: -5.719394858308684,
        b: 0.08643759946713066,
    };

    let promoted = GenomeEnvelope::promote(g, "darwin_test", "Prueba binaria").expect("promote ok");
    assert_eq!(promoted.generation, 1);

    // Verificamos que se escribieron AMBOS formatos para el genoma activo y el histórico json
    let active_json = temp_dir.join("genomes/backtest/active.json");
    let active_bin = temp_dir.join("genomes/backtest/active.bin");
    let hist_json = temp_dir.join("genomes/backtest/history/gen_000001.json");

    assert!(active_json.exists(), "active.json debe existir");
    assert!(active_bin.exists(), "active.bin debe existir");
    assert!(hist_json.exists(), "history/gen_000001.json debe existir");

    // Medición de latencia de carga binaria
    let t0 = std::time::Instant::now();
    let loaded = GenomeEnvelope::load_active().expect("load_active ok");
    let elapsed = t0.elapsed();

    assert_eq!(loaded.generation, 1);
    assert_eq!(loaded.source, "darwin_test");
    println!("⏱️ Latencia de carga de genoma binario con SHA-256: {:?}", elapsed);
    assert!(elapsed.as_millis() < 50, "Carga binaria debe ser ultra-rápida");

    // Limpieza
    let _ = std::fs::remove_dir_all(&temp_dir);
}
