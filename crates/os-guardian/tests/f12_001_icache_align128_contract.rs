//! # F12-001: Contratos Formales de Aislamiento a 128 Bytes, Lamport SPSC & Precalentamiento Fantasma L1I/BTB
//!
//! Certifica los 6 invariantes físicos y algorítmicos del módulo `icache_sheaf_silicon`:
//! - **C1:** Alineación física estricta de 128 bytes (`align_of == 128`, `addr % 128 == 0`) y separación en arreglos contiguos.
//! - **C2:** Aislamiento espacial de cursores en `LamportRing128` y exactitud FIFO con rechazo limpio en saturación.
//! - **C3:** Resistencia de secuencia monotónica bajo 1,000 ciclos de *wraparound* circular sin pérdida ni corrupción.
//! - **C4:** Determinismo criptográfico y garantía de cero efectos secundarios (`side_effects_committed == 0`) en `PhantomBranchWarmer`.
//! - **C5:** Inmunidad a pánicos e inspección de ramas bajo entradas adversariales (`NaN`, apalancamiento de liquidación, edge insuficiente).
//! - **C6:** Exactitud bit a bit del empaquetado binario Little-Endian de 48 bytes (`encode_phantom_wire_frame`).

use os_guardian::{
    CacheAligned128, LamportRing128, PhantomBranchWarmer, PhantomOrderCandidate,
    SPATIAL_PREFETCH_LINE_BYTES,
};
use std::mem::{align_of, size_of};

#[test]
fn test_c1_cache_aligned_128_physical_alignment_and_spacing() {
    assert_eq!(
        align_of::<CacheAligned128<u64>>(),
        SPATIAL_PREFETCH_LINE_BYTES
    );
    assert_eq!(
        size_of::<CacheAligned128<u64>>(),
        SPATIAL_PREFETCH_LINE_BYTES
    );

    let slots: [CacheAligned128<u64>; 8] = std::array::from_fn(|i| CacheAligned128::new(i as u64));
    for (idx, slot) in slots.iter().enumerate() {
        assert!(slot.is_aligned_128());
        assert_eq!(**slot, idx as u64);
        if idx + 1 < slots.len() {
            let delta = slots[idx + 1].addr().abs_diff(slot.addr());
            assert_eq!(delta, SPATIAL_PREFETCH_LINE_BYTES);
        }
    }
}

#[test]
fn test_c2_lamport_ring_128_layout_isolation_and_fifo_exactness() {
    let ring: LamportRing128<u64, 16> = LamportRing128::new();
    assert!(ring.verify_layout_isolation());
    assert_eq!(ring.capacity(), 16);
    assert!(ring.is_empty());

    for i in 0..16u64 {
        assert_eq!(ring.try_push(i + 100), Ok(()));
    }
    assert_eq!(ring.len(), 16);
    assert!(!ring.is_empty());

    // El elemento 17 debe rechazarse limpiamente sin sobrescribir la FIFO
    assert_eq!(ring.try_push(999), Err(999));

    for i in 0..16u64 {
        assert_eq!(ring.try_pop(), Some(i + 100));
    }
    assert!(ring.is_empty());
    assert_eq!(ring.try_pop(), None);
}

#[test]
fn test_c3_lamport_ring_wraparound_monotonic_sequence_endurance() {
    let ring: LamportRing128<u32, 8> = LamportRing128::new();
    for cycle in 0..1_000u32 {
        let base = cycle * 5;
        for offset in 0..5u32 {
            assert_eq!(ring.try_push(base + offset), Ok(()));
        }
        assert_eq!(ring.len(), 5);
        for offset in 0..5u32 {
            assert_eq!(ring.try_pop(), Some(base + offset));
        }
        assert!(ring.is_empty());
    }
}

#[test]
fn test_c4_phantom_branch_warmer_zero_side_effects_and_determinism() {
    let warmer_a = PhantomBranchWarmer::new([0x42; 32]);
    let warmer_b = PhantomBranchWarmer::new([0x42; 32]);
    let candidate = PhantomOrderCandidate::default();

    let rep_a = warmer_a.warm_hot_path_branches(&candidate, 32);
    let rep_b = warmer_b.warm_hot_path_branches(&candidate, 32);

    assert_eq!(rep_a.iterations_executed, 32);
    // 4 ramas tomadas por iteración válida (finitud, liquidación, edge>=3x, firma HMAC)
    assert_eq!(rep_a.branches_exercised, 32 * 4);
    assert_eq!(rep_a.side_effects_committed, 0);
    assert_eq!(warmer_a.committed_side_effects(), 0);
    assert_eq!(warmer_a.total_warmups(), 32);
    assert_ne!(rep_a.last_digest_tag, 0);
    assert_eq!(rep_a, rep_b);
}

#[test]
fn test_c5_phantom_branch_warmer_adversarial_nan_and_liquidation_veto() {
    let warmer = PhantomBranchWarmer::default();

    // Caso 1: Precio NaN y apalancamiento excesivo + edge insuficiente -> solo toma la rama 4 (firma HMAC en seco)
    let bad_candidate = PhantomOrderCandidate {
        symbol_id: 7,
        price: f64::NAN,
        quantity: -1.0,
        leverage: 50.0,
        stop_distance_pct: 0.05,
        maintenance_margin_rate: 0.02,
        expected_edge_pct: 0.0001,
        roundtrip_friction_pct: 0.0010,
        timestamp_ns: 99,
    };

    let rep = warmer.warm_hot_path_branches(&bad_candidate, 16);
    assert_eq!(rep.iterations_executed, 16);
    assert_eq!(rep.branches_exercised, 16); // Solo la cuarta rama (empaquetado + HMAC) se contabiliza
    assert_eq!(rep.side_effects_committed, 0);
}

#[test]
fn test_c6_wire_frame_encoding_exact_48_byte_layout() {
    let candidate = PhantomOrderCandidate {
        symbol_id: 42,
        price: 50_000.0,
        quantity: 0.001, // nocional = 50.0 >= 5.0
        leverage: 10.0,
        stop_distance_pct: 0.02,
        maintenance_margin_rate: 0.005,
        expected_edge_pct: 0.005,
        roundtrip_friction_pct: 0.001,
        timestamp_ns: 123_456_789,
    };

    let frame = PhantomBranchWarmer::encode_phantom_wire_frame(&candidate);
    assert_eq!(
        u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]),
        42
    );

    let mut price_bytes = [0u8; 8];
    price_bytes.copy_from_slice(&frame[4..12]);
    assert_eq!(f64::from_bits(u64::from_le_bytes(price_bytes)), 50_000.0);

    let mut ts_bytes = [0u8; 8];
    ts_bytes.copy_from_slice(&frame[36..44]);
    assert_eq!(u64::from_le_bytes(ts_bytes), 123_456_789);

    let flag = u32::from_le_bytes([frame[44], frame[45], frame[46], frame[47]]);
    assert_eq!(flag, 0x0102_0304);
}
