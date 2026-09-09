//! # Thread-Safe Epigenoma Store — Zero-Copy Atomic Edition
//!
//! Proporciona acceso global thread-safe a parámetros genéticos dinámicos.
//! Utiliza linear probing lock-free O(1) con discriminación explícita de inicialización
//! para soportar valores 0.0 legítimos y eliminar colisiones destructivas.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const MAX_EPIGENES: usize = 1024;

struct EpigenomaSlot {
    key_hash: AtomicU64,
    value: AtomicU64,
    initialized: AtomicBool,
}

impl EpigenomaSlot {
    const fn new() -> Self {
        Self {
            key_hash: AtomicU64::new(0),
            value: AtomicU64::new(0),
            initialized: AtomicBool::new(false),
        }
    }
}

static EPIGENOMA_TABLE: [EpigenomaSlot; MAX_EPIGENES] = {
    const INIT: EpigenomaSlot = EpigenomaSlot::new();
    [INIT; MAX_EPIGENES]
};

#[inline(always)]
fn fnv1a_hash(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    if hash == 0 {
        1
    } else {
        hash
    }
}

#[inline(always)]
fn find_slot_read(key: &str) -> Option<usize> {
    let target_hash = fnv1a_hash(key);
    let start = (target_hash as usize) % MAX_EPIGENES;

    for i in 0..MAX_EPIGENES {
        let idx = (start + (i * (i + 1)) / 2) % MAX_EPIGENES;
        let slot_hash = EPIGENOMA_TABLE[idx].key_hash.load(Ordering::Acquire);
        if slot_hash == target_hash {
            return Some(idx);
        }
        if slot_hash == 0 {
            return None;
        }
    }
    None
}

#[inline(always)]
fn find_slot_write(key: &str) -> Option<usize> {
    let target_hash = fnv1a_hash(key);
    let start = (target_hash as usize) % MAX_EPIGENES;

    for i in 0..MAX_EPIGENES {
        let idx = (start + (i * (i + 1)) / 2) % MAX_EPIGENES;
        let slot_hash = EPIGENOMA_TABLE[idx].key_hash.load(Ordering::Acquire);
        if slot_hash == target_hash {
            return Some(idx);
        }
        if slot_hash == 0 {
            // Intenta reclamar el slot vacío con CAS atómico
            if EPIGENOMA_TABLE[idx]
                .key_hash
                .compare_exchange(0, target_hash, Ordering::SeqCst, Ordering::Acquire)
                .is_ok()
            {
                return Some(idx);
            }
            // Si otro hilo lo reclamó concurrentemente, verificamos si es el mismo target_hash
            if EPIGENOMA_TABLE[idx].key_hash.load(Ordering::Acquire) == target_hash {
                return Some(idx);
            }
        }
    }
    None
}

/// Sets dynamic gene value in global Epigenoma store (lock-free, O(1))
#[inline(always)]
pub fn set_epigenoma_gene(key: &str, value: f64) {
    if !value.is_finite() {
        return;
    }
    // FIX #868: Solo escribir si se encontró o reclamó un slot exclusivo para key
    if let Some(slot) = find_slot_write(key) {
        EPIGENOMA_TABLE[slot]
            .value
            .store(value.to_bits(), Ordering::Release);
        EPIGENOMA_TABLE[slot]
            .initialized
            .store(true, Ordering::Release);
    }
}

/// Reads dynamic gene value from global Epigenoma store with fallback to default (lock-free, O(1))
#[inline(always)]
pub fn read_epigenoma_gene(key: &str, default_val: f64) -> f64 {
    if let Some(slot) = find_slot_read(key) {
        if EPIGENOMA_TABLE[slot].initialized.load(Ordering::Acquire) {
            return f64::from_bits(EPIGENOMA_TABLE[slot].value.load(Ordering::Acquire));
        }
    }
    default_val
}

/// Inicializa un gene con su valor por defecto (solo si el slot está sin inicializar).
#[inline]
pub fn init_epigenoma_gene(key: &str, default_val: f64) {
    if !default_val.is_finite() {
        return;
    }
    if let Some(slot) = find_slot_write(key) {
        if !EPIGENOMA_TABLE[slot].initialized.load(Ordering::Acquire) {
            EPIGENOMA_TABLE[slot]
                .value
                .store(default_val.to_bits(), Ordering::Release);
            EPIGENOMA_TABLE[slot]
                .initialized
                .store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epigenoma_store_zero_value_and_distinct_keys() {
        let key1 = "alpha_scalp_threshold";
        let key2 = "beta_swing_factor";

        set_epigenoma_gene(key1, 0.0);
        set_epigenoma_gene(key2, 42.5);

        // El valor 0.0 debe ser recuperado exactamente, no el default_val
        let read1 = read_epigenoma_gene(key1, 999.0);
        assert_eq!(read1, 0.0);

        let read2 = read_epigenoma_gene(key2, 999.0);
        assert_eq!(read2, 42.5);

        // Clave no inicializada debe retornar el default
        let uninit = read_epigenoma_gene("uninitialized_gene_xyz", 13.0);
        assert_eq!(uninit, 13.0);
    }

    #[test]
    fn test_epigenoma_store_nan_immunity() {
        let key = "gene_nan_test";
        set_epigenoma_gene(key, 10.0);
        set_epigenoma_gene(key, f64::NAN);
        let read = read_epigenoma_gene(key, 999.0);
        assert_eq!(
            read, 10.0,
            "Setting NaN must be rejected, preserving existing valid value"
        );
    }
}
