//! # Thread-Safe Epigenoma Store — Zero-Copy Atomic Edition
//!
//! Proporciona acceso global thread-safe a parámetros genéticos dinámicos.
//! Migrado de RwLock<HashMap<String, f64>> a slots atómicos lock-free O(1).

use std::sync::atomic::{AtomicU64, Ordering};

/// Slots atómicos pre-alocados para genes epigenéticos.
/// Cada slot almacena un f64 como bits atómicos (Zero-Copy, Zero-Lock).
const MAX_EPIGENES: usize = 128;

static EPIGENOMA_SLOTS: [AtomicU64; MAX_EPIGENES] = {
    // Inicialización const de array atómico
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; MAX_EPIGENES]
};

static EPIGENOMA_DEFAULTS: [AtomicU64; MAX_EPIGENES] = {
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; MAX_EPIGENES]
};

/// Mapeo de nombres de genes a slots.
/// Usa un hash FNV-1a ultra-rápido para indexar en O(1).
#[inline(always)]
fn gene_slot(key: &str) -> usize {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    (hash as usize) % MAX_EPIGENES
}

/// Sets dynamic gene value in global Epigenoma store (lock-free, O(1))
#[inline(always)]
pub fn set_epigenoma_gene(key: &str, value: f64) {
    let slot = gene_slot(key);
    EPIGENOMA_SLOTS[slot].store(value.to_bits(), Ordering::Relaxed);
}

/// Reads dynamic gene value from global Epigenoma store with fallback to default (lock-free, O(1))
#[inline(always)]
pub fn read_epigenoma_gene(key: &str, default_val: f64) -> f64 {
    let slot = gene_slot(key);
    let bits = EPIGENOMA_SLOTS[slot].load(Ordering::Relaxed);
    if bits == 0 {
        default_val
    } else {
        f64::from_bits(bits)
    }
}

/// Inicializa un gene con su valor por defecto (solo si el slot está vacío).
/// Útil durante el bootstrap del sistema.
#[inline]
pub fn init_epigenoma_gene(key: &str, default_val: f64) {
    let slot = gene_slot(key);
    EPIGENOMA_DEFAULTS[slot].store(default_val.to_bits(), Ordering::Relaxed);
    // Solo escribir si el slot está vacío (CAS)
    let _ = EPIGENOMA_SLOTS[slot].compare_exchange(
        0,
        default_val.to_bits(),
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
}
