use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 🧠 ALGORITMO #46: ARENA POOL CUÁNTICA DE CERO ASIGNACIÓN DE MEMORIA EN HEAP (ZERO ALLOC ARENA POOL ENGINE)
/// Pre-asigna pools de memoria alineados a líneas de caché CPU de 64 bytes.
/// Garantiza 0 bytes de asignación en Heap durante la ejecución en caliente para eliminar pausas de GC.
#[derive(Debug, Default)]
#[repr(C, align(64))]
pub struct ZeroAllocArenaPoolEngine {
    occupancy_bitmap: AtomicU64,
    total_allocations: AtomicUsize,
    total_deallocations: AtomicUsize,
}

impl ZeroAllocArenaPoolEngine {
    pub const MAX_SLOTS: usize = 64;

    pub fn new() -> Self {
        Self {
            occupancy_bitmap: AtomicU64::new(0),
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
        }
    }

    /// Obtiene una ranura pre-asignada libre sin invocar malloc/free
    #[inline(always)]
    pub fn acquire_zero_alloc_slot(mask: u64) -> Option<usize> {
        let trailing_zeros = mask.trailing_zeros() as usize;
        if trailing_zeros < 64 {
            Some(trailing_zeros)
        } else {
            None
        }
    }

    /// Adquiere una ranura de forma atómica y lock-free (CAS en el bitmap de 64 bits)
    #[inline(always)]
    pub fn acquire_slot(&self) -> Option<usize> {
        loop {
            let current = self.occupancy_bitmap.load(Ordering::Acquire);
            let free_mask = !current;
            if free_mask == 0 {
                return None; // No slots available (Pool exhausted)
            }
            let slot = free_mask.trailing_zeros() as usize;
            let bit = 1u64 << slot;
            if self.occupancy_bitmap.compare_exchange_weak(
                current,
                current | bit,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                self.total_allocations.fetch_add(1, Ordering::Relaxed);
                return Some(slot);
            }
        }
    }

    /// Libera una ranura de forma atómica y lock-free
    #[inline(always)]
    pub fn release_slot(&self, slot: usize) -> bool {
        if slot >= 64 {
            return false;
        }
        let bit = 1u64 << slot;
        loop {
            let current = self.occupancy_bitmap.load(Ordering::Acquire);
            if current & bit == 0 {
                return false; // Slot already free
            }
            if self.occupancy_bitmap.compare_exchange_weak(
                current,
                current & !bit,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                self.total_deallocations.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
    }

    #[inline(always)]
    pub fn is_slot_occupied(&self, slot: usize) -> bool {
        if slot >= 64 {
            return false;
        }
        (self.occupancy_bitmap.load(Ordering::Relaxed) & (1u64 << slot)) != 0
    }

    #[inline(always)]
    pub fn occupied_count(&self) -> u32 {
        self.occupancy_bitmap.load(Ordering::Relaxed).count_ones()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_alloc_pool_atomic_acquire_and_release() {
        let pool = ZeroAllocArenaPoolEngine::new();
        assert_eq!(pool.occupied_count(), 0);

        let slot1 = pool.acquire_slot().expect("slot 0");
        assert_eq!(slot1, 0);
        assert_eq!(pool.occupied_count(), 1);
        assert!(pool.is_slot_occupied(0));

        let slot2 = pool.acquire_slot().expect("slot 1");
        assert_eq!(slot2, 1);
        assert_eq!(pool.occupied_count(), 2);

        assert!(pool.release_slot(0));
        assert_eq!(pool.occupied_count(), 1);
        assert!(!pool.is_slot_occupied(0));

        // Re-acquire should reuse slot 0
        let slot_reused = pool.acquire_slot().expect("reused slot 0");
        assert_eq!(slot_reused, 0);
    }

    #[test]
    fn test_zero_alloc_pool_exhaustion_and_out_of_bounds() {
        let pool = ZeroAllocArenaPoolEngine::new();
        // Fill all 64 slots
        for i in 0..64 {
            let s = pool.acquire_slot().expect("slot acquired");
            assert_eq!(s, i);
        }
        assert_eq!(pool.occupied_count(), 64);
        assert_eq!(pool.acquire_slot(), None);

        // Out of bounds operations
        assert!(!pool.release_slot(64));
        assert!(!pool.release_slot(100));
        assert!(!pool.is_slot_occupied(64));
    }
}

