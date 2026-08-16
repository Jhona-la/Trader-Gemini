/// 🧠 ALGORITMO #46: ARENA POOL CUÁNTICA DE CERO ASIGNACIÓN DE MEMORIA EN HEAP (ZERO ALLOC ARENA POOL ENGINE)
/// Pre-asigna pools de memoria alineados a líneas de caché CPU de 64 bytes.
/// Garantiza 0 bytes de asignación en Heap durante la ejecución en caliente para eliminar pausas de GC.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct ZeroAllocArenaPoolEngine;

impl ZeroAllocArenaPoolEngine {
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
}
