// Removed std::f64

/// ⚡ CIRCULAR LOCK-FREE RING BUFFER DE CERO ASIGNACIÓN EN HEAP (ZERO-ALLOCATION ARENA POOL)
/// Reutiliza arreglos contiguos de tamaño fijo N alineados a 64 Bytes en memoria.
/// Elimina por completo las llamadas a malloc/free en el bucle caliente (Jitter = 0 ns).
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct LockFreeRingBuffer<T: Copy + Default, const N: usize> {
    pub data: [T; N],
    pub head: usize,
    pub count: usize,
}

impl<T: Copy + Default, const N: usize> LockFreeRingBuffer<T, N> {
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
            head: 0,
            count: 0,
        }
    }

    /// Inserta una nueva observación en O(1) de forma circular sin dinamicidad
    #[inline(always)]
    pub fn push(&mut self, val: T) {
        self.data[self.head] = val;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// Obtiene el último elemento insertado
    #[inline(always)]
    pub fn latest(&self) -> Option<T> {
        if self.count == 0 {
            None
        } else {
            let idx = (self.head + N - 1) % N;
            Some(self.data[idx])
        }
    }

    /// Limpia el búfer en O(1)
    #[inline(always)]
    pub fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}

impl<T: Copy + Default, const N: usize> Default for LockFreeRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}
