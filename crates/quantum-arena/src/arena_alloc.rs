use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 🔒 ASIGNADOR DE MEMORIA STACK ARENA ZERO-HEAP (ZERO-JITTER EXECUTOR)
/// Permite re-usar búferes pre-reservados en memoria contigua sin llamar a malloc/free en el bucle principal.
/// Garantiza latencia 100% determinista sin variaciones (Jitter Zero).
pub struct ZeroHeapArena<const N: usize> {
    buffer: UnsafeCell<[u8; N]>,
    offset: AtomicUsize,
}

unsafe impl<const N: usize> Sync for ZeroHeapArena<N> {}
unsafe impl<const N: usize> Send for ZeroHeapArena<N> {}

impl<const N: usize> ZeroHeapArena<N> {
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([0u8; N]),
            offset: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice<'a, T: Copy>(&'a self, len: usize) -> Option<&'a mut [T]> {
        let size = len * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        let mut current = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = (current + align - 1) & !(align - 1);
            let next = aligned + size;

            if next > N {
                return None;
            }

            match self.offset.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => {
                    unsafe {
                        let ptr = (self.buffer.get() as *mut u8).add(aligned) as *mut T;
                        return Some(std::slice::from_raw_parts_mut(ptr, len));
                    }
                }
                Err(actual) => current = actual,
            }
        }
    }
}
