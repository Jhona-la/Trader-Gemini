use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

const RING_CAPACITY: usize = 65536;

pub struct ZeroCopyRing {
    head: AtomicUsize,
    tail: AtomicUsize,
    // (Timestamp/Value or Name/Latency)
    buffer: [MaybeUninit<(&'static str, u64)>; RING_CAPACITY],
}

impl ZeroCopyRing {
    pub fn new() -> Self {
        // Inicializar sin memoria asignada dinámicamente.
        // Array fijo en la estructura.
        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            buffer: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn lock_in_ram(&self) {
        #[cfg(windows)]
        unsafe {
            let _ = os_guardian::memory_compaction::lock_critical_memory(&self.buffer);
        }
    }

    /// O(1) wait-free SPSC push
    #[inline(always)]
    pub fn push(&self, item: (&'static str, u64)) -> Result<(), ()> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Verifica si está lleno
        if head.wrapping_sub(tail) >= RING_CAPACITY {
            return Err(());
        }

        let index = head % RING_CAPACITY;

        unsafe {
            let ptr = self.buffer[index].as_ptr() as *mut (&'static str, u64);
            ptr.write(item);
        }

        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// O(1) wait-free SPSC pop
    #[inline(always)]
    pub fn pop(&self) -> Option<(&'static str, u64)> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Verifica si está vacío
        if tail == head {
            return None;
        }

        let index = tail % RING_CAPACITY;

        let item = unsafe {
            let ptr = self.buffer[index].as_ptr();
            ptr.read()
        };

        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }
}

// Implement Sync since it's an SPSC safe structure.
unsafe impl Sync for ZeroCopyRing {}
unsafe impl Send for ZeroCopyRing {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_ring_push_pop() {
        let ring = ZeroCopyRing::new();
        assert!(ring.pop().is_none());

        assert!(ring.push(("latency_ns", 42)).is_ok());
        assert!(ring.push(("signal_calc", 150)).is_ok());

        let item1 = ring.pop().unwrap();
        assert_eq!(item1.0, "latency_ns");
        assert_eq!(item1.1, 42);

        let item2 = ring.pop().unwrap();
        assert_eq!(item2.0, "signal_calc");
        assert_eq!(item2.1, 150);

        assert!(ring.pop().is_none());
    }

    #[test]
    fn test_zero_copy_ring_fifo_order_and_lock_in_ram() {
        let ring = ZeroCopyRing::new();
        ring.lock_in_ram();

        for i in 0..100 {
            assert!(ring.push(("event", i)).is_ok());
        }

        for i in 0..100 {
            let item = ring.pop().unwrap();
            assert_eq!(item.0, "event");
            assert_eq!(item.1, i);
        }

        assert!(ring.pop().is_none());
    }
}
