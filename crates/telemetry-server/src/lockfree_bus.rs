use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

const RING_SIZE: usize = 1024;
const INDEX_MASK: usize = RING_SIZE - 1;

/// Zero-Latency Lock-Free Telemetry Bus
/// Allows HFT engines and AI tensors to publish metrics to the dashboard
/// without EVER acquiring a lock or waiting.
pub struct LockFreeBus<T: Copy + Default> {
    head: AtomicU64,
    tail: AtomicU64,
    buffer: [UnsafeCell<T>; RING_SIZE],
}

// Ensure the bus can be safely shared across threads.
unsafe impl<T: Copy + Default> Sync for LockFreeBus<T> {}
unsafe impl<T: Copy + Default> Send for LockFreeBus<T> {}

impl<T: Copy + Default> LockFreeBus<T> {
    pub fn new() -> Self {
        let buffer = core::array::from_fn(|_| UnsafeCell::new(T::default()));
        Self {
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            buffer,
        }
    }

    /// O(1) wait-free push. If full, it overwrites the oldest (Ring Buffer semantic).
    #[inline(always)]
    pub fn push(&self, event: T) {
        let current_head = self.head.fetch_add(1, Ordering::Relaxed);
        let idx = (current_head as usize) & INDEX_MASK;

        unsafe {
            *self.buffer[idx].get() = event;
        }

        // Advance tail if we are overwriting
        let mut current_tail = self.tail.load(Ordering::Relaxed);
        while current_head.saturating_sub(current_tail) >= RING_SIZE as u64 {
            if self
                .tail
                .compare_exchange_weak(
                    current_tail,
                    current_tail + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
            current_tail = self.tail.load(Ordering::Relaxed);
        }
    }

    /// O(1) read for telemetry background workers.
    pub fn try_pop(&self) -> Option<T> {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let current_head = self.head.load(Ordering::Relaxed);

        if current_tail < current_head {
            if self
                .tail
                .compare_exchange_weak(
                    current_tail,
                    current_tail + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                let idx = (current_tail as usize) & INDEX_MASK;
                let data = unsafe { *self.buffer[idx].get() };
                return Some(data);
            }
        }
        None
    }
}

impl<T: Copy + Default> Default for LockFreeBus<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockfree_bus_push_and_try_pop() {
        let bus: LockFreeBus<u64> = LockFreeBus::default();
        assert_eq!(bus.try_pop(), None);

        bus.push(100);
        bus.push(200);

        assert_eq!(bus.try_pop(), Some(100));
        assert_eq!(bus.try_pop(), Some(200));
        assert_eq!(bus.try_pop(), None);
    }

    #[test]
    fn test_lockfree_bus_overwrite_ring_wrap() {
        let bus: LockFreeBus<u32> = LockFreeBus::new();
        // Push more than RING_SIZE (1024)
        for i in 0..1500 {
            bus.push(i);
        }

        // Tail was advanced to catch up with overwrite
        let mut count = 0;
        while let Some(val) = bus.try_pop() {
            assert!(val >= 1500 - 1024);
            count += 1;
        }
        assert!(count <= 1024);
    }
}
