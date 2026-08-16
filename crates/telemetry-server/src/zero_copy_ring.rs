use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;

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
        {
            let ptr = self.buffer.as_ptr() as *const u8;
            let size = std::mem::size_of_val(&self.buffer);
            let _ = os_guardian::lock_critical_memory(ptr, size);
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
