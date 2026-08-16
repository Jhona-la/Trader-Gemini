use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;

const TENSOR_RING_SIZE: usize = 128; // Potencia de 2 para bitwise masking rápido

pub struct TensorTelemetryRing {
    head: AtomicUsize,
    tail: AtomicUsize,
    // [f64; 54] feature tensors and a timestamp
    data: [MaybeUninit<(u64, [f64; 54])>; TENSOR_RING_SIZE],
}

impl TensorTelemetryRing {
    pub const fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            data: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    pub fn lock_in_ram(&self) {
        #[cfg(windows)]
        {
            let ptr = self.data.as_ptr() as *const u8;
            let size = std::mem::size_of_val(&self.data);
            let _ = os_guardian::lock_critical_memory(ptr, size);
        }
    }

    /// O(1) Lock-free push desde el Hot Path (Engine/ML) sin heap allocations
    #[inline(always)]
    pub fn push_tensor(&self, timestamp_ms: u64, features: &[f64; 54]) -> Result<(), ()> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        
        if head.wrapping_sub(tail) >= TENSOR_RING_SIZE {
            return Err(());
        }
        
        let idx = head & (TENSOR_RING_SIZE - 1);
        unsafe {
            let ptr = self.data[idx].as_ptr() as *mut (u64, [f64; 54]);
            ptr.write((timestamp_ms, *features));
        }
        
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// O(1) Lock-free pop para Telemetría (Background Thread)
    pub fn pop_tensor(&self) -> Option<(u64, [f64; 54])> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail == head {
            return None;
        }
        
        let idx = tail & (TENSOR_RING_SIZE - 1);
        let item = unsafe {
            let ptr = self.data[idx].as_ptr();
            ptr.read()
        };
        
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }
}

// Global estática para evitar Arc<> y latencia de deref
pub static GLOBAL_TENSOR_TELEMETRY: TensorTelemetryRing = TensorTelemetryRing::new();

unsafe impl Sync for TensorTelemetryRing {}
unsafe impl Send for TensorTelemetryRing {}
