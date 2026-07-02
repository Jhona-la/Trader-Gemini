use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Axioma XIX: La Ley del Flight Recorder Omnisciente
/// Registro lock-free y zero-allocation mediante mapeo de memoria directa.
/// Totalmente opaco para el SO y operando a velocidad de caché L1/L2.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct FlightEvent {
    pub timestamp: u64,
    pub trace_id: u64,
    pub event_type: u8,
    pub payload: [u8; 47], // 8 + 8 + 1 + 47 = 64 bytes exactos (1 Cache Line)
}

pub struct FlightRecorder {
    mmap: *mut u8, // Using raw pointer to allow Sync
    head: AtomicUsize,
    capacity: usize,
    // Maintain ownership of mmap to drop it properly
    _mmap_guard: MmapMut,
}

// Prometemos que es seguro compartir entre hilos (O(1) atómico lock-free)
unsafe impl Sync for FlightRecorder {}
unsafe impl Send for FlightRecorder {}

impl FlightRecorder {
    pub fn new(path: &str, capacity_events: usize) -> Self {
        let file_size = capacity_events * std::mem::size_of::<FlightEvent>();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .expect("Fallo al crear archivo de Flight Recorder");
            
        file.set_len(file_size as u64).expect("Fallo al reservar espacio de Flight Recorder");
        
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file).expect("Fallo al hacer mmap") };
        let mmap_ptr = mmap.as_mut_ptr();
        
        Self {
            mmap: mmap_ptr,
            head: AtomicUsize::new(0),
            capacity: capacity_events,
            _mmap_guard: mmap,
        }
    }

    /// Registra un evento en nanosegundos (lock-free)
    #[inline(always)]
    pub fn record(&self, event: FlightEvent) {
        let idx = self.head.fetch_add(1, Ordering::Relaxed) % self.capacity;
        let offset = idx * std::mem::size_of::<FlightEvent>();
        
        unsafe {
            let dest = self.mmap.add(offset);
            std::ptr::copy_nonoverlapping(
                &event as *const FlightEvent as *const u8,
                dest,
                std::mem::size_of::<FlightEvent>(),
            );
        }
    }
}
