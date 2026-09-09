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
        // FIX #1441: Clamping defensivo de capacidad para evitar división por cero en record()
        let safe_capacity = capacity_events.clamp(1, 10_000_000);
        let file_size = safe_capacity * std::mem::size_of::<FlightEvent>();
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // FIX #587: truncate(false) para preservar el registro forense previo tras reinicios
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .expect("Fallo al crear archivo de Flight Recorder");

        let meta = file
            .metadata()
            .expect("Fallo al leer metadatos de Flight Recorder");
        if meta.len() < file_size as u64 {
            file.set_len(file_size as u64)
                .expect("Fallo al reservar espacio de Flight Recorder");
        }

        let mut mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .expect("Fallo al hacer mmap")
        };
        let mmap_ptr = mmap.as_mut_ptr();

        Self {
            mmap: mmap_ptr,
            head: AtomicUsize::new(0),
            capacity: safe_capacity,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_flight_recorder_mmap() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_telemetry_flight_recorder.bin");
        let path_str = path.to_string_lossy().to_string();

        let recorder = FlightRecorder::new(&path_str, 16);
        let event = FlightEvent {
            timestamp: 1672531200000,
            trace_id: 42,
            event_type: 1,
            payload: [0u8; 47],
        };
        recorder.record(event);
        assert_eq!(recorder.head.load(Ordering::Relaxed), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_flight_recorder_circular_wrap_around_and_zero_capacity_clamping() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_telemetry_fr_wrap.bin");
        let path_str = path.to_string_lossy().to_string();

        // 0 capacity is clamped safely to 1
        let recorder = FlightRecorder::new(&path_str, 0);
        assert_eq!(recorder.capacity, 1);

        for i in 0..10 {
            recorder.record(FlightEvent {
                timestamp: i,
                trace_id: i * 100,
                event_type: 2,
                payload: [i as u8; 47],
            });
        }
        assert_eq!(recorder.head.load(Ordering::Relaxed), 10);

        let _ = std::fs::remove_file(path);
    }
}
