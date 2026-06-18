use std::sync::atomic::{AtomicU64, Ordering};

pub const RING_CAPACITY: usize = 1024;
pub const FEATURE_SIZE: usize = 576 / 4; // 144 floats (576 bytes)

#[repr(C, align(64))]
pub struct QuantumRingBuffer {
    pub seqlock: AtomicU64,
    pub data: [[f32; FEATURE_SIZE]; RING_CAPACITY],
    pub lap_violation_count: AtomicU64,
}

impl QuantumRingBuffer {
    pub fn new() -> Self {
        Self {
            seqlock: AtomicU64::new(0),
            data: [[0.0; FEATURE_SIZE]; RING_CAPACITY],
            lap_violation_count: AtomicU64::new(0),
        }
    }

    /// Tries to write into the ring buffer. Implements Lap Detection.
    /// reader_idx is the index currently held by the reader.
    pub fn write_tick(&mut self, reader_idx: usize, payload: &[f32; FEATURE_SIZE]) -> bool {
        let current_seq = self.seqlock.load(Ordering::Relaxed);
        let writer_idx = ((current_seq / 2) as usize) % RING_CAPACITY;
        
        // LAP DETECTION: Don't overwrite what the reader is currently reading.
        // We drop the tick instead of corrupting the read.
        if (writer_idx + 1) % RING_CAPACITY == reader_idx {
            self.lap_violation_count.fetch_add(1, Ordering::Relaxed);
            return false; // Drop the tick
        }

        // 1. Mark as writing (odd)
        self.seqlock.store(current_seq + 1, Ordering::Release);

        // 2. Write data (SIMD mapped in Cython, here standard array copy)
        self.data[writer_idx].copy_from_slice(payload);

        // 3. Mark as complete (even)
        self.seqlock.store(current_seq + 2, Ordering::Release);
        true
    }

    /// Reads from the ring buffer safely using SeqLock pattern.
    pub fn read_tick(&self, read_idx: usize, out_payload: &mut [f32; FEATURE_SIZE]) -> bool {
        let mut retries = 0;
        loop {
            let seq1 = self.seqlock.load(Ordering::Acquire);
            
            // If odd, a write is in progress.
            if seq1 % 2 != 0 {
                retries += 1;
                if retries > 100 { return false; } // Degrade to older state or fail
                std::hint::spin_loop();
                continue;
            }

            out_payload.copy_from_slice(&self.data[read_idx]);

            let seq2 = self.seqlock.load(Ordering::Acquire);
            if seq1 == seq2 {
                return true; // Safe read
            }
        }
    }
}
