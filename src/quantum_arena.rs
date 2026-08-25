use std::sync::atomic::{AtomicU64, Ordering};

pub const RING_CAPACITY: usize = 1024;
pub const FEATURE_SIZE: usize = 576 / 4; // 144 floats (576 bytes)

#[repr(C, align(64))]
pub struct QuantumStateArena {
    pub prices: *const f32,
    pub volumes: *const f32,
    pub tensor_len: usize,
    pub mempool_panic_score: f32,
    pub net_liq_pressure: f32,
    pub timestamp_ns: i64,
}

#[repr(C)]
pub struct TradeDecision {
    pub action: i32,
    pub position_size: f32,
    pub stop_loss: f32,
    pub take_profit: f32,
    pub confidence: f32,
    pub error_code: i32,
    pub mempool_panic: f32,
    pub net_liq_pressure: f32,
    pub liquidation_cascade: f32,
}

#[repr(C, align(64))]
pub struct QuantumRingBuffer {
    pub seqlock: AtomicU64,
    pub data: [[f32; FEATURE_SIZE]; RING_CAPACITY],
    pub lap_violation_count: AtomicU64,
}

impl Default for QuantumRingBuffer {
    fn default() -> Self {
        Self::new()
    }
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
        // FIX #1471: Modulo seguro sobre reader_idx
        if (writer_idx + 1) % RING_CAPACITY == (reader_idx % RING_CAPACITY) {
            self.lap_violation_count.fetch_add(1, Ordering::Relaxed);
            return false; // Drop the tick
        }

        // 1. Mark as writing (odd)
        self.seqlock.store(current_seq + 1, Ordering::Release);

        // 2. Write data (SIMD mapped in Cython, here standard array copy)
        // FIX #1471: Sanitización de floats antes de escribir en el ring buffer
        for (dst, src) in self.data[writer_idx].iter_mut().zip(payload.iter()) {
            *dst = if src.is_finite() { *src } else { 0.0 };
        }

        // 3. Mark as complete (even)
        self.seqlock.store(current_seq + 2, Ordering::Release);
        true
    }

    /// Reads from the ring buffer safely using SeqLock pattern.
    pub fn read_tick(&self, read_idx: usize, out_payload: &mut [f32; FEATURE_SIZE]) -> bool {
        // FIX #1418: Modulo de capacidad para prevenir pánicos por desbordamiento de índice
        let safe_idx = read_idx % RING_CAPACITY;
        let mut retries = 0;
        loop {
            let seq1 = self.seqlock.load(Ordering::Acquire);

            // If odd, a write is in progress.
            if !seq1.is_multiple_of(2) {
                retries += 1;
                if retries > 100 {
                    return false;
                } // Degrade to older state or fail
                std::hint::spin_loop();
                continue;
            }

            out_payload.copy_from_slice(&self.data[safe_idx]);

            let seq2 = self.seqlock.load(Ordering::Acquire);
            if seq1 == seq2 {
                return true; // Safe read
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_ring_buffer_write_and_read() {
        let mut ring = QuantumRingBuffer::new();
        let mut payload = [0.0f32; FEATURE_SIZE];
        payload[0] = 60000.0;
        payload[1] = 1.5;

        assert!(ring.write_tick(100, &payload));

        let mut read_payload = [0.0f32; FEATURE_SIZE];
        assert!(ring.read_tick(0, &mut read_payload));
        assert_eq!(read_payload[0], 60000.0);
        assert_eq!(read_payload[1], 1.5);
    }

    #[test]
    fn test_quantum_ring_buffer_lap_prevention() {
        let mut ring = QuantumRingBuffer::new();
        let payload = [1.0f32; FEATURE_SIZE];

        // Reader is at index 1, writer is currently at 0 (writer_idx + 1 == reader_idx)
        let written = ring.write_tick(1, &payload);
        assert!(!written); // Dropped tick to prevent lap violation
        assert_eq!(ring.lap_violation_count.load(Ordering::Relaxed), 1);
    }
}
