use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const TENSOR_RING_SIZE: usize = 128; // Potencia de 2 para bitwise masking rápido

struct Slot {
    sequence: AtomicUsize,
    timestamp: AtomicU64,
    features: [UnsafeCell<f64>; 54],
}

unsafe impl Sync for Slot {}
unsafe impl Send for Slot {}

impl Slot {
    const fn new(seq: usize) -> Self {
        const ZERO_CELL: UnsafeCell<f64> = UnsafeCell::new(0.0);
        Self {
            sequence: AtomicUsize::new(seq),
            timestamp: AtomicU64::new(0),
            features: [ZERO_CELL; 54],
        }
    }
}

pub struct TensorTelemetryRing {
    head: AtomicUsize,
    tail: AtomicUsize,
    slots: [Slot; TENSOR_RING_SIZE],
}

impl TensorTelemetryRing {
    pub const fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            slots: [
                Slot::new(0),
                Slot::new(1),
                Slot::new(2),
                Slot::new(3),
                Slot::new(4),
                Slot::new(5),
                Slot::new(6),
                Slot::new(7),
                Slot::new(8),
                Slot::new(9),
                Slot::new(10),
                Slot::new(11),
                Slot::new(12),
                Slot::new(13),
                Slot::new(14),
                Slot::new(15),
                Slot::new(16),
                Slot::new(17),
                Slot::new(18),
                Slot::new(19),
                Slot::new(20),
                Slot::new(21),
                Slot::new(22),
                Slot::new(23),
                Slot::new(24),
                Slot::new(25),
                Slot::new(26),
                Slot::new(27),
                Slot::new(28),
                Slot::new(29),
                Slot::new(30),
                Slot::new(31),
                Slot::new(32),
                Slot::new(33),
                Slot::new(34),
                Slot::new(35),
                Slot::new(36),
                Slot::new(37),
                Slot::new(38),
                Slot::new(39),
                Slot::new(40),
                Slot::new(41),
                Slot::new(42),
                Slot::new(43),
                Slot::new(44),
                Slot::new(45),
                Slot::new(46),
                Slot::new(47),
                Slot::new(48),
                Slot::new(49),
                Slot::new(50),
                Slot::new(51),
                Slot::new(52),
                Slot::new(53),
                Slot::new(54),
                Slot::new(55),
                Slot::new(56),
                Slot::new(57),
                Slot::new(58),
                Slot::new(59),
                Slot::new(60),
                Slot::new(61),
                Slot::new(62),
                Slot::new(63),
                Slot::new(64),
                Slot::new(65),
                Slot::new(66),
                Slot::new(67),
                Slot::new(68),
                Slot::new(69),
                Slot::new(70),
                Slot::new(71),
                Slot::new(72),
                Slot::new(73),
                Slot::new(74),
                Slot::new(75),
                Slot::new(76),
                Slot::new(77),
                Slot::new(78),
                Slot::new(79),
                Slot::new(80),
                Slot::new(81),
                Slot::new(82),
                Slot::new(83),
                Slot::new(84),
                Slot::new(85),
                Slot::new(86),
                Slot::new(87),
                Slot::new(88),
                Slot::new(89),
                Slot::new(90),
                Slot::new(91),
                Slot::new(92),
                Slot::new(93),
                Slot::new(94),
                Slot::new(95),
                Slot::new(96),
                Slot::new(97),
                Slot::new(98),
                Slot::new(99),
                Slot::new(100),
                Slot::new(101),
                Slot::new(102),
                Slot::new(103),
                Slot::new(104),
                Slot::new(105),
                Slot::new(106),
                Slot::new(107),
                Slot::new(108),
                Slot::new(109),
                Slot::new(110),
                Slot::new(111),
                Slot::new(112),
                Slot::new(113),
                Slot::new(114),
                Slot::new(115),
                Slot::new(116),
                Slot::new(117),
                Slot::new(118),
                Slot::new(119),
                Slot::new(120),
                Slot::new(121),
                Slot::new(122),
                Slot::new(123),
                Slot::new(124),
                Slot::new(125),
                Slot::new(126),
                Slot::new(127),
            ],
        }
    }

    pub fn lock_in_ram(&self) {
        #[cfg(windows)]
        unsafe {
            let _ = os_guardian::memory_compaction::lock_critical_memory(&self.slots);
        }
    }

    /// O(1) Lock-free thread-safe MPSC push desde el Hot Path (Engine/ML) sin heap allocations
    #[inline(always)]
    pub fn push_tensor(&self, timestamp_ms: u64, features: &[f64; 54]) -> Result<(), ()> {
        // FIX #1443: Sanitización de valores finitos en tensores de telemetría
        let mut safe_features = [0.0; 54];
        for i in 0..54 {
            safe_features[i] = if features[i].is_finite() {
                features[i]
            } else {
                0.0
            };
        }

        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[head & (TENSOR_RING_SIZE - 1)];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = (seq as isize).wrapping_sub(head as isize);
            if diff == 0 {
                match self.head.compare_exchange_weak(
                    head,
                    head.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        slot.timestamp.store(timestamp_ms, Ordering::Relaxed);
                        for i in 0..54 {
                            unsafe {
                                *slot.features[i].get() = safe_features[i];
                            }
                        }
                        slot.sequence.store(head.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => head = actual,
                }
            } else if diff < 0 {
                // Ring buffer is full
                return Err(());
            } else {
                head = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// O(1) Lock-free pop para Telemetría (Background Thread)
    pub fn pop_tensor(&self) -> Option<(u64, [f64; 54])> {
        let mut tail = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[tail & (TENSOR_RING_SIZE - 1)];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = (seq as isize).wrapping_sub((tail.wrapping_add(1)) as isize);
            if diff == 0 {
                match self.tail.compare_exchange_weak(
                    tail,
                    tail.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let ts = slot.timestamp.load(Ordering::Relaxed);
                        let mut feat = [0.0; 54];
                        for i in 0..54 {
                            feat[i] = unsafe { *slot.features[i].get() };
                        }
                        slot.sequence
                            .store(tail.wrapping_add(TENSOR_RING_SIZE), Ordering::Release);
                        return Some((ts, feat));
                    }
                    Err(actual) => tail = actual,
                }
            } else if diff < 0 {
                // Ring buffer is empty
                return None;
            } else {
                tail = self.tail.load(Ordering::Relaxed);
            }
        }
    }
}

// Global estática para evitar Arc<> y latencia de deref
pub static GLOBAL_TENSOR_TELEMETRY: TensorTelemetryRing = TensorTelemetryRing::new();

unsafe impl Sync for TensorTelemetryRing {}
unsafe impl Send for TensorTelemetryRing {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_telemetry_push_pop_and_nan_sanitization() {
        let ring = TensorTelemetryRing::new();
        assert!(ring.pop_tensor().is_none());

        let mut feats = [1.0; 54];
        feats[0] = f64::NAN;
        feats[1] = f64::INFINITY;

        assert!(ring.push_tensor(1672531200000, &feats).is_ok());

        let (ts, popped_feats) = ring.pop_tensor().unwrap();
        assert_eq!(ts, 1672531200000);
        assert_eq!(popped_feats[0], 0.0);
        assert_eq!(popped_feats[1], 0.0);
        assert_eq!(popped_feats[2], 1.0);

        assert!(ring.pop_tensor().is_none());
    }

    #[test]
    fn test_tensor_telemetry_ring_capacity_and_lock_in_ram() {
        let ring = TensorTelemetryRing::new();
        ring.lock_in_ram();

        let feats = [0.5; 54];
        // Fill the ring completely (TENSOR_RING_SIZE = 128)
        for i in 0..128 {
            assert!(ring.push_tensor(i as u64, &feats).is_ok());
        }

        // Ring is full -> next push returns Err(())
        assert!(ring.push_tensor(999, &feats).is_err());

        // Drain 1 item
        let (ts, _) = ring.pop_tensor().unwrap();
        assert_eq!(ts, 0);

        // Now push succeeds again
        assert!(ring.push_tensor(999, &feats).is_ok());
    }
}
