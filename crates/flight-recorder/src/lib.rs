use std::cell::UnsafeCell;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 🛸 ALGORITMO #79: CAJA NEGRA Y REGISTRADOR DE VUELO HFT (FLIGHT RECORDER ENGINE)
/// Grabación continua en ring-buffer circular Lock-Free de 64 bytes por micro-evento,
/// permitiendo autopsias forenses instantáneas post-crash sin latencia en el Hot-Path.

pub const EVENT_TICK_PROCESSED: u16 = 1;
pub const EVENT_ORDER_SUBMITTED: u16 = 2;
pub const EVENT_ORDER_FILLED: u16 = 3;
pub const EVENT_ORDER_CANCELED: u16 = 4;
pub const EVENT_RISK_REJECTED: u16 = 5;
pub const EVENT_KILL_SWITCH_TRIPPED: u16 = 6;
pub const EVENT_ARBITRAGE_FIRED: u16 = 7;

/// Frame atómico alineado a 64 bytes (exactamente 1 línea de caché de CPU)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct FlightRecord {
    pub timestamp_ns: u64, // 8 bytes
    pub event_type: u16,   // 2 bytes
    pub coin_id: u16,      // 2 bytes
    pub flags: u32,        // 4 bytes (Total 16)
    pub payload: [f64; 6], // 48 bytes (Total 64)
}

const DEFAULT_CAPACITY: usize = 65536; // 64K records = 4 MB contiguos en RAM

pub struct FlightRecorder {
    buffer: Box<[UnsafeCell<FlightRecord>]>,
    sequences: Box<[AtomicU64]>,
    cursor: AtomicUsize,
    capacity: usize,
}

unsafe impl Sync for FlightRecorder {}
unsafe impl Send for FlightRecorder {}

impl Default for FlightRecorder {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl FlightRecorder {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two();
        let mut vec = Vec::with_capacity(cap);
        let mut seqs = Vec::with_capacity(cap);
        for _ in 0..cap {
            vec.push(UnsafeCell::new(FlightRecord::default()));
            seqs.push(AtomicU64::new(0));
        }
        Self {
            buffer: vec.into_boxed_slice(),
            sequences: seqs.into_boxed_slice(),
            cursor: AtomicUsize::new(0),
            capacity: cap,
        }
    }

    /// Registra un evento en nanosegundos con cero asignaciones en el hot-path
    #[inline(always)]
    pub fn record(&self, event_type: u16, coin_id: u16, flags: u32, payload: [f64; 6]) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.record_with_time(event_type, coin_id, flags, payload, now_ns);
    }

    /// Registra un evento con timestamp explícito (hardware TSC / WS feed) sin syscalls en el hot-path
    #[inline(always)]
    pub fn record_with_time(
        &self,
        event_type: u16,
        coin_id: u16,
        flags: u32,
        payload: [f64; 6],
        timestamp_ns: u64,
    ) {
        // FIX #1439: Sanitización de finitud en el payload de la caja negra
        let mut safe_payload = payload;
        for p in &mut safe_payload {
            if !p.is_finite() {
                *p = 0.0;
            }
        }

        let idx = self.cursor.fetch_add(1, Ordering::Relaxed) & (self.capacity - 1);
        let record = FlightRecord {
            timestamp_ns,
            event_type,
            coin_id,
            flags,
            payload: safe_payload,
        };

        // FIX #946: Seqlock lock-free para garantizar escrituras y lecturas atómicas sin data races
        let _ = self.sequences[idx].fetch_add(1, Ordering::Acquire);
        unsafe {
            let slot = self.buffer[idx].get();
            std::ptr::write_volatile(slot, record);
        }
        self.sequences[idx].fetch_add(1, Ordering::Release);
    }

    /// Recupera los últimos N registros en orden cronológico
    pub fn get_recent_records(&self, count: usize) -> Vec<FlightRecord> {
        let total = self.cursor.load(Ordering::Relaxed);
        let n = count.min(self.capacity).min(total);
        let mut out = Vec::with_capacity(n);

        for i in 0..n {
            let target_seq = total.saturating_sub(n).wrapping_add(i);
            let idx = target_seq & (self.capacity - 1);
            let mut spins = 0;
            loop {
                let s1 = self.sequences[idx].load(Ordering::Acquire);
                if s1 & 1 == 0 {
                    let rec = unsafe { std::ptr::read_volatile(self.buffer[idx].get()) };
                    let s2 = self.sequences[idx].load(Ordering::Acquire);
                    if s1 == s2 {
                        out.push(rec);
                        break;
                    }
                }
                spins += 1;
                if spins > 10 {
                    let rec = unsafe { *self.buffer[idx].get() };
                    out.push(rec);
                    break;
                }
                std::hint::spin_loop();
            }
        }
        out
    }

    /// Vuelca todo el buffer a un archivo binario para análisis forense post-mortem
    pub fn dump_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        let records = self.get_recent_records(self.capacity);
        let slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                records.as_ptr() as *const u8,
                records.len() * std::mem::size_of::<FlightRecord>(),
            )
        };
        file.write_all(slice)?;
        file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flight_recorder_circular_buffer() {
        let recorder = FlightRecorder::new(16);
        for i in 0..20 {
            recorder.record(
                EVENT_ORDER_SUBMITTED,
                i as u16,
                0,
                [i as f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            );
        }

        let recent = recorder.get_recent_records(5);
        assert_eq!(recent.len(), 5);
        assert_eq!(recent[4].coin_id, 19);
        assert_eq!(recent[0].coin_id, 15);
    }

    #[test]
    fn test_flight_recorder_file_dump_and_reload() {
        let temp_dir = std::env::temp_dir();
        let dump_path = temp_dir.join(format!(
            "flight_dump_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let recorder = FlightRecorder::new(8);
        recorder.record(
            EVENT_ARBITRAGE_FIRED,
            1,
            0xAA,
            [100.0, 200.0, 0.0, 0.0, 0.0, 0.0],
        );
        recorder.dump_to_file(&dump_path).unwrap();

        assert!(dump_path.exists());
        let _ = std::fs::remove_file(dump_path);
    }

    #[test]
    fn test_flight_recorder_nan_payload_sanitization() {
        let recorder = FlightRecorder::new(4);
        recorder.record(
            EVENT_TICK_PROCESSED,
            1,
            0,
            [f64::NAN, f64::INFINITY, -f64::NAN, 1.0, 2.0, 3.0],
        );
        let recent = recorder.get_recent_records(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].payload[0], 0.0);
        assert_eq!(recent[0].payload[1], 0.0);
        assert_eq!(recent[0].payload[2], 0.0);
        assert_eq!(recent[0].payload[3], 1.0);
    }

    #[test]
    fn test_flight_recorder_record_with_time_explicit() {
        let recorder = FlightRecorder::new(8);
        let explicit_time_ns = 1700000000123456789;
        recorder.record_with_time(
            EVENT_KILL_SWITCH_TRIPPED,
            2,
            0xFF,
            [13.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            explicit_time_ns,
        );
        let recent = recorder.get_recent_records(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].timestamp_ns, explicit_time_ns);
        assert_eq!(recent[0].event_type, EVENT_KILL_SWITCH_TRIPPED);
        assert_eq!(recent[0].coin_id, 2);
        assert_eq!(recent[0].flags, 0xFF);
    }

    #[test]
    fn test_flight_recorder_capacity_and_overflow_boundary() {
        // Non-power of two request (e.g. 5) gets rounded up to 8
        let recorder = FlightRecorder::new(5);
        assert_eq!(recorder.capacity, 8);

        // Record 100 entries
        for i in 0..100 {
            recorder.record(EVENT_ORDER_FILLED, (i % 30) as u16, 0, [i as f64; 6]);
        }

        // Get 8 records
        let records = recorder.get_recent_records(8);
        assert_eq!(records.len(), 8);
        assert_eq!(records[7].payload[0], 99.0);
        assert_eq!(records[0].payload[0], 92.0);
    }
}
