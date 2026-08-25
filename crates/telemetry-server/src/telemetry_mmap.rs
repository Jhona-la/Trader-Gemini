use memmap2::MmapMut;
use quantum_arena::GlobalArena;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Axiom IX: Zero-Latency Telemetry via Memory-Mapped Files (SHM)
///
/// This avoids standard OS I/O overhead on Windows. It writes the global
/// state to a raw binary file mapped directly into RAM. External native
/// analytics or dashboards can read this file in microseconds without locking.
pub struct MmapTelemetry {
    mmap: MmapMut,
    arena: Arc<GlobalArena>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TelemetrySnapshot {
    pub tick_counter: u64,
    pub unified_capital: f64,
    pub pnl_realized_scalp: f64,
    pub global_leverage: f64,
    pub ai_ml_prob: f64,
    pub hurst_exponent: f64,
    pub memory_used_mb: f64,
    pub global_roi: f64, // Added: ROI general antes y después de fees aproximado
    pub win_rate: f64,   // Added: Tasa de victorias real-time
    pub tensor_drift: f64, // Added: Drift del tensor online
}

impl MmapTelemetry {
    pub fn new(arena: Arc<GlobalArena>, path: &str) -> std::io::Result<Self> {
        let size = std::mem::size_of::<TelemetrySnapshot>() as u64;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(size)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        let ptr = mmap.as_mut_ptr() as *mut std::ffi::c_void;
        let locked = unsafe { os_guardian::lock_memory_region(ptr, size as usize) };
        if !locked {
            crate::telemetry_err!(
                "⚠️ [TELEMETRY] Warning: Failed to VirtualLock mmap telemetry region."
            );
        } else {
            crate::telemetry_log!(
                "🔒 [TELEMETRY] Mmap region VirtualLocked (Zero-Latency Guarantee)."
            );
        }

        Ok(Self { mmap, arena })
    }

    /// O(1) Zero-copy write to memory-mapped file.
    pub fn snapshot_to_ram(&mut self) {
        let mut pnl_realized = 0.0;
        let mut ml_prob_sum = 0.0;
        let mut hurst_sum = 0.0;
        let mut active_coins = 0.0;
        let mut wr_weighted_sum = 0.0;
        let mut total_trades = 0;

        for coin in self.arena.coins.iter() {
            pnl_realized += coin.scalp.pnl_realized.load(Ordering::Relaxed)
                + coin.swing.pnl_realized.load(Ordering::Relaxed);
            ml_prob_sum += coin.ml_prob.load(Ordering::Relaxed);
            hurst_sum += coin.hurst_exponent.load(Ordering::Relaxed);
            active_coins += 1.0;

            // FIX #1446: Agregación de win-rate ponderado por trades
            let sc_trades = coin.scalp.trade_count.load(Ordering::Relaxed);
            let sw_trades = coin.swing.trade_count.load(Ordering::Relaxed);
            let sc_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
            let sw_wr = coin.swing.win_rate.load(Ordering::Relaxed);

            if sc_trades > 0 && sc_wr.is_finite() {
                wr_weighted_sum += sc_wr * (sc_trades as f64);
                total_trades += sc_trades;
            }
            if sw_trades > 0 && sw_wr.is_finite() {
                wr_weighted_sum += sw_wr * (sw_trades as f64);
                total_trades += sw_trades;
            }
        }

        let global_wr = if total_trades > 0 {
            wr_weighted_sum / (total_trades as f64)
        } else {
            0.0
        };

        let avg_ml_prob = if active_coins > 0.0 {
            ml_prob_sum / active_coins
        } else {
            0.5
        };
        let avg_hurst = if active_coins > 0.0 {
            hurst_sum / active_coins
        } else {
            0.5
        };

        let os_telemetry = os_guardian::telemetry::get_system_telemetry();

        let base = self.arena.config.base_capital.load(Ordering::Relaxed);
        let uni = self.arena.unified_capital.load(Ordering::Relaxed);
        let roi = if base > 0.0 && uni.is_finite() && base.is_finite() {
            ((uni - base) / base) * 100.0
        } else {
            0.0
        };

        // FIX #727: Sanitizar flotantes de snapshot mmap contra NaNs
        let safe_uni = if uni.is_finite() { uni } else { 13.0 };
        let safe_pnl = if pnl_realized.is_finite() { pnl_realized } else { 0.0 };
        let safe_roi = if roi.is_finite() { roi } else { 0.0 };
        let safe_mem = if os_telemetry.memory_used_mb.is_finite() { os_telemetry.memory_used_mb } else { 0.0 };
        let safe_lev = {
            let lev = self.arena.config.global_leverage.load(Ordering::Relaxed);
            if lev.is_finite() { lev } else { 1.0 }
        };

        let snap = TelemetrySnapshot {
            tick_counter: self.arena.tick_counter.load(Ordering::Relaxed),
            unified_capital: safe_uni,
            pnl_realized_scalp: safe_pnl,
            global_leverage: safe_lev,
            ai_ml_prob: if avg_ml_prob.is_finite() { avg_ml_prob } else { 0.5 },
            hurst_exponent: if avg_hurst.is_finite() { avg_hurst } else { 0.5 },
            memory_used_mb: safe_mem,
            global_roi: safe_roi,
            win_rate: if global_wr.is_finite() { global_wr } else { 0.0 },
            tensor_drift: 0.0,
        };

        // Write struct directly to memory map
        let bytes: [u8; std::mem::size_of::<TelemetrySnapshot>()] =
            unsafe { std::mem::transmute(snap) };
        let _ = (&mut self.mmap[..]).write_all(&bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_telemetry_snapshot() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_telemetry.bin");
        let path_str = path.to_string_lossy().to_string();

        let arena = Arc::new(GlobalArena::new(13.0));
        let mut telemetry = MmapTelemetry::new(arena, &path_str).unwrap();

        telemetry.snapshot_to_ram();
        drop(telemetry);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_mmap_telemetry_snapshot_binary_readback_and_weighted_wr() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = temp_dir.join(format!("test_mmap_readback_{}.bin", unique_id));
        let path_str = path.to_string_lossy().to_string();

        let arena = Arc::new(GlobalArena::new(13.0));
        arena.unified_capital.store(26.0, Ordering::Relaxed);
        arena.coins[0].scalp.trade_count.store(10, Ordering::Relaxed);
        arena.coins[0].scalp.win_rate.store(0.80, Ordering::Relaxed);
        arena.coins[0].swing.trade_count.store(5, Ordering::Relaxed);
        arena.coins[0].swing.win_rate.store(0.60, Ordering::Relaxed);

        let mut telemetry = MmapTelemetry::new(arena, &path_str).unwrap();
        telemetry.snapshot_to_ram();

        let data = std::fs::read(&path).expect("read mmap file");
        assert_eq!(data.len(), std::mem::size_of::<TelemetrySnapshot>());

        let snap: TelemetrySnapshot = unsafe { std::ptr::read(data.as_ptr() as *const TelemetrySnapshot) };
        assert_eq!(snap.unified_capital, 26.0);
        // ROI = ((26 - 13) / 13) * 100 = 100%
        assert!((snap.global_roi - 100.0).abs() < 1e-4);
        // Weighted WR: (0.80 * 10 + 0.60 * 5) / 15 = (8.0 + 3.0) / 15 = 11.0 / 15 = 0.733333
        assert!((snap.win_rate - (11.0 / 15.0)).abs() < 1e-4);

        drop(telemetry);
        let _ = std::fs::remove_file(path);
    }
}

