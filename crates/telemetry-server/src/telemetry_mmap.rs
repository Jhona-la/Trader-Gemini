use quantum_arena::GlobalArena;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io::Write;

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
    pub global_roi: f64,          // Added: ROI general antes y después de fees aproximado
    pub win_rate: f64,            // Added: Tasa de victorias real-time
    pub tensor_drift: f64,        // Added: Drift del tensor online
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
            crate::telemetry_err!("⚠️ [TELEMETRY] Warning: Failed to VirtualLock mmap telemetry region.");
        } else {
            crate::telemetry_log!("🔒 [TELEMETRY] Mmap region VirtualLocked (Zero-Latency Guarantee).");
        }
        
        Ok(Self { mmap, arena })
    }

    /// O(1) Zero-copy write to memory-mapped file.
    pub fn snapshot_to_ram(&mut self) {
        let mut pnl_realized = 0.0;
        let mut ml_prob_sum = 0.0;
        let mut hurst_sum = 0.0;
        let mut active_coins = 0.0;
        
        for coin in self.arena.coins.iter() {
            pnl_realized += coin.scalp.pnl_realized.load(Ordering::Relaxed) + coin.swing.pnl_realized.load(Ordering::Relaxed);
            ml_prob_sum += coin.ml_prob.load(Ordering::Relaxed);
            hurst_sum += coin.hurst_exponent.load(Ordering::Relaxed);
            active_coins += 1.0;
        }

        let avg_ml_prob = if active_coins > 0.0 { ml_prob_sum / active_coins } else { 0.5 };
        let avg_hurst = if active_coins > 0.0 { hurst_sum / active_coins } else { 0.5 };
        
        let os_telemetry = os_guardian::telemetry::get_system_telemetry();
        
        let base = self.arena.config.base_capital.load(Ordering::Relaxed);
        let uni = self.arena.unified_capital.load(Ordering::Relaxed);
        let roi = if base > 0.0 { ((uni - base) / base) * 100.0 } else { 0.0 };

        let snap = TelemetrySnapshot {
            tick_counter: self.arena.tick_counter.load(Ordering::Relaxed),
            unified_capital: uni,
            pnl_realized_scalp: pnl_realized,
            global_leverage: self.arena.config.global_leverage.load(Ordering::Relaxed),
            ai_ml_prob: avg_ml_prob,
            hurst_exponent: avg_hurst,
            memory_used_mb: os_telemetry.memory_used_mb,
            global_roi: roi,
            win_rate: 0.0, // To be fed dynamically
            tensor_drift: 0.0,
        };
        
        // Write struct directly to memory map
        let bytes: [u8; std::mem::size_of::<TelemetrySnapshot>()] = unsafe { std::mem::transmute(snap) };
        let _ = (&mut self.mmap[..]).write_all(&bytes);
    }
}
