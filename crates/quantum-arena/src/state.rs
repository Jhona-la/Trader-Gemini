use crate::atomic_float::AtomicF64;
use crate::config::QuantumConfig;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Axioma V: Cohesión Celular Absoluta.
/// Los motores Scalp y Swing no se pisan porque operan en structs aislados, 
/// pero unidos dentro del mismo bloque contiguo de RAM (GlobalArena).

#[repr(C, align(64))]
pub struct ScalpState {
    pub pnl_realized: AtomicF64,
    pub pnl_unrealized: AtomicF64,
    pub active_positions: AtomicUsize,
    pub win_rate: AtomicF64,
    pub profit_factor: AtomicF64,
    pub kelly_fraction: AtomicF64,
}

impl Default for ScalpState {
    fn default() -> Self {
        Self {
            pnl_realized: AtomicF64::new(0.0),
            pnl_unrealized: AtomicF64::new(0.0),
            active_positions: AtomicUsize::new(0),
            win_rate: AtomicF64::new(0.55),     // Asumimos 55% inicial para que Kelly sea > 0
            profit_factor: AtomicF64::new(1.5), // Profit factor rentable
            kelly_fraction: AtomicF64::new(0.0),
        }
    }
}

#[repr(C, align(64))]
pub struct SwingState {
    pub pnl_realized: AtomicF64,
    pub pnl_unrealized: AtomicF64,
    pub active_positions: AtomicUsize,
    pub win_rate: AtomicF64,
    pub profit_factor: AtomicF64,
    pub kelly_fraction: AtomicF64,
}

impl Default for SwingState {
    fn default() -> Self {
        Self {
            pnl_realized: AtomicF64::new(0.0),
            pnl_unrealized: AtomicF64::new(0.0),
            active_positions: AtomicUsize::new(0),
            win_rate: AtomicF64::new(0.55),
            profit_factor: AtomicF64::new(1.5),
            kelly_fraction: AtomicF64::new(0.0),
        }
    }
}

/// GlobalArena: El hipergrafo en memoria que todos los hilos leen y escriben.
/// Contiene configuración atómica y estado aislado por horizonte de tiempo.
#[repr(C, align(64))]
pub struct GlobalArena {
    pub config: QuantumConfig,
    pub scalp: ScalpState,
    pub swing: SwingState,
    pub positions: crate::position::PositionManager,
    pub unified_capital: AtomicF64,
    pub tick_counter: AtomicU64,
}

impl GlobalArena {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            config: QuantumConfig::default(),
            scalp: ScalpState::default(),
            swing: SwingState::default(),
            positions: crate::position::PositionManager::default(),
            unified_capital: AtomicF64::new(initial_capital),
            tick_counter: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn increment_tick(&self) -> u64 {
        self.tick_counter.fetch_add(1, Ordering::Relaxed)
    }
}
