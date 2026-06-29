pub mod atomic_float;
pub mod config;
pub mod state;
pub mod position;

// Re-exportamos los componentes principales
pub use atomic_float::AtomicF64;
pub use config::QuantumConfig;
pub use state::{GlobalArena, ScalpState, SwingState};
pub use position::{PositionManager, Position};
