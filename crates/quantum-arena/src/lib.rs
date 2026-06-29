pub mod atomic_float;
pub mod config;
pub mod state;

// Re-exportamos los componentes principales
pub use atomic_float::AtomicF64;
pub use config::QuantumConfig;
pub use state::{GlobalArena, ScalpState, SwingState};
