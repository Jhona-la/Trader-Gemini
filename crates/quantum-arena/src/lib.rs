pub mod atomic_float;
pub mod config;
pub mod state;
pub mod position;
pub mod tick_source;

// Re-exportamos los componentes principales
pub use atomic_float::AtomicF64;
pub use config::QuantumConfig;
pub use state::{GlobalArena, ScalpState, SwingState};
pub use position::{PositionManager, Position};
pub use tick_source::{TickSource, TickEvent};
