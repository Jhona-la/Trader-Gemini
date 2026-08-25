pub mod active_universe;
pub mod adaptive_quantiles;
pub mod arena_alloc;
pub mod atomic_float;
pub mod config;
pub mod genome;
pub mod genome_store;
pub mod position;
pub mod ring_buffer;
pub mod state;
pub mod state_continuity;
pub mod symbol_registry;
pub mod symbols;
pub mod tick_source;
pub mod zero_alloc_pool;
pub mod zero_copy_stream;

// Re-exportamos los componentes principales
pub use atomic_float::AtomicF64;
pub use config::QuantumConfig;
pub use position::{Position, PositionManager};
pub use state::{GlobalArena, ScalpState, SwingState};
pub use tick_source::{TickEvent, TickSource};
