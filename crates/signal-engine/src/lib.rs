pub use strategy_core::scalp;
pub use strategy_core::swing;
pub use strategy_core::maker;
pub use strategy_core::stat_arb;

// Re-export specific structs that other crates expect from signal_engine
pub use scalp::{ScalpEngine, ScalpEngine as ScalpSignalEngine};
pub use swing::{SwingEngine, SwingEngine as SwingSignalEngine};
pub use maker::{MakerEngine, MakerQuote};
pub use stat_arb::StatArbEngine;
pub use strategy_core::{SignalIntent, SignalType};
