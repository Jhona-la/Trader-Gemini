pub use strategy_core::scalp;
pub use strategy_core::swing;
pub use strategy_core::maker;
pub use strategy_core::stat_arb;

// Re-export specific structs that other crates expect from signal_engine
pub mod turbo_scalper;
pub mod coaxial_breakout;
pub mod orchestrator;
pub mod micro_scalp_trigger;
pub mod swing_conformal_filter;
pub mod hawkes_bessel;
pub mod quantum_oscillator;
pub mod soliton_wave;
pub mod supersonic_shockwave;
pub mod game_theoretic_nash;
pub mod stochastic_resonance;
pub mod trend_runner;
pub mod perceptron_gate;

pub use scalp::{ScalpEngine, ScalpEngine as ScalpSignalEngine};
pub use swing::{SwingEngine, SwingEngine as SwingSignalEngine};
pub use maker::{MakerEngine, MakerQuote};
pub use stat_arb::StatArbEngine;
pub use strategy_core::{SignalIntent, SignalType, TradeHorizon};
