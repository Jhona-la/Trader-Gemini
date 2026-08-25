pub use strategy_core::maker;
pub use strategy_core::scalp;
pub use strategy_core::stat_arb;
pub use strategy_core::swing;

// Re-export specific structs that other crates expect from signal_engine
pub mod coaxial_breakout;
pub mod game_theoretic_nash;
pub mod hawkes_bessel;
pub mod micro_scalp_trigger;
pub mod orchestrator;
pub mod perceptron_gate;
pub mod quantum_oscillator;
pub mod renyi_tsallis_entropy;
pub mod soliton_wave;
pub mod stochastic_resonance;
pub mod supersonic_shockwave;
pub mod swing_conformal_filter;
pub mod trend_runner;
pub mod turbo_scalper;

pub use maker::{MakerEngine, MakerQuote};
pub use renyi_tsallis_entropy::RenyiTsallisEntropyEngine;
pub use scalp::{ScalpEngine, ScalpEngine as ScalpSignalEngine};
pub use stat_arb::StatArbEngine;
pub use strategy_core::{SignalIntent, SignalType, TradeHorizon};
pub use swing::{SwingEngine, SwingEngine as SwingSignalEngine};
