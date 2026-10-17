pub use strategy_core::maker;
pub use strategy_core::stat_arb;

// Re-export specific structs that other crates expect from signal_engine
pub mod coaxial_breakout;
// U-ERR-1: `micro_scalp_trigger`, `swing_conformal_filter` y `turbo_scalper`
// nombraban BANDAS DE HORIZONTE («micro-scalp», «swing», «turbo») que ninguno
// de los tres motores decide: los tres declaran `TradeHorizon::Continuous`.
// Ahora se llaman por la magnitud que miden.
pub mod conformal_reversion_filter;
pub mod contagion_modulator;
pub mod feynman_propagator;
pub mod flow_excitation_confluence;
pub mod flow_impulse;
pub mod game_theoretic_nash;
pub mod hawkes_bessel;
pub mod navier_stokes_manifold;
pub mod orchestrator;
pub mod perceptron_gate;
pub mod skill_motores;
pub mod voto_espectral;
pub mod proyeccion_espectral;
pub mod quantum_oscillator;
pub mod renyi_tsallis_entropy;
pub mod soliton_wave;
pub mod stochastic_resonance;
pub mod supersonic_shockwave;
pub mod trend_runner;

pub use feynman_propagator::{FeynmanPropagatorEngine, FeynmanPropagatorParams};
pub use hawkes_bessel::{HawkesBesselEngine, HawkesBesselParams};
pub use maker::{MakerEngine, MakerQuote};
pub use navier_stokes_manifold::{NavierStokesManifoldEngine, NavierStokesManifoldParams};
pub use orchestrator::compute_ensemble_boost;
pub use perceptron_gate::{PerceptronGateEngine, PerceptronGateParams};
pub use proyeccion_espectral::{ProyeccionEspectralEngine, ProyeccionEspectralParams};
pub use quantum_oscillator::{QuantumOscillatorEngine, QuantumOscillatorParams};
pub use renyi_tsallis_entropy::{RenyiTsallisEntropyEngine, RenyiTsallisEntropyParams};
pub use soliton_wave::{SolitonWaveEngine, SolitonWaveParams};
pub use stat_arb::StatArbEngine;
pub use stochastic_resonance::{StochasticResonanceEngine, StochasticResonanceParams};
pub use strategy_core::{SignalIntent, SignalType, TradeHorizon};
pub use supersonic_shockwave::{SupersonicShockwaveEngine, SupersonicShockwaveParams};
