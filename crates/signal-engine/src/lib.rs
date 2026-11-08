pub use strategy_core::maker;
pub use strategy_core::stat_arb;

// Re-export specific structs that other crates expect from signal_engine
pub mod coaxial_breakout;
pub mod continuous_volatility_operator;
pub mod derham_hodge_currents;
// U-ERR-1: `micro_scalp_trigger`, `swing_conformal_filter` y `turbo_scalper`
// nombraban BANDAS DE HORIZONTE («micro-scalp», «swing», «turbo») que ninguno
// de los tres motores decide: los tres declaraban `TradeHorizon::Continuous`
// (enum de una sola variante, retirado en LG-29).
// Ahora se llaman por la magnitud que miden.
pub mod conformal_reversion_filter;
pub mod contagion_modulator;
pub mod feynman_path_integral_routing;
pub mod feynman_propagator;
pub mod flow_excitation_confluence;
pub mod flow_impulse;
pub mod fractional_rough_volatility;
pub mod game_theoretic_nash;
pub mod hawkes_bessel;
pub mod jarzynski_crooks_thermodynamics;
pub mod kdv_soliton_dispersion;
pub mod mean_field_games_psychology;
pub mod navier_stokes_kolmogorov;
pub mod navier_stokes_manifold;
pub mod orchestrator;
pub mod perceptron_gate;
pub mod proyeccion_espectral;
pub mod quantum_behavioral_field;
pub mod quantum_oscillator;
pub mod quantum_slot_orchestrator;
pub mod renyi_tsallis_entropy;
pub mod riemann_burgers_shockwave;
pub mod ruelle_frobenius_transfer;
pub mod skill_motores;
pub mod soliton_wave;
pub mod srb_kolmogorov_entropy;
pub mod stochastic_resonance;
pub mod supersonic_shockwave;
pub mod trend_runner;
pub mod voto_espectral;
pub mod wilson_exact_renormalization;

pub use continuous_volatility_operator::{
    ContinuousVolatilityOperatorEngine, ContinuousVolatilityParams, ContinuousVolatilityState,
    VOL_SPECTRUM_SCALES,
};
pub use derham_hodge_currents::{
    DeRhamHodgeCurrentsEngine, DeRhamHodgeParams, DeRhamHodgeReport, DERHAM_SIMPLICIAL_NODES,
};
pub use feynman_path_integral_routing::{
    Complex64, FeynmanPathIntegralRoutingEngine, FeynmanRoutingParams, FeynmanRoutingReport,
    ROUTING_PATH_CANDIDATES, ROUTING_PATH_SLICES,
};
pub use feynman_propagator::{FeynmanPropagatorEngine, FeynmanPropagatorParams};
pub use fractional_rough_volatility::{
    FractionalRoughVolatilityEngine, FractionalRoughVolatilityParams,
    FractionalRoughVolatilityReport, ROUGH_VOL_WINDOW,
};
pub use hawkes_bessel::{HawkesBesselEngine, HawkesBesselParams};
pub use jarzynski_crooks_thermodynamics::{
    JarzynskiCrooksParams, JarzynskiCrooksReport, JarzynskiCrooksThermodynamicsEngine,
    JarzynskiTrajectorySample, CRITICAL_ENTROPY_PRODUCTION, DEFAULT_KB_BOLTZMANN,
    JARZYNSKI_WINDOW_CAPACITY,
};
pub use kdv_soliton_dispersion::{
    KdvSolitonDispersionEngine, KdvSolitonParams, KdvSolitonReport, KDV_GRID_POINTS,
};
pub use maker::{MakerEngine, MakerQuote};
pub use mean_field_games_psychology::{
    MeanFieldGamesParams, MeanFieldGamesPsychologyEngine, MeanFieldGamesReport, MFG_SPATIAL_NODES,
};
pub use navier_stokes_kolmogorov::{
    NavierStokesKolmogorovEngine, NavierStokesParams, NavierStokesReport, NS_GRID_DIM,
};
pub use navier_stokes_manifold::{NavierStokesManifoldEngine, NavierStokesManifoldParams};
pub use orchestrator::{
    compute_ensemble_boost, compute_ensemble_boost_effective,
    compute_rmt_voter_effective_bets_ratio, RMT_MAX_SOURCES, RMT_MIN_SAMPLES, RMT_SAMPLE_STRIDE,
    RMT_WINDOW_SAMPLES,
};
pub use perceptron_gate::{PerceptronGateEngine, PerceptronGateParams};
pub use proyeccion_espectral::{ProyeccionEspectralEngine, ProyeccionEspectralParams};
pub use quantum_behavioral_field::{
    BehavioralMarketInput, BehavioralStateReport, QuantumBehavioralFieldEngine,
    QuantumBehavioralParams,
};
pub use quantum_oscillator::{QuantumOscillatorEngine, QuantumOscillatorParams};
pub use quantum_slot_orchestrator::{
    MultiSlotEvaluationReport, MultiSlotOrderEmission, QuantumSlotDecision,
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams, SlotOrderProposal, SpectralSlot,
};
pub use renyi_tsallis_entropy::{RenyiTsallisEntropyEngine, RenyiTsallisEntropyParams};
pub use riemann_burgers_shockwave::{
    BurgersShockwaveReport, RiemannBurgersParams, RiemannBurgersShockwaveEngine,
};
pub use ruelle_frobenius_transfer::{
    RuelleFrobeniusParams, RuelleFrobeniusReport, RuelleFrobeniusTransferEngine, RUELLE_GRID_SIZE,
    RUELLE_HISTORY_CAPACITY,
};
pub use soliton_wave::{SolitonWaveEngine, SolitonWaveParams};
pub use srb_kolmogorov_entropy::{
    SrbEntropyParams, SrbEntropyReport, SrbKolmogorovEntropyEngine, CHAOS_EMBEDDING_DIM,
};
pub use stochastic_resonance::{StochasticResonanceEngine, StochasticResonanceParams};
pub use strategy_core::{SignalIntent, SignalType};
pub use supersonic_shockwave::{SupersonicShockwaveEngine, SupersonicShockwaveParams};
pub use wilson_exact_renormalization::{
    RgScaleCoupling, WilsonExactRenormalizationEngine, WilsonExactRgParams, WilsonExactRgReport,
    RG_MOMENTUM_SCALES,
};
