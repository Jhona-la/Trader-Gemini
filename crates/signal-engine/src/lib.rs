pub use strategy_core::maker;
pub use strategy_core::stat_arb;

// Re-export specific structs that other crates expect from signal_engine
pub mod ads_cft_holographic;
pub mod berry_phase_holonomy;
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
pub mod mfdfa_multifractal_hurst;
pub mod mpdo_dissipative_dmrg;
pub mod navier_stokes_kolmogorov;
pub mod navier_stokes_manifold;
pub mod nls_peregrine_soliton;
pub mod non_abelian_braiding_jones;
pub mod orchestrator;
pub mod perceptron_gate;
pub mod proyeccion_espectral;
pub mod quantum_behavioral_field;
pub mod quantum_many_body_scars;
pub mod quantum_oscillator;
pub mod quantum_slot_orchestrator;
pub mod renyi_tsallis_entropy;
pub mod riemann_burgers_shockwave;
pub mod ruelle_frobenius_transfer;
pub mod skill_motores;
pub mod soliton_wave;
pub mod srb_kolmogorov_entropy;
pub mod starobinsky_inflationary_attractor;
pub mod stochastic_resonance;
pub mod supersonic_shockwave;
pub mod supersymmetry_nicolai_map;
pub mod trend_runner;
pub mod voto_espectral;
pub mod wheeler_dewitt_minisuperspace;
pub mod wilson_exact_renormalization;

pub use ads_cft_holographic::{AdSCftHolographicEngine, AdSCftHolographicReport, AdSCftParams};
pub use berry_phase_holonomy::{BerryPhaseConfig, BerryPhaseEngine, BerryPhaseReport};
pub use continuous_volatility_operator::{
    ContinuousVolatilityOperatorEngine, ContinuousVolatilityParams, ContinuousVolatilityState,
    VOL_SPECTRUM_SCALES,
};
pub use derham_hodge_currents::{
    DERHAM_SIMPLICIAL_NODES, DeRhamHodgeCurrentsEngine, DeRhamHodgeParams, DeRhamHodgeReport,
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
    CRITICAL_ENTROPY_PRODUCTION, DEFAULT_KB_BOLTZMANN, JARZYNSKI_WINDOW_CAPACITY,
    JarzynskiCrooksParams, JarzynskiCrooksReport, JarzynskiCrooksThermodynamicsEngine,
    JarzynskiTrajectorySample,
};
pub use kdv_soliton_dispersion::{
    KDV_GRID_POINTS, KdvSolitonDispersionEngine, KdvSolitonParams, KdvSolitonReport,
};
pub use maker::{MakerEngine, MakerQuote};
pub use mean_field_games_psychology::{
    MFG_SPATIAL_NODES, MeanFieldGamesParams, MeanFieldGamesPsychologyEngine, MeanFieldGamesReport,
};
pub use mfdfa_multifractal_hurst::{
    MFDFA_ORDERS, MFDFA_SCALES, MFDFA_WINDOW_CAPACITY, MfDfaHurstReport,
    MfDfaMultifractalHurstEngine, MfDfaParams,
};
pub use mpdo_dissipative_dmrg::{
    LocalDensityMatrix, MPDO_CHANNELS, MpdoDissipativeConfig, MpdoDissipativeDmrgEngine,
    MpdoDissipativeReport,
};
pub use navier_stokes_kolmogorov::{
    NS_GRID_DIM, NavierStokesKolmogorovEngine, NavierStokesParams, NavierStokesReport,
};
pub use navier_stokes_manifold::{NavierStokesManifoldEngine, NavierStokesManifoldParams};
pub use nls_peregrine_soliton::{
    NLS_GRID_POINTS, NlsPeregrineParams, NlsPeregrineReport, NlsPeregrineSolitonEngine,
};
pub use non_abelian_braiding_jones::{
    BRAID_HISTORY_LEN, BRAID_STRANDS, NonAbelianBraidingConfig, NonAbelianBraidingJonesEngine,
    NonAbelianBraidingReport,
};
pub use orchestrator::{
    RMT_MAX_SOURCES, RMT_MIN_SAMPLES, RMT_SAMPLE_STRIDE, RMT_WINDOW_SAMPLES,
    compute_ensemble_boost, compute_ensemble_boost_effective,
    compute_rmt_voter_effective_bets_ratio,
};
pub use perceptron_gate::{PerceptronGateEngine, PerceptronGateParams};
pub use proyeccion_espectral::{ProyeccionEspectralEngine, ProyeccionEspectralParams};
pub use quantum_behavioral_field::{
    BehavioralMarketInput, BehavioralStateReport, QuantumBehavioralFieldEngine,
    QuantumBehavioralParams,
};
pub use quantum_many_body_scars::{
    QuantumManyBodyScarsConfig, QuantumManyBodyScarsEngine, QuantumManyBodyScarsReport,
    SCARS_HILBERT_DIM,
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
    RUELLE_GRID_SIZE, RUELLE_HISTORY_CAPACITY, RuelleFrobeniusParams, RuelleFrobeniusReport,
    RuelleFrobeniusTransferEngine,
};
pub use soliton_wave::{SolitonWaveEngine, SolitonWaveParams};
pub use srb_kolmogorov_entropy::{
    CHAOS_EMBEDDING_DIM, SrbEntropyParams, SrbEntropyReport, SrbKolmogorovEntropyEngine,
};
pub use starobinsky_inflationary_attractor::{
    StarobinskyConfig, StarobinskyInflationaryEngine, StarobinskyReport,
};
pub use stochastic_resonance::{StochasticResonanceEngine, StochasticResonanceParams};
pub use strategy_core::{SignalIntent, SignalType};
pub use supersonic_shockwave::{SupersonicShockwaveEngine, SupersonicShockwaveParams};
pub use supersymmetry_nicolai_map::{
    SupersymmetryNicolaiConfig, SupersymmetryNicolaiEngine, SupersymmetryNicolaiReport,
};
pub use wheeler_dewitt_minisuperspace::{
    WheelerDeWittConfig, WheelerDeWittMinisuperspaceEngine, WheelerDeWittReport,
};
pub use wilson_exact_renormalization::{
    RG_MOMENTUM_SCALES, RgScaleCoupling, WilsonExactRenormalizationEngine, WilsonExactRgParams,
    WilsonExactRgReport,
};
