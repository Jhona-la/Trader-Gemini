pub mod active_universe;
pub mod adaptive_quantiles;
pub mod atomic_float;
pub mod config;
pub mod espectral_multiactivo;
pub mod evalues;
pub mod fee_schedule;
pub mod feed_health;
pub mod genome;
pub mod genome_store;
pub mod horizon_policy;
pub mod net_multiplexer;
pub mod paths;
pub mod position;
pub mod protection_health;
pub mod quantum_phase;
pub mod state;
pub mod state_continuity;
pub mod symbol_ranker_engine;
pub mod symbol_registry;
pub mod symbols;
pub mod spectral_regime;
pub mod spectral_tape;
pub mod temporal_spectrum;
pub mod tick_source;
pub mod topological_persistence;
pub mod wasserstein_kantorovich;

// Re-exportamos los componentes principales
pub use adaptive_quantiles::{
    AdaptiveQuantileEngine, P2Quantile, RingQuantile, CAPACIDAD_VENTANA_CUANTIL,
    MUESTRAS_MADURAS_CUANTIL,
};
pub use atomic_float::AtomicF64;
pub use config::QuantumConfig;
pub use fee_schedule::{FeeSchedule, FeeScheduleError, SymbolFees};
pub use espectral_multiactivo::{
    credibilidad_continua, familia_honesta, EspectralMultiactivo, EspectralMultiactivoParams,
    SnapshotParEscala,
};
pub use genome::{GenomeError, SuperGenotype};
pub use net_multiplexer::get_global_client;
pub use position::{Position, PositionManager};
pub use quantum_phase::{
    calcular_factor_cuantico, clasificar_fase, evaluar_fase_cuantica, FaseCuantica,
    QuantumCouplingParams, QuantumPhaseState,
};
pub use spectral_regime::{SpectralRegimeField, SpectralRegimeParams};
pub use state::GlobalArena;
pub use tick_source::TickEvent;
pub use topological_persistence::{
    MicroPoint3D, TopologicalMetrics, TopologicalPersistenceEngine,
    NUM_FILTRATION_SCALES, TOPOLOGICAL_WINDOW_CAP,
};
pub use wasserstein_kantorovich::{
    exact_empirical_w2_distance_8, wasserstein_hermite_c1, WassersteinKantorovichEngine,
    WassersteinKantorovichMetrics, WASSERSTEIN_HALF_CAP, WASSERSTEIN_WINDOW_CAP,
};
