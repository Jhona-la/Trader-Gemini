pub mod correlation;
pub mod ewma;
pub mod hawkes;
pub mod kalman;
pub mod lead_lag;
pub mod microstructure;
pub mod multifractal;
pub mod normalizer;
pub mod omni_strategies;
pub mod quantum_tensor_store;
pub mod shannon_entropy;
pub mod simd_neural_network;
pub mod spectral;
pub mod tensor_ring;
pub mod welford;

// Re-exportar primitivas matemáticas O(1)
pub use correlation::MarketCorrelationHeatmap;
pub use ewma::Ewma;
pub use hawkes::HawkesProcessEngine;
pub use kalman::KalmanFilter1D;
pub use lead_lag::LeadLagAlphaEngine;
pub use microstructure::{
    obi_acceleration, order_book_imbalance, AdaptiveResonanceClustering,
    InstitutionalVolumeTracker, OFIModel, OrderFlowTracker,
};
pub use multifractal::{MultifractalSpectrumEngine, MultiScaleHurstConfluence};
pub use normalizer::{GarmanKlassVolatilityEstimator, StatisticalNormalizer};
pub use omni_strategies::OmniStrategyEngine;
pub use shannon_entropy::ShannonEntropyEngine;
pub use simd_neural_network::SimdNeuralNet;
pub use spectral::SpectralCycleEngine;
pub use welford::WelfordOnline;

