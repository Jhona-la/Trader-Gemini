pub mod connes_clifford_kan;
pub mod copulas;
pub mod correlation;
pub mod entrelazamiento;
pub mod ewma;
pub mod feynman_path_instanton;
pub mod firma_maxent;
pub mod grothendieck_sheaves;
pub mod hawkes;
pub mod hawkes_cross;
pub mod hodge_flow;
pub mod hurst_dfa;
pub mod matriz_rmt;
pub mod kalman;
pub mod koopman_dmd;
pub mod lead_lag;
pub mod microstructure;
pub mod mps;
pub mod multifractal;
pub mod navier_stokes;
pub mod normalizer;
pub mod omni_strategies;
pub mod path_signatures;
pub mod random_matrix_wilson;
pub mod rough_ergodic;
pub mod simd_neural_network;
pub mod spectral;
pub mod tensor_ring;
pub mod transfer_entropy;
pub mod welford;

// Re-exportar primitivas matemáticas O(1)
pub use connes_clifford_kan::{
    CliffordMultivector3D, ConnesCliffordKanEngine, ConnesCliffordKanReport, ConnesSpectralReport,
    ConnesSpectralTripleEngine, CubicBSplineKanEdge, HudsonParthasarathyEvolutor,
    HudsonParthasarathyState, NonCommutativeRegime, CONNES_HILBERT_DIM, KAN_SPLINE_BASIS_COUNT,
};
pub use correlation::MarketCorrelationHeatmap;
pub use ewma::Ewma;
pub use feynman_path_instanton::{
    FeynmanInstantonEngine, InstantonParams, InstantonRegime, InstantonTunnelingReport,
    MAX_PATH_POINTS,
};
pub use grothendieck_sheaves::{
    AtiyahSingerIndexReport, CellularSheafEngine, CellularSheafReport, GrothendieckSheafParams,
    SheafLaplacianWorkspace, SheafRestrictionEdge, SheafTopologicalRegime, MAX_SHEAF_EDGES,
    MAX_SHEAF_VERTICES,
};
pub use hawkes::HawkesProcessEngine;
pub use hodge_flow::{
    CrossMarketHodgeReport, HelmholtzHodgeFlowEngine, HelmholtzHodgeParams,
    HodgeDecompositionResult, HodgeRegime, MAX_HODGE_ASSETS,
};
pub use kalman::KalmanFilter1D;
pub use lead_lag::LeadLagAlphaEngine;
pub use microstructure::{
    obi_acceleration, order_book_imbalance, AdaptiveResonanceClustering,
    InstitutionalVolumeTracker, OFIModel, OrderFlowTracker, SpoofingDetector,
};
pub use multifractal::{
    smooth_persistence_degree, MultiScaleHurstConfluence, MultifractalParams,
    MultifractalSpectrumEngine,
};
pub use navier_stokes::{HydrodynamicRegime, NavierStokesParams, NavierStokesReynoldsEngine};
pub use normalizer::{GarmanKlassVolatilityEstimator, StatisticalNormalizer};
pub use omni_strategies::OmniStrategyEngine;
pub use random_matrix_wilson::{
    jacobi_eigensolver_symmetric, ContinuousLyapunovEstimator, DenoisedCorrelationResult,
    LyapunovDiagnostic, MarchenkoPasturBounds, MarchenkoPasturFilter, RandomMatrixWilsonEngine,
    WilsonOperatorClass, WilsonRGEngine, WilsonRGScaleReport, MAX_ASSETS, MAX_DYADIC_SCALES,
};
pub use rough_ergodic::{
    ErgodicGrowthEngine, ErgodicGrowthResult, MalliavinEvaluator, MalliavinRoughGreeks,
    RoughBergomiPredictor, ROUGH_FACTORS_COUNT,
};
pub use simd_neural_network::SimdNeuralNet;
pub use spectral::SpectralCycleEngine;
pub use tensor_ring::TensorRing;
pub use welford::WelfordOnline;
