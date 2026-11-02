pub mod connes_clifford_kan;
pub mod copulas;
pub mod correlation;
pub mod entrelazamiento;
pub mod ewma;
pub mod feynman_path_instanton;
pub mod firma_maxent;
pub mod grothendieck_sheaves;
pub mod hairer_landauer_anderson;
pub mod hawkes;
pub mod hawkes_cross;
pub mod hodge_flow;
pub mod hurst_dfa;
pub mod kalman;
pub mod koopman_dmd;
pub mod lead_lag;
pub mod matriz_rmt;
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
pub mod simd_yoshida_replicator;
pub mod soa_shadow_otoc;
pub mod spectral;
pub mod spot_futures_abl_iceberg;
pub mod tensor_ring;
pub mod transfer_entropy;
pub mod tropical_kdv_soliton;
pub mod tzcnt_lut_kpz;
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
pub use hairer_landauer_anderson::{
    AndersonLocalizationEngine, AndersonLocalizationReport, HairerLandauerAndersonMasterEngine,
    HairerLandauerAndersonReport, HairerRegularityEngine, HairerRegularityReport,
    LandauerButtikerEngine, LandauerButtikerReport, MesoscopicTransportRegime, MESOSCOPIC_CHANNELS,
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
pub use simd_yoshida_replicator::{
    Avx512Lane8F64, EvolutionaryReplicatorSimplex, HamiltonianPhaseState, ReplicatorSimplexReport,
    SimdYoshidaReplicatorEngine, SimdYoshidaReplicatorReport, SymplecticEvolutionRegime,
    SymplecticOrbitReport, Yoshida4SymplecticIntegrator, SIMD_F64_LANES, YOSHIDA_C, YOSHIDA_D,
    YOSHIDA_W0, YOSHIDA_W1,
};
pub use soa_shadow_otoc::{
    InFlightMarginEntry, InFlightShadowLedger, OtocShadowRegime, QuantumOtocEngine,
    QuantumOtocReport, SoaOrderBookL2, SoaShadowOtocMasterEngine, SoaShadowOtocReport, OTOC_OP_DIM,
};
pub use spectral::SpectralCycleEngine;
pub use spot_futures_abl_iceberg::{
    AblWeakValueReport, AharonovBergmannLebowitzEngine, IcebergHunterL3Engine, IcebergL3Report,
    SpotFuturesAblIcebergMasterEngine, SpotFuturesAblIcebergReport, SpotFuturesAblRegime,
    SpotFuturesLeadLagEngine, SpotFuturesLeadLagReport, ABL_STATE_DIM, LEAD_LAG_WINDOW,
};
pub use tensor_ring::TensorRing;
pub use tropical_kdv_soliton::{
    tropical_add, tropical_mul, KdvSolitonIstAnalyzer, KdvSolitonReport, TropicalKdvCycleReport,
    TropicalKdvEngine, TropicalMaxPlusMatrix, TropicalSolitonRegime, TropicalSpectralReport,
    KDV_GRID_DIM, TROPICAL_ONE, TROPICAL_ZERO,
};
pub use tzcnt_lut_kpz::{
    BitmapTickScanner256, FastAsciiLutEncoder, KpzBitmapRegime, KpzSurfaceGrowthEngine,
    KpzSurfaceReport, TzcntLutKpzMasterEngine, TzcntLutKpzReport, DIGIT_PAIRS_LUT, KPZ_DYNAMIC_Z,
    KPZ_GROWTH_BETA, KPZ_ROUGHNESS_ALPHA, KPZ_SURFACE_SITES,
};
pub use welford::WelfordOnline;
