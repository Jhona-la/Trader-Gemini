pub mod amplituhedron_order_flow;
pub mod ashtekar_lqg_area;
pub mod bergman_fock_reproducing;
pub mod bouchaud_mezard_ewl;
pub mod calabi_yau_seiberg_witten;
pub mod caputo_fraccional;
pub mod chern_simons_witten_knot;
pub mod connes_clifford_kan;
pub mod copulas;
pub mod correlation;
pub mod cover_orabona_wfr_ville;
pub mod entrelazamiento;
pub mod ewma;
pub mod feynman_path_instanton;
pub mod firma_maxent;
pub mod fisher_rao;
pub mod flujo_rg;
pub mod gross_pitaevskii_bec;
pub mod grothendieck_sheaves;
pub mod hairer_landauer_anderson;
pub mod hawkes;
pub mod hawkes_cross;
pub mod hodge_flow;
pub mod hurst_dfa;
pub mod hyperbolic_poincare_hodge;
pub mod jt_schwarzian_gravity;
pub mod kalman;
pub mod kam_eth_tur_stein;
pub mod koopman_dmd;
pub mod kostant_souriau_prequantum;
pub mod landauer_bec_hopfield;
pub mod lattice_qcd_wilson_polyakov;
pub mod lead_lag;
pub mod liouville_quantum_gravity;
pub mod lppls_sornette;
pub mod matrix_model_kp_hierarchy;
pub mod matriz_rmt;
pub mod metrica_fisher_empirica; // GLM-221 (C1): Fisher empirica q-gaussiana con beta_q
pub mod microstructure;
pub mod monge_kantorovich_wasserstein;
pub mod mps;
pub mod multifractal;
pub mod navier_stokes;
pub mod norma_hinf;
pub mod normalizer;
pub mod omni_strategies;
pub mod path_signatures;
pub mod penrose_spin_network;
pub mod peps_hypergraph_ricci;
pub mod perelman_ricci_surgery;
pub mod quantum_geodesic_phase_space;
pub mod random_matrix_wilson;
pub mod ricci_empirica;
pub mod rough_ergodic;
pub mod simd_neural_network;
pub mod simd_yoshida_replicator;
pub mod soa_shadow_otoc;
pub mod spectral;
pub mod spot_futures_abl_iceberg;
pub mod stoikov_microprice;
pub mod symplectic_floer_cohomology;
pub mod tensor_ring;
pub mod tqft_anyons;
pub mod transfer_entropy;
pub mod tropical_kdv_soliton;
pub mod tzcnt_lut_kpz;
pub mod vietoris_rips_persistence;
pub mod virasoro_cft_bootstrap;
pub mod voiculescu_jarzynski_svgd;
pub mod welford;
pub mod zakai_doob_amplituhedron;

// Re-exportar primitivas matemáticas O(1)
pub use amplituhedron_order_flow::{
    AmplituhedronOrderFlowConfig, AmplituhedronOrderFlowEngine, AmplituhedronOrderFlowReport,
};
pub use ashtekar_lqg_area::{
    ASHTEKAR_SPIN_CHANNELS, AshtekarLqgAreaEngine, AshtekarLqgParams, AshtekarLqgReport,
    BARBERO_IMMIRZI_GAMMA, PLANCK_AREA_SCALE,
};
pub use bergman_fock_reproducing::{
    BERGMAN_COMPLEX_DIM, BergmanFockParams, BergmanFockReport, BergmanFockReproducingEngine,
};
pub use bouchaud_mezard_ewl::{
    BouchaudMezardEngine, BouchaudMezardParams, BouchaudMezardReport, EwlQuantumGameEngine,
    EwlQuantumGameReport, LiquidityCondensationRegime, MAX_BM_LEVELS, QuantumStrategySU2,
};
pub use calabi_yau_seiberg_witten::{
    CY_COMPLEX_DIM, CY_HISTORY_LEN, CalabiYauParams, CalabiYauSeibergWittenEngine,
    CalabiYauSeibergWittenReport, ComplexCoord,
};
pub use chern_simons_witten_knot::{
    ChernSimonsWittenConfig, ChernSimonsWittenKnotEngine, ChernSimonsWittenReport, Crossing,
    KNOT_BRAID_STRANDS, KNOT_CROSSING_HISTORY,
};
pub use connes_clifford_kan::{
    CONNES_HILBERT_DIM, CliffordMultivector3D, ConnesCliffordKanEngine, ConnesCliffordKanReport,
    ConnesSpectralReport, ConnesSpectralTripleEngine, CubicBSplineKanEdge,
    HudsonParthasarathyEvolutor, HudsonParthasarathyState, KAN_SPLINE_BASIS_COUNT,
    NonCommutativeRegime,
};
pub use correlation::MarketCorrelationHeatmap;
pub use cover_orabona_wfr_ville::{
    ASSET_UNIVERSE, COVER_SIMPLEX_POINTS, CoverOrabonaWfrVilleMasterEngine,
    CoverOrabonaWfrVilleReport, CoverVovkReport, CoverVovkUniversalPortfolioEngine,
    FILOCHE_GRID_SIZE, FilocheMayborodaLocalizationEngine, FilocheMayborodaReport,
    GrossmanZhouKyleFloorEngine, GrossmanZhouKyleReport, OrabonaCoinBettingEngine,
    OrabonaCoinBettingReport, VilleMaximalSupermartingaleEngine, VilleSupermartingaleReport,
    WFR_BINS, WassersteinFisherRaoEngine, WfrReport,
};
pub use ewma::Ewma;
pub use feynman_path_instanton::{
    FeynmanInstantonEngine, InstantonParams, InstantonRegime, InstantonTunnelingReport,
    MAX_PATH_POINTS,
};
pub use gross_pitaevskii_bec::{
    BEC_ANGULAR_SECTORS, GrossPitaevskiiBecEngine, GrossPitaevskiiBecReport, GrossPitaevskiiParams,
    HBAR_EFF, PARTICLE_MASS_M,
};
pub use grothendieck_sheaves::{
    AtiyahSingerIndexReport, CellularSheafEngine, CellularSheafReport, GrothendieckSheafParams,
    MAX_SHEAF_EDGES, MAX_SHEAF_VERTICES, SheafLaplacianWorkspace, SheafRestrictionEdge,
    SheafTopologicalRegime,
};
pub use hairer_landauer_anderson::{
    AndersonLocalizationEngine, AndersonLocalizationReport, HairerLandauerAndersonMasterEngine,
    HairerLandauerAndersonReport, HairerRegularityEngine, HairerRegularityReport,
    LandauerButtikerEngine, LandauerButtikerReport, MESOSCOPIC_CHANNELS, MesoscopicTransportRegime,
};
pub use hawkes::HawkesProcessEngine;
pub use hodge_flow::{
    CrossMarketHodgeReport, HelmholtzHodgeFlowEngine, HelmholtzHodgeParams,
    HodgeDecompositionResult, HodgeRegime, MAX_HODGE_ASSETS,
};
pub use hyperbolic_poincare_hodge::{
    HyperbolicPoincareHodgeEngine, HyperbolicPoincareHodgeParams, HyperbolicPoincareHodgeReport,
    POINCARE_HISTORY_LEN, PoincarePoint,
};
pub use jt_schwarzian_gravity::{JtSchwarzianConfig, JtSchwarzianEngine, JtSchwarzianReport};
pub use kalman::KalmanFilter1D;
pub use kam_eth_tur_stein::{
    BaratoSeifertTurEngine, ETH_MAX_LEVELS, EthMblOrderbookEngine, EthMblRegime, EthMblReport,
    JS_DIM, JamesSteinTweedieReport, JamesSteinTweedieShrinkageEngine, KamChirikovRegime,
    KamChirikovReport, KamChirikovResonanceEngine, KamEthTurSteinMasterEngine,
    KamEthTurSteinReport, LATTICE_D1Q3_NODES, LatticeBoltzmannD1Q3Engine, LatticeBoltzmannReport,
    TurReport,
};
pub use kostant_souriau_prequantum::{
    KostantSouriauConfig, KostantSouriauPrequantumEngine, KostantSouriauReport,
};
pub use landauer_bec_hopfield::{
    BEC_ENERGY_LEVELS, BOLTZMANN_K_MKT, BecOrderbookReport, BecPhaseRegime,
    BoseEinsteinOrderbookEngine, HOPFIELD_FEATURE_DIM, HOPFIELD_PROTOTYPE_COUNT,
    KbnCompensatedAccumulator, LN_2, LandauerBecHopfieldMasterEngine, LandauerBecHopfieldReport,
    LandauerDissipationRegime, LandauerThermodynamicEngine, LandauerThermodynamicReport,
    ModernContinuousHopfieldEngine, ModernHopfieldReport,
};
pub use lattice_qcd_wilson_polyakov::{
    LATTICE_DIM_SPATIAL, LATTICE_DIM_TEMPORAL, LATTICE_PLAQUETTE_COUNT, LatticeQcdConfig,
    LatticeQcdReport, LatticeQcdWilsonPolyakovEngine, Su2Matrix,
};
pub use lead_lag::LeadLagAlphaEngine;
pub use liouville_quantum_gravity::{
    LiouvilleCft2dQuantumGravityEngine, LiouvilleCftParams, LiouvilleQuantumGravityReport,
};
pub use matrix_model_kp_hierarchy::{
    KP_MATRIX_DIM, KP_MOMENTS_COUNT, MatrixModelKpConfig, MatrixModelKpHierarchyEngine,
    MatrixModelKpReport,
};
pub use microstructure::{
    AdaptiveResonanceClustering, InstitutionalVolumeTracker, OFIModel, OrderFlowTracker,
    SpoofingDetector, obi_acceleration, order_book_imbalance,
};
pub use monge_kantorovich_wasserstein::{
    MongeKantorovichWassersteinEngine, WASSERSTEIN_NODES, WassersteinParams, WassersteinReport,
};
pub use multifractal::{
    MultiScaleHurstConfluence, MultifractalParams, MultifractalSpectrumEngine,
    smooth_persistence_degree,
};
pub use navier_stokes::{HydrodynamicRegime, NavierStokesParams, NavierStokesReynoldsEngine};
pub use normalizer::{GarmanKlassVolatilityEstimator, StatisticalNormalizer};
pub use omni_strategies::OmniStrategyEngine;
pub use penrose_spin_network::{
    MAX_SPIN_NODES, PenroseSpinNetworkEngine, PenroseSpinNetworkParams, PenroseSpinNetworkReport,
    SPIN_HISTORY_CAPACITY, SpinNode,
};
pub use peps_hypergraph_ricci::{
    PEPS_BOND_DIM, PEPS_COLS, PEPS_ROWS, PepsHypergraphRicciEngine, PepsHypergraphRicciReport,
    PepsRicciParams,
};
pub use perelman_ricci_surgery::{
    MAX_MANIFOLD_DIM, PerelmanRicciParams, PerelmanRicciSurgeryEngine, PerelmanSurgeryReport,
};
pub use quantum_geodesic_phase_space::{
    GeodesicPhaseSpaceParams, GeodesicPhaseSpaceReport, PHASE_SPACE_HISTORY_LEN, PhaseSpacePoint,
    QuantumGeodesicPhaseSpaceEngine,
};
pub use random_matrix_wilson::{
    ContinuousLyapunovEstimator, DenoisedCorrelationResult, LyapunovDiagnostic, MAX_ASSETS,
    MAX_DYADIC_SCALES, MarchenkoPasturBounds, MarchenkoPasturFilter, RandomMatrixWilsonEngine,
    WilsonOperatorClass, WilsonRGEngine, WilsonRGScaleReport, jacobi_eigensolver_symmetric,
};
pub use rough_ergodic::{
    ErgodicGrowthEngine, ErgodicGrowthResult, MalliavinEvaluator, MalliavinRoughGreeks,
    ROUGH_FACTORS_COUNT, RoughBergomiPredictor,
};
pub use simd_neural_network::SimdNeuralNet;
pub use simd_yoshida_replicator::{
    Avx512Lane8F64, EvolutionaryReplicatorSimplex, HamiltonianPhaseState, ReplicatorSimplexReport,
    SIMD_F64_LANES, SimdYoshidaReplicatorEngine, SimdYoshidaReplicatorReport,
    SymplecticEvolutionRegime, SymplecticOrbitReport, YOSHIDA_C, YOSHIDA_D, YOSHIDA_W0, YOSHIDA_W1,
    Yoshida4SymplecticIntegrator,
};
pub use soa_shadow_otoc::{
    InFlightMarginEntry, InFlightShadowLedger, OTOC_OP_DIM, OtocShadowRegime, QuantumOtocEngine,
    QuantumOtocReport, SoaOrderBookL2, SoaShadowOtocMasterEngine, SoaShadowOtocReport,
};
pub use spectral::SpectralCycleEngine;
pub use spot_futures_abl_iceberg::{
    ABL_STATE_DIM, AblWeakValueReport, AharonovBergmannLebowitzEngine, IcebergHunterL3Engine,
    IcebergL3Report, LEAD_LAG_WINDOW, SpotFuturesAblIcebergMasterEngine,
    SpotFuturesAblIcebergReport, SpotFuturesAblRegime, SpotFuturesLeadLagEngine,
    SpotFuturesLeadLagReport,
};
pub use stoikov_microprice::{
    IMBALANCE_STATES, RTT_WINDOW_SIZE, StoikovMicropriceEngine, StoikovMicropriceParams,
    StoikovMicropriceReport,
};
pub use symplectic_floer_cohomology::{
    SymplecticFloerCohomologyEngine, SymplecticFloerConfig, SymplecticFloerReport,
};
pub use tensor_ring::TensorRing;
pub use tqft_anyons::{
    AnyonFusionState2D, Complex64, PHI_GOLDEN, TqftAnyonReport, TqftFibonacciAnyonEngine,
};
pub use tropical_kdv_soliton::{
    KDV_GRID_DIM, KdvSolitonIstAnalyzer, KdvSolitonReport, TROPICAL_ONE, TROPICAL_ZERO,
    TropicalKdvCycleReport, TropicalKdvEngine, TropicalMaxPlusMatrix, TropicalSolitonRegime,
    TropicalSpectralReport, tropical_add, tropical_mul,
};
pub use tzcnt_lut_kpz::{
    BitmapTickScanner256, DIGIT_PAIRS_LUT, FastAsciiLutEncoder, KPZ_DYNAMIC_Z, KPZ_GROWTH_BETA,
    KPZ_ROUGHNESS_ALPHA, KPZ_SURFACE_SITES, KpzBitmapRegime, KpzSurfaceGrowthEngine,
    KpzSurfaceReport, TzcntLutKpzMasterEngine, TzcntLutKpzReport,
};
pub use vietoris_rips_persistence::{
    TDA_FILTRATION_LEVELS, TDA_POINT_CLOUD_LEN, TdaMicroPoint, VietorisRipsParams,
    VietorisRipsReport, VietorisRipsTopologicalEngine,
};
pub use virasoro_cft_bootstrap::{
    CFT_WINDOW_CAPACITY, ConformalRegime, VirasoroCftBootstrapEngine, VirasoroCftParams,
    VirasoroCftReport,
};
pub use voiculescu_jarzynski_svgd::{
    JarzynskiCrooksEngine, JarzynskiRegime, JarzynskiReport, MAX_FREE_EIGENVALUES,
    MAX_JARZYNSKI_SAMPLES, MAX_SVGD_PARTICLES, SVGD_DIM, SteinVariationalGradientDescentEngine,
    SvgdReport, VoiculescuFreeProbabilityEngine, VoiculescuFreeSpectrumReport,
    VoiculescuJarzynskiSvgdMasterEngine, VoiculescuJarzynskiSvgdReport,
};
pub use welford::WelfordOnline;
pub use zakai_doob_amplituhedron::{
    AMPLITUHEDRON_K, AMPLITUHEDRON_N, AmplituhedronReport, DISPERSION_FREQS,
    DoobMeyerCompensatorEngine, DoobMeyerReport, DoobMeyerToxicityRegime,
    KramersKronigDispersionEngine, KramersKronigReport, PLUCKER_COORDS_COUNT,
    PositiveGrassmannianAmplituhedronEngine, ZAKAI_GRID_SIZE, ZakaiDoobAmplituhedronMasterEngine,
    ZakaiDoobAmplituhedronReport, ZakaiFilterEngine, ZakaiFilterReport,
};
