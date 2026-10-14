//! # Metacortex Engine — Self-Modifying Organism Seed Crate
//!
//! The immutable seed of Trader Gemini. Generates code in `cerebro/cortex/`, executes immune tests
//! in `sistema_inmune/tests_vivos/`, invokes the compilation sandbox (`cargo build`), and hot-swaps modules.

pub mod cazador_constantes;
pub mod compiler_sandbox;
pub mod consejo_seniors;
pub mod epigenoma_store;
pub mod evolutionary_templates;
pub mod fases_autonomous;
pub mod hot_swap_controller;
pub mod immune_system;
pub mod online_learning;
pub mod prospect_theory;
pub mod reminiscence_and_adn;
pub mod shadow_graph_auditor;

use std::path::{Path, PathBuf};
use std::time::Instant;

pub use cazador_constantes::CazadorConstantes;
pub use compiler_sandbox::{CompilationConfig, CompilationResult, CompilerSandbox};
pub use consejo_seniors::{
    evaluate_causal_manipulation_veto, evaluate_ente_mercado_opinion, evaluate_execution_slippage,
    evaluate_execution_slippage_with_bounds, evaluate_metacognitive_opinion,
    evaluate_microstructure_flow_confidence, evaluate_ml_opinion_with_bounds,
    evaluate_risk_drawdown_and_conviction, evaluate_series_temporales_ym,
    evaluate_volatility_regime, evaluate_volatility_regime_spectral,
    evaluate_volatility_regime_with_bounds, ConsejoDeliberacion, ConsensusResult,
    MarketSnapshotPayload, SeniorOpinion, SeniorRole, DEFAULT_CASCADE_SEVERITY_BREAKER,
    DEFAULT_CAUSAL_ALIGNED_MAX_RISK, DEFAULT_CAUSAL_MIN_BOOK_IMBALANCE,
    DEFAULT_CAUSAL_VETO_THRESHOLD, DEFAULT_CROWD_IMBALANCE_DECAY, DEFAULT_CROWD_MIN_FLOOR,
    DEFAULT_FAST_SLIPPAGE_FLOOR, DEFAULT_HODGE_CURL_ATTENUATION, DEFAULT_HODGE_LAMINAR_FLOOR,
    DEFAULT_LIQ_DECAY_MULT, DEFAULT_LIQ_MIN_FLOOR, DEFAULT_MAX_DRAWDOWN_LIMIT,
    DEFAULT_META_DD_PENALTY_MULT, DEFAULT_META_MAJORITY_FACTOR, DEFAULT_META_MAX_DD_PENALTY,
    DEFAULT_META_SOLITARY_FACTOR, DEFAULT_ML_DEAD_ZONE, DEFAULT_ML_FULL_EDGE,
    DEFAULT_SLOW_SLIPPAGE_SPAN, DEFAULT_STREAK_DECAY_RATE, DEFAULT_WHALE_MIN_FLOOR,
    DEFAULT_WHALE_Z_THRESHOLD, DEFAULT_YM_MAX_WEIGHT, DEFAULT_YM_REF_CURRENT,
};
pub use prospect_theory::ProspectTheoryEngine;
pub use epigenoma_store::{read_epigenoma_gene, set_epigenoma_gene};
pub use evolutionary_templates::{
    ContinuumStrategyParams, EvolutionaryTemplateEngine, VolumeFundingParams, WaveletFeatureParams,
    WaveletType,
};
pub use fases_autonomous::{FaseAutonomous, FaseAutonomousManager, HealthMetrics};
pub use hot_swap_controller::{EpigenomaSymbolParams, HotSwapController};
pub use immune_system::{LivingImmuneSystem, TraumaRecord};
pub use online_learning::OnlineLearningModule;
pub use reminiscence_and_adn::{AdnBackupCatalog, GenerationMetadata, ReminiscenceModule};
pub use shadow_graph_auditor::{ShadowEvent, ShadowGraphAuditor};

pub struct MetacortexEngine {
    pub workspace_root: PathBuf,
    pub sandbox: CompilerSandbox,
    pub template_engine: EvolutionaryTemplateEngine,
    pub immune_system: LivingImmuneSystem,
    pub hot_swap: HotSwapController,
    pub consejo: ConsejoDeliberacion,
    pub phase_manager: FaseAutonomousManager,
    pub shadow_auditor: std::sync::Arc<ShadowGraphAuditor>,
    pub last_mutation_time: Option<Instant>,
    pub enforce_rate_limit: bool,
}

impl MetacortexEngine {
    pub fn new<P: AsRef<Path>>(workspace_root: P) -> Self {
        let root = workspace_root.as_ref().to_path_buf();
        let cortex_dir = root.join("cerebro").join("cortex");

        let config = CompilationConfig::default();
        let sandbox = CompilerSandbox::new(&root, config);
        let template_engine = EvolutionaryTemplateEngine::new(&cortex_dir);
        let immune_system = LivingImmuneSystem::new(&root);
        let hot_swap = HotSwapController::new(&root);
        let consejo = ConsejoDeliberacion::new();
        let phase_manager = FaseAutonomousManager::new();
        let shadow_auditor = std::sync::Arc::new(ShadowGraphAuditor::new());

        Self {
            workspace_root: root,
            sandbox,
            template_engine,
            immune_system,
            hot_swap,
            consejo,
            phase_manager,
            shadow_auditor,
            last_mutation_time: None,
            enforce_rate_limit: true,
        }
    }

    pub fn without_rate_limit(mut self) -> Self {
        self.enforce_rate_limit = false;
        self
    }

    /// Triggers a full mutation cycle for a new Wavelet Feature
    pub fn trigger_wavelet_mutation(
        &mut self,
        params: WaveletFeatureParams,
    ) -> Result<CompilationResult, String> {
        // Enforce hourly mutation frequency safeguard if enabled
        if self.enforce_rate_limit {
            if let Some(last) = self.last_mutation_time {
                if last.elapsed().as_secs() < 3600 {
                    return Err(
                        "Mutation rate limit exceeded (max 1 mutation per hour)".to_string()
                    );
                }
            }
        }

        // 1. Generate code in cortex/
        let target_file = self
            .template_engine
            .generate_wavelet_feature(&params)
            .map_err(|e| format!("Failed to generate wavelet feature code: {}", e))?;

        // 2. Generate immune test suite
        let _ = self.immune_system.generate_immune_tests();

        // 3. Compile in sandbox
        let result = self.sandbox.compile_package("trader-gemini-v5");
        if result.success {
            self.last_mutation_time = Some(Instant::now());
        } else {
            // Clean up uncompilable file on failure
            let _ = std::fs::remove_file(target_file);
        }

        Ok(result)
    }
}
