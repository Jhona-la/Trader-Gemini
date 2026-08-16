//! # Metacortex Engine — Self-Modifying Organism Seed Crate
//!
//! The immutable seed of Trader Gemini. Generates code in `cerebro/cortex/`, executes immune tests
//! in `sistema_inmune/tests_vivos/`, invokes the compilation sandbox (`cargo build`), and hot-swaps modules.

pub mod compiler_sandbox;
pub mod evolutionary_templates;
pub mod immune_system;
pub mod hot_swap_controller;
pub mod shadow_graph_auditor;
pub mod consejo_seniors;
pub mod cazador_constantes;
pub mod quantum_evolver;
pub mod fases_autonomous;
pub mod epigenoma_store;
pub mod reminiscence_and_adn;
pub mod online_learning;

use std::path::{Path, PathBuf};
use std::time::Instant;

pub use compiler_sandbox::{CompilerSandbox, CompilationConfig, CompilationResult};
pub use evolutionary_templates::{
    EvolutionaryTemplateEngine, WaveletFeatureParams, VolumeFundingParams, DualHorizonStrategyParams, WaveletType
};
pub use immune_system::{LivingImmuneSystem, TraumaRecord};
pub use hot_swap_controller::{HotSwapController, EpigenomaSymbolParams};
pub use consejo_seniors::{ConsejoDeliberacion, SeniorRole, SeniorOpinion, ConsensusResult, MarketSnapshotPayload};
pub use cazador_constantes::CazadorConstantes;
pub use quantum_evolver::{QuantumEvolver, QuantumState};
pub use fases_autonomous::{FaseAutonomousManager, FaseAutonomous, HealthMetrics};
pub use epigenoma_store::{set_epigenoma_gene, read_epigenoma_gene};
pub use reminiscence_and_adn::{AdnBackupCatalog, GenerationMetadata, ReminiscenceModule};
pub use online_learning::OnlineLearningModule;

pub struct MetacortexEngine {
    pub workspace_root: PathBuf,
    pub sandbox: CompilerSandbox,
    pub template_engine: EvolutionaryTemplateEngine,
    pub immune_system: LivingImmuneSystem,
    pub hot_swap: HotSwapController,
    pub consejo: ConsejoDeliberacion,
    pub quantum_evolver: QuantumEvolver,
    pub phase_manager: FaseAutonomousManager,
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
        let quantum_evolver = QuantumEvolver::new();
        let phase_manager = FaseAutonomousManager::new();

        Self {
            workspace_root: root,
            sandbox,
            template_engine,
            immune_system,
            hot_swap,
            consejo,
            quantum_evolver,
            phase_manager,
            last_mutation_time: None,
            enforce_rate_limit: true,
        }
    }

    pub fn without_rate_limit(mut self) -> Self {
        self.enforce_rate_limit = false;
        self
    }

    /// Triggers a full mutation cycle for a new Wavelet Feature
    pub fn trigger_wavelet_mutation(&mut self, params: WaveletFeatureParams) -> Result<CompilationResult, String> {
        // Enforce hourly mutation frequency safeguard if enabled
        if self.enforce_rate_limit {
            if let Some(last) = self.last_mutation_time {
                if last.elapsed().as_secs() < 3600 {
                    return Err("Mutation rate limit exceeded (max 1 mutation per hour)".to_string());
                }
            }
        }

        // 1. Generate code in cortex/
        let target_file = self.template_engine.generate_wavelet_feature(&params)
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
