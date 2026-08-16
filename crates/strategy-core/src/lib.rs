pub mod maker;
pub mod scalp;
pub mod stat_arb;
pub mod swing;
pub mod types;

use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
pub use types::*;

pub trait QuantumStrategy: Send + Sync {
    fn name(&self) -> &str;

    /// Initializes and registers strategy-specific parameters in the OmniscientRegistry
    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String>;

    /// Main entry point to evaluate the strategy state on a new tick/bar
    fn evaluate(&self) -> f64; // returns signal strength or alpha
}

pub struct StrategyOrchestrator {
    strategies: Vec<Box<dyn QuantumStrategy>>,
    registry: Arc<OmniscientRegistry>,
}

impl StrategyOrchestrator {
    pub fn new(registry: Arc<OmniscientRegistry>) -> Self {
        Self {
            strategies: Vec::new(),
            registry,
        }
    }

    pub fn add_strategy(&mut self, mut strategy: Box<dyn QuantumStrategy>) -> Result<(), String> {
        strategy.init(self.registry.clone())?;
        self.strategies.push(strategy);
        Ok(())
    }
}
