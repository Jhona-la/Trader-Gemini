pub mod conformal;
pub mod maker;
pub mod momentum_booster;
pub mod multivariate_coint;
pub mod scalp;
pub mod stat_arb;
pub mod strategy_telemetry;
pub mod swing;
pub mod types;
pub mod vecm_arbitrage;

use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
pub use multivariate_coint::MultivariateCointegrationEngine;
pub use types::*;

pub trait QuantumStrategy: Send + Sync {
    fn name(&self) -> &str;

    /// Initializes and registers strategy-specific parameters in the OmniscientRegistry
    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String>;

    /// Main entry point to evaluate the strategy state on a new tick/bar
    fn evaluate(&self) -> f64; // returns signal strength or alpha

    /// D-101 & D-111: Entrada multiactivo escopada por moneda y símbolo
    fn evaluate_for_coin(&self, _coin_id: usize, _symbol: &str) -> f64 {
        self.evaluate()
    }

    /// Evaluación escopada opcional por coin_id y símbolo
    fn evaluate_scoped(&self, coin_id: Option<usize>, symbol: Option<&str>) -> f64 {
        if let (Some(cid), Some(sym)) = (coin_id, symbol) {
            self.evaluate_for_coin(cid, sym)
        } else {
            self.evaluate()
        }
    }

    /// Target operational trading horizon (Scalp vs Swing)
    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Scalp
    }
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
