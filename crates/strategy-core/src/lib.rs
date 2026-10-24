pub mod birch_swinnerton_dyer;
pub mod kahler_hodge;
pub mod maker;
pub mod momentum_booster;
pub mod multivariate_coint;
pub mod prueba_reversion;
pub mod stat_arb;
pub mod types;
pub mod vecm_arbitrage;
pub mod yang_mills_gauge;

pub use birch_swinnerton_dyer::{
    bsd_hermite_c1, frobenius_trace_mod_p, legendre_symbol, BirchSwinnertonDyerEngine,
    BirchSwinnertonDyerMetrics, BSD_NUM_PRIMES, BSD_ODD_PRIMES,
};
pub use kahler_hodge::{KahlerHodgeEngine, KahlerHodgeMetrics, KahlerHodgeParams};
pub use multivariate_coint::MultivariateCointegrationEngine;
pub use vecm_arbitrage::{ContinuousOrnsteinUhlenbeckSde, JohansenVecmEngine, VecmArbitrageParams};
pub use yang_mills_gauge::{YangMillsGaugeEngine, YangMillsParams};
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
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

    /// OLA Ω145: Declara si la estrategia es contextualmente elegible/operable para este activo en este tick.
    /// Si una estrategia requiere condiciones de datos o dimensionales no cumplidas (e.g. universo
    /// multiactivo N >= 3, feed spot síncrono, o inputs ausentes), devuelve `false`.
    /// Estrategias inoperables se excluyen del denominador de fuentes del ensamble (no diluyen
    /// artificialmente la convicción del consenso). Toda estrategia operable que decida abstenerse
    /// (emitiendo 0.0) mantiene `is_operable_for_coin == true` y sí diluye la convicción como
    /// abstención legítima de mercado.
    fn is_operable_for_coin(&self, _coin_id: usize, _symbol: &str) -> bool {
        true
    }
}
