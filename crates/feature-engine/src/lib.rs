pub mod correlation;
pub mod ewma;
pub mod microstructure;
pub mod omni_strategies;
pub mod welford;

// Re-exportar primitivas matemáticas O(1)
pub use correlation::MarketCorrelationHeatmap;
pub use ewma::Ewma;
pub use microstructure::{obi_acceleration, order_book_imbalance, OFIModel, OrderFlowTracker};
pub use omni_strategies::OmniStrategyEngine;
pub use welford::WelfordOnline;

pub mod normalizer;
