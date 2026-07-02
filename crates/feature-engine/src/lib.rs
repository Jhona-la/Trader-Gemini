pub mod ewma;
pub mod microstructure;
pub mod welford;
pub mod omni_strategies;
pub mod correlation;

// Re-exportar primitivas matemáticas O(1)
pub use ewma::Ewma;
pub use microstructure::{obi_acceleration, order_book_imbalance, OrderFlowTracker, OFIModel};
pub use welford::WelfordOnline;
pub use omni_strategies::OmniStrategyEngine;
pub use correlation::MarketCorrelationHeatmap;

