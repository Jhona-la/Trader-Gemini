

pub mod omni_multiplexer;
pub mod parser;
pub mod ws_client;
pub mod multiplexer;
pub mod historical;
pub mod bypass;
pub mod macro_data;
pub mod market_context;

// Exportar la conexión
pub use ws_client::BinanceStreamer;
pub use macro_data::MacroFetcher;
pub use market_context::MarketContextFetcher;


