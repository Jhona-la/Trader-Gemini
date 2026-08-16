pub mod bypass;
pub mod historical;
pub mod macro_data;
pub mod macro_feed;
pub mod market_context;
pub mod multiplexer;
pub mod omni_multiplexer;
pub mod onchain_feed;
pub mod parser;
pub mod persistence;
pub mod state_db;
pub mod ws_client;

// Exportar la conexión
pub use macro_data::MacroFetcher;
pub use market_context::MarketContextFetcher;
pub use ws_client::BinanceStreamer;

pub mod dynamic_ranker;
pub mod lakehouse_mmap;
