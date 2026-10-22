pub mod api_client;
pub mod asset_selector;
pub mod bypass;
pub mod cero_dato;
pub mod chaos_monkey;
pub mod dynamic_ranker;
pub mod feature_vm;
pub mod historical;
pub mod http_cliente;
pub mod lakehouse_mmap;
// RB-R1 (Codex RB, 2026-10-19): `macro_data` (MacroFetcher) y
// `market_context` (MarketContextFetcher) ya no viven aquí. Eran los únicos
// usuarios de `polars` y `yahoo_finance_api` en este crate, y por ellos la
// pila polars entraba en el grafo de `god-engine-core`, `backtest-engine` y
// `evolution-engine`. MacroFetcher se retiró (duplicaba `macro_history_sync`
// y nadie leía sus MACRO_*.parquet); el fetcher de funding vive en la
// herramienta que lo usa (`src/bin/download_history/market_context.rs`).
pub mod macro_feed;
pub mod multiplexer;
pub mod omni_multiplexer;
pub mod onchain_feed;
pub mod parser;
pub mod persistence;
pub mod resilient_stream;
pub mod spot_feed;
pub mod state_db;
pub mod storage;
pub mod telemetry_bus;
pub mod teleonomia;
pub mod validation;
pub mod ws_client;

// Re-exports principales
pub use resilient_stream::ResilientStreamManager;
pub use spot_feed::start_spot_feed_sync;
pub use ws_client::BinanceStreamer;
