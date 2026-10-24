pub mod api_client;
pub mod cero_dato;
pub mod dynamic_ranker;
pub mod feature_vm;
pub mod historical;
pub mod http_cliente;
// RB-R1 (Codex RB, 2026-10-19): `macro_data` (MacroFetcher) y
// `market_context` (MarketContextFetcher) ya no viven aquí. Eran los únicos
// usuarios de `polars` y `yahoo_finance_api` en este crate, y por ellos la
// pila polars entraba en el grafo de `god-engine-core`, `backtest-engine` y
// `evolution-engine`. MacroFetcher se retiró (duplicaba `macro_history_sync`
// y nadie leía sus MACRO_*.parquet); el fetcher de funding vive en la
// herramienta que lo usa (`src/bin/download_history/market_context.rs`).
pub mod multiplexer;
pub mod omni_multiplexer;
pub mod parser;
pub mod spot_feed;
pub mod validation;
pub mod ws_client;

// Re-exports principales
pub use spot_feed::start_spot_feed_sync;
pub use ws_client::BinanceStreamer;
