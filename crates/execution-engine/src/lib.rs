pub mod binance_api;
pub mod executor;
pub mod client;
pub mod simulator;
pub mod shadow;

#[derive(Debug, Clone)]
pub struct ExecutionPayload {
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub order_type: String,
    pub time_in_force: String,
    pub signature: String,
    pub timestamp: u64,
}
