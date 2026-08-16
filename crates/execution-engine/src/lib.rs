pub mod binance_api;
pub mod executor;
pub mod client;
pub mod simulator;
pub mod router;
pub mod hot_swap;

#[derive(Debug, Clone)]
pub struct ExecutionPayload {
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub order_type: String,
    pub time_in_force: String,
    pub position_side: String,
    pub signature: String,
    pub timestamp: u64,
}
