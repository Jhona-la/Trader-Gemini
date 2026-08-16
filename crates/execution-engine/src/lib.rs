pub mod binance_api;
pub mod client;
pub mod executor;
pub mod hot_swap;
pub mod router;
pub mod simulator;

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
