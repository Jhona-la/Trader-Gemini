pub mod binance_api;
pub mod executor;

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
