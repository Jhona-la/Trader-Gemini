#[derive(Debug, Clone, Copy)]
pub struct TickEvent {
    pub coin_id: usize,
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}
