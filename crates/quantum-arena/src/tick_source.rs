#[derive(Debug, Clone, Copy)]
pub struct TickEvent {
    pub coin_id: usize,
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

/// Axioma VII: Unificación Backtest = Producción (El Espejo Perfecto)
/// Este trait permite inyectar ticks al motor sin importar su origen.
pub trait TickSource: Send + Sync {
    #[allow(async_fn_in_trait)]
    async fn next_tick(&mut self) -> Option<TickEvent>;
}
