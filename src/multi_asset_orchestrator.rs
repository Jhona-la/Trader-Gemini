use quantum_arena::GlobalArena;
use signal_engine::{MakerEngine, MakerQuote, SignalIntent, SignalType, StatArbEngine};
use std::sync::Arc;

pub struct MultiAssetOrchestrator {
    pub arena: Arc<GlobalArena>,
    maker_engine: MakerEngine,
    stat_arb_engine: StatArbEngine,

    // Últimos precios
    pub btc_bid: f64,
    pub btc_ask: f64,
    pub eth_bid: f64,
    pub eth_ask: f64,

    // Estado de inventario
    pub btc_inventory_usd: f64,
}

impl MultiAssetOrchestrator {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        Self {
            arena,
            maker_engine: MakerEngine::new(0.0002), // 2 bps base spread
            stat_arb_engine: StatArbEngine::new(100, 2.0), // 100 ticks, umbral Z = 2.0
            btc_bid: 0.0,
            btc_ask: 0.0,
            eth_bid: 0.0,
            eth_ask: 0.0,
            btc_inventory_usd: 0.0,
        }
    }

    /// Procesa el tick de cualquier símbolo y retorna las intenciones de Quote (Maker) y Arb (StatArb).
    pub fn on_tick(
        &mut self,
        symbol: &str,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
    ) -> (Option<MakerQuote>, Option<SignalIntent>) {
        // FIX #1459: Validación estricta de finitud y positividad de precios de tick
        if !bid.is_finite()
            || !ask.is_finite()
            || !bid_qty.is_finite()
            || !ask_qty.is_finite()
            || bid <= 0.0
            || ask <= 0.0
            || bid_qty < 0.0
            || ask_qty < 0.0
        {
            return (None, None);
        }

        let mut quote: Option<MakerQuote> = None;
        let mut arb_signal: Option<SignalIntent> = None;

        let sym_lower = symbol.to_lowercase();

        if sym_lower.starts_with("btc") {
            self.btc_bid = bid;
            self.btc_ask = ask;

            // FIX #1419: Market Making usando parámetros dinámicos del genoma activo
            use std::sync::atomic::Ordering;
            let volatility = if bid > 0.0 && bid.is_finite() && ask.is_finite() {
                ((ask - bid) / bid).max(0.0)
            } else {
                0.0001
            };
            let maker_spread_pct = self
                .arena
                .config
                .maker_spread_pct
                .load(Ordering::Relaxed)
                .clamp(0.00005, 0.01);
            let maker_obi_threshold = self
                .arena
                .config
                .dynamic_ofi_threshold
                .load(Ordering::Relaxed)
                .clamp(0.1, 0.99);
            let tensor_poly_a = self
                .arena
                .config
                .tensor_poly_a
                .load(Ordering::Relaxed)
                .clamp(0.001, 1.0);
            let tensor_poly_b = self
                .arena
                .config
                .tensor_poly_b
                .load(Ordering::Relaxed)
                .clamp(0.0001, 0.1);

            let new_quote = self.maker_engine.generate_quote(
                bid,
                ask,
                bid_qty,
                ask_qty,
                self.btc_inventory_usd,
                volatility,
                maker_spread_pct,
                maker_obi_threshold,
                tensor_poly_a,
                tensor_poly_b,
            );
            quote = Some(new_quote);
        } else if sym_lower.starts_with("eth") {
            self.eth_bid = bid;
            self.eth_ask = ask;
        }

        // Si tenemos datos de ambos, calculamos la Cointegración (Arbitraje Estadístico)
        if self.btc_bid > 0.0 && self.btc_ask > 0.0 && self.eth_bid > 0.0 && self.eth_ask > 0.0 {
            let btc_mid = (self.btc_bid + self.btc_ask) / 2.0;
            let eth_mid = (self.eth_bid + self.eth_ask) / 2.0;

            if btc_mid.is_finite() && eth_mid.is_finite() && btc_mid > 0.0 && eth_mid > 0.0 {
                let arb = self.stat_arb_engine.update(btc_mid, eth_mid);
                // Si hay señal (Long/Short) o orden de salida (Flat con confidence 1.0)
                if arb.signal != SignalType::Flat || arb.confidence == 1.0 {
                    arb_signal = Some(arb);
                }
            }
        }

        (quote, arb_signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_asset_orchestrator_on_tick_btc_and_eth() {
        let arena = Arc::new(GlobalArena::new(13.0));
        let mut orch = MultiAssetOrchestrator::new(arena);

        let (q_btc, arb1) = orch.on_tick("BTCUSDT", 60000.0, 60001.0, 1.0, 1.0);
        assert!(q_btc.is_some());
        assert!(arb1.is_none());

        let (q_eth, _arb2) = orch.on_tick("ETHUSDT", 3000.0, 3000.5, 5.0, 5.0);
        assert!(q_eth.is_none()); // Quotes generated only for BTC in this orchestrator
    }

    #[test]
    fn test_multi_asset_orchestrator_nan_immunity() {
        let arena = Arc::new(GlobalArena::new(13.0));
        let mut orch = MultiAssetOrchestrator::new(arena);

        let (q_nan, arb_nan) = orch.on_tick("BTCUSDT", f64::NAN, 60001.0, 1.0, 1.0);
        assert!(q_nan.is_none());
        assert!(arb_nan.is_none());
    }
}
