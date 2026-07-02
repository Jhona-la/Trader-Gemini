use std::sync::Arc;
use quantum_arena::GlobalArena;
use signal_engine::{MakerEngine, StatArbEngine, MakerQuote, SignalIntent, SignalType};

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
    pub fn on_tick(&mut self, symbol: &str, bid: f64, ask: f64, bid_qty: f64, ask_qty: f64) -> (Option<MakerQuote>, Option<SignalIntent>) {
        let mut quote: Option<MakerQuote> = None;
        let mut arb_signal: Option<SignalIntent> = None;
        
        let sym_lower = symbol.to_lowercase();
        
        if sym_lower.starts_with("btc") {
            self.btc_bid = bid;
            self.btc_ask = ask;
            
            // Market Making: Cotizamos alrededor de BTCUSDT usando Order Flow Imbalance
            let volatility = (ask - bid) / bid; 
            let maker_spread_pct = 0.0005; // Fallback since MAO doesn't use the arena config fully
            let maker_obi_threshold = 0.7;
            let new_quote = self.maker_engine.generate_quote(bid, ask, bid_qty, ask_qty, self.btc_inventory_usd, volatility, maker_spread_pct, maker_obi_threshold);
            quote = Some(new_quote);
        } else if sym_lower.starts_with("eth") {
            self.eth_bid = bid;
            self.eth_ask = ask;
        }
        
        // Si tenemos datos de ambos, calculamos la Cointegración (Arbitraje Estadístico)
        if self.btc_bid > 0.0 && self.eth_bid > 0.0 {
            let btc_mid = (self.btc_bid + self.btc_ask) / 2.0;
            let eth_mid = (self.eth_bid + self.eth_ask) / 2.0;
            
            let arb = self.stat_arb_engine.update(btc_mid, eth_mid);
            // Si hay señal (Long/Short) o orden de salida (Flat con confidence 1.0)
            if arb.signal != SignalType::Flat || arb.confidence == 1.0 {
                arb_signal = Some(arb);
            }
        }
        
        (quote, arb_signal)
    }
}
