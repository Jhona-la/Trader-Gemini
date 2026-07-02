use feature_engine::microstructure::OFIModel;

#[derive(Debug, Clone)]
pub struct MakerQuote {
    pub bid_price: f64,
    pub ask_price: f64,
}

pub struct MakerEngine {
    ofi_model: OFIModel,
    _base_spread_bps: f64,
}

impl MakerEngine {
    pub fn new(base_spread_bps: f64) -> Self {
        Self {
            ofi_model: OFIModel::new(), // Lookback de 10 ticks interno
            _base_spread_bps: base_spread_bps,
        }
    }

    #[inline(always)]
    pub fn generate_quote(
        &mut self,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
        inventory_delta_usd: f64,
        volatility: f64,
        genome_spread_pct: f64,
        genome_obi_threshold: f64,
    ) -> MakerQuote {
        let _ofi = self.ofi_model.update(bid, ask, bid_qty, ask_qty);
        let total_vol = bid_qty + ask_qty;
        let obi = if total_vol > 0.0 { (bid_qty - ask_qty) / total_vol } else { 0.0 };
        
        let mid = (bid + ask) / 2.0;
        
        // Ampliamos el spread si la volatilidad es alta para protegernos de toxicidad
        // Usamos el spread base dinámico proveniente del genoma
        let dynamic_spread_pct = genome_spread_pct + (volatility * 0.005);
        let half_spread = mid * dynamic_spread_pct;
        
        // Skews
        // Si OBI > threshold (gran presión compradora), subimos los precios asimétricamente
        let mut obi_skew = 0.0;
        if obi > genome_obi_threshold {
            obi_skew = mid * 0.0002; // Subir precio 0.02%
        } else if obi < -genome_obi_threshold {
            obi_skew = -mid * 0.0002; // Bajar precio 0.02%
        }
        
        // Si inventory > 0 (estamos Long), bajamos los precios para salir rápido y evitar acumular
        let inv_skew = inventory_delta_usd * (mid * 0.000005); 
        
        let total_skew = obi_skew - inv_skew;
        
        let optimal_bid = mid - half_spread + total_skew;
        let optimal_ask = mid + half_spread + total_skew;
        
        // Regla estricta de Market Maker: NUNCA cruzar el spread real de mercado (eso pagaría Taker fee)
        let final_bid = optimal_bid.min(bid);
        let final_ask = optimal_ask.max(ask);
        
        MakerQuote {
            bid_price: final_bid,
            ask_price: final_ask,
        }
    }
}
