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
    #[allow(clippy::too_many_arguments)]
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
        tensor_poly_a: f64,
        tensor_poly_b: f64,
    ) -> MakerQuote {
        if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= bid || !bid_qty.is_finite() || !ask_qty.is_finite() {
            return MakerQuote {
                bid_price: if bid.is_finite() && bid > 0.0 { bid } else { 1e-8 },
                ask_price: if ask.is_finite() && ask > 0.0 { ask } else { 2e-8 },
            };
        }

        // FIX #690: Sanitizar parámetros flotantes
        let safe_inv = if inventory_delta_usd.is_finite() { inventory_delta_usd } else { 0.0 };
        let safe_vol = if volatility.is_finite() && volatility >= 0.0 { volatility } else { 0.0 };
        let safe_spread = if genome_spread_pct.is_finite() && genome_spread_pct >= 0.0 { genome_spread_pct } else { 0.001 };
        let safe_obi_th = if genome_obi_threshold.is_finite() { genome_obi_threshold } else { 0.5 };
        let safe_poly_a = if tensor_poly_a.is_finite() { tensor_poly_a } else { 1.0 };
        let safe_poly_b = if tensor_poly_b.is_finite() { tensor_poly_b } else { 1.0 };

        let _ofi = self.ofi_model.update(bid, ask, bid_qty, ask_qty);
        let total_vol = bid_qty + ask_qty;
        let obi = if total_vol > 0.0 {
            (bid_qty - ask_qty) / total_vol
        } else {
            0.0
        };

        let mid = (bid + ask) / 2.0;

        // Ampliamos el spread si la volatilidad es alta para protegernos de toxicidad
        // Usamos tensor_poly_b en vez del viejo hardcode "0.005"
        let dynamic_spread_pct = safe_spread + (safe_vol * safe_poly_b);
        let half_spread = mid * dynamic_spread_pct;

        // Skews
        // Si OBI > threshold (gran presión compradora), subimos los precios asimétricamente
        // Usamos tensor_poly_a * 0.01 para representar el sesgo (ej: 0.01 a 0.2% dictado por ML) en vez de "0.0002"
        let mut obi_skew = 0.0;
        let dynamic_obi_skew = mid * (safe_poly_a * 0.01);
        if obi > safe_obi_th {
            obi_skew = dynamic_obi_skew;
        } else if obi < -safe_obi_th {
            obi_skew = -dynamic_obi_skew;
        }

        // Si inventory > 0 (estamos Long), bajamos los precios para salir rápido y evitar acumular
        // Normalizamos el inventario en USD de forma adimensional para evitar distorsión cuadrática en BTC vs Altcoins
        let norm_inv = (safe_inv / 100.0).clamp(-5.0, 5.0);
        let inv_skew = norm_inv * mid * (safe_poly_b * 0.0005);

        let total_skew = obi_skew - inv_skew;

        let optimal_bid = mid - half_spread + total_skew;
        let optimal_ask = mid + half_spread + total_skew;

        // Regla estricta de Market Maker: NUNCA cruzar el spread real de mercado (eso pagaría Taker fee)
        let final_bid = if optimal_bid.is_finite() {
            optimal_bid.min(bid).max(1e-8)
        } else {
            bid.max(1e-8)
        };
        let final_ask = if optimal_ask.is_finite() {
            optimal_ask.max(ask).max(final_bid + 1e-8)
        } else {
            ask.max(final_bid + 1e-8)
        };

        MakerQuote {
            bid_price: final_bid,
            ask_price: final_ask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maker_engine_generate_quote_nominal() {
        let mut engine = MakerEngine::new(5.0);
        let quote = engine.generate_quote(
            100.0, 100.2, 10.0, 10.0, 0.0, 0.01, 0.001, 0.5, 1.0, 1.0
        );

        assert!(quote.bid_price <= 100.0, "Maker bid must not cross best bid");
        assert!(quote.ask_price >= 100.2, "Maker ask must not cross best ask");
        assert!(quote.bid_price < quote.ask_price);
    }

    #[test]
    fn test_maker_engine_never_crosses_market_spread() {
        let mut engine = MakerEngine::new(5.0);
        // Extreme inventory or OBI skew should still never cross market spread
        let quote = engine.generate_quote(
            50000.0, 50001.0, 100.0, 1.0, 1000.0, 0.05, 0.005, 0.2, 5.0, 5.0
        );

        assert!(quote.bid_price <= 50000.0);
        assert!(quote.ask_price >= 50001.0);
    }

    #[test]
    fn test_maker_engine_nan_and_negative_immunity() {
        let mut engine = MakerEngine::new(5.0);
        let quote = engine.generate_quote(
            f64::NAN, 100.0, 10.0, 10.0, 0.0, 0.01, 0.001, 0.5, 1.0, 1.0
        );
        assert!(quote.bid_price.is_finite() && quote.ask_price.is_finite());
    }
}

