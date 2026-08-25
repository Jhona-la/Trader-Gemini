/// Microestructura del Mercado (Order Book)
/// Axioma II: O(1) puro y #[inline(always)]

#[derive(Clone, Copy)]
pub struct OrderFlowTracker {
    pub cumulative_buy_vol: f64,
    pub cumulative_sell_vol: f64,
    pub short_ema_obi: f64,
    pub long_ema_obi: f64,
    pub velocity_obi: f64,
}

impl Default for OrderFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderFlowTracker {
    pub fn new() -> Self {
        Self {
            cumulative_buy_vol: 0.0,
            cumulative_sell_vol: 0.0,
            short_ema_obi: 0.0,
            long_ema_obi: 0.0,
            velocity_obi: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, volume: f64, is_buyer_maker: bool) -> f64 {
        // FIX #1457: Sanitización de volumen
        if !volume.is_finite() || volume < 0.0 {
            return 0.0;
        }

        // En Binance: buyer_maker = true significa que el trade fue ejecutado contra el BID (Taker Sell)
        // buyer_maker = false significa que el trade fue ejecutado contra el ASK (Taker Buy)
        let (buy_v, sell_v) = if is_buyer_maker {
            (0.0, volume)
        } else {
            (volume, 0.0)
        };
        
        self.cumulative_buy_vol += buy_v;
        self.cumulative_sell_vol += sell_v;
        
        // Instant Imbalance for this tick
        let tick_imbalance = if volume == 0.0 { 0.0 } else { (buy_v - sell_v) / volume };
        
        // EWMA update
        let short_alpha = 0.1; // Fast
        let long_alpha = 0.01; // Slow
        
        let prev_short = self.short_ema_obi;
        self.short_ema_obi = (tick_imbalance - self.short_ema_obi) * short_alpha + self.short_ema_obi;
        self.long_ema_obi = (tick_imbalance - self.long_ema_obi) * long_alpha + self.long_ema_obi;
        
        self.velocity_obi = self.short_ema_obi - prev_short;
        
        tick_imbalance
    }

    #[inline(always)]
    pub fn get_volume_delta_ratio(&self) -> f64 {
        let total = self.cumulative_buy_vol + self.cumulative_sell_vol;
        if total <= 0.0 {
            0.0
        } else {
            ((self.cumulative_buy_vol - self.cumulative_sell_vol) / total).clamp(-1.0, 1.0)
        }
    }
}

/// Cont et al. (2014) Level-1 Order Flow Imbalance (OFI)
/// Mide la presión direccional neta calculando el cambio neto en volumen
/// condicionado al movimiento del precio en el nivel BBO (Best Bid/Offer).
#[derive(Clone, Copy)]
pub struct OFIModel {
    pub prev_bid_price: f64,
    pub prev_bid_qty: f64,
    pub prev_ask_price: f64,
    pub prev_ask_qty: f64,
    pub ema_ofi: f64,
}

impl Default for OFIModel {
    fn default() -> Self {
        Self::new()
    }
}

impl OFIModel {
    pub fn new() -> Self {
        Self {
            prev_bid_price: 0.0,
            prev_bid_qty: 0.0,
            prev_ask_price: 0.0,
            prev_ask_qty: 0.0,
            ema_ofi: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) -> f64 {
        // FIX #1457: Sanitización estricta de inputs del libro BBO
        if !bid_price.is_finite() || !ask_price.is_finite() || !bid_qty.is_finite() || !ask_qty.is_finite() || bid_price <= 0.0 || ask_price <= 0.0 {
            return self.ema_ofi;
        }

        if self.prev_bid_price == 0.0 {
            self.prev_bid_price = bid_price;
            self.prev_bid_qty = bid_qty;
            self.prev_ask_price = ask_price;
            self.prev_ask_qty = ask_qty;
            return 0.0;
        }

        // Flujo neto en BID (e_bid)
        let e_bid = if bid_price > self.prev_bid_price {
            bid_qty
        } else if bid_price == self.prev_bid_price {
            bid_qty - self.prev_bid_qty
        } else {
            -self.prev_bid_qty
        };

        // Flujo neto en ASK (e_ask)
        let e_ask = if ask_price < self.prev_ask_price {
            ask_qty
        } else if ask_price == self.prev_ask_price {
            ask_qty - self.prev_ask_qty
        } else {
            -self.prev_ask_qty
        };

        // OFI Tick (Direccionalidad neta)
        let ofi = e_bid - e_ask;

        // Actualizar estados pasados
        self.prev_bid_price = bid_price;
        self.prev_bid_qty = bid_qty;
        self.prev_ask_price = ask_price;
        self.prev_ask_qty = ask_qty;

        // Suavizado EWMA para evitar ruido (alpha = 0.1)
        self.ema_ofi = (ofi - self.ema_ofi) * 0.1 + self.ema_ofi;
        
        self.ema_ofi
    }
}

/// Calcula el Order Book Imbalance (OBI).
#[inline(always)]
pub fn order_book_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    // FIX #1457: Sanitización de volúmenes de imbalance
    if !bid_vol.is_finite() || !ask_vol.is_finite() || bid_vol < 0.0 || ask_vol < 0.0 {
        return 0.0;
    }
    let total_vol = bid_vol + ask_vol;
    if total_vol <= 0.0 {
        0.0
    } else {
        ((bid_vol - ask_vol) / total_vol).clamp(-1.0, 1.0)
    }
}

#[inline(always)]
pub fn obi_acceleration(current_obi: f64, previous_obi: f64) -> f64 {
    // FIX #1457: Sanitización de aceleración OBI
    let c = if current_obi.is_finite() { current_obi } else { 0.0 };
    let p = if previous_obi.is_finite() { previous_obi } else { 0.0 };
    c - p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_flow_tracker_and_delta_ratio() {
        let mut tracker = OrderFlowTracker::new();
        // Buyer maker = false -> Taker Buy
        tracker.update(10.0, false);
        // Buyer maker = true -> Taker Sell
        tracker.update(5.0, true);

        assert_eq!(tracker.cumulative_buy_vol, 10.0);
        assert_eq!(tracker.cumulative_sell_vol, 5.0);
        // (10 - 5) / 15 = 5/15 = 1/3 ~ 0.3333
        let ratio = tracker.get_volume_delta_ratio();
        assert!((ratio - (1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ofi_model_and_obi_acceleration() {
        let mut ofi = OFIModel::new();
        let _ = ofi.update(60000.0, 60001.0, 10.0, 10.0);
        // Bid price goes up, volume = 15.0 -> e_bid = 15.0
        let res = ofi.update(60000.5, 60001.0, 15.0, 10.0);
        assert!(res > 0.0, "OFI should be positive when bid price rises");

        let obi = order_book_imbalance(15.0, 5.0);
        assert_eq!(obi, 0.5);

        let accel = obi_acceleration(0.5, 0.2);
        assert!((accel - 0.3).abs() < 1e-10);
    }
}

