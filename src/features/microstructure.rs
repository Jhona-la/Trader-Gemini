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
        if total == 0.0 {
            0.0
        } else {
            (self.cumulative_buy_vol - self.cumulative_sell_vol) / total
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
    let total_vol = bid_vol + ask_vol;
    if total_vol == 0.0 {
        0.0
    } else {
        (bid_vol - ask_vol) / total_vol
    }
}

/// Aceleración de Liquidez (Derivada del OBI)
#[inline(always)]
pub fn obi_acceleration(current_obi: f64, previous_obi: f64) -> f64 {
    current_obi - previous_obi
}
