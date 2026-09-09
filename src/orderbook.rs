use std::cmp::Ordering;
use std::collections::BTreeMap;

// A custom float wrapper to allow using f64 as keys in BTreeMap
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        // En un Orderbook HFT cuántico, garantizamos que no entran NaNs (Estasis Predictiva).
        // Aún así, hacemos fallback a Equal para no causar Pánicos en BTreeMap.
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

pub struct OrderBook {
    pub symbol: String,
    // BTreeMap keeps prices sorted. For bids, we want descending order to get the best (highest) bid easily.
    // For asks, we want ascending order to get the best (lowest) ask.
    // Rust's BTreeMap is ascending by default.
    bids: BTreeMap<OrderedFloat, f64>,
    asks: BTreeMap<OrderedFloat, f64>,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    pub fn update_bid(&mut self, price: f64, quantity: f64) {
        // FIX #1458: Validación estricta contra NaNs en BTreeMap de libro L2
        if !price.is_finite() || !quantity.is_finite() || price <= 0.0 || quantity < 0.0 {
            return;
        }
        if quantity == 0.0 {
            self.bids.remove(&OrderedFloat(price));
        } else {
            self.bids.insert(OrderedFloat(price), quantity);
        }
    }

    pub fn update_ask(&mut self, price: f64, quantity: f64) {
        // FIX #1458: Validación estricta contra NaNs en BTreeMap de libro L2
        if !price.is_finite() || !quantity.is_finite() || price <= 0.0 || quantity < 0.0 {
            return;
        }
        if quantity == 0.0 {
            self.asks.remove(&OrderedFloat(price));
        } else {
            self.asks.insert(OrderedFloat(price), quantity);
        }
    }

    pub fn best_bid(&self) -> Option<(f64, f64)> {
        // Bids: highest price is best. Since BTreeMap is ascending, it's the last element.
        self.bids.iter().next_back().map(|(p, q)| (p.0, *q))
    }

    pub fn best_ask(&self) -> Option<(f64, f64)> {
        // Asks: lowest price is best. It's the first element.
        self.asks.iter().next().map(|(p, q)| (p.0, *q))
    }

    pub fn micro_price(&self) -> Option<f64> {
        let (bb_price, bb_qty) = self.best_bid()?;
        let (ba_price, ba_qty) = self.best_ask()?;

        let total_qty = bb_qty + ba_qty;
        // FIX #735: Guarda estricta contra volumen nulo en micro-price
        if total_qty <= 1e-9 || !total_qty.is_finite() {
            return None;
        }

        // Micro-price calculation: weighted by opposite volume
        let mp = (bb_price * ba_qty + ba_price * bb_qty) / total_qty;
        if mp.is_finite() && mp > 0.0 {
            Some(mp)
        } else {
            None
        }
    }

    /// FIX #735: Cálculo de desbalance L2 (Order Book Imbalance) con guardas numéricas
    pub fn order_book_imbalance(&self) -> f64 {
        let (_bb_price, bb_qty) = match self.best_bid() {
            Some(v) => v,
            None => return 0.0,
        };
        let (_ba_price, ba_qty) = match self.best_ask() {
            Some(v) => v,
            None => return 0.0,
        };
        let total_vol = bb_qty + ba_qty;
        if total_vol <= 1e-9 || !total_vol.is_finite() {
            return 0.0;
        }
        let obi = (bb_qty - ba_qty) / total_vol;
        if obi.is_finite() {
            obi.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_bids_asks_micro_price_and_obi() {
        let mut ob = OrderBook::new("BTCUSDT".to_string());
        assert!(ob.best_bid().is_none());
        assert!(ob.best_ask().is_none());
        assert_eq!(ob.order_book_imbalance(), 0.0);

        ob.update_bid(60000.0, 2.0);
        ob.update_bid(59990.0, 1.0);
        ob.update_ask(60010.0, 1.0);
        ob.update_ask(60020.0, 3.0);

        let (bb, bb_q) = ob.best_bid().unwrap();
        assert_eq!(bb, 60000.0);
        assert_eq!(bb_q, 2.0);

        let (ba, ba_q) = ob.best_ask().unwrap();
        assert_eq!(ba, 60010.0);
        assert_eq!(ba_q, 1.0);

        let mp = ob.micro_price().unwrap();
        // micro_price = (60000 * 1 + 60010 * 2) / 3 = (60000 + 120020) / 3 = 180020 / 3 = 60006.666...
        assert!((mp - 60006.666666666664).abs() < 1e-6);

        let obi = ob.order_book_imbalance();
        // obi = (2 - 1) / (2 + 1) = 1 / 3 = 0.333...
        assert!((obi - 0.3333333333333333).abs() < 1e-6);

        // Removal on 0 quantity
        ob.update_bid(60000.0, 0.0);
        let (bb2, _) = ob.best_bid().unwrap();
        assert_eq!(bb2, 59990.0);
    }

    #[test]
    fn test_orderbook_nan_and_negative_immunity() {
        let mut ob = OrderBook::new("BTCUSDT".to_string());
        ob.update_bid(f64::NAN, 1.0);
        ob.update_bid(-50.0, 1.0);
        ob.update_ask(60000.0, f64::NAN);
        ob.update_ask(60000.0, -1.0);

        assert!(ob.best_bid().is_none());
        assert!(ob.best_ask().is_none());
    }
}
