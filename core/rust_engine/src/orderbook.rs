use std::collections::BTreeMap;
use std::cmp::Ordering;

// A custom float wrapper to allow using f64 as keys in BTreeMap
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
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

    pub fn update_bid(&mut self, price: f64, quantity: f64) {
        if quantity == 0.0 {
            self.bids.remove(&OrderedFloat(price));
        } else {
            self.bids.insert(OrderedFloat(price), quantity);
        }
    }

    pub fn update_ask(&mut self, price: f64, quantity: f64) {
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
        if total_qty == 0.0 {
            return None;
        }

        // Micro-price calculation: weighted by opposite volume
        let mp = (bb_price * ba_qty + ba_price * bb_qty) / total_qty;
        Some(mp)
    }
}
