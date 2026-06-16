use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use std::collections::BTreeMap;

#[pyclass]
pub struct OrderBookSoA {
    // Structure of Arrays (SoA) para maximizar Cache L1
    pub bid_prices: Vec<f64>,
    pub bid_vols: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub ask_vols: Vec<f64>,
}

#[pymethods]
impl OrderBookSoA {
    #[new]
    pub fn new() -> Self {
        OrderBookSoA {
            bid_prices: Vec::with_capacity(1000),
            bid_vols: Vec::with_capacity(1000),
            ask_prices: Vec::with_capacity(1000),
            ask_vols: Vec::with_capacity(1000),
        }
    }

    pub fn update_level(&mut self, is_bid: bool, price: f64, vol: f64) {
        let (prices, vols) = if is_bid {
            (&mut self.bid_prices, &mut self.bid_vols)
        } else {
            (&mut self.ask_prices, &mut self.ask_vols)
        };

        // Binary search to find price or insertion point
        // For bids, we want descending order. For asks, ascending order.
        let pos = if is_bid {
            prices.binary_search_by(|p| p.partial_cmp(&price).unwrap().reverse())
        } else {
            prices.binary_search_by(|p| p.partial_cmp(&price).unwrap())
        };

        match pos {
            Ok(idx) => {
                if vol == 0.0 {
                    prices.remove(idx);
                    vols.remove(idx);
                } else {
                    vols[idx] = vol;
                }
            }
            Err(idx) => {
                if vol > 0.0 {
                    prices.insert(idx, price);
                    vols.insert(idx, vol);
                }
            }
        }
    }

    pub fn get_bbo(&self) -> (f64, f64) {
        let best_bid = self.bid_prices.first().copied().unwrap_or(0.0);
        let best_ask = self.ask_prices.first().copied().unwrap_or(0.0);
        (best_bid, best_ask)
    }

    pub fn get_imbalance(&self, levels: usize) -> f64 {
        let bid_vol: f64 = self.bid_vols.iter().take(levels).sum();
        let ask_vol: f64 = self.ask_vols.iter().take(levels).sum();
        if bid_vol + ask_vol == 0.0 {
            return 0.0;
        }
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }
}

#[pymodule]
fn nano_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OrderBookSoA>()?;
    Ok(())
}

