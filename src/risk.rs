use std::sync::RwLock;
use std::collections::HashMap;
use lazy_static::lazy_static;
use crate::portfolio::{Portfolio, GLOBAL_PORTFOLIO, Position};
use crate::math_kernels::QuarterKelly;

#[derive(Debug)]
pub struct RiskManager {
    pub max_drawdown: f64,
    pub max_risk_per_trade: f64,
    pub kelly_trackers: HashMap<String, QuarterKelly>,
}

lazy_static! {
    pub static ref GLOBAL_RISK: RwLock<RiskManager> = RwLock::new(RiskManager::new(0.05, 1.0));
}

impl RiskManager {
    pub fn new(max_drawdown: f64, max_risk_per_trade: f64) -> Self {
        Self {
            max_drawdown,
            max_risk_per_trade,
            kelly_trackers: HashMap::new(),
        }
    }

    pub fn update_limits_local(&mut self, max_drawdown: f64, max_risk_per_trade: f64) {
        self.max_drawdown = max_drawdown;
        self.max_risk_per_trade = max_risk_per_trade;
    }

    pub fn update_limits(max_drawdown: f64, max_risk_per_trade: f64) {
        if let Ok(mut risk) = GLOBAL_RISK.write() {
            risk.update_limits_local(max_drawdown, max_risk_per_trade);
        }
    }

    pub fn report_trade_result_local(&mut self, symbol: &str, is_win: bool, pnl_pct: f64) {
        let kelly = self.kelly_trackers.entry(symbol.to_string()).or_insert_with(QuarterKelly::new);
        kelly.update(is_win, pnl_pct);
    }

    pub fn report_trade_result(symbol: &str, is_win: bool, pnl_pct: f64) {
        if let Ok(mut risk) = GLOBAL_RISK.write() {
            risk.report_trade_result_local(symbol, is_win, pnl_pct);
        }
    }

    pub fn get_symbol_constraints(symbol: &str) -> (f64, f64, f64) {
        // Returns (minQty, stepSize, minNotional)
        // Values approximate Binance Futures limits as of 2026
        match symbol.to_uppercase().as_str() {
            "BTCUSDT" => (0.001, 0.001, 5.0),
            "ETHUSDT" => (0.001, 0.001, 5.0),
            "SOLUSDT" => (1.0, 1.0, 5.0),
            "ADAUSDT" => (1.0, 1.0, 5.0),
            "DOGEUSDT" => (1.0, 1.0, 5.0),
            "XRPUSDT" => (1.0, 1.0, 5.0),
            "BNBUSDT" => (0.01, 0.01, 5.0),
            "AVAXUSDT" => (0.1, 0.1, 5.0),
            "DOTUSDT" => (0.1, 0.1, 5.0),
            "LINKUSDT" => (0.1, 0.1, 5.0),
            _ => (1.0, 1.0, 5.0), // Default fallback
        }
    }

    pub fn calculate_micro_position_size_local(&self, symbol: &str, current_price: f64, leverage: f64, capital: f64) -> Option<f64> {
        let (min_qty, step_size, min_notional) = Self::get_symbol_constraints(symbol);
        
        // Target risk based on Kelly Criterion
        let kelly_fraction = {
            if let Some(kelly) = self.kelly_trackers.get(symbol) {
                kelly.sizing_fraction()
            } else {
                0.10 // Default 10% if no history
            }
        };

        // Enforce max risk ceiling
        let safe_kelly = kelly_fraction.clamp(0.01, self.max_risk_per_trade);

        // Required capital per trade
        let max_loss_capital = capital * safe_kelly;

        // Assumed average SL distance (e.g., 1.5%) - Phase 13 tuning
        let avg_sl_pct = 0.015;
        let notional_target = max_loss_capital / avg_sl_pct;

        let effective_notional = notional_target.max(min_notional);
        let mut raw_qty = effective_notional / current_price;
        
        // Enforce minQty
        if raw_qty < min_qty {
            raw_qty = min_qty;
        }
        
        // Round down to stepSize
        let step_multiplier = 1.0 / step_size;
        let final_qty = (raw_qty * step_multiplier).floor() / step_multiplier;
        
        // Check if final required margin exceeds available capital
        let required_margin = (final_qty * current_price) / leverage;

        if required_margin > capital {
            return None; // Cannot afford the minimum trade
        }
        
        Some(final_qty)
    }

    pub fn calculate_micro_position_size(symbol: &str, current_price: f64, leverage: f64, capital: f64) -> Option<f64> {
        let risk = GLOBAL_RISK.read().unwrap();
        risk.calculate_micro_position_size_local(symbol, current_price, leverage, capital)
    }

    pub fn can_open_position(horizon: i32, requested_qty: f64, current_price: f64, dynamic_leverage: f64) -> bool {
        let balance = Portfolio::get_balance();
        if balance <= 0.0 {
            return false;
        }

        let required_margin = (requested_qty * current_price) / dynamic_leverage;
        let risk_guard = GLOBAL_RISK.read().unwrap();
        
        let max_capital_allowed = balance * risk_guard.max_risk_per_trade;
        
        // Check if there is already a position in that horizon
        let existing_pos = Portfolio::get_position(horizon);
        if existing_pos.is_some() {
            // Already have a position, don't average down or pyramid
            return false;
        }
        
        // Basic Margin check (Assumes isolated margin / strict risk)
        if required_margin > max_capital_allowed {
            return false;
        }
        
        true
    }

    pub fn check_drawdown(current_price: f64) -> bool {
        // Kill switch logic
        let balance = Portfolio::get_balance();
        let scalping_pos = Portfolio::get_position(0);
        let swing_pos = Portfolio::get_position(1);
        
        let mut total_unrealized_pnl = 0.0;
        
        if let Some(pos) = scalping_pos {
            let pnl = if pos.side == 1 {
                (current_price - pos.entry_price) * pos.qty
            } else {
                (pos.entry_price - current_price) * pos.qty
            };
            total_unrealized_pnl += pnl;
        }
        
        if let Some(pos) = swing_pos {
            let pnl = if pos.side == 1 {
                (current_price - pos.entry_price) * pos.qty
            } else {
                (pos.entry_price - current_price) * pos.qty
            };
            total_unrealized_pnl += pnl;
        }
        
        let risk_guard = GLOBAL_RISK.read().unwrap();
        let max_loss = balance * risk_guard.max_drawdown;
        
        if total_unrealized_pnl < -max_loss {
            // Kill Switch triggered
            return false; 
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_capital_btc_sizing() {
        // BTCUSDT at $60,000, Leverage 50x, Capital 13 USD.
        // target_notional = 1.3 USD (10% of 13) * 50 = 65 USD.
        // effective_notional = max(65, 5.0) = 65 USD.
        // raw_qty = 65 / 60000 = 0.001083.
        // step_size = 0.001. final_qty = 0.001.
        // required_margin = 0.001 * 60000 / 50 = 1.2 USD. 1.2 <= 13.
        
        let qty = RiskManager::calculate_micro_position_size("BTCUSDT", 60000.0, 50.0, 13.0);
        assert_eq!(qty, Some(0.001));
    }

    #[test]
    fn test_micro_capital_ada_sizing() {
        // ADAUSDT at $0.45, Leverage 50x, Capital 13 USD.
        // target_notional = 1.3 * 50 = 65 USD.
        // raw_qty = 65 / 0.45 = 144.44.
        // step_size = 1.0. final_qty = 144.0.
        // required_margin = 144 * 0.45 / 50 = 1.296 USD. 1.296 <= 13.
        
        let qty = RiskManager::calculate_micro_position_size("ADAUSDT", 0.45, 50.0, 13.0);
        assert_eq!(qty, Some(144.0));
    }
    
    #[test]
    fn test_insufficient_funds_rejected() {
        // BTCUSDT at $60,000, Leverage 1x, Capital 13 USD.
        // minimum notional is 60 USD.
        // required_margin at 1x is 60 USD > 13 USD -> should return None.
        let qty = RiskManager::calculate_micro_position_size("BTCUSDT", 60000.0, 1.0, 13.0);
        assert_eq!(qty, None);
    }
}
