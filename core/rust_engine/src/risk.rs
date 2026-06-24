use std::sync::RwLock;
use lazy_static::lazy_static;
use crate::portfolio::{Portfolio, GLOBAL_PORTFOLIO, Position};

#[derive(Debug)]
pub struct RiskManager {
    pub max_drawdown: f64,
    pub max_risk_per_trade: f64,
}

lazy_static! {
    pub static ref GLOBAL_RISK: RwLock<RiskManager> = RwLock::new(RiskManager {
        max_drawdown: 0.05, // 5% max drawdown
        max_risk_per_trade: 0.1, // 10% max risk per trade
    });
}

impl RiskManager {
    pub fn update_limits(max_drawdown: f64, max_risk_per_trade: f64) {
        if let Ok(mut risk) = GLOBAL_RISK.write() {
            risk.max_drawdown = max_drawdown;
            risk.max_risk_per_trade = max_risk_per_trade;
        }
    }

    pub fn can_open_position(horizon: i32, requested_qty: f64, current_price: f64) -> bool {
        let balance = Portfolio::get_balance();
        if balance <= 0.0 {
            return false;
        }

        let assumed_leverage = if horizon == 0 { 50.0 } else { 30.0 };
        let required_margin = (requested_qty * current_price) / assumed_leverage;
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
