use std::sync::RwLock;
use lazy_static::lazy_static;

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub horizon: i32, // 0 = Scalping, 1 = Swing
    pub side: i32,    // 1 = Long, -1 = Short
    pub entry_price: f64,
    pub qty: f64,
}

#[derive(Debug)]
pub struct Portfolio {
    pub usdt_balance: f64,
    pub scalping_position: Option<Position>,
    pub swing_position: Option<Position>,
}

lazy_static! {
    pub static ref GLOBAL_PORTFOLIO: RwLock<Portfolio> = RwLock::new(Portfolio {
        usdt_balance: 13.0, // Starting capital
        scalping_position: None,
        swing_position: None,
    });
}

impl Portfolio {
    pub fn update_balance(new_balance: f64) {
        if let Ok(mut port) = GLOBAL_PORTFOLIO.write() {
            port.usdt_balance = new_balance;
        }
    }

    pub fn get_balance() -> f64 {
        if let Ok(port) = GLOBAL_PORTFOLIO.read() {
            return port.usdt_balance;
        }
        0.0
    }

    pub fn set_position(horizon: i32, side: i32, entry_price: f64, qty: f64) {
        if let Ok(mut port) = GLOBAL_PORTFOLIO.write() {
            let pos = Position { horizon, side, entry_price, qty };
            if horizon == 0 {
                port.scalping_position = Some(pos);
            } else {
                port.swing_position = Some(pos);
            }
        }
    }

    pub fn clear_position(horizon: i32) {
        if let Ok(mut port) = GLOBAL_PORTFOLIO.write() {
            if horizon == 0 {
                port.scalping_position = None;
            } else {
                port.swing_position = None;
            }
        }
    }

    pub fn get_position(horizon: i32) -> Option<Position> {
        if let Ok(port) = GLOBAL_PORTFOLIO.read() {
            if horizon == 0 {
                return port.scalping_position;
            } else {
                return port.swing_position;
            }
        }
        None
    }

    pub fn get_heat(current_price: f64) -> f64 {
        if let Ok(port) = GLOBAL_PORTFOLIO.read() {
            if port.usdt_balance <= 0.0 { return 1.0; }
            let mut used_margin = 0.0;
            if let Some(pos) = port.scalping_position {
                used_margin += (pos.qty * current_price) / 50.0; // Phase 13: approx 50x
            }
            if let Some(pos) = port.swing_position {
                used_margin += (pos.qty * current_price) / 15.0; // Phase 13: approx 15x
            }
            return used_margin / port.usdt_balance;
        }
        0.0
    }
}
