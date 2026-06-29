// trailing.rs
// QUANTUM TRAILING ENGINE - Zero-copy, sub-microsecond trailing evaluation

#[repr(C)]
pub struct TrailingResult {
    pub stop_price: f64,
    pub force_close: bool,
    pub new_phase: i32,
    pub max_pnl_pct: f64,
    pub mfe_atr: f64,
}

#[no_mangle]
pub extern "C" fn evaluate_quantum_trailing(
    pos_side: i32,       // 1 for LONG, -1 for SHORT
    entry_price: f64,
    current_price: f64,
    current_atr: f64,
    mut current_phase: i32,
    mut mfe_atr: f64,
    mut max_pnl_pct: f64,
    current_trail_stop: f64,
    // Profile configs
    pullback_tol: f64,
    trail_f1: f64,
    trail_f2: f64,
    trail_f3: f64,
    trail_runner: f64,
) -> TrailingResult {
    if current_atr <= 0.0 || entry_price <= 0.0 {
        return TrailingResult {
            stop_price: current_trail_stop,
            force_close: false,
            new_phase: current_phase,
            max_pnl_pct,
            mfe_atr,
        };
    }

    // 1. Calculate PnL (ATR and Pct)
    let pnl_pct = if pos_side == 1 {
        (current_price - entry_price) / entry_price
    } else {
        (entry_price - current_price) / entry_price
    };

    let pnl_atr = if pos_side == 1 {
        (current_price - entry_price) / current_atr
    } else {
        (entry_price - current_price) / current_atr
    };

    // 2. Update State
    if pnl_atr > mfe_atr {
        mfe_atr = pnl_atr;
    }
    if pnl_pct > max_pnl_pct {
        max_pnl_pct = pnl_pct;
    }

    // 3. Phase Transitions
    if current_phase == 0 && pnl_atr >= 0.5 {
        current_phase = 1;
    } else if current_phase == 1 && pnl_atr >= 1.5 {
        current_phase = 2;
    } else if current_phase == 2 && pnl_atr >= 3.0 {
        current_phase = 3;
    } else if current_phase == 3 && mfe_atr >= 4.0 {
        current_phase = 4;
    }

    // 4. Mechanism Proposals
    let mut best_stop = if current_trail_stop > 1e-9 { current_trail_stop } else { 0.0 };
    let mut proposals = [0.0; 3];
    let mut prop_count = 0;

    // T1: ATR Step Trailing
    if current_phase != 0 {
        let dist_atr = match current_phase {
            1 => trail_f1,
            2 => trail_f2,
            3 => trail_f3,
            4 => trail_runner,
            _ => 2.0,
        };
        
        let mut t1_stop = if pos_side == 1 {
            current_price - (dist_atr * current_atr)
        } else {
            current_price + (dist_atr * current_atr)
        };

        // Escudo Cuántico (Breakeven Lock)
        let fee_rate = 0.000375 * 2.0;
        if max_pnl_pct >= 0.01 {
            if pos_side == 1 {
                let breakeven_price = entry_price * (1.0 + fee_rate);
                if t1_stop < breakeven_price {
                    t1_stop = breakeven_price;
                }
                if max_pnl_pct >= 0.015 {
                    let profit_lock = entry_price * (1.0 + 0.005);
                    if t1_stop < profit_lock {
                        t1_stop = profit_lock;
                    }
                }
            } else {
                let breakeven_price = entry_price * (1.0 - fee_rate);
                if t1_stop == 0.0 || t1_stop > breakeven_price {
                    t1_stop = breakeven_price;
                }
                if max_pnl_pct >= 0.015 {
                    let profit_lock = entry_price * (1.0 - 0.005);
                    if t1_stop == 0.0 || t1_stop > profit_lock {
                        t1_stop = profit_lock;
                    }
                }
            }
        }
        proposals[prop_count] = t1_stop;
        prop_count += 1;
    }

    // T3: Parabolic Trailing
    if mfe_atr >= 3.0 {
        let mut factor = 0.02 + (mfe_atr - 3.0) * 0.05;
        if factor > 0.20 { factor = 0.20; }
        
        let mut dist_parabolic = trail_f3 - (mfe_atr * factor);
        if dist_parabolic < 0.5 { dist_parabolic = 0.5; }

        let t3_stop = if pos_side == 1 {
            current_price - (dist_parabolic * current_atr)
        } else {
            current_price + (dist_parabolic * current_atr)
        };
        proposals[prop_count] = t3_stop;
        prop_count += 1;
    }

    // T5: Volatility Contraction
    if current_phase != 0 {
        let dist_vol = 1.5 * current_atr;
        let t5_stop = if pos_side == 1 {
            current_price - dist_vol
        } else {
            current_price + dist_vol
        };
        proposals[prop_count] = t5_stop;
        prop_count += 1;
    }

    // Evaluate best stop
    for i in 0..prop_count {
        let p = proposals[i];
        if pos_side == 1 {
            if best_stop == 0.0 || p > best_stop {
                best_stop = p;
            }
        } else {
            if best_stop == 0.0 || p < best_stop {
                best_stop = p;
            }
        }
    }

    // Force Close Check
    let dd_atr = mfe_atr - pnl_atr;
    let mut current_tol = pullback_tol;
    if current_phase == 3 || current_phase == 4 {
        current_tol *= 0.8;
    }

    let mut force_close = false;
    if mfe_atr > 1.0 && dd_atr > current_tol {
        force_close = true;
    }

    TrailingResult {
        stop_price: best_stop,
        force_close,
        new_phase: current_phase,
        max_pnl_pct,
        mfe_atr,
    }
}
