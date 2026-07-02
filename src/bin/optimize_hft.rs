use std::fs::File;
use std::io::Read;
use std::time::Instant;
use rayon::prelude::*;
use quantum_engine::stateful_engine::StatefulEngine;

const BPS_4: f64 = 0.0004;
const BPS_2_SL: f64 = -0.0002;
const HORIZON_TICKS: usize = 2000;

#[derive(Clone, Copy, Debug)]
struct HftParams {
    cvpin_threshold: f64,
    vol_delta_threshold: f64,
    hurst_max: f64,
}

#[derive(Default, Debug)]
struct BacktestResult {
    trades: usize,
    wins: usize,
    losses: usize,
}

struct TickRecord {
    price: f64,
    cvpin: f64,
    vol_delta: f64,
    hurst: f64,
}

fn evaluate_params(
    params: &HftParams,
    records: &[TickRecord],
) -> BacktestResult {
    let mut res = BacktestResult::default();
    let num_ticks = records.len();

    for i in 0..num_ticks.saturating_sub(HORIZON_TICKS) {
        let rec = &records[i];
        
        // Quantum Threshold Logic:
        // CVPIN > T (High informed trading proxy)
        // VolDelta > V (High directional momentum)
        // Hurst < H (Mean-reverting / high volatility)
        if rec.cvpin > params.cvpin_threshold && rec.vol_delta.abs() > params.vol_delta_threshold && rec.hurst < params.hurst_max {
            let entry_price = rec.price;
            let is_long = rec.vol_delta < 0.0; // FADE the extreme imbalance (Mean Reversion)
            
            let target_price = if is_long { entry_price * (1.0 + BPS_4) } else { entry_price * (1.0 - BPS_4) };
            let sl_price = if is_long { entry_price * (1.0 + BPS_2_SL) } else { entry_price * (1.0 - BPS_2_SL) };
            
            let mut is_win = false;
            let mut is_loss = false;
            
            for j in 1..=HORIZON_TICKS {
                let future_price = records[i + j].price;
                
                if is_long {
                    if future_price <= sl_price {
                        is_loss = true;
                        break;
                    }
                    if future_price >= target_price {
                        is_win = true;
                        break;
                    }
                } else {
                    if future_price >= sl_price {
                        is_loss = true;
                        break;
                    }
                    if future_price <= target_price {
                        is_win = true;
                        break;
                    }
                }
            }
            
            if is_win {
                res.wins += 1;
                res.trades += 1;
            } else if is_loss {
                res.losses += 1;
                res.trades += 1;
            } else {
                res.losses += 1;
                res.trades += 1;
            }
        }
    }
    
    res
}

fn main() {
    println!("============================================================");
    println!("🔬 HFT DETERMINISTIC OPTIMIZER (CVPIN/VolDelta/Hurst)");
    println!("============================================================");

    let start_load = Instant::now();
    let mut file = match File::open("data/BTCUSDT_ticks.bin") {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Failed to open tick data");
            return;
        }
    };
    
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let num_ticks = buf.len() / (4 * 8); // 4 f64s per tick
    
    if num_ticks == 0 {
        println!("❌ No data found.");
        return;
    }
    
    let ptr = buf.as_ptr() as *const f64;
    let prices = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 1), num_ticks) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 2), num_ticks) };
    let is_buyer_maker = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 3), num_ticks) };

    println!("🌳 Data loaded: {} total ticks in {:?}", num_ticks, start_load.elapsed());
    
    let mut engine = StatefulEngine::new();
    let mut records = Vec::with_capacity(num_ticks);

    let start_proc = Instant::now();
    for i in 0..num_ticks {
        let price = prices[i];
        let qty = volumes[i];
        let is_buyer = is_buyer_maker[i] > 0.5;

        engine.process_tick(price, qty);
        engine.update_trade_flow(qty, is_buyer);
        
        let features = engine.get_features();
        
        records.push(TickRecord {
            price,
            cvpin: features[7] as f64,        // offset 7 is cvpin
            vol_delta: features[6] as f64,  // offset 6 is VolDelta
            hurst: features[1] as f64,      // offset 1 is Hurst
        });
    }
    println!("⚙️ Feature Engineering finished in {:?}", start_proc.elapsed());

    // Define Grid
    // Define Grid
    let mut param_grid = Vec::new();
    let cvpin_steps = [50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0];
    let vol_delta_steps = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95];
    let hurst_steps = [0.3, 0.35, 0.4, 0.45, 0.5];

    for &c in &cvpin_steps {
        for &v in &vol_delta_steps {
            for &h in &hurst_steps {
                param_grid.push(HftParams {
                    cvpin_threshold: c,
                    vol_delta_threshold: v,
                    hurst_max: h
                });
            }
        }
    }

    println!("⚙️ Testing {} parameter combinations...", param_grid.len());
    let start_opt = Instant::now();

    let mut results: Vec<(HftParams, BacktestResult)> = param_grid.into_par_iter()
        .map(|params| {
            let res = evaluate_params(&params, &records);
            (params, res)
        })
        .collect();

    println!("✅ Optimization finished in {:?}", start_opt.elapsed());

    // Filter and Sort by Win Rate
    results.retain(|(_, r)| r.trades >= 10); // Lowered minimum sample significance
    results.sort_by(|(_, a), (_, b)| {
        let wr_a = a.wins as f64 / a.trades as f64;
        let wr_b = b.wins as f64 / b.trades as f64;
        wr_b.partial_cmp(&wr_a).unwrap()
    });

    println!("\n🏆 TOP 15 REGIMES FOR SCALPING (Target: 4bps, SL: 2bps)");
    for (i, (params, res)) in results.iter().take(15).enumerate() {
        let wr = (res.wins as f64 / res.trades as f64) * 100.0;
        println!("#{}: WR: {:.2}% | Trades: {} | CVPIN > {}, VolDelta > {}, Hurst < {}",
            i + 1, wr, res.trades, params.cvpin_threshold, params.vol_delta_threshold, params.hurst_max
        );
    }
}
