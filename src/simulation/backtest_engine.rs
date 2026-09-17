#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::needless_range_loop)]

pub use backtest_engine::{
    UnifiedConfig,
    run_backtest_native,
    ffi_run_unified_backtest,
    ffi_run_unified_backtest_mmap,
    ffi_run_polars_backtest_mmap,
};

use crate::telemetry::tick_recorder::RecordedTick;

pub fn run_backtest_tick_level(
    ticks: &[RecordedTick],
    genome: &crate::core::config::Genome,
) -> f64 {
    use std::sync::Arc;
    use crate::core::god_engine_core::GodEngineCore;
    use crate::core::state::GlobalArena;
    use std::sync::atomic::Ordering;

    let config = crate::core::config::QuantumConfig::new_from_genome(genome);
    let arena = Arc::new(GlobalArena::new(config));
    
    let mut core = GodEngineCore::new(arena.clone());
    core.refresh_models();
    
    // Process ALL ticks directly to guarantee 100% parity with Production
    // BUG FIX: Use real OmniState initialization so ML models train on the exact same base scales as live_trader!
    let mut synthetic_omni = crate::data::omni_multiplexer::OmniState::new().get_features(); 
    
    // Fast RNG for synthetic drift in backtest
    let mut rng_state = 123456789u64;
    
    for t in ticks {
        // Pseudo-random drift for backtest variance
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        
        let noise = ((rng_state % 1000) as f64 / 1000.0) - 0.5;
        
        // Drift the CORRECT macro indices (21: DXY, 22: SP500, 23: NASDAQ, 24: VIX, 25: US10Y)
        synthetic_omni[21] += noise * 0.01;  // DXY
        synthetic_omni[22] += noise * 0.5;   // SP500
        synthetic_omni[23] += noise * 1.5;   // NASDAQ
        synthetic_omni[24] += noise * 0.01;  // VIX
        synthetic_omni[25] += noise * 0.001; // US10Y
        
        let _ = core.process_tick(
            t.coin_id,
            t.tick.bid_price,
            t.tick.ask_price,
            t.tick.bid_qty,
            t.tick.ask_qty,
            t.tick.timestamp_ms,
            &synthetic_omni,
        );
    }
    
    // Sum all PnL from all coins
    let mut total_pnl = 0.0;
    for i in 0..30 {
        total_pnl += arena.coins[i].scalp.pnl_realized.load(Ordering::Relaxed);
        total_pnl += arena.coins[i].swing.pnl_realized.load(Ordering::Relaxed);
    }
    
    total_pnl
}
