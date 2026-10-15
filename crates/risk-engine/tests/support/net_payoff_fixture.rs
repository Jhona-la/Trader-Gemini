//! Synthetic close observations used only to isolate non-EV admission gates.
//! Returns are net PnL / entry notional, not a score or a TP-hit probability.
use signal_engine::{NetPayoffEstimate, NetPayoffSource, SignalType};

pub fn measured(coin_id: usize, side: SignalType, returns: &[f64]) -> NetPayoffEstimate {
    assert!(!returns.is_empty());
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let variance = returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    NetPayoffEstimate {
        mean_net_return_pct: mean,
        standard_deviation: variance.sqrt(),
        sample_count: n,
        coin_id,
        side,
        sample_start_ms: 1_000,
        sample_end_ms: n as u64 * 1_000,
        as_of_ms: n as u64 * 1_000,
        source: NetPayoffSource::LocalSimulation,
    }
}

pub fn positive(coin_id: usize, side: SignalType) -> NetPayoffEstimate {
    // 100 synthetic historical closes: 75 wins +4%, 25 losses -6% net.
    // WR=.75, aggregate PF=2 at equal notionals, mean=.015, population SD.
    // Historical exit geometries may differ from the next order's brackets.
    let returns: [f64; 100] =
        std::array::from_fn(|index| if index % 4 == 3 { -0.06 } else { 0.04 });
    measured(coin_id, side, &returns)
}
