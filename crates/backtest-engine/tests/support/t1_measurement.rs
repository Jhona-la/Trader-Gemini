//! Shared T-1 measurement helpers. The legacy fixture/comparator are preserved
//! deliberately: changing them would invalidate historical ratchet comparisons.
use backtest_engine::STATS_LEN;

/// Serie sintética determinista histórica: tendencia lenta + ciclo + ruido.
/// GO: reproducible NO implica ruido centrado ni muestra representativa.
pub fn serie(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut seed = 0x5DEECE66Du64;
    let mut p = 60_000.0f64;
    for i in 0..n {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
        // Historia (Ola XLI·A1): el fixture anterior (ATR ~11 pb/min)
        // dejaba sigma(tau)-fricción(7 pb) por debajo del spread(4 pb),
        // provocando INVIABLE perpetuo. Se adoptaron ciclo 30 pb y ruido
        // nominal 40 pb para permitir actividad tras el calentamiento.
        // GO: se preservan coeficientes, seed y operaciones. El ruido de
        // esta fórmula es negativo, no centrado; no cambiarlo para pasar T-1.
        let ciclo = (i as f64 / 180.0).sin() * 0.0030;
        p *= 1.0 + ciclo + u * 0.0040 + 0.00004;
        closes.push(p);
        highs.push(p * (1.0 + 0.0018 + u.abs() * 0.0012));
        lows.push(p * (1.0 - 0.0018 - u.abs() * 0.0012));
        vols.push(900.0 + u.abs() * 2_000.0);
    }
    (closes, highs, lows, vols)
}

pub fn difiere(a: &[f64; STATS_LEN], b: &[f64; STATS_LEN]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| {
        if !x.is_finite() || !y.is_finite() {
            return x.is_finite() != y.is_finite();
        }
        (x - y).abs() > 1e-9 * x.abs().max(1.0)
    })
}

pub fn furthest_endpoint(base: f64, lo: f64, hi: f64) -> f64 {
    let dist_lo = (base - lo).abs();
    let dist_hi = (hi - base).abs();
    if dist_hi >= dist_lo { hi } else { lo }
}

pub fn changed_slots(before: &[f64], after: &[f64]) -> Vec<usize> {
    assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter_map(|(i, (a, b))| ((a - b).abs() > 1e-9 * a.abs().max(1.0)).then_some(i))
        .collect()
}

pub fn diagnostic_stats_line(stats: &[f64; STATS_LEN]) -> String {
    // run_backtest_native: [net_wr, trades, final_capital, dd, sharpe,
    // gross_pnl, net_pnl, gross_wr]. This is diagnosis, not a new fitness.
    format!(
        "[T1-DIAG] stats: trades={} pnl={} wr={} capital={}",
        stats[1], stats[6], stats[0], stats[2]
    )
}
