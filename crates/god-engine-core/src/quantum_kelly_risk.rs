//! # QuantumKellyRiskEngine — Lock-Free Auto-Evolutionary Adaptive Risk Manager
//!
//! Computes mathematical expectancy E[X], dynamic optimal Kelly fraction f*,
//! and volatility-adaptive compounding position sizes for a dynamically fetched base capital.

use std::sync::atomic::Ordering;
use quantum_arena::atomic_float::AtomicF64;

pub struct QuantumKellyRiskEngine {
    pub rolling_wins: AtomicF64,
    pub rolling_losses: AtomicF64,
    pub sum_win_pct: AtomicF64,
    pub sum_loss_pct: AtomicF64,
    pub peak_capital: AtomicF64,
    pub current_win_streak: AtomicF64,
    pub current_loss_streak: AtomicF64,
}

impl std::fmt::Debug for QuantumKellyRiskEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumKellyRiskEngine")
            .field("rolling_wins", &self.rolling_wins.load(Ordering::Relaxed))
            .field("rolling_losses", &self.rolling_losses.load(Ordering::Relaxed))
            .field("sum_win_pct", &self.sum_win_pct.load(Ordering::Relaxed))
            .field("sum_loss_pct", &self.sum_loss_pct.load(Ordering::Relaxed))
            .field("peak_capital", &self.peak_capital.load(Ordering::Relaxed))
            .field("current_win_streak", &self.current_win_streak.load(Ordering::Relaxed))
            .field("current_loss_streak", &self.current_loss_streak.load(Ordering::Relaxed))
            .finish()
    }
}

impl QuantumKellyRiskEngine {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            rolling_wins: AtomicF64::new(0.0),
            rolling_losses: AtomicF64::new(0.0),
            sum_win_pct: AtomicF64::new(0.0),
            sum_loss_pct: AtomicF64::new(0.0),
            peak_capital: AtomicF64::new(initial_capital.max(1.0)),
            current_win_streak: AtomicF64::new(0.0),
            current_loss_streak: AtomicF64::new(0.0),
        }
    }

    /// Record closed trade for real-time expectancy and Kelly updating (Lock-Free & Thread-Safe)
    #[inline]
    pub fn update_trade_outcome(&self, pnl_pct: f64, current_capital: f64) {
        let peak = self.peak_capital.load(Ordering::Relaxed);
        if current_capital > peak {
            self.peak_capital.store(current_capital, Ordering::Relaxed);
        }

        if pnl_pct > 0.0 {
            self.rolling_wins.fetch_add(1.0, Ordering::Relaxed);
            self.sum_win_pct.fetch_add(pnl_pct, Ordering::Relaxed);
            self.current_win_streak.fetch_add(1.0, Ordering::Relaxed);
            self.current_loss_streak.store(0.0, Ordering::Relaxed);
        } else if pnl_pct < 0.0 {
            self.rolling_losses.fetch_add(1.0, Ordering::Relaxed);
            self.sum_loss_pct.fetch_add(pnl_pct.abs(), Ordering::Relaxed);
            self.current_loss_streak.fetch_add(1.0, Ordering::Relaxed);
            self.current_win_streak.store(0.0, Ordering::Relaxed);
        }

        // Exponential window decay (half-life ~100 trades)
        let wins = self.rolling_wins.load(Ordering::Relaxed);
        let losses = self.rolling_losses.load(Ordering::Relaxed);
        let total_trades = wins + losses;
        
        if total_trades > 100.0 {
            let decay = 0.98;
            self.rolling_wins.store(wins * decay, Ordering::Relaxed);
            self.rolling_losses.store(losses * decay, Ordering::Relaxed);
            let win_pnl = self.sum_win_pct.load(Ordering::Relaxed);
            let loss_pnl = self.sum_loss_pct.load(Ordering::Relaxed);
            self.sum_win_pct.store(win_pnl * decay, Ordering::Relaxed);
            self.sum_loss_pct.store(loss_pnl * decay, Ordering::Relaxed);
        }
    }

    /// Computes real-time Kelly optimal fraction f*, Win Rate, and Mathematical Expectancy in bps
    #[inline]
    pub fn compute_dynamic_kelly(&self, current_capital: f64, base_capital: f64, regime_hurst: f64, global_covariance: f64, neural_confidence: f64) -> (f64, f64, f64) {
        let wins = self.rolling_wins.load(Ordering::Relaxed);
        let losses = self.rolling_losses.load(Ordering::Relaxed);
        let total_trades = wins + losses;
        
        let (win_rate, avg_win, avg_loss) = if total_trades >= 3.0 {
            let wr = wins / total_trades;
            let win_sum = self.sum_win_pct.load(Ordering::Relaxed);
            let loss_sum = self.sum_loss_pct.load(Ordering::Relaxed);
            let win = if wins > 0.0 { win_sum / wins } else { 0.0055 };
            let loss = if losses > 0.0 { loss_sum / losses } else { 0.0022 };
            (wr, win, loss.max(0.0005))
        } else {
            (0.50, 0.003, 0.003) // Neutral Bayesian prior to prevent early over-leveraging
        };

        let win_loss_ratio = (avg_win / avg_loss).max(0.5);
        let expectancy_bps = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) * 10000.0;

        let raw_kelly = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio;
        
        // Drawdown de-risking multiplier
        let peak = self.peak_capital.load(Ordering::Relaxed).max(current_capital).max(1.0);
        let current_dd = if peak > 0.0 { (peak - current_capital) / peak } else { 0.0 };
        let dd_de_risk_factor = (1.0 - (current_dd / 0.15)).clamp(0.10, 1.0);

        // --- FASE XLII: Capital Evolutionary Derisking (SRE Growth) ---
        // Curva logarítmica continua: decae asintóticamente a medida que el capital crece respecto al base_capital
        let capital_derisk_factor = 1.5 / (1.0 + (current_capital / base_capital.max(1.0)).max(1.0).ln());

        // Regime-adaptive Kelly multiplier
        let regime_mult = if regime_hurst > 0.55 {
            1.25 // Trending regime: Scale up Kelly for compound growth
        } else if regime_hurst < 0.42 {
            0.85 // Mean-reverting: Conservative Kelly
        } else {
            1.00
        };
        
        // --- STREAK MULTIPLIER (PHASE 46) ---
        let win_streak = self.current_win_streak.load(Ordering::Relaxed);
        let loss_streak = self.current_loss_streak.load(Ordering::Relaxed);
        let streak_mult = if win_streak > 0.0 {
            1.0 + (win_streak * 0.20).min(1.5) // Acelera hasta 2.5x en rachas ganadoras prolongadas
        } else if loss_streak > 0.0 {
            (1.0 / (1.0 + loss_streak * 0.5)).max(0.1) // Se comprime agresivamente ante rachas perdedoras (Protección Capital)
        } else {
            1.0
        };

        // --- FASE 52: Multivariate Kelly Tensor Field ---
        // Attenuate kelly if global cross-correlation is high (global covariance).
        let topology_attenuation = (1.0 - (global_covariance.abs() * 0.8)).clamp(0.20, 1.00);

        // --- FASE 4: SIMD Neural Confidence Multiplier ---
        // Amplificar interés compuesto solo si la red neuronal está altamente segura (>80%)
        let neural_mult = if neural_confidence > 0.80 {
            1.0 + (neural_confidence * 1.5) // Acelera agresivamente
        } else if neural_confidence < 0.40 {
            0.1 // Protege el capital si hay duda sistémica
        } else {
            1.0
        };

        let optimal_kelly = (raw_kelly * dd_de_risk_factor * capital_derisk_factor * regime_mult * streak_mult * topology_attenuation * neural_mult).clamp(0.01, 0.50);

        (expectancy_bps, win_rate, optimal_kelly)
    }
}
