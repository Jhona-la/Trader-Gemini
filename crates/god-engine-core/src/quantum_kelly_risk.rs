//! # QuantumKellyRiskEngine — Lock-Free Auto-Evolutionary Adaptive Risk Manager
//!
//! Computes mathematical expectancy E[X], dynamic optimal Kelly fraction f*,
//! and volatility-adaptive compounding position sizes for a dynamically fetched base capital.
//!
//! #653 (Ola 53) — ISLA SIN CONSUMIDORES (auditor B): este módulo no tiene
//! ningún caller en el árbol vivo (sólo `pub mod` y tests); el sizing real
//! vive en risk-engine (quantum_kelly inline + leverage_matrix). Decisión
//! del consejo (#605, opción A): mantener anotado — NO conectarlo sin una
//! ola dedicada con oráculo (duplicaría el Kelly del risk-engine).

use quantum_arena::atomic_float::AtomicF64;
use std::sync::atomic::Ordering;

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
            .field(
                "rolling_losses",
                &self.rolling_losses.load(Ordering::Relaxed),
            )
            .field("sum_win_pct", &self.sum_win_pct.load(Ordering::Relaxed))
            .field("sum_loss_pct", &self.sum_loss_pct.load(Ordering::Relaxed))
            .field("peak_capital", &self.peak_capital.load(Ordering::Relaxed))
            .field(
                "current_win_streak",
                &self.current_win_streak.load(Ordering::Relaxed),
            )
            .field(
                "current_loss_streak",
                &self.current_loss_streak.load(Ordering::Relaxed),
            )
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
        if !pnl_pct.is_finite() {
            return;
        }
        if current_capital.is_finite() && current_capital > 0.0 {
            let peak = self.peak_capital.load(Ordering::Relaxed);
            if current_capital > peak {
                self.peak_capital.store(current_capital, Ordering::Relaxed);
            }
        }

        if pnl_pct > 0.0 {
            self.rolling_wins.fetch_add(1.0, Ordering::Relaxed);
            self.sum_win_pct.fetch_add(pnl_pct, Ordering::Relaxed);
            self.current_win_streak.fetch_add(1.0, Ordering::Relaxed);
            self.current_loss_streak.store(0.0, Ordering::Relaxed);
        } else if pnl_pct < 0.0 {
            self.rolling_losses.fetch_add(1.0, Ordering::Relaxed);
            self.sum_loss_pct
                .fetch_add(pnl_pct.abs(), Ordering::Relaxed);
            self.current_loss_streak.fetch_add(1.0, Ordering::Relaxed);
            self.current_win_streak.store(0.0, Ordering::Relaxed);
        }

        // Continuous exponential window decay for trades beyond warm-up (half-life ~140 trades)
        let wins = self.rolling_wins.load(Ordering::Relaxed);
        let losses = self.rolling_losses.load(Ordering::Relaxed);
        let total_trades = wins + losses;

        if total_trades > 100.0 {
            // FIX #573 & #1203: Decaimiento suave atómico mediante fetch_update (elimina races concurrentes)
            let decay = 0.995;
            let _ = self
                .rolling_wins
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v * decay));
            let _ = self
                .rolling_losses
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v * decay));
            let _ = self
                .sum_win_pct
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v * decay));
            let _ = self
                .sum_loss_pct
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v * decay));
        }
    }

    /// Computes real-time Kelly optimal fraction f*, Win Rate, and Mathematical Expectancy in bps
    #[inline]
    pub fn compute_dynamic_kelly(
        &self,
        current_capital: f64,
        base_capital: f64,
        regime_hurst: f64,
        global_covariance: f64,
        neural_confidence: f64,
    ) -> (f64, f64, f64) {
        // FIX #1432: Sanitización de argumentos entrantes para estabilidad matemática cuántica
        let safe_current_cap = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        let safe_base_cap = if base_capital.is_finite() && base_capital > 0.0 {
            base_capital
        } else {
            13.0
        };
        let safe_hurst = if regime_hurst.is_finite() {
            regime_hurst
        } else {
            0.5
        };
        let safe_cov = if global_covariance.is_finite() {
            global_covariance
        } else {
            0.0
        };
        let safe_neural = if neural_confidence.is_finite() {
            neural_confidence
        } else {
            0.5
        };

        let wins = self.rolling_wins.load(Ordering::Relaxed);
        let losses = self.rolling_losses.load(Ordering::Relaxed);
        let total_trades = wins + losses;

        let neural_directional_wr = if safe_neural > 0.50 {
            safe_neural
        } else {
            1.0 - safe_neural
        };

        let (win_rate, avg_win, avg_loss) = if total_trades >= 10.0 {
            let hist_wr = wins / total_trades;
            let win_sum = self.sum_win_pct.load(Ordering::Relaxed);
            let loss_sum = self.sum_loss_pct.load(Ordering::Relaxed);
            let win = if wins > 0.0 { win_sum / wins } else { 0.0055 };
            let loss = if losses > 0.0 {
                loss_sum / losses
            } else {
                0.0035
            };
            // Fusión Bayesiana Cuántica: 35% histórico + 65% convicción neuronal instantánea
            let fused_wr = hist_wr * 0.35 + neural_directional_wr * 0.65;
            (fused_wr, win.max(0.0045), loss.max(0.0025))
        } else {
            (neural_directional_wr.max(0.55), 0.0055, 0.0035) // Prior Bayesiano impulsado por la red neuronal
        };

        let win_loss_ratio = (avg_win / avg_loss).max(0.5);
        let expectancy_bps = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) * 10000.0;

        let raw_kelly = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio;

        // Drawdown de-risking multiplier
        let peak = self
            .peak_capital
            .load(Ordering::Relaxed)
            .max(safe_current_cap)
            .max(1.0);
        let current_dd = if peak > 0.0 {
            (peak - safe_current_cap) / peak
        } else {
            0.0
        };
        let dd_de_risk_factor = (1.0 - (current_dd / 0.85)).clamp(0.15, 1.0);

        // --- FASE XLII: Capital Evolutionary Derisking (SRE Growth) ---
        // FIX #575: Ponderar la aceleración inicial por la significancia estadística de la muestra
        let sample_confidence = (total_trades / 20.0).clamp(0.20, 1.0);
        let capital_ratio = (safe_current_cap / safe_base_cap.max(1.0)).max(1.0);
        let capital_derisk_factor = (1.0 + 0.5 * sample_confidence) / (1.0 + capital_ratio.ln());


        // --- FASE 52: Multivariate Kelly Tensor Field ---
        // Attenuate kelly if global cross-correlation is high (global covariance).
        let topology_attenuation = (1.0 - (safe_cov.abs() * 0.8)).clamp(0.20, 1.00);

        // AGY-AUD-005: the accelerators (streak/neural/regime that exceed 1.0)
        // were dead code — .min(raw_kelly) guaranteed they never increased the
        // output beyond raw Kelly. Only de-risking factors actually bind.
        // Compute the product of de-risking factors only; each ∈ (0, 1].
        let derisking_product = dd_de_risk_factor
            * capital_derisk_factor
            * topology_attenuation
            // Hurst-adaptive: persistent regime preserves full Kelly,
            // mean-reverting regime compresses (the only regime effect that
            // isn't dead code under the raw_kelly cap).
            * if safe_hurst < 0.42 { 0.85 } else { 1.0 }
            // Loss streak compression (the de-risk branch of streak logic).
            * if self.current_loss_streak.load(Ordering::Relaxed) > 0.0 {
                let ls = self.current_loss_streak.load(Ordering::Relaxed);
                (1.0 / (1.0 + ls * 0.5)).max(0.1)
            } else {
                1.0
            }
            // Low neural confidence: protect capital.
            * if safe_neural < 0.40 { 0.1 } else { 1.0 };

        let optimal_kelly = (raw_kelly * derisking_product).min(raw_kelly);
        let safe_expectancy = if expectancy_bps.is_finite() {
            expectancy_bps
        } else {
            0.0
        };
        let safe_wr = if win_rate.is_finite() {
            win_rate.clamp(0.0, 1.0)
        } else {
            0.50
        };
        let safe_kelly =
            if safe_expectancy <= 0.0 || !optimal_kelly.is_finite() || optimal_kelly <= 0.0 {
                0.0 // Abstinencia matemática estricta cuando no hay ventaja estadística
            } else {
                optimal_kelly.clamp(0.0, 0.50)
            };

        (safe_expectancy, safe_wr, safe_kelly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_kelly_risk_nominal_updates_and_expectancy() {
        let engine = QuantumKellyRiskEngine::new(13.0);
        // Initially 0 trades: returns neural prior
        let (exp, wr, _kelly) = engine.compute_dynamic_kelly(13.0, 13.0, 0.55, 0.0, 0.70);
        assert_eq!(wr, 0.70);
        assert!(exp >= 0.0);

        // Record 5 winning trades and 1 losing trade
        for _ in 0..5 {
            engine.update_trade_outcome(0.015, 14.0);
        }
        engine.update_trade_outcome(-0.005, 13.9);

        let (exp2, wr2, kelly2) = engine.compute_dynamic_kelly(14.0, 13.0, 0.60, 0.1, 0.85);
        assert!(wr2 > 0.70);
        assert!(exp2 > 0.0);
        assert!(kelly2 > 0.0 && kelly2 <= 0.50);
    }

    #[test]
    fn test_quantum_kelly_risk_nan_and_negative_inputs_immunity() {
        let engine = QuantumKellyRiskEngine::new(13.0);
        let (exp, wr, kelly) =
            engine.compute_dynamic_kelly(f64::NAN, -10.0, f64::NAN, f64::NAN, f64::NAN);
        assert!(exp.is_finite());
        assert!(wr.is_finite());
        assert!(kelly.is_finite());
        assert!(kelly >= 0.0 && kelly <= 0.50);
    }

    #[test]
    fn test_quantum_kelly_risk_win_loss_streaks_and_decay() {
        let engine = QuantumKellyRiskEngine::new(13.0);
        for _ in 0..105 {
            engine.update_trade_outcome(0.01, 15.0);
        }
        assert_eq!(engine.current_loss_streak.load(Ordering::Relaxed), 0.0);
        assert!(engine.current_win_streak.load(Ordering::Relaxed) >= 100.0);

        // Loss resets win streak
        engine.update_trade_outcome(-0.01, 14.8);
        assert_eq!(engine.current_win_streak.load(Ordering::Relaxed), 0.0);
        assert_eq!(engine.current_loss_streak.load(Ordering::Relaxed), 1.0);
    }

    #[test]
    fn test_quantum_kelly_risk_update_trade_outcome_nan_immunity() {
        let engine = QuantumKellyRiskEngine::new(13.0);
        engine.update_trade_outcome(f64::NAN, f64::NAN);
        assert_eq!(engine.rolling_wins.load(Ordering::Relaxed), 0.0);
        assert_eq!(engine.rolling_losses.load(Ordering::Relaxed), 0.0);
    }
}
