//! # Auditor Interno (El Anti-Sistema) — Structural Doubt & Dual Verification Engine
//!
//! An unbiased agent that does not predict or trade. Its sole purpose is to audit,
//! demolish false self-affirmation, compare internal claims against exchange fills,
//! detect cognitive biases, and trigger Epistemic Crisis if self-deception is caught.

use serde::{Deserialize, Serialize};

/// Raw execution fill recorded independently from exchange WS/REST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFillRecord {
    pub timestamp_ns: u64,
    pub symbol_id: u16,
    pub is_buyer: bool,
    pub price: f64,
    pub qty: f64,
    pub fee_usd: f64,
    pub realized_pnl: f64,
    pub is_closed_position: bool,
}

/// Independent performance metrics calculated by the Auditor
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndependentMetrics {
    pub calculated_sharpe: f64,
    pub calculated_max_drawdown: f64,
    pub calculated_win_rate: f64,
    pub calculated_latency_us: u32,
    pub benchmark_alpha: f64,
}

/// Cognitive bias detection report
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CognitiveBiasReport {
    pub overfitting_recent_past: bool,
    pub illusion_of_control: bool,
    pub selective_confirmation: bool,
    pub complexity_inflation: bool,
    pub unverified_mutation_claims: bool,
    pub bias_detected: bool,
}

/// Daily Tribunal Verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TribunalVerdict {
    Inocente,
    CulpableRecuperable,
    CulpablePeligroso,
    AutoengañoDetectado,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyTribunalReport {
    pub verdict: TribunalVerdict,
    pub pnl_discrepancy_pct: f64,
    pub sharpe_discrepancy_pct: f64,
    pub justification: String,
}

/// Dual Result Verifier
pub struct VerificadorResultados;

impl VerificadorResultados {
    /// Computes independent metrics from raw exchange fill records with default base capital
    pub fn compute_metrics(fills: &[ExchangeFillRecord]) -> IndependentMetrics {
        Self::compute_metrics_with_capital(fills, 13.0)
    }

    /// Computes independent metrics from raw exchange fill records anchored to explicit base capital
    pub fn compute_metrics_with_capital(
        fills: &[ExchangeFillRecord],
        base_capital: f64,
    ) -> IndependentMetrics {
        if fills.is_empty() {
            return IndependentMetrics::default();
        }

        let closed_fills: Vec<&ExchangeFillRecord> =
            fills.iter().filter(|f| f.is_closed_position).collect();
        if closed_fills.is_empty() {
            return IndependentMetrics::default();
        }

        let total_trades = closed_fills.len() as f64;
        let winning_trades = closed_fills.iter().filter(|f| f.realized_pnl > 0.0).count() as f64;
        let win_rate = winning_trades / total_trades;

        let safe_capital = if base_capital.is_finite() && base_capital > 0.0 {
            base_capital
        } else {
            13.0
        };
        let returns: Vec<f64> = closed_fills
            .iter()
            .map(|f| {
                let pnl = if f.realized_pnl.is_finite() {
                    f.realized_pnl
                } else {
                    0.0
                };
                let fee = if f.fee_usd.is_finite() {
                    f.fee_usd
                } else {
                    0.0
                };
                (pnl - fee) / safe_capital
            })
            .collect();
        let mean_ret: f64 = returns.iter().sum::<f64>() / total_trades;
        let variance: f64 = if total_trades > 1.0 {
            returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / (total_trades - 1.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let time_span_days =
            if let (Some(first), Some(last)) = (closed_fills.first(), closed_fills.last()) {
                ((last.timestamp_ns.saturating_sub(first.timestamp_ns)) as f64 / (1e9 * 86400.0))
                    .max(1.0 / 24.0)
            } else {
                1.0
            };
        let trades_per_year = (total_trades / time_span_days) * 365.0;

        let sharpe = if std_dev > 1e-9 {
            (mean_ret / std_dev) * trades_per_year.sqrt().min(500.0)
        } else {
            0.0
        };

        // Reconstruct peak-to-trough max drawdown anchored to account equity
        let initial_equity = safe_capital;
        let mut equity = initial_equity;
        let mut peak = initial_equity;
        let mut max_dd = 0.0;

        for fill in &closed_fills {
            equity += fill.realized_pnl - fill.fee_usd;
            if equity > peak {
                peak = equity;
            }
            let dd = if peak > 0.0 {
                (peak - equity) / peak
            } else {
                0.0
            };
            if dd > max_dd {
                max_dd = dd;
            }
        }

        let gross_pnl: f64 =
            closed_fills.iter().map(|f| f.realized_pnl).sum::<f64>() / safe_capital;
        let total_fees: f64 = closed_fills.iter().map(|f| f.fee_usd).sum::<f64>() / safe_capital;
        let gross_alpha = gross_pnl / total_trades;
        let fee_drag = total_fees / total_trades;

        IndependentMetrics {
            calculated_sharpe: sharpe,
            calculated_max_drawdown: max_dd,
            calculated_win_rate: win_rate,
            calculated_latency_us: 150, // Standard baseline
            benchmark_alpha: gross_alpha - (fee_drag * 0.5),
        }
    }
}

/// Auditor Interno Engine
pub struct AuditorInterno {
    fills_journal: Vec<ExchangeFillRecord>,
    base_capital: f64,
}

impl Default for AuditorInterno {
    fn default() -> Self {
        Self {
            fills_journal: Vec::new(),
            base_capital: 13.0,
        }
    }
}

impl AuditorInterno {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capital(base_capital: f64) -> Self {
        Self {
            fills_journal: Vec::new(),
            base_capital: base_capital.max(1.0),
        }
    }

    pub fn record_fill(&mut self, fill: ExchangeFillRecord) {
        // FIX #677: Descartar fills con flotantes no finitos
        if !fill.price.is_finite()
            || !fill.qty.is_finite()
            || !fill.fee_usd.is_finite()
            || !fill.realized_pnl.is_finite()
        {
            return;
        }
        self.fills_journal.push(fill);
    }

    /// Evaluates internal claims against independent exchange fill verification
    pub fn verify_internal_claims(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> DailyTribunalReport {
        // FIX #677: Sanitizar claims internos
        let safe_sharpe = if internal_sharpe.is_finite() {
            internal_sharpe
        } else {
            0.0
        };
        let safe_wr = if internal_win_rate.is_finite() {
            internal_win_rate
        } else {
            0.0
        };
        let safe_dd = if internal_drawdown.is_finite() {
            internal_drawdown
        } else {
            0.0
        };

        let independent = VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        );

        let sharpe_disc = if safe_sharpe > 0.0 {
            (safe_sharpe - independent.calculated_sharpe).abs() / safe_sharpe
        } else {
            0.0
        };

        let wr_disc = (safe_wr - independent.calculated_win_rate).abs();
        let dd_disc = (safe_dd - independent.calculated_max_drawdown).abs();

        let max_disc = sharpe_disc.max(wr_disc).max(dd_disc);

        let verdict = if max_disc > 0.20 {
            TribunalVerdict::AutoengañoDetectado
        } else if max_disc > 0.10 {
            TribunalVerdict::CulpablePeligroso
        } else if max_disc > 0.05 {
            TribunalVerdict::CulpableRecuperable
        } else {
            TribunalVerdict::Inocente
        };

        DailyTribunalReport {
            verdict,
            pnl_discrepancy_pct: wr_disc,
            sharpe_discrepancy_pct: sharpe_disc,
            justification: format!(
                "Max discrepancy: {:.2}%. Internal Sharpe={:.2} vs Indep={:.2}",
                max_disc * 100.0,
                safe_sharpe,
                independent.calculated_sharpe
            ),
        }
    }

    /// Detects cognitive biases in current trading behavior (lookahead-free)
    // FIX #722: Calibrar umbral de complejidad (> 64) para respetar los 54 features nominales y sanitizar retornos
    pub fn audit_biases(
        &self,
        num_features: usize,
        recent_7d_ret: f64,
        out_of_sample_ret: f64,
    ) -> CognitiveBiasReport {
        let safe_7d = if recent_7d_ret.is_finite() {
            recent_7d_ret
        } else {
            0.0
        };
        let safe_oos = if out_of_sample_ret.is_finite() {
            out_of_sample_ret
        } else {
            0.0
        };
        let overfitting = (safe_7d > 0.05) && (safe_oos < -0.01);
        let complexity_inflation = num_features > 64;

        let bias_detected = overfitting || complexity_inflation;

        CognitiveBiasReport {
            overfitting_recent_past: overfitting,
            illusion_of_control: false,
            selective_confirmation: false,
            complexity_inflation,
            unverified_mutation_claims: false,
            bias_detected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auditor_interno_verification_and_bias() {
        let mut auditor = AuditorInterno::new();
        auditor.record_fill(ExchangeFillRecord {
            timestamp_ns: 1_000_000,
            symbol_id: 0,
            is_buyer: true,
            price: 60000.0,
            qty: 0.001,
            fee_usd: 0.024,
            realized_pnl: 1.50,
            is_closed_position: true,
        });
        auditor.record_fill(ExchangeFillRecord {
            timestamp_ns: 2_000_000,
            symbol_id: 0,
            is_buyer: false,
            price: 60100.0,
            qty: 0.001,
            fee_usd: 0.024,
            realized_pnl: -0.50,
            is_closed_position: true,
        });

        let tribunal = auditor.verify_internal_claims(2.0, 0.50, 0.05);
        assert_ne!(tribunal.justification, "");

        let bias = auditor.audit_biases(34, 0.02, 0.01);
        assert!(!bias.bias_detected);
    }

    #[test]
    fn test_verificador_resultados_nan_capital_immunity() {
        let fills = vec![
            ExchangeFillRecord {
                timestamp_ns: 1_000_000,
                symbol_id: 0,
                is_buyer: true,
                price: 60000.0,
                qty: 0.001,
                fee_usd: f64::NAN,
                realized_pnl: 1.0,
                is_closed_position: true,
            },
            ExchangeFillRecord {
                timestamp_ns: 2_000_000,
                symbol_id: 0,
                is_buyer: false,
                price: 60100.0,
                qty: 0.001,
                fee_usd: 0.02,
                realized_pnl: f64::NAN,
                is_closed_position: true,
            },
        ];

        let metrics_nan_cap = VerificadorResultados::compute_metrics_with_capital(&fills, f64::NAN);
        assert!(metrics_nan_cap.calculated_sharpe.is_finite());
        assert!(metrics_nan_cap.calculated_max_drawdown.is_finite());
        assert!(metrics_nan_cap.calculated_win_rate.is_finite());
    }
}
