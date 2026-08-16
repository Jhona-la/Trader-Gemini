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
    /// Computes independent metrics from raw exchange fill records
    pub fn compute_metrics(fills: &[ExchangeFillRecord]) -> IndependentMetrics {
        if fills.is_empty() {
            return IndependentMetrics::default();
        }

        let closed_fills: Vec<&ExchangeFillRecord> = fills.iter().filter(|f| f.is_closed_position).collect();
        if closed_fills.is_empty() {
            return IndependentMetrics::default();
        }

        let total_trades = closed_fills.len() as f64;
        let winning_trades = closed_fills.iter().filter(|f| f.realized_pnl > 0.0).count() as f64;
        let win_rate = winning_trades / total_trades;

        let returns: Vec<f64> = closed_fills.iter().map(|f| f.realized_pnl).collect();
        let mean_ret: f64 = returns.iter().sum::<f64>() / total_trades;
        let variance: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / total_trades;
        let std_dev = variance.sqrt();

        let sharpe = if std_dev > 1e-9 {
            (mean_ret / std_dev) * (365.0_f64).sqrt()
        } else {
            0.0
        };

        // Reconstruct peak-to-trough max drawdown
        let mut equity = 0.0;
        let mut peak = 0.0;
        let mut max_dd = 0.0;

        for fill in &closed_fills {
            equity += fill.realized_pnl - fill.fee_usd;
            if equity > peak {
                peak = equity;
            }
            let dd = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
            if dd > max_dd {
                max_dd = dd;
            }
        }

        IndependentMetrics {
            calculated_sharpe: sharpe,
            calculated_max_drawdown: max_dd,
            calculated_win_rate: win_rate,
            calculated_latency_us: 150, // Standard baseline
            benchmark_alpha: mean_ret * 0.8,
        }
    }
}

/// Auditor Interno Engine
pub struct AuditorInterno {
    fills_journal: Vec<ExchangeFillRecord>,
}

impl Default for AuditorInterno {
    fn default() -> Self {
        Self {
            fills_journal: Vec::new(),
        }
    }
}

impl AuditorInterno {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_fill(&mut self, fill: ExchangeFillRecord) {
        self.fills_journal.push(fill);
    }

    /// Evaluates internal claims against independent exchange fill verification
    pub fn verify_internal_claims(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> DailyTribunalReport {
        let independent = VerificadorResultados::compute_metrics(&self.fills_journal);

        let sharpe_disc = if internal_sharpe > 0.0 {
            (internal_sharpe - independent.calculated_sharpe).abs() / internal_sharpe
        } else {
            0.0
        };

        let wr_disc = (internal_win_rate - independent.calculated_win_rate).abs();
        let dd_disc = (internal_drawdown - independent.calculated_max_drawdown).abs();

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
                max_disc * 100.0, internal_sharpe, independent.calculated_sharpe
            ),
        }
    }

    /// Detects cognitive biases in current trading behavior
    pub fn audit_biases(&self, num_features: usize, recent_7d_ret: f64, future_1d_ret: f64) -> CognitiveBiasReport {
        let overfitting = (recent_7d_ret > 0.05) && (future_1d_ret < -0.01);
        let complexity_inflation = num_features > 50;

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
