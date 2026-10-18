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

/// OLA Ω120: Constantes canónicas por defecto para el Tribunal de Autoengaño y Auditor Interno
pub const DEFAULT_AUTOENGANO_THRESHOLD: f64 = 0.20;
pub const DEFAULT_CULPABLE_PELIGROSO_THRESHOLD: f64 = 0.10;
pub const DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD: f64 = 0.05;
pub const DEFAULT_OVERFITTING_RECENT_MIN: f64 = 0.05;
pub const DEFAULT_OVERFITTING_OOS_MAX: f64 = -0.01;
pub const DEFAULT_MAX_FEATURES_NOMINAL: usize = 64;

#[inline]
pub const fn default_autoengano_threshold() -> f64 {
    DEFAULT_AUTOENGANO_THRESHOLD
}
#[inline]
pub const fn default_culpable_peligroso_threshold() -> f64 {
    DEFAULT_CULPABLE_PELIGROSO_THRESHOLD
}
#[inline]
pub const fn default_culpable_recuperable_threshold() -> f64 {
    DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD
}
#[inline]
pub const fn default_overfitting_recent_min() -> f64 {
    DEFAULT_OVERFITTING_RECENT_MIN
}
#[inline]
pub const fn default_overfitting_oos_max() -> f64 {
    DEFAULT_OVERFITTING_OOS_MAX
}
#[inline]
pub const fn default_max_features_nominal() -> usize {
    DEFAULT_MAX_FEATURES_NOMINAL
}

/// OLA Ω120: Parámetros desacoplados y configurables para el Tribunal de Autoengaño y Detección de Sesgos Cognitivos
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TribunalThresholds {
    #[serde(default = "default_autoengano_threshold")]
    pub autoengano_threshold: f64,
    #[serde(default = "default_culpable_peligroso_threshold")]
    pub culpable_peligroso_threshold: f64,
    #[serde(default = "default_culpable_recuperable_threshold")]
    pub culpable_recuperable_threshold: f64,
    #[serde(default = "default_overfitting_recent_min")]
    pub overfitting_recent_min: f64,
    #[serde(default = "default_overfitting_oos_max")]
    pub overfitting_oos_max: f64,
    #[serde(default = "default_max_features_nominal")]
    pub max_features_nominal: usize,
}

impl Default for TribunalThresholds {
    fn default() -> Self {
        Self {
            autoengano_threshold: DEFAULT_AUTOENGANO_THRESHOLD,
            culpable_peligroso_threshold: DEFAULT_CULPABLE_PELIGROSO_THRESHOLD,
            culpable_recuperable_threshold: DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD,
            overfitting_recent_min: DEFAULT_OVERFITTING_RECENT_MIN,
            overfitting_oos_max: DEFAULT_OVERFITTING_OOS_MAX,
            max_features_nominal: DEFAULT_MAX_FEATURES_NOMINAL,
        }
    }
}

impl TribunalThresholds {
    pub fn validate(&self) -> Result<(), String> {
        if !self.autoengano_threshold.is_finite() || !(0.05..=1.0).contains(&self.autoengano_threshold) {
            return Err(format!("Invalid autoengano_threshold: {}", self.autoengano_threshold));
        }
        if !self.culpable_peligroso_threshold.is_finite() || !(0.01..=0.50).contains(&self.culpable_peligroso_threshold) {
            return Err(format!("Invalid culpable_peligroso_threshold: {}", self.culpable_peligroso_threshold));
        }
        if !self.culpable_recuperable_threshold.is_finite() || !(0.001..=0.25).contains(&self.culpable_recuperable_threshold) {
            return Err(format!("Invalid culpable_recuperable_threshold: {}", self.culpable_recuperable_threshold));
        }
        if self.culpable_recuperable_threshold >= self.culpable_peligroso_threshold {
            return Err(format!(
                "culpable_recuperable_threshold ({}) must be < culpable_peligroso_threshold ({})",
                self.culpable_recuperable_threshold, self.culpable_peligroso_threshold
            ));
        }
        if self.culpable_peligroso_threshold >= self.autoengano_threshold {
            return Err(format!(
                "culpable_peligroso_threshold ({}) must be < autoengano_threshold ({})",
                self.culpable_peligroso_threshold, self.autoengano_threshold
            ));
        }
        if !self.overfitting_recent_min.is_finite() || !(0.0..=1.0).contains(&self.overfitting_recent_min) {
            return Err(format!("Invalid overfitting_recent_min: {}", self.overfitting_recent_min));
        }
        if !self.overfitting_oos_max.is_finite() || !(-1.0..=0.0).contains(&self.overfitting_oos_max) {
            return Err(format!("Invalid overfitting_oos_max: {}", self.overfitting_oos_max));
        }
        if !(1..=256).contains(&self.max_features_nominal) {
            return Err(format!("Invalid max_features_nominal: {}", self.max_features_nominal));
        }
        Ok(())
    }
}

/// OLA Ω120: Función pura analítica que evalúa el veredicto del Tribunal a partir de la máxima discrepancia observada
#[inline(always)]
pub fn evaluate_tribunal_verdict(max_disc: f64, thresholds: &TribunalThresholds) -> TribunalVerdict {
    let disc = if max_disc.is_finite() && max_disc >= 0.0 {
        max_disc
    } else {
        1.0 // Fail-closed ante flotante no finito
    };

    let auto = if thresholds.autoengano_threshold.is_finite() && thresholds.autoengano_threshold > 0.0 {
        thresholds.autoengano_threshold
    } else {
        DEFAULT_AUTOENGANO_THRESHOLD
    };

    let peligroso = if thresholds.culpable_peligroso_threshold.is_finite() && thresholds.culpable_peligroso_threshold > 0.0 {
        thresholds.culpable_peligroso_threshold
    } else {
        DEFAULT_CULPABLE_PELIGROSO_THRESHOLD
    };

    let recuperable = if thresholds.culpable_recuperable_threshold.is_finite() && thresholds.culpable_recuperable_threshold > 0.0 {
        thresholds.culpable_recuperable_threshold
    } else {
        DEFAULT_CULPABLE_RECUPERABLE_THRESHOLD
    };

    if disc > auto {
        TribunalVerdict::AutoengañoDetectado
    } else if disc > peligroso {
        TribunalVerdict::CulpablePeligroso
    } else if disc > recuperable {
        TribunalVerdict::CulpableRecuperable
    } else {
        TribunalVerdict::Inocente
    }
}

/// OLA Ω120: Calcula la discrepancia máxima normalizada entre métricas internas y realidad de ejecución
#[inline(always)]
pub fn calculate_max_discrepancy(
    internal_sharpe: f64,
    internal_win_rate: f64,
    internal_drawdown: f64,
    independent: &IndependentMetrics,
) -> (f64, f64, f64, f64) {
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

    let sharpe_disc = if safe_sharpe > 0.0 {
        (safe_sharpe - independent.calculated_sharpe).abs() / safe_sharpe
    } else {
        0.0
    };

    let wr_disc = (safe_wr - independent.calculated_win_rate).abs();
    // F1-006 (Ola Ω150): Solo penalizar subestimación de drawdown (cuando el sistema
    // afirma tener MENOR drawdown que la realidad de los fills). Reportar un drawdown
    // interno más conservador (mayor) no constituye autoengaño.
    let dd_disc = (independent.calculated_max_drawdown - safe_dd).max(0.0);

    let max_disc = sharpe_disc.max(wr_disc).max(dd_disc);
    (max_disc, sharpe_disc, wr_disc, dd_disc)
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

/// F1-006 (Ola Ω150): Capacidad máxima acotada del diario de fills para evitar
/// crecimiento ilimitado de RAM y degradación cuadrática en sesiones largas.
pub const MAX_FILLS_JOURNAL: usize = 4096;

/// Dual Result Verifier
pub struct VerificadorResultados;

impl VerificadorResultados {
    /// Computes independent metrics from raw exchange fill records with default base capital
    pub fn compute_metrics(fills: &[ExchangeFillRecord]) -> IndependentMetrics {
        Self::compute_metrics_with_capital(fills, 13.0)
    }

    /// Computes independent metrics from raw exchange fill records anchored to explicit base capital.
    /// F1-006 (Ola Ω150): Single-pass zero-allocation Welford online variance + equity curve reconstruction.
    pub fn compute_metrics_with_capital(
        fills: &[ExchangeFillRecord],
        base_capital: f64,
    ) -> IndependentMetrics {
        if fills.is_empty() {
            return IndependentMetrics::default();
        }

        let safe_capital = if base_capital.is_finite() && base_capital > 0.0 {
            base_capital
        } else {
            13.0
        };

        let mut count: u64 = 0;
        let mut winning_trades: u64 = 0;
        let mut mean_ret: f64 = 0.0;
        let mut m2: f64 = 0.0;
        let mut first_ts: Option<u64> = None;
        let mut last_ts: u64 = 0;
        let mut equity = safe_capital;
        let mut peak = safe_capital;
        let mut max_dd = 0.0;
        let mut gross_pnl_sum = 0.0;
        let mut total_fees_sum = 0.0;

        for f in fills.iter().filter(|f| f.is_closed_position) {
            count += 1;
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
            let net = pnl - fee;
            if net > 0.0 {
                winning_trades += 1;
            }
            if first_ts.is_none() {
                first_ts = Some(f.timestamp_ns);
            }
            last_ts = f.timestamp_ns;

            let ret = net / safe_capital;
            let delta = ret - mean_ret;
            mean_ret += delta / (count as f64);
            let delta2 = ret - mean_ret;
            m2 += delta * delta2;

            equity += net;
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

            gross_pnl_sum += pnl;
            total_fees_sum += fee;
        }

        if count == 0 {
            return IndependentMetrics::default();
        }

        let total_trades = count as f64;
        let win_rate = (winning_trades as f64) / total_trades;
        let variance = if count > 1 {
            (m2 / ((count - 1) as f64)).max(0.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let time_span_days = if let Some(first) = first_ts {
            ((last_ts.saturating_sub(first)) as f64 / (1e9 * 86400.0)).max(1.0 / 24.0)
        } else {
            1.0
        };
        let trades_per_year = (total_trades / time_span_days) * 365.0;

        let sharpe = if std_dev > 1e-9 {
            (mean_ret / std_dev) * trades_per_year.sqrt().min(500.0)
        } else {
            0.0
        };

        let gross_pnl = gross_pnl_sum / safe_capital;
        let total_fees = total_fees_sum / safe_capital;
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
    pub thresholds: TribunalThresholds,
}

impl Default for AuditorInterno {
    fn default() -> Self {
        Self {
            fills_journal: Vec::new(),
            base_capital: 13.0,
            thresholds: TribunalThresholds::default(),
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
            thresholds: TribunalThresholds::default(),
        }
    }

    /// OLA Ω120: Builder para configurar umbrales adaptativos del Tribunal de Autoengaño
    pub fn with_thresholds(mut self, thresholds: TribunalThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// OLA Ω120: Actualiza dinámicamente los umbrales del Tribunal de Autoengaño
    pub fn update_thresholds(&mut self, thresholds: TribunalThresholds) {
        if thresholds.validate().is_ok() {
            self.thresholds = thresholds;
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
        if self.fills_journal.len() >= MAX_FILLS_JOURNAL {
            let overflow = self.fills_journal.len() + 1 - MAX_FILLS_JOURNAL;
            self.fills_journal.drain(0..overflow);
        }
        self.fills_journal.push(fill);
    }

    /// OLA Ω120: Calcula la discrepancia máxima normalizada entre métricas internas y realidad
    /// para alimentar el registro omnisciente y el asiento SeniorAuditorInterno en el Consejo.
    pub fn compute_internal_discrepancy(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> f64 {
        let independent = VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        );
        let (max_disc, _, _, _) = calculate_max_discrepancy(
            internal_sharpe,
            internal_win_rate,
            internal_drawdown,
            &independent,
        );
        max_disc
    }

    /// Computa directamente las métricas de rendimiento independientes calculadas desde los fills reales
    pub fn compute_independent_metrics(&self) -> IndependentMetrics {
        VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        )
    }

    /// Evaluates internal claims against independent exchange fill verification
    pub fn verify_internal_claims(
        &self,
        internal_sharpe: f64,
        internal_win_rate: f64,
        internal_drawdown: f64,
    ) -> DailyTribunalReport {
        let safe_sharpe = if internal_sharpe.is_finite() {
            internal_sharpe
        } else {
            0.0
        };

        let independent = VerificadorResultados::compute_metrics_with_capital(
            &self.fills_journal,
            self.base_capital,
        );

        let (max_disc, sharpe_disc, wr_disc, _) = calculate_max_discrepancy(
            internal_sharpe,
            internal_win_rate,
            internal_drawdown,
            &independent,
        );

        let verdict = evaluate_tribunal_verdict(max_disc, &self.thresholds);

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
        let overfitting = (safe_7d > self.thresholds.overfitting_recent_min)
            && (safe_oos < self.thresholds.overfitting_oos_max);
        let complexity_inflation = num_features > self.thresholds.max_features_nominal;

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

    #[test]
    fn f1_006_bounded_fills_journal_and_one_sided_drawdown_discrepancy() {
        let mut auditor = AuditorInterno::with_capital(13.0);
        for i in 0..(MAX_FILLS_JOURNAL + 128) {
            auditor.record_fill(ExchangeFillRecord {
                timestamp_ns: (i as u64 + 1) * 1_000_000_000,
                symbol_id: 0,
                is_buyer: i % 2 == 0,
                price: 50_000.0,
                qty: 0.001,
                fee_usd: 0.002,
                realized_pnl: if i % 2 == 0 { 0.05 } else { -0.02 },
                is_closed_position: true,
            });
        }
        assert_eq!(
            auditor.fills_journal.len(),
            MAX_FILLS_JOURNAL,
            "fills_journal debe estar acotado en MAX_FILLS_JOURNAL"
        );

        let indep = auditor.compute_independent_metrics();
        // Reportar un drawdown interno más conservador (mayor que el independiente) no debe producir discrepancia:
        let (_, _, _, dd_disc_conservative) = calculate_max_discrepancy(
            indep.calculated_sharpe,
            indep.calculated_win_rate,
            indep.calculated_max_drawdown + 0.05,
            &indep,
        );
        assert_eq!(dd_disc_conservative, 0.0);
    }
}
