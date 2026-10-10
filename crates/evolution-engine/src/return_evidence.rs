//! Descriptive evidence and one-way safety authority for realized-return batches.
//! No IID, Gaussian, time-uniform, or causal guarantee is inferred from a score.
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeanStatistic {
    /// mean / sample_std * sqrt(n); not an annualized Sharpe or a p-value.
    Studentized(f64),
    /// All values are equal. The Student statistic is undefined, not zero.
    Constant,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReturnEvidence {
    pub n: usize,
    pub mean: f64,
    pub sample_std: f64,
    pub statistic: MeanStatistic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceError {
    InsufficientObservations,
    NonFiniteObservation { index: usize },
    NumericalRange,
}

impl ReturnEvidence {
    pub fn studentized(self) -> Option<f64> {
        match self.statistic {
            MeanStatistic::Studentized(t) => Some(t),
            MeanStatistic::Constant => None,
        }
    }

    /// A constant negative observed batch is loss evidence, not a valid t-test.
    /// The consumer must enforce its sample policy and must not call this a p-value.
    pub fn is_constant_loss(self) -> bool {
        self.statistic == MeanStatistic::Constant && self.mean < 0.0
    }
}

/// Retains all tails and rejects the WHOLE batch on nonfinite observations.
/// Normalize first to avoid an absolute variance floor and square overflow.
/// Two-pass variance uses n-1; studentization uses sqrt(n), not sqrt(n-1).
pub fn summarize_returns(returns: &[f64]) -> Result<ReturnEvidence, EvidenceError> {
    if returns.len() < 2 {
        return Err(EvidenceError::InsufficientObservations);
    }
    let mut scale = 0.0_f64;
    for (index, &value) in returns.iter().enumerate() {
        if !value.is_finite() {
            return Err(EvidenceError::NonFiniteObservation { index });
        }
        scale = scale.max(value.abs());
    }
    if returns.iter().all(|&value| value == returns[0]) {
        return Ok(ReturnEvidence {
            n: returns.len(),
            mean: returns[0],
            sample_std: 0.0,
            statistic: MeanStatistic::Constant,
        });
    }
    let n = returns.len() as f64;
    // Compensated sum in normalized units limits cancellation error.
    let mut sum = 0.0;
    let mut correction = 0.0;
    for &value in returns {
        let corrected = value / scale - correction;
        let next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
    }
    let mean_scaled = sum / n;
    let variance_scaled = returns
        .iter()
        .map(|value| (value / scale - mean_scaled).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let std_scaled = variance_scaled.sqrt();
    let t = (mean_scaled / std_scaled) * n.sqrt();
    let mean = mean_scaled * scale;
    let sample_std = std_scaled * scale;
    if !t.is_finite()
        || !mean.is_finite()
        || !sample_std.is_finite()
        || std_scaled <= 0.0
        || sample_std <= 0.0
    {
        // A nonconstant batch that loses variance numerically is not Constant.
        return Err(EvidenceError::NumericalRange);
    }
    Ok(ReturnEvidence {
        n: returns.len(),
        mean,
        sample_std,
        statistic: MeanStatistic::Studentized(t),
    })
}

/// This is smoothing over NEW observation revisions, not accumulation of
/// independent evidence and not an anytime-valid test. Overlap still matters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvidenceEwma {
    last_revision: Option<u64>,
    value: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwmaError {
    InvalidScore,
    InvalidWeight,
}

impl EvidenceEwma {
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Duplicate/older snapshots are idempotent; zero is a valid initialized value.
    pub fn observe(&mut self, revision: u64, score: f64, weight: f64) -> Result<bool, EwmaError> {
        if !score.is_finite() {
            return Err(EwmaError::InvalidScore);
        }
        if !weight.is_finite() || weight <= 0.0 || weight > 1.0 {
            return Err(EwmaError::InvalidWeight);
        }
        if self
            .last_revision
            .is_some_and(|previous| revision <= previous)
        {
            return Ok(false);
        }
        let next = match self.value {
            Some(previous) => weight * score + (1.0 - weight) * previous,
            None => score,
        };
        if !next.is_finite() {
            return Err(EwmaError::InvalidScore);
        }
        self.value = Some(next);
        self.last_revision = Some(revision);
        Ok(true)
    }
}

/// May LATCH the shared stop, never clear it. Recovery needs the authority that
/// owns all stop reasons, which an aggregate strategy score does not provide.
/// Returns true only if this call changed false to true.
pub fn latch_degradation(stop: &AtomicBool, degraded: bool) -> bool {
    degraded && !stop.swap(true, Ordering::AcqRel)
}

use risk_engine::VilleEProcess;

/// Secuenciador de evidencia anytime-valid basado en supermartingalas de Ville (Ω31).
/// Inmune al sesgo de parada opcional (*optional stopping*) mediante la cota maximal de Ville:
///
/// $$\mathbb{P}\left(\sup_{t \ge 0} M_t \ge \frac{1}{\alpha}\right) \le \alpha$$
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialVilleEvidence {
    pub process: VilleEProcess,
}

impl SequentialVilleEvidence {
    pub fn new(alpha: f64) -> Result<Self, &'static str> {
        let process = VilleEProcess::new(alpha)?;
        Ok(Self { process })
    }

    pub fn with_bounds(alpha: f64, lambda_min: f64, lambda_max: f64) -> Result<Self, &'static str> {
        let process = VilleEProcess::with_bounds(alpha, lambda_min, lambda_max)?;
        Ok(Self { process })
    }

    /// Ingiere una nueva observación de retorno o innovación de señal.
    pub fn observe(&mut self, ret: f64) -> f64 {
        self.process.update(ret)
    }

    /// Certifica si el proceso ha superado el umbral de Ville $M_t \ge 1 / \alpha$.
    pub fn is_edge_certified(&self) -> bool {
        self.process.is_edge_certified()
    }

    /// Indica si el proceso ha agotado el capital de prueba o revertido bajo su baseline.
    pub fn is_exhausted(&self) -> bool {
        self.process.is_exhausted()
    }

    /// Drawdown fraccional de evidencia acumulada respecto a su máximo histórico en [0.0, 1.0].
    pub fn evidence_drawdown(&self) -> f64 {
        self.process.evidence_drawdown()
    }

    /// Indica si la evidencia estadística ha sufrido un deterioro mayor a `max_drawdown`.
    pub fn is_evidence_decayed(&self, max_drawdown: f64) -> bool {
        self.process.is_evidence_decayed(max_drawdown)
    }

    /// Cota superior del p-valor en cualquier momento secuencial.
    pub fn anytime_p_value(&self) -> f64 {
        self.process.anytime_p_value()
    }
}
