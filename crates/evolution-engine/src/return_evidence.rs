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

/// LF-3: pago máximo posible en R-múltiplos para la geometría `(tp, sl)` de
/// cada horizonte dado: `max(tp/sl)` sobre los pares finitos y positivos, nunca
/// menor que 1 (un TP no puede pagar menos de lo que arriesga el soporte por
/// abajo). Con las curvas TP/SL del genoma, potencias de τ, el cociente es
/// monótono en ln τ y basta evaluarlo en los dos extremos de la banda.
pub fn soporte_superior_en_r(pares_tp_sl: &[(f64, f64)]) -> f64 {
    pares_tp_sl
        .iter()
        .filter(|(tp, sl)| tp.is_finite() && sl.is_finite() && *tp > 0.0 && *sl > 0.0)
        .map(|(tp, sl)| tp / sl)
        .filter(|rr| rr.is_finite())
        .fold(1.0_f64, f64::max)
}

/// LF-3: vigilancia anytime-valid de la DEGRADACIÓN de un genoma promovido.
///
/// El vigilante anterior alimentaba un e-proceso que apuesta A FAVOR del
/// genoma y revertía cuando su riqueza caía un 50 % desde el pico. Ese
/// criterio no es una prueba: la riqueza de cualquier e-proceso con varianza
/// sufre tarde o temprano una caída del 50 % desde su máximo, también con
/// ventaja positiva (Monte Carlo, 400 trayectorias i.i.d. de RR 2,25: revierte
/// el 100 % de los genomas con 35 % y con 40 % de acierto, esperanza +0,14 R
/// y +0,30 R). Además recortaba cada TP a +1 R, con lo que vigilaba la tasa de
/// acierto contra 0,5 y no la esperanza.
///
/// Aquí la hipótesis nula es la que protege al genoma, H0: E[X] ≥ 0, con X el
/// retorno de la operación en R-múltiplos sobre el soporte [−1, `soporte_r`].
/// Se apuesta EN CONTRA: Y = −X / soporte_r ∈ [−1/… , 1] cumple el contrato de
/// `VilleEProcess` (H0: E[Y] ≤ 0, Y ≥ −1) y la degradación se declara cuando
/// su riqueza supera 1/α. Por la desigualdad maximal de Ville, la probabilidad
/// de revertir un genoma con esperanza no negativa es ≤ α en cualquier
/// instante de inspección.
#[derive(Clone, Debug, PartialEq)]
pub struct VigilanciaDegradacion {
    proceso: VilleEProcess,
    soporte_r: f64,
}

impl VigilanciaDegradacion {
    /// `soporte_r` es el pago máximo posible en R-múltiplos (≥ 1): el TP/SL
    /// máximo de la geometría vigente (`soporte_superior_en_r`).
    pub fn new(
        alpha: f64,
        lambda_min: f64,
        lambda_max: f64,
        soporte_r: f64,
    ) -> Result<Self, &'static str> {
        if !soporte_r.is_finite() || soporte_r < 1.0 {
            return Err("soporte_r debe ser finito y mayor o igual que 1.0");
        }
        let proceso = VilleEProcess::with_bounds(alpha, lambda_min, lambda_max)?;
        Ok(Self { proceso, soporte_r })
    }

    /// Ingiere el retorno de una operación en R-múltiplos. Se acota al soporte
    /// [−1, soporte_r]; un valor no finito se ignora.
    pub fn observe_r(&mut self, x_r: f64) {
        if !x_r.is_finite() {
            return;
        }
        let x = x_r.clamp(-1.0, self.soporte_r);
        self.proceso.update(-x / self.soporte_r);
    }

    /// Evidencia anytime-valid de que la esperanza del genoma es negativa:
    /// la riqueza de la apuesta en contra alcanzó 1/α en algún instante.
    pub fn degradada(&self) -> bool {
        self.proceso.peak_e_value >= self.proceso.threshold
    }

    /// Riqueza actual de la apuesta en contra del genoma.
    pub fn e_valor(&self) -> f64 {
        self.proceso.e_value
    }

    /// Observaciones admitidas.
    pub fn observaciones(&self) -> usize {
        self.proceso.count
    }

    pub fn soporte_r(&self) -> f64 {
        self.soporte_r
    }
}
