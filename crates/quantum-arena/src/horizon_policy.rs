//! Shared read policy for the existing runtime phenotype (FMT-134).
//!
//! These limits are compatibility/risk policy, not laws of market dynamics.
//! A horizon is a continuous coordinate in milliseconds; the two historical
//! anchors only parameterize power laws, not separate operating engines.
//! This module does not authorize an order or certify evidence at a scale.
use crate::temporal_spectrum::{HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};

/// Identifies the read formulas, not a transactional genome generation.
pub const HORIZON_READ_POLICY_VERSION: &str = "runtime-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonParameter {
    TakeProfit,
    StopLoss,
    Kelly,
    TrailMultiplier,
    TrailActivation,
    TrailStep,
    ObiThreshold,
}

impl HorizonParameter {
    /// Existing runtime bounds; None means no output clipping at this layer.
    pub const fn output_bounds(self) -> Option<(f64, f64)> {
        match self {
            Self::TakeProfit | Self::StopLoss => None,
            Self::Kelly => Some((0.01, 3.0)),
            Self::TrailMultiplier | Self::TrailActivation => Some((1.5, 6.0)),
            Self::TrailStep => Some((0.5, 4.0)),
            Self::ObiThreshold => Some((0.10, 0.95)),
        }
    }

    /// exp(a + b*ln(tau / 1 ms)), followed by the existing output bounds.
    /// The runtime floor at 1 ns is preserved, without adding an upper bound.
    /// This compatibility API is NOT an input validator: f64::max maps NaN
    /// and negative tau to the floor, and exponentiation can overflow.
    #[inline(always)]
    pub fn evaluate(self, curve: HorizonCurve, tau_ms: f64) -> f64 {
        self.evaluate_log(curve, tau_ms.max(1e-6).ln())
    }

    #[inline(always)]
    fn evaluate_log(self, curve: HorizonCurve, ln_tau: f64) -> f64 {
        let raw = (curve.a + curve.b * ln_tau).exp();
        match self.output_bounds() {
            Some((lo, hi)) => raw.clamp(lo, hi),
            None => raw,
        }
    }
}

/// Derived curves keep the existing serialization/anchor convention. They
/// must not override their authoritative scalar genes when a cache is stale.
#[inline]
pub(crate) fn curve_from_anchors(fast: f64, slow: f64) -> HorizonCurve {
    HorizonCurve::through_two_points(
        TAU_ANCHOR_FAST_MS,
        fast.max(0.001),
        TAU_ANCHOR_SLOW_MS,
        slow.max(0.001),
    )
}

/// Multipliers in ATR units. Compute ln(tau) once, as in the old runtime.
#[inline(always)]
pub(crate) fn trailing_at_tau(
    mult: HorizonCurve,
    act: HorizonCurve,
    step: HorizonCurve,
    tau_ms: f64,
) -> (f64, f64, f64, f64) {
    let ln_tau = tau_ms.max(1e-6).ln();
    let mult = HorizonParameter::TrailMultiplier.evaluate_log(mult, ln_tau);
    let act = HorizonParameter::TrailActivation.evaluate_log(act, ln_tau);
    let step = HorizonParameter::TrailStep.evaluate_log(step, ln_tau);
    let maximum = (act * 1.5).clamp(2.0, 7.0);
    (mult, act, step, maximum)
}
