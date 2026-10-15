//! Economic evidence has a target and units, independently of a win label.
use crate::types::SignalType;

/// These are local close observations, never an exchange-settlement certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetPayoffSource {
    LocalSimulation,
    ExchangeLocalEstimate,
}

/// Empirical mean of net PnL divided by each closed trade's entry notional.
///
/// The cohort is one coin, side and source. It pools historical scores,
/// continuous horizons and exit geometries; it does not claim conditional
/// calibration for the next order or a lower confidence bound. Fees already
/// included in the observations must not be charged a second time as base cost.
/// `Some` with a zero mean is evidence; `None` on the intent means unknown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetPayoffEstimate {
    pub mean_net_return_pct: f64,
    pub standard_deviation: f64,
    pub sample_count: usize,
    pub coin_id: usize,
    pub side: SignalType,
    pub sample_start_ms: u64,
    pub sample_end_ms: u64,
    pub as_of_ms: u64,
    pub source: NetPayoffSource,
}

impl NetPayoffEstimate {
    /// Validate the declared target's numeric, cohort and availability contract.
    /// This does not establish predictive accuracy or settlement provenance.
    #[inline]
    pub fn is_valid_for(&self, coin_id: usize, side: SignalType) -> bool {
        self.mean_net_return_pct.is_finite()
            && self.standard_deviation.is_finite()
            && self.standard_deviation >= 0.0
            && self.sample_count > 0
            && self.coin_id == coin_id
            && self.side == side
            && side != SignalType::Flat
            && self.sample_start_ms <= self.sample_end_ms
            && self.sample_end_ms <= self.as_of_ms
    }
}
