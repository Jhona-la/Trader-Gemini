//! LG5: empirical net return per closed entry notional, never a first-hit label.
//!
//! One bounded, uniformly weighted window per coin, queried by side and source.
//! Scores, continuous horizons and initial targets are frozen provenance, NOT
//! conditioning variables. This is descriptive evidence, not an optimal Kelly
//! fraction, a confidence bound or proof of exchange settlement.
use crate::calibration::WINDOW;
use crate::outcome_context::OutcomeContext;
use quantum_arena::position::MAX_SPECTRAL_SLOTS;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use strategy_core::{NetPayoffEstimate, NetPayoffSource, SignalType};

pub const WATERMARK_COUNT: usize = MAX_SPECTRAL_SLOTS * 2;

#[derive(Serialize, Deserialize)]
#[serde(remote = "NetPayoffSource")]
enum StoredSource {
    LocalSimulation,
    ExchangeLocalEstimate,
}

pub fn source_for(context: OutcomeContext) -> NetPayoffSource {
    match context {
        OutcomeContext::IsolatedSimulation => NetPayoffSource::LocalSimulation,
        OutcomeContext::ExchangeLocalEstimate => NetPayoffSource::ExchangeLocalEstimate,
    }
}

/// Slot identity is supplied separately from the signal/learning ledger.
/// No heap allocation is needed to capture or consume this record.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EntryPayoffEvidence {
    pub slot: usize,
    pub generation: u64,
    pub is_long: bool,
    pub entry_time_ms: u64,
    pub score: f64,
    pub tau_ms: u64,
    pub entry_price: f64,
    pub initial_tp_price: f64,
    pub initial_sl_price: f64,
    #[serde(with = "StoredSource")]
    pub source: NetPayoffSource,
}

impl EntryPayoffEvidence {
    pub fn is_valid(&self) -> bool {
        self.slot < MAX_SPECTRAL_SLOTS
            && self.score.is_finite()
            && (0.0..=1.0).contains(&self.score)
            && self.tau_ms > 0
            && [
                self.entry_price,
                self.initial_tp_price,
                self.initial_sl_price,
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
            && if self.is_long {
                self.initial_tp_price > self.entry_price && self.initial_sl_price < self.entry_price
            } else {
                self.initial_tp_price < self.entry_price && self.initial_sl_price > self.entry_price
            }
    }

    fn key(&self) -> PayoffKey {
        PayoffKey {
            entry_time_ms: self.entry_time_ms,
            generation: self.generation,
        }
    }

    fn watermark_index(&self) -> usize {
        self.slot
            + match self.source {
                NetPayoffSource::LocalSimulation => 0,
                NetPayoffSource::ExchangeLocalEstimate => MAX_SPECTRAL_SLOTS,
            }
    }
}

/// Ordering is opening event time first, generation second. Slots and sources
/// have independent watermarks; one concurrent trade cannot retire another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PayoffKey {
    pub entry_time_ms: u64,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PayoffObservation {
    pub entry: EntryPayoffEvidence,
    pub closed_entry_price: f64,
    pub closed_quantity: f64,
    pub net_pnl_usd: f64,
    pub close_time_ms: u64,
    pub observed_at_ms: u64,
}

impl PayoffObservation {
    /// Both fees already belong to net_pnl_usd. Neither margin nor leverage nor
    /// a live stop participates in this denominator. Partial fills use the
    /// actual entry price and quantity returned by the close transition.
    pub fn net_return_pct(&self) -> Option<f64> {
        if !self.entry.is_valid()
            || !self.closed_entry_price.is_finite()
            || self.closed_entry_price <= 0.0
            || !self.closed_quantity.is_finite()
            || self.closed_quantity <= 0.0
            || !self.net_pnl_usd.is_finite()
            || self.entry.entry_time_ms > self.close_time_ms
            || self.close_time_ms > self.observed_at_ms
        {
            return None;
        }
        let notional = self.closed_entry_price * self.closed_quantity;
        if !notional.is_finite() || notional <= 0.0 {
            return None;
        }
        let r = self.net_pnl_usd / notional;
        r.is_finite().then_some(r)
    }
}

/// Additive persisted state. Default means no measured magnitudes; legacy
/// Platt booleans cannot populate it. Watermarks and clock outlive the window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayoffSnapshot {
    pub rows: Vec<PayoffObservation>,
    pub clock_ms: Option<u64>,
    pub watermarks: Vec<Option<PayoffKey>>,
}

impl Default for PayoffSnapshot {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            clock_ms: None,
            watermarks: vec![None; WATERMARK_COUNT],
        }
    }
}

impl PayoffSnapshot {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.rows.len() > WINDOW || self.watermarks.len() != WATERMARK_COUNT {
            return Err("net_payoff window or watermark length");
        }
        if self.clock_ms.is_none()
            && (!self.rows.is_empty() || self.watermarks.iter().any(Option::is_some))
        {
            return Err("net_payoff missing evidence clock");
        }
        for key in self.watermarks.iter().flatten() {
            if self.clock_ms.is_none_or(|clock| key.entry_time_ms > clock) {
                return Err("net_payoff watermark after evidence clock");
            }
        }
        for (i, row) in self.rows.iter().enumerate() {
            if row.net_return_pct().is_none() {
                return Err("net_payoff invalid numeric, geometry or time context");
            }
            if self.clock_ms.is_none_or(|clock| row.observed_at_ms > clock) {
                return Err("net_payoff row after evidence clock");
            }
            if self.watermarks[row.entry.watermark_index()].is_none_or(|key| row.entry.key() > key)
            {
                return Err("net_payoff missing or regressed slot watermark");
            }
            if self.rows[..i].iter().any(|old| {
                old.entry.watermark_index() == row.entry.watermark_index()
                    && old.entry.key() == row.entry.key()
            }) {
                return Err("net_payoff duplicate slot outcome");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NetPayoffModel {
    coin_id: usize,
    rows: VecDeque<PayoffObservation>,
    clock_ms: Option<u64>,
    watermarks: [Option<PayoffKey>; WATERMARK_COUNT],
}

impl NetPayoffModel {
    pub fn new(coin_id: usize) -> Self {
        Self {
            coin_id,
            rows: VecDeque::with_capacity(WINDOW),
            clock_ms: None,
            watermarks: [None; WATERMARK_COUNT],
        }
    }

    /// Admission is all-or-nothing and allocation-free after construction.
    /// A repeated/older occupant of the SAME slot/source cannot be counted
    /// again, including after its observation has left the bounded window.
    pub fn observe(
        &mut self,
        entry: EntryPayoffEvidence,
        net_pnl_usd: f64,
        closed_entry_price: f64,
        closed_quantity: f64,
        close_time_ms: u64,
        observed_at_ms: u64,
    ) -> bool {
        let row = PayoffObservation {
            entry,
            closed_entry_price,
            closed_quantity,
            net_pnl_usd,
            close_time_ms,
            observed_at_ms,
        };
        if row.net_return_pct().is_none() {
            return false;
        }
        let index = entry.watermark_index();
        let key = entry.key();
        if self.watermarks[index].is_some_and(|previous| key <= previous) {
            return false;
        }
        if self.rows.len() == WINDOW {
            self.rows.pop_front();
        }
        self.rows.push_back(row);
        self.watermarks[index] = Some(key);
        self.clock_ms = Some(
            self.clock_ms
                .map_or(observed_at_ms, |old| old.max(observed_at_ms)),
        );
        true
    }

    /// Uniform empirical mean and population standard deviation of matching
    /// rows, pooled over frozen score/tau/geometry. No extrapolated global
    /// fallback, prior, clipping, confidence bound or inferred first-hit label.
    /// Query and all moment passes allocate no memory.
    pub fn estimate(
        &self,
        coin_id: usize,
        side: SignalType,
        source: NetPayoffSource,
        as_of_ms: u64,
    ) -> Option<NetPayoffEstimate> {
        if coin_id != self.coin_id
            || side == SignalType::Flat
            || self.clock_ms.is_none_or(|clock| as_of_ms < clock)
        {
            return None;
        }
        let selected = |row: &&PayoffObservation| {
            row.entry.is_long == (side == SignalType::Long)
                && row.entry.source == source
                && row.observed_at_ms <= as_of_ms
        };
        let mut sample_count = 0;
        let mut sample_start_ms = u64::MAX;
        let mut sample_end_ms = 0;
        let mut scale: f64 = 0.0;
        for row in self.rows.iter().filter(selected) {
            sample_count += 1;
            sample_start_ms = sample_start_ms.min(row.close_time_ms);
            sample_end_ms = sample_end_ms.max(row.close_time_ms);
            scale = scale.max(row.net_return_pct()?.abs());
        }
        if sample_count == 0 {
            return None;
        }
        let (mean_net_return_pct, standard_deviation) = if scale == 0.0 {
            (0.0, 0.0)
        } else {
            // Normalize for numerical range only; every observation keeps unit
            // statistical weight. No dollar weighting or return clipping.
            let sum: f64 = self
                .rows
                .iter()
                .filter(selected)
                .map(|row| row.net_return_pct().unwrap() / scale)
                .sum();
            let normalized_mean = sum / sample_count as f64;
            let sq: f64 = self
                .rows
                .iter()
                .filter(selected)
                .map(|row| {
                    let delta = row.net_return_pct().unwrap() / scale - normalized_mean;
                    delta * delta
                })
                .sum();
            (
                scale * normalized_mean,
                scale * (sq / sample_count as f64).sqrt(),
            )
        };
        let estimate = NetPayoffEstimate {
            mean_net_return_pct,
            standard_deviation,
            sample_count,
            coin_id,
            side,
            sample_start_ms,
            sample_end_ms,
            as_of_ms,
            source,
        };
        estimate.is_valid_for(coin_id, side).then_some(estimate)
    }

    pub fn snapshot(&self) -> PayoffSnapshot {
        PayoffSnapshot {
            rows: self.rows.iter().copied().collect(),
            clock_ms: self.clock_ms,
            watermarks: self.watermarks.to_vec(),
        }
    }

    /// Reject the entire input before touching either evidence or watermarks.
    pub fn restore(&mut self, snapshot: &PayoffSnapshot) -> Result<(), &'static str> {
        snapshot.validate()?;
        self.rows.clear();
        self.rows.extend(snapshot.rows.iter().copied());
        self.clock_ms = snapshot.clock_ms;
        self.watermarks.copy_from_slice(&snapshot.watermarks);
        Ok(())
    }
}
