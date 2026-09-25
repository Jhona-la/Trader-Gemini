//! Research-only sampled-path evidence. Not an execution/fill simulator.
//! File timestamps are explicitly declared milliseconds by the caller.
//! A finite grid of requested horizons is not a proof of continuous support.
use serde::Serialize;

pub const RECORD_BYTES: usize = 40;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    TradeDerivedUnverifiedBook,
    CandleSynthetic,
    LegacyUnknown,
}

#[derive(Debug, Clone, Copy)]
pub struct Record {
    pub timestamp_ms: u64,
    pub bid: f64,
    pub ask: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}
impl Record {
    pub fn mid(self) -> f64 {
        self.bid + (self.ask - self.bid) * 0.5
    }
}

/// Validated immutable borrowed bytes; no alignment cast or numeric imputation.
pub struct Tape<'a> {
    payload: &'a [u8],
    pub provenance: Provenance,
}
impl<'a> Tape<'a> {
    pub fn parse(bytes: &'a [u8], allow_legacy: bool) -> Result<Self, String> {
        let (payload, provenance) = if bytes.starts_with(b"TGMTICK1") {
            (&bytes[8..], Provenance::TradeDerivedUnverifiedBook)
        } else if bytes.starts_with(b"TGMSYNT1") {
            (&bytes[8..], Provenance::CandleSynthetic)
        } else if bytes.starts_with(b"TGM") || !allow_legacy {
            return Err("unknown/missing tick header; legacy requires explicit opt-in".into());
        } else {
            (bytes, Provenance::LegacyUnknown)
        };
        if payload.is_empty() || payload.len() % RECORD_BYTES != 0 {
            return Err("empty or truncated 40-byte little-endian tick payload".into());
        }
        let tape = Self {
            payload,
            provenance,
        };
        let mut previous = None;
        for i in 0..tape.len() {
            let t = tape.get(i).unwrap();
            if ![t.bid, t.ask, t.bid_qty, t.ask_qty]
                .iter()
                .all(|x| x.is_finite())
                || t.bid <= 0.0
                || t.ask < t.bid
                || t.bid_qty < 0.0
                || t.ask_qty < 0.0
                || !(t.bid_qty + t.ask_qty).is_finite()
                || previous.is_some_and(|p| t.timestamp_ms < p)
            {
                return Err(format!(
                    "invalid market values or timestamp order at record {i}"
                ));
            }
            previous = Some(t.timestamp_ms);
        }
        Ok(tape)
    }
    pub fn len(&self) -> usize {
        self.payload.len() / RECORD_BYTES
    }
    pub fn is_empty(&self) -> bool {
        self.payload.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<Record> {
        let start = index.checked_mul(RECORD_BYTES)?;
        let b = self.payload.get(start..start.checked_add(RECORD_BYTES)?)?;
        let value = |i| f64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Some(Record {
            timestamp_ms: u64::from_le_bytes(b[..8].try_into().unwrap()),
            bid: value(8),
            ask: value(16),
            bid_qty: value(24),
            ask_qty: value(32),
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BarrierSpec {
    /// Positive fractional change in entry midpoint; not net profit or a fill.
    pub take_profit_return: f64,
    pub stop_loss_return: f64,
}
impl BarrierSpec {
    pub fn validate(self) -> Result<(), String> {
        if !self.take_profit_return.is_finite()
            || !self.stop_loss_return.is_finite()
            || self.take_profit_return <= 0.0
            || self.take_profit_return >= 1.0
            || self.stop_loss_return <= 0.0
            || self.stop_loss_return >= 1.0
        {
            return Err(
                "midpoint barrier returns must be finite and strictly between 0 and 1".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    TakeProfit,
    StopLoss,
    NoObservedHit,
    RightCensored,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Outcome {
    pub kind: OutcomeKind,
    pub event_time_ms: Option<u64>,
    pub event_record: Option<usize>,
    /// Availability of this outcome, including horizon-coverage confirmation.
    pub information_end_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HorizonEvidence {
    pub horizon_ms: u64,
    pub deadline_ms: u64,
    pub horizon_covered: bool,
    pub confirmation_time_ms: Option<u64>,
    pub last_observation_ms: u64,
    pub largest_observed_gap_ms: u64,
    /// Return of the last sampled midpoint <= deadline; not an interpolated fill.
    pub last_observed_return: f64,
    pub long: Outcome,
    pub short: Outcome,
    /// Conservative end for the entire row, not just an early barrier hit.
    pub information_end_ms: u64,
}

pub fn validate_horizons(horizons: &[u64]) -> Result<(), String> {
    if horizons.is_empty() || horizons[0] == 0 || horizons.windows(2).any(|p| p[0] >= p[1]) {
        return Err("horizons must be positive, strictly increasing milliseconds".into());
    }
    Ok(())
}

/// One forward scan across all requested horizons for this anchor.
/// Outcomes refer only to observations in tape order, not an unobserved path.
/// Budget counts both record visits and emitted horizons; exhaustion is an error.
pub fn label_surface(
    tape: &Tape<'_>,
    anchor: usize,
    horizons: &[u64],
    spec: BarrierSpec,
    remaining_work: &mut u64,
) -> Result<Vec<HorizonEvidence>, String> {
    spec.validate()?;
    validate_horizons(horizons)?;
    let entry = tape.get(anchor).ok_or("anchor out of range")?;
    entry
        .timestamp_ms
        .checked_add(*horizons.last().unwrap())
        .ok_or("deadline overflow")?;
    let p0 = entry.mid();
    let mut next = anchor + 1;
    let mut last = entry;
    let mut gap = 0;
    let mut ret = 0.0;
    let mut long_hit: Option<Outcome> = None;
    let mut short_hit: Option<Outcome> = None;
    let mut result = Vec::new();
    for &horizon_ms in horizons {
        *remaining_work = remaining_work
            .checked_sub(1)
            .ok_or("label work budget exhausted")?;
        let deadline_ms = entry.timestamp_ms + horizon_ms;
        while let Some(t) = tape.get(next) {
            if t.timestamp_ms > deadline_ms {
                break;
            }
            *remaining_work = remaining_work
                .checked_sub(1)
                .ok_or("label work budget exhausted")?;
            gap = gap.max(t.timestamp_ms - last.timestamp_ms);
            ret = (t.mid() - p0) / p0;
            if !ret.is_finite() {
                return Err("nonfinite midpoint return".into());
            }
            let event = |kind| Outcome {
                kind,
                event_time_ms: Some(t.timestamp_ms),
                event_record: Some(next),
                information_end_ms: t.timestamp_ms,
            };
            if long_hit.is_none() {
                if ret >= spec.take_profit_return {
                    long_hit = Some(event(OutcomeKind::TakeProfit));
                } else if ret <= -spec.stop_loss_return {
                    long_hit = Some(event(OutcomeKind::StopLoss));
                }
            }
            if short_hit.is_none() {
                if ret <= -spec.take_profit_return {
                    short_hit = Some(event(OutcomeKind::TakeProfit));
                } else if ret >= spec.stop_loss_return {
                    short_hit = Some(event(OutcomeKind::StopLoss));
                }
            }
            last = t;
            next += 1;
        }
        // A later timestamp confirms coverage, but its price is never read into
        // this horizon's return/barrier outcomes. Include its time for purging.
        let confirmation = if last.timestamp_ms == deadline_ms {
            Some(last.timestamp_ms)
        } else {
            tape.get(next).map(|t| t.timestamp_ms)
        };
        let covered = confirmation.is_some();
        let info_end = confirmation.unwrap_or(last.timestamp_ms);
        let absent = Outcome {
            kind: if covered {
                OutcomeKind::NoObservedHit
            } else {
                OutcomeKind::RightCensored
            },
            event_time_ms: None,
            event_record: None,
            information_end_ms: info_end,
        };
        result.push(HorizonEvidence {
            horizon_ms,
            deadline_ms,
            horizon_covered: covered,
            confirmation_time_ms: confirmation,
            last_observation_ms: last.timestamp_ms,
            largest_observed_gap_ms: gap.max(info_end - last.timestamp_ms),
            last_observed_return: ret,
            long: long_hit.unwrap_or(absent),
            short: short_hit.unwrap_or(absent),
            information_end_ms: info_end,
        });
    }
    Ok(result)
}
