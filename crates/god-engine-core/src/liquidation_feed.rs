//! P-4 (PREDICTORES / MOTOR UNIVERSAL) — FEED DE LIQUIDACIONES.
//!
//! XXXIII operational path: checked USD-M snapshot -> instance/symbol-scoped
//! LiquidationState -> non-consuming as-of view shared by features and council.
//! No observation is not proof of a healthy feed. Scores are heuristic, not
//! probabilities. The existing default half-life is NOT genome-calibrated.
//!
//! Legacy compatibility buffer below is NOT the operational delivery path.
//! The following historical notes describe the old implementation only:
//! Historical design: canal global de severidad de liquidaciones. Los eventos
//! `!forceOrder` del WS (y cualquier otro productor de cascadas, p.ej. el
//! sniffer de Hyperliquid) depositan aquí su severidad normalizada; el
//! camino per-tick del core la consume UNA vez (swap) y la inyecta como
//! `dex_severity` a `update_macro_features` — el `ExponentialDecayTensor`
//! (dark_alpha) la decae con su constante genómica.
//!
//! POR QUÉ: `dex_severity` llevaba 0.0 constante desde el origen (sin
//! productor): la dimensión [9] del vector universal está zerificada a
//! ambos lados por paridad (el trainer no tiene historia de
//! liquidaciones), pero el TENSOR 54D de la NN, el registry y el futuro
//! asiento "Ente del Mercado" del consejo SÍ consumen la señal viva.
//!
//! SEMÁNTICA EVENT-DRIVEN: `bump` deposita max(pendiente, sev) — nunca
//! acumula niveles; el consumo es swap(0): cada evento alimenta un solo
//! apply_event del tensor de decaimiento. Severidad normalizada [0,1]:
//! |notional liquidado| / umbral de cascada (≈$1M), log-escalada.
use quantum_arena::atomic_float::AtomicF64;
use std::sync::atomic::Ordering;

static PENDING_SEVERITY: AtomicF64 = AtomicF64::new(0.0);

/// Deposita la severidad de un evento de liquidación (clamp [0,1]).
/// API histórica; el productor forceOrder operativo ya no la usa (XXXIII).
/// CAS-loop por si dos hilos depositan a la vez: gana el mayor.
pub fn bump(severity: f64) {
    if severity.is_finite() && severity > 0.0 {
        let sev = severity.min(1.0);
        let _ = PENDING_SEVERITY.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(if cur >= sev { cur } else { sev })
        });
    }
}

/// Consume la severidad pendiente (una sola lectura por evento).
#[inline]
pub fn take_pending() -> f64 {
    PENDING_SEVERITY.swap(0.0, Ordering::Relaxed)
}

/// Lee la severidad pendiente SIN consumirla — para observadores
/// (payload del consejo, telemetría) que no deben robarle el evento al
/// camino per-tick.
#[inline]
pub fn peek_pending() -> f64 {
    PENDING_SEVERITY.load(Ordering::Relaxed)
}

/// Normaliza un notional liquidado (USD) a severidad [0,1]:
/// log-escala contra umbral de cascada de $1M — $10k=2/3, $100k=5/6,
/// $1M=1.0 (cap). This is a heuristic relative to a 1 USD reference,
/// not an empirical percentile or a probability of cascade.
pub fn severity_from_notional(notional_usd: f64) -> f64 {
    if !notional_usd.is_finite() || notional_usd <= 0.0 {
        return 0.0;
    }
    (notional_usd.ln() / 1_000_000f64.ln()).clamp(0.0, 1.0)
}

/// Reported UM execution snapshot, not an incremental fill or complete market tape.
#[derive(Debug, Clone, PartialEq)]
pub struct LiquidationObservation {
    pub symbol: String,
    pub event_time_ms: u64,
    pub trade_time_ms: u64,
    pub is_buy: bool,
    pub reported_filled_notional: f64,
}

impl LiquidationObservation {
    fn validate(&self) -> Result<(), &'static str> {
        if self.symbol.is_empty()
            || self.trade_time_ms == 0
            || self.event_time_ms < self.trade_time_ms
            || !self.reported_filled_notional.is_finite()
            || self.reported_filled_notional < 0.0
        {
            Err("invalid liquidation observation")
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationUpdate {
    Accepted,
    Duplicate,
    Older,
    ConflictingTimestamp,
}

/// Decaying MAX envelope of reported snapshot scores. Never sums cumulative z.
/// x(t)=max_i s_i exp(-lambda*(t-E_i)); E is the exchange observation time.
/// It is a heuristic pressure score, not probability, intensity or total volume.
#[derive(Debug, Clone)]
pub struct LiquidationState {
    latest: LiquidationObservation,
    peak: f64,
    decay_lambda_per_ms: f64,
}

impl LiquidationState {
    pub fn new(
        observation: LiquidationObservation,
        decay_lambda_per_ms: f64,
    ) -> Result<Self, &'static str> {
        observation.validate()?;
        if !decay_lambda_per_ms.is_finite() || decay_lambda_per_ms <= 0.0 {
            return Err("invalid liquidation decay rate");
        }
        Ok(Self {
            peak: severity_from_notional(observation.reported_filled_notional),
            latest: observation,
            decay_lambda_per_ms,
        })
    }

    pub fn latest(&self) -> &LiquidationObservation {
        &self.latest
    }

    /// Repeated reads do not consume or refresh evidence. Earlier as-of clocks
    /// are invalid: never leak an observation from the future into a decision.
    pub fn severity_at(&self, as_of_ms: u64) -> Result<f64, &'static str> {
        if as_of_ms < self.latest.event_time_ms {
            return Err("liquidation evidence is in the future");
        }
        let age = (as_of_ms - self.latest.event_time_ms) as f64;
        Ok(self.peak * (-self.decay_lambda_per_ms * age).exp())
    }

    pub fn observe(
        &mut self,
        observation: LiquidationObservation,
        decay_lambda_per_ms: f64,
    ) -> Result<ObservationUpdate, &'static str> {
        observation.validate()?;
        if !decay_lambda_per_ms.is_finite() || decay_lambda_per_ms <= 0.0 {
            return Err("invalid liquidation decay rate");
        }
        if observation.symbol != self.latest.symbol {
            return Err("liquidation symbol mismatch");
        }
        if observation.event_time_ms < self.latest.event_time_ms {
            return Ok(ObservationUpdate::Older);
        }
        if observation.event_time_ms == self.latest.event_time_ms {
            if observation == self.latest {
                return Ok(ObservationUpdate::Duplicate);
            }
            // Millisecond equality is not order identity. Preserve the strongest
            // snapshot without summing it or advancing the clock; report ambiguity.
            self.peak = self
                .peak
                .max(severity_from_notional(observation.reported_filled_notional));
            self.latest = observation;
            return Ok(ObservationUpdate::ConflictingTimestamp);
        }
        let aged = self.severity_at(observation.event_time_ms)?;
        self.peak = aged.max(severity_from_notional(observation.reported_filled_notional));
        self.latest = observation;
        self.decay_lambda_per_ms = decay_lambda_per_ms;
        Ok(ObservationUpdate::Accepted)
    }
}

#[derive(Debug, Default)]
pub struct LiquidationDiagnostics {
    pub accepted: u64,
    pub duplicates: u64,
    pub older: u64,
    pub conflicting: u64,
    pub invalid: u64,
    pub unknown_symbol: u64,
    pub invalid_as_of: u64,
}
